"""Persistent, capacity-growing storage for compressed TurboPolar blocks."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import mlx.core as mx

from rfsn_v11.quant.polar.payload import PolarKeyBlock
from rfsn_v11.quant.v_quant.encoder import QuantizedVBlock


def _expand_block_dim(x: mx.array) -> mx.array:
    """Expand a 4D single-block tensor to 5D by adding a singleton block axis."""
    return mx.expand_dims(x, axis=2)


@dataclass
class PolarKBlockStorage:
    """Persistent storage for compressed key blocks.

    Buffers are allocated lazily on first append with capacity 1 and grow by
    exactly one block on each append that exceeds capacity. Valid blocks occupy
    indices [0, block_count).
    """
    radii: Optional[mx.array] = None
    angle_codes_l1: Optional[mx.array] = None
    angle_codes_deep: Optional[mx.array] = None
    radii_scales: Optional[mx.array] = None
    block_count: int = 0
    capacity: int = 0
    block_size: int = 0
    head_dim: int = 0
    metadata: dict = field(default_factory=dict)
    reallocation_count: int = 0
    bytes_copied_during_growth: int = 0

    def _init_buffers(self, block: PolarKeyBlock, initial_capacity: int):
        self.metadata = block.metadata
        self.block_size = block.block_size
        self.head_dim = block.head_dim
        radii = _expand_block_dim(block.radii)
        angle_l1 = _expand_block_dim(block.angle_codes_l1)
        angle_deep = _expand_block_dim(block.angle_codes_deep)

        def grow(x):
            shape = x.shape[:2] + (initial_capacity,) + x.shape[3:]
            zeros = mx.zeros(shape, dtype=x.dtype)
            zeros[:, :, 0:1, :, :] = x
            return zeros

        self.radii = grow(radii)
        self.angle_codes_l1 = grow(angle_l1)
        self.angle_codes_deep = grow(angle_deep)
        if block.radii_scales is not None:
            scales = _expand_block_dim(block.radii_scales)
            self.radii_scales = grow(scales)
        else:
            self.radii_scales = None
        self.capacity = initial_capacity
        self.block_count = 1

    def _grow(self):
        new_capacity = self.capacity + 1
        def grow(x):
            shape = x.shape[:2] + (1,) + x.shape[3:]
            extra = mx.zeros(shape, dtype=x.dtype)
            return mx.concatenate([x, extra], axis=2)
        self.bytes_copied_during_growth += int(self.radii.size * self.radii.itemsize)
        self.radii = grow(self.radii)
        self.bytes_copied_during_growth += int(self.angle_codes_l1.size * self.angle_codes_l1.itemsize)
        self.angle_codes_l1 = grow(self.angle_codes_l1)
        self.bytes_copied_during_growth += int(self.angle_codes_deep.size * self.angle_codes_deep.itemsize)
        self.angle_codes_deep = grow(self.angle_codes_deep)
        if self.radii_scales is not None:
            self.bytes_copied_during_growth += int(self.radii_scales.size * self.radii_scales.itemsize)
            self.radii_scales = grow(self.radii_scales)
        self.capacity = new_capacity
        self.reallocation_count += 1

    def append(self, block: PolarKeyBlock, initial_capacity: int = 1):
        if self.radii is None:
            self._init_buffers(block, initial_capacity)
            return
        if self.block_count >= self.capacity:
            self._grow()
        idx = self.block_count
        self.radii[:, :, idx:idx + 1, :, :] = _expand_block_dim(block.radii)
        self.angle_codes_l1[:, :, idx:idx + 1, :, :] = _expand_block_dim(block.angle_codes_l1)
        self.angle_codes_deep[:, :, idx:idx + 1, :, :] = _expand_block_dim(block.angle_codes_deep)
        if self.radii_scales is not None and block.radii_scales is not None:
            self.radii_scales[:, :, idx:idx + 1, :, :] = _expand_block_dim(block.radii_scales)
        self.block_count += 1

    def to_unified_block(self, shape: Tuple[int, ...]) -> PolarKeyBlock:
        """Return a PolarKeyBlock view of the valid compressed blocks."""
        if self.radii is None or self.block_count == 0:
            raise ValueError("No compressed blocks to materialize")
        return PolarKeyBlock(
            radii=self.radii[:, :, :self.block_count, :, :],
            angle_codes_l1=self.angle_codes_l1[:, :, :self.block_count, :, :],
            angle_codes_deep=self.angle_codes_deep[:, :, :self.block_count, :, :],
            radii_scales=self.radii_scales[:, :, :self.block_count, :, :] if self.radii_scales is not None else None,
            shape=shape,
            block_size=self.block_size,
            head_dim=self.head_dim,
            metadata=self.metadata,
        )


@dataclass
class QuantVBlockStorage:
    """Persistent storage for quantized value blocks.

    Buffers are allocated lazily on first append with capacity 1 and grow by
    exactly one block on each append that exceeds capacity. Valid blocks occupy
    indices [0, block_count).
    """
    codes: Optional[mx.array] = None
    scales: Optional[mx.array] = None
    group_size: int = 32
    block_count: int = 0
    capacity: int = 0
    reallocation_count: int = 0
    bytes_copied_during_growth: int = 0

    def _init_buffers(self, block: QuantizedVBlock, initial_capacity: int):
        codes = _expand_block_dim(block.codes)
        scales = _expand_block_dim(block.scales)

        def grow(x):
            shape = x.shape[:2] + (initial_capacity,) + x.shape[3:]
            zeros = mx.zeros(shape, dtype=x.dtype)
            zeros[:, :, 0:1, :, :] = x
            return zeros

        self.codes = grow(codes)
        self.scales = grow(scales)
        self.group_size = block.group_size
        self.capacity = initial_capacity
        self.block_count = 1

    def _grow(self):
        new_capacity = self.capacity + 1
        def grow(x):
            shape = x.shape[:2] + (1,) + x.shape[3:]
            extra = mx.zeros(shape, dtype=x.dtype)
            return mx.concatenate([x, extra], axis=2)
        self.bytes_copied_during_growth += int(self.codes.size * self.codes.itemsize)
        self.codes = grow(self.codes)
        self.bytes_copied_during_growth += int(self.scales.size * self.scales.itemsize)
        self.scales = grow(self.scales)
        self.capacity = new_capacity
        self.reallocation_count += 1

    def append(self, block: QuantizedVBlock, initial_capacity: int = 1):
        if self.codes is None:
            self._init_buffers(block, initial_capacity)
            return
        if self.block_count >= self.capacity:
            self._grow()
        idx = self.block_count
        self.codes[:, :, idx:idx + 1, :, :] = _expand_block_dim(block.codes)
        self.scales[:, :, idx:idx + 1, :, :] = _expand_block_dim(block.scales)
        self.block_count += 1

    def to_quantized_block(self) -> QuantizedVBlock:
        """Return a QuantizedVBlock view of the valid compressed blocks."""
        if self.codes is None or self.block_count == 0:
            raise ValueError("No quantized V blocks to materialize")
        # Stored shape includes a singleton block-detail axis; squeeze it.
        return QuantizedVBlock(
            codes=mx.squeeze(self.codes[:, :, :self.block_count, :, :], axis=3),
            scales=mx.squeeze(self.scales[:, :, :self.block_count, :, :], axis=3),
            group_size=self.group_size,
        )

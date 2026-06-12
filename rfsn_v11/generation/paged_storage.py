"""Paged storage for compressed TurboPolar KV cache blocks.

Fixed-size pages eliminate the quadratic historical copying that occurs when
appending blocks one-at-a-time to a monolithic array.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx

from rfsn_v11.quant.polar.payload import PolarKeyBlock
from rfsn_v11.quant.v_quant.encoder import QuantizedVBlock


# Recommended page capacity in blocks.
DEFAULT_PAGE_CAPACITY_BLOCKS = 16


def _nbytes(x: mx.array) -> int:
    return int(x.size * x.itemsize)


@dataclass
class PolarKPage:
    """One fixed-size page of compressed key blocks."""

    radii: mx.array
    angle_codes_l1: mx.array
    angle_codes_deep: mx.array
    radii_scales: Optional[mx.array]
    valid_blocks: int
    capacity_blocks: int


@dataclass
class QuantVPage:
    """One fixed-size page of quantized value blocks."""

    codes: mx.array
    scales: mx.array
    valid_blocks: int
    capacity_blocks: int


@dataclass
class PagedPolarKStorage:
    """Paged storage for compressed key blocks.

    Pages are allocated on demand.  When a page fills, a new empty page is
    allocated; previously filled pages are never copied.
    """

    pages: List[PolarKPage] = field(default_factory=list)
    block_size: int = 0
    head_dim: int = 0
    metadata: dict = field(default_factory=dict)
    page_allocations: int = 0
    bytes_copied_during_growth: int = 0
    total_valid_blocks: int = 0

    def _allocate_page(self, radii_shape, angle_l1_shape, angle_deep_shape, radii_scales_shape, capacity: int):
        radii = mx.zeros(radii_shape[:2] + (capacity,) + radii_shape[2:], dtype=mx.int8)
        angle_l1 = mx.zeros(angle_l1_shape[:2] + (capacity,) + angle_l1_shape[2:], dtype=mx.uint8)
        angle_deep = mx.zeros(angle_deep_shape[:2] + (capacity,) + angle_deep_shape[2:], dtype=mx.uint8)
        radii_scales = mx.zeros(radii_scales_shape[:2] + (capacity,) + radii_scales_shape[2:], dtype=mx.float16) if radii_scales_shape else None
        page = PolarKPage(
            radii=radii,
            angle_codes_l1=angle_l1,
            angle_codes_deep=angle_deep,
            radii_scales=radii_scales,
            valid_blocks=0,
            capacity_blocks=capacity,
        )
        self.pages.append(page)
        self.page_allocations += 1

    def append(self, block: PolarKeyBlock):
        if not self.pages:
            self.metadata = block.metadata
            self.block_size = block.block_size
            self.head_dim = block.head_dim
            self._allocate_page(
                block.radii.shape,
                block.angle_codes_l1.shape,
                block.angle_codes_deep.shape,
                block.radii_scales.shape if block.radii_scales is not None else None,
                DEFAULT_PAGE_CAPACITY_BLOCKS,
            )

        last_page = self.pages[-1]
        if last_page.valid_blocks >= last_page.capacity_blocks:
            self._allocate_page(
                last_page.radii.shape[:2] + (1,) + last_page.radii.shape[3:],
                last_page.angle_codes_l1.shape[:2] + (1,) + last_page.angle_codes_l1.shape[3:],
                last_page.angle_codes_deep.shape[:2] + (1,) + last_page.angle_codes_deep.shape[3:],
                last_page.radii_scales.shape[:2] + (1,) + last_page.radii_scales.shape[3:] if last_page.radii_scales is not None else None,
                DEFAULT_PAGE_CAPACITY_BLOCKS,
            )
            last_page = self.pages[-1]

        idx = last_page.valid_blocks
        last_page.radii = _set_block(last_page.radii, idx, mx.expand_dims(block.radii, axis=2))
        last_page.angle_codes_l1 = _set_block(last_page.angle_codes_l1, idx, mx.expand_dims(block.angle_codes_l1, axis=2))
        last_page.angle_codes_deep = _set_block(last_page.angle_codes_deep, idx, mx.expand_dims(block.angle_codes_deep, axis=2))
        if block.radii_scales is not None:
            last_page.radii_scales = _set_block(last_page.radii_scales, idx, mx.expand_dims(block.radii_scales, axis=2))
        last_page.valid_blocks += 1
        self.total_valid_blocks += 1

    def to_unified_block(self, shape: Tuple[int, ...]) -> PolarKeyBlock:
        """Return a single PolarKeyBlock by concatenating all valid pages.

        Note: this is a transitional API.  Future kernels will process pages
        directly without this concatenation step.
        """
        if not self.pages or self.total_valid_blocks == 0:
            raise ValueError("No compressed blocks to materialize")

        all_radii = []
        all_angle_l1 = []
        all_angle_deep = []
        all_scales = []
        for page in self.pages:
            if page.valid_blocks == 0:
                continue
            all_radii.append(page.radii[:, :, :page.valid_blocks, :, :])
            all_angle_l1.append(page.angle_codes_l1[:, :, :page.valid_blocks, :, :])
            all_angle_deep.append(page.angle_codes_deep[:, :, :page.valid_blocks, :, :])
            if page.radii_scales is not None:
                all_scales.append(page.radii_scales[:, :, :page.valid_blocks, :, :])

        radii = mx.concatenate(all_radii, axis=2)
        angle_l1 = mx.concatenate(all_angle_l1, axis=2)
        angle_deep = mx.concatenate(all_angle_deep, axis=2)
        radii_scales = mx.concatenate(all_scales, axis=2) if all_scales else None

        return PolarKeyBlock(
            radii=radii,
            angle_codes_l1=angle_l1,
            angle_codes_deep=angle_deep,
            radii_scales=radii_scales,
            shape=shape,
            block_size=self.block_size,
            head_dim=self.head_dim,
            metadata=self.metadata,
        )

    def get_memory_stats(self) -> Tuple[int, int]:
        """Return (logical_payload_bytes, allocated_capacity_bytes)."""
        logical = 0
        allocated = 0
        for page in self.pages:
            for arr in (page.radii, page.angle_codes_l1, page.angle_codes_deep):
                allocated += _nbytes(arr)
                if page.valid_blocks > 0:
                    logical += _nbytes(arr[:, :, :page.valid_blocks, :, :])
            if page.radii_scales is not None:
                allocated += _nbytes(page.radii_scales)
                if page.valid_blocks > 0:
                    logical += _nbytes(page.radii_scales[:, :, :page.valid_blocks, :, :])
        return logical, allocated


@dataclass
class PagedQuantVStorage:
    """Paged storage for quantized value blocks."""

    pages: List[QuantVPage] = field(default_factory=list)
    group_size: int = 32
    page_allocations: int = 0
    bytes_copied_during_growth: int = 0
    total_valid_blocks: int = 0

    def _allocate_page(self, codes_shape, scales_shape, capacity: int):
        codes = mx.zeros(codes_shape[:2] + (capacity,) + codes_shape[2:], dtype=mx.int8)
        scales = mx.zeros(scales_shape[:2] + (capacity,) + scales_shape[2:], dtype=mx.float16)
        page = QuantVPage(
            codes=codes,
            scales=scales,
            valid_blocks=0,
            capacity_blocks=capacity,
        )
        self.pages.append(page)
        self.page_allocations += 1

    def append(self, block: QuantizedVBlock):
        if not self.pages:
            self.group_size = block.group_size
            self._allocate_page(block.codes.shape, block.scales.shape, DEFAULT_PAGE_CAPACITY_BLOCKS)

        last_page = self.pages[-1]
        if last_page.valid_blocks >= last_page.capacity_blocks:
            self._allocate_page(last_page.codes.shape[:2] + (1,) + last_page.codes.shape[3:],
                                last_page.scales.shape[:2] + (1,) + last_page.scales.shape[3:],
                                DEFAULT_PAGE_CAPACITY_BLOCKS)
            last_page = self.pages[-1]

        idx = last_page.valid_blocks
        last_page.codes = _set_block(last_page.codes, idx, mx.expand_dims(block.codes, axis=2))
        last_page.scales = _set_block(last_page.scales, idx, mx.expand_dims(block.scales, axis=2))
        last_page.valid_blocks += 1
        self.total_valid_blocks += 1

    def to_quantized_block(self) -> QuantizedVBlock:
        """Return a single QuantizedVBlock by concatenating all valid pages."""
        if not self.pages or self.total_valid_blocks == 0:
            raise ValueError("No quantized V blocks to materialize")

        all_codes = []
        all_scales = []
        for page in self.pages:
            if page.valid_blocks == 0:
                continue
            all_codes.append(page.codes[:, :, :page.valid_blocks, :, :])
            all_scales.append(page.scales[:, :, :page.valid_blocks, :, :])

        codes = mx.concatenate(all_codes, axis=2)
        scales = mx.concatenate(all_scales, axis=2)
        return QuantizedVBlock(
            codes=mx.squeeze(codes, axis=3),
            scales=mx.squeeze(scales, axis=3),
            group_size=self.group_size,
        )

    def get_memory_stats(self) -> Tuple[int, int]:
        """Return (logical_payload_bytes, allocated_capacity_bytes)."""
        logical = 0
        allocated = 0
        for page in self.pages:
            for arr in (page.codes, page.scales):
                allocated += _nbytes(arr)
                if page.valid_blocks > 0:
                    logical += _nbytes(arr[:, :, :page.valid_blocks, :, :])
        return logical, allocated


def _set_block(dest: mx.array, idx: int, src: mx.array) -> mx.array:
    """Write src into dest[:, :, idx:idx+1, ...] and return the updated array."""
    # MLX supports in-place slice assignment.
    dest[:, :, idx : idx + 1, ...] = src
    return dest

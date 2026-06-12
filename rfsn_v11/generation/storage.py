"""Persistent, capacity-growing storage for compressed TurboPolar blocks.

Internally uses fixed-size pages (16 blocks per page) to eliminate the
quadratic historical copying that occurs with single-block-at-a-time growth.
The external API remains unchanged for backwards compatibility.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import mlx.core as mx

from rfsn_v11.generation.paged_storage import (
    PagedPolarKStorage,
    PagedQuantVStorage,
)
from rfsn_v11.quant.polar.payload import PolarKeyBlock
from rfsn_v11.quant.v_quant.encoder import QuantizedVBlock


def _nbytes(x: mx.array) -> int:
    return int(x.size * x.itemsize)


@dataclass
class PolarKBlockStorage:
    """Persistent storage for compressed key blocks using fixed-size pages.

    Internally delegates to PagedPolarKStorage.  The exposed arrays are cached
    after each append so repeated reads do not re-concatenate pages.
    """

    _paged: PagedPolarKStorage = field(default_factory=PagedPolarKStorage)

    # Cached views of the concatenated valid blocks.
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

    def _refresh_cache(self):
        if self._paged.total_valid_blocks == 0:
            self.radii = None
            self.angle_codes_l1 = None
            self.angle_codes_deep = None
            self.radii_scales = None
            self.block_count = 0
            self.capacity = 0
            return

        B = self._paged.pages[0].radii.shape[0]
        H = self._paged.pages[0].radii.shape[1]
        L = self._paged.block_size
        D = self._paged.head_dim
        S = self._paged.total_valid_blocks
        unified = self._paged.to_unified_block((B, H, S * L, D))
        self.radii = unified.radii
        self.angle_codes_l1 = unified.angle_codes_l1
        self.angle_codes_deep = unified.angle_codes_deep
        self.radii_scales = unified.radii_scales
        self.block_count = S
        self.capacity = sum(p.capacity_blocks for p in self._paged.pages)
        self.block_size = self._paged.block_size
        self.head_dim = self._paged.head_dim
        self.metadata = self._paged.metadata
        self.reallocation_count = self._paged.page_allocations
        self.bytes_copied_during_growth = self._paged.bytes_copied_during_growth

    def append(self, block: PolarKeyBlock, initial_capacity: int = 1):
        del initial_capacity  # ignored; paged storage manages capacity
        self._paged.append(block)
        self._refresh_cache()

    def to_unified_block(self, shape: Tuple[int, ...]) -> PolarKeyBlock:
        return self._paged.to_unified_block(shape)


@dataclass
class QuantVBlockStorage:
    """Persistent storage for quantized value blocks using fixed-size pages.

    Internally delegates to PagedQuantVStorage.  The exposed arrays are cached
    after each append so repeated reads do not re-concatenate pages.
    """

    _paged: PagedQuantVStorage = field(default_factory=PagedQuantVStorage)

    # Cached views of the concatenated valid blocks.
    codes: Optional[mx.array] = None
    scales: Optional[mx.array] = None
    group_size: int = 32
    block_count: int = 0
    capacity: int = 0
    reallocation_count: int = 0
    bytes_copied_during_growth: int = 0

    def _refresh_cache(self):
        if self._paged.total_valid_blocks == 0:
            self.codes = None
            self.scales = None
            self.block_count = 0
            self.capacity = 0
            return

        unified = self._paged.to_quantized_block()
        self.codes = unified.codes
        self.scales = unified.scales
        self.group_size = self._paged.group_size
        self.block_count = self._paged.total_valid_blocks
        self.capacity = sum(p.capacity_blocks for p in self._paged.pages)
        self.reallocation_count = self._paged.page_allocations
        self.bytes_copied_during_growth = self._paged.bytes_copied_during_growth

    def append(self, block: QuantizedVBlock, initial_capacity: int = 1):
        del initial_capacity  # ignored; paged storage manages capacity
        self._paged.append(block)
        self._refresh_cache()

    def to_quantized_block(self) -> QuantizedVBlock:
        return self._paged.to_quantized_block()

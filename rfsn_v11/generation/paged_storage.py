"""Paged storage for compressed TurboPolar KV cache blocks.

Fixed-size pages eliminate the quadratic historical copying that occurs when
appending blocks one-at-a-time to a monolithic array.

Every page is allocated from an explicit immutable layout.  Page shapes are never
derived from an existing page tensor.
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


@dataclass(frozen=True)
class PolarPageLayout:
    """Immutable layout for every PolarKPage in a storage instance."""

    batch_size: int
    num_kv_heads: int
    page_capacity_blocks: int
    block_size: int
    head_dim: int
    pair_count: int
    radii_shape: Tuple[int, ...]
    angle_l1_shape: Tuple[int, ...]
    angle_deep_shape: Tuple[int, ...]
    radii_scales_shape: Optional[Tuple[int, ...]]


@dataclass(frozen=True)
class QuantVPageLayout:
    """Immutable layout for every QuantVPage in a storage instance."""

    batch_size: int
    num_kv_heads: int
    page_capacity_blocks: int
    block_size: int
    head_dim: int
    group_size: int
    codes_shape: Tuple[int, ...]
    scales_shape: Tuple[int, ...]


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
    group_size: int = 32


def compute_polar_page_layout(
    block: PolarKeyBlock, page_capacity_blocks: int = DEFAULT_PAGE_CAPACITY_BLOCKS
) -> PolarPageLayout:
    """Compute an immutable layout from a single reference block.

    Block radii is 4-D [B, H, L, pair_count]; page layout inserts capacity at axis 2.
    """
    B, H, L, _ = block.radii.shape
    pair_count = block.radii.shape[-1]
    # Page shapes: insert capacity at axis 2.
    radii_shape = (B, H, page_capacity_blocks, L, pair_count)
    angle_l1_shape = (B, H, page_capacity_blocks, L, block.angle_codes_l1.shape[-1])
    angle_deep_shape = (B, H, page_capacity_blocks, L, block.angle_codes_deep.shape[-1])
    radii_scales_shape = None
    if block.radii_scales is not None:
        radii_scales_shape = (B, H, page_capacity_blocks, 1, 1)
    return PolarPageLayout(
        batch_size=B,
        num_kv_heads=H,
        page_capacity_blocks=page_capacity_blocks,
        block_size=L,
        head_dim=block.head_dim,
        pair_count=pair_count,
        radii_shape=radii_shape,
        angle_l1_shape=angle_l1_shape,
        angle_deep_shape=angle_deep_shape,
        radii_scales_shape=radii_scales_shape,
    )


def compute_quant_v_page_layout(
    block: QuantizedVBlock, page_capacity_blocks: int = DEFAULT_PAGE_CAPACITY_BLOCKS
) -> QuantVPageLayout:
    """Compute an immutable layout from a single reference block."""
    B, H, _, L, D = block.codes.shape
    return QuantVPageLayout(
        batch_size=B,
        num_kv_heads=H,
        page_capacity_blocks=page_capacity_blocks,
        block_size=L,
        head_dim=D,
        group_size=block.group_size,
        codes_shape=(B, H, page_capacity_blocks, L, D),
        scales_shape=(B, H, page_capacity_blocks, L, D // block.group_size),
    )


def allocate_polar_page(layout: PolarPageLayout) -> PolarKPage:
    """Allocate a new PolarKPage from an explicit layout."""
    radii = mx.zeros(layout.radii_shape, dtype=mx.int8)
    angle_l1 = mx.zeros(layout.angle_l1_shape, dtype=mx.uint8)
    angle_deep = mx.zeros(layout.angle_deep_shape, dtype=mx.uint8)
    radii_scales = None
    if layout.radii_scales_shape is not None:
        radii_scales = mx.zeros(layout.radii_scales_shape, dtype=mx.float16)
    return PolarKPage(
        radii=radii,
        angle_codes_l1=angle_l1,
        angle_codes_deep=angle_deep,
        radii_scales=radii_scales,
        valid_blocks=0,
        capacity_blocks=layout.page_capacity_blocks,
    )


def allocate_quant_v_page(layout: QuantVPageLayout) -> QuantVPage:
    """Allocate a new QuantVPage from an explicit layout."""
    codes = mx.zeros(layout.codes_shape, dtype=mx.int8)
    scales = mx.zeros(layout.scales_shape, dtype=mx.float16)
    return QuantVPage(
        codes=codes,
        scales=scales,
        valid_blocks=0,
        capacity_blocks=layout.page_capacity_blocks,
        group_size=layout.group_size,
    )


def validate_polar_page_shape(page: PolarKPage, layout: PolarPageLayout) -> None:
    """Validate that a page matches its layout exactly."""
    if page.radii.shape != layout.radii_shape:
        raise ValueError(
            f"PolarKPage radii shape mismatch: {page.radii.shape} != {layout.radii_shape}"
        )
    if page.angle_codes_l1.shape != layout.angle_l1_shape:
        raise ValueError(
            f"PolarKPage angle_l1 shape mismatch: {page.angle_codes_l1.shape} != {layout.angle_l1_shape}"
        )
    if page.angle_codes_deep.shape != layout.angle_deep_shape:
        raise ValueError(
            f"PolarKPage angle_deep shape mismatch: {page.angle_codes_deep.shape} != {layout.angle_deep_shape}"
        )
    if layout.radii_scales_shape is not None:
        if page.radii_scales is None:
            raise ValueError("PolarKPage missing radii_scales, layout requires them")
        if page.radii_scales.shape != layout.radii_scales_shape:
            raise ValueError(
                f"PolarKPage radii_scales shape mismatch: {page.radii_scales.shape} != {layout.radii_scales_shape}"
            )
    else:
        if page.radii_scales is not None:
            raise ValueError("PolarKPage has radii_scales, layout expects None")


def validate_quant_v_page_shape(page: QuantVPage, layout: QuantVPageLayout) -> None:
    """Validate that a page matches its layout exactly."""
    if page.codes.shape != layout.codes_shape:
        raise ValueError(
            f"QuantVPage codes shape mismatch: {page.codes.shape} != {layout.codes_shape}"
        )
    if page.scales.shape != layout.scales_shape:
        raise ValueError(
            f"QuantVPage scales shape mismatch: {page.scales.shape} != {layout.scales_shape}"
        )


@dataclass
class PagedPolarKStorage:
    """Paged storage for compressed key blocks.

    Pages are allocated on demand from an explicit immutable layout.  When a page
    fills, a new empty page is allocated from the same layout; previously filled
    pages are never copied.
    """

    pages: List[PolarKPage] = field(default_factory=list)
    layout: Optional[PolarPageLayout] = None
    block_size: int = 0
    head_dim: int = 0
    metadata: dict = field(default_factory=dict)
    page_allocations: int = 0
    bytes_copied_during_growth: int = 0
    total_valid_blocks: int = 0

    def _allocate_page(self):
        if self.layout is None:
            raise RuntimeError("PagedPolarKStorage layout not set")
        page = allocate_polar_page(self.layout)
        self.pages.append(page)
        self.page_allocations += 1

    def append(self, block: PolarKeyBlock):
        if not self.pages:
            self.metadata = block.metadata
            self.block_size = block.block_size
            self.head_dim = block.head_dim
            self.layout = compute_polar_page_layout(block, DEFAULT_PAGE_CAPACITY_BLOCKS)
            self._allocate_page()

        last_page = self.pages[-1]
        if last_page.valid_blocks >= last_page.capacity_blocks:
            self._allocate_page()
            last_page = self.pages[-1]

        idx = last_page.valid_blocks
        # block fields are 4-D [B, H, L, ...]; page fields are 5-D [B, H, C, L, ...]
        last_page.radii = _set_block(
            last_page.radii, idx, mx.expand_dims(block.radii, axis=2)
        )
        last_page.angle_codes_l1 = _set_block(
            last_page.angle_codes_l1, idx, mx.expand_dims(block.angle_codes_l1, axis=2)
        )
        last_page.angle_codes_deep = _set_block(
            last_page.angle_codes_deep,
            idx,
            mx.expand_dims(block.angle_codes_deep, axis=2),
        )
        if block.radii_scales is not None:
            last_page.radii_scales = _set_block(
                last_page.radii_scales, idx, mx.expand_dims(block.radii_scales, axis=2)
            )
        last_page.valid_blocks += 1
        self.total_valid_blocks += 1

    def debug_materialize_all_blocks(self, shape: Tuple[int, ...]) -> PolarKeyBlock:
        """Return a single PolarKeyBlock by concatenating all valid pages.

        This is a debug/export utility only.  Production kernels must process
        pages directly without this concatenation step.
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
            all_radii.append(page.radii[:, :, : page.valid_blocks, :, :])
            all_angle_l1.append(page.angle_codes_l1[:, :, : page.valid_blocks, :, :])
            all_angle_deep.append(
                page.angle_codes_deep[:, :, : page.valid_blocks, :, :]
            )
            if page.radii_scales is not None:
                all_scales.append(page.radii_scales[:, :, : page.valid_blocks, :, :])

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

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def get_memory_stats(self) -> Tuple[int, int]:
        """Return (logical_payload_bytes, allocated_capacity_bytes)."""
        logical = 0
        allocated = 0
        for page in self.pages:
            for arr in (page.radii, page.angle_codes_l1, page.angle_codes_deep):
                allocated += _nbytes(arr)
                if page.valid_blocks > 0:
                    logical += _nbytes(arr[:, :, : page.valid_blocks, :, :])
            if page.radii_scales is not None:
                allocated += _nbytes(page.radii_scales)
                if page.valid_blocks > 0:
                    logical += _nbytes(
                        page.radii_scales[:, :, : page.valid_blocks, :, :]
                    )
        return logical, allocated

    def get_page_block(self, page_index: int, block_index: int) -> PolarKeyBlock:
        """Return a single PolarKeyBlock from a specific page and block index."""
        page = self.pages[page_index]
        if block_index >= page.valid_blocks:
            raise IndexError(
                f"Block index {block_index} out of range (page has {page.valid_blocks} valid blocks)"
            )
        return PolarKeyBlock(
            radii=page.radii[:, :, block_index : block_index + 1, :, :],
            angle_codes_l1=page.angle_codes_l1[
                :, :, block_index : block_index + 1, :, :
            ],
            angle_codes_deep=page.angle_codes_deep[
                :, :, block_index : block_index + 1, :, :
            ],
            radii_scales=page.radii_scales[:, :, block_index : block_index + 1, :, :]
            if page.radii_scales is not None
            else None,
            shape=(
                self.layout.batch_size,
                self.layout.num_kv_heads,
                self.block_size,
                self.head_dim,
            ),
            block_size=self.block_size,
            head_dim=self.head_dim,
            metadata=self.metadata,
        )


@dataclass
class PagedQuantVStorage:
    """Paged storage for quantized value blocks."""

    pages: List[QuantVPage] = field(default_factory=list)
    layout: Optional[QuantVPageLayout] = None
    group_size: int = 32
    page_allocations: int = 0
    bytes_copied_during_growth: int = 0
    total_valid_blocks: int = 0

    def _allocate_page(self):
        if self.layout is None:
            raise RuntimeError("PagedQuantVStorage layout not set")
        page = allocate_quant_v_page(self.layout)
        self.pages.append(page)
        self.page_allocations += 1

    def append(self, block: QuantizedVBlock):
        if not self.pages:
            self.group_size = block.group_size
            self.layout = compute_quant_v_page_layout(
                block, DEFAULT_PAGE_CAPACITY_BLOCKS
            )
            self._allocate_page()

        last_page = self.pages[-1]
        if last_page.valid_blocks >= last_page.capacity_blocks:
            self._allocate_page()
            last_page = self.pages[-1]

        idx = last_page.valid_blocks
        last_page.codes = _set_block(last_page.codes, idx, block.codes)
        last_page.scales = _set_block(last_page.scales, idx, block.scales)
        last_page.valid_blocks += 1
        self.total_valid_blocks += 1

    def debug_materialize_all_blocks(self) -> QuantizedVBlock:
        """Return a single QuantizedVBlock by concatenating all valid pages.

        This is a debug/export utility only.  Production kernels must process
        pages directly without this concatenation step.
        """
        if not self.pages or self.total_valid_blocks == 0:
            raise ValueError("No quantized V blocks to materialize")

        all_codes = []
        all_scales = []
        for page in self.pages:
            if page.valid_blocks == 0:
                continue
            all_codes.append(page.codes[:, :, : page.valid_blocks, :, :])
            all_scales.append(page.scales[:, :, : page.valid_blocks, :, :])

        codes = mx.concatenate(all_codes, axis=2)
        scales = mx.concatenate(all_scales, axis=2)
        return QuantizedVBlock(
            codes=codes,
            scales=scales,
            group_size=self.group_size,
        )

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def get_memory_stats(self) -> Tuple[int, int]:
        """Return (logical_payload_bytes, allocated_capacity_bytes)."""
        logical = 0
        allocated = 0
        for page in self.pages:
            for arr in (page.codes, page.scales):
                allocated += _nbytes(arr)
                if page.valid_blocks > 0:
                    logical += _nbytes(arr[:, :, : page.valid_blocks, :, :])
        return logical, allocated

    def get_page_block(self, page_index: int, block_index: int) -> QuantizedVBlock:
        """Return a single QuantizedVBlock from a specific page and block index."""
        page = self.pages[page_index]
        if block_index >= page.valid_blocks:
            raise IndexError(
                f"Block index {block_index} out of range (page has {page.valid_blocks} valid blocks)"
            )
        return QuantizedVBlock(
            codes=page.codes[:, :, block_index : block_index + 1, :, :],
            scales=page.scales[:, :, block_index : block_index + 1, :, :],
            group_size=self.group_size,
        )


def _set_block(dest: mx.array, idx: int, src: mx.array) -> mx.array:
    """Write src into dest[:, :, idx:idx+1, ...] and return the updated array."""
    dest[:, :, idx : idx + 1, ...] = src
    return dest

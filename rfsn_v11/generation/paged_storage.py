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
    zero_points_shape: Optional[Tuple[int, ...]] = None


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
    zero_points: Optional[mx.array] = None


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
    num_groups = D // block.group_size
    zp_shape = None
    if block.zero_points is not None:
        zp_shape = (B, H, page_capacity_blocks, L, num_groups)
    return QuantVPageLayout(
        batch_size=B,
        num_kv_heads=H,
        page_capacity_blocks=page_capacity_blocks,
        block_size=L,
        head_dim=D,
        group_size=block.group_size,
        codes_shape=(B, H, page_capacity_blocks, L, D),
        scales_shape=(B, H, page_capacity_blocks, L, num_groups),
        zero_points_shape=zp_shape,
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
    zero_points = None
    if layout.zero_points_shape is not None:
        zero_points = mx.zeros(layout.zero_points_shape, dtype=mx.float16)
    return QuantVPage(
        codes=codes,
        scales=scales,
        valid_blocks=0,
        capacity_blocks=layout.page_capacity_blocks,
        group_size=layout.group_size,
        zero_points=zero_points,
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
    """Paged storage for compressed key blocks using field-specific slab storage.

    Internally uses one large preallocated array per field that grows by doubling.
    Virtual pages are sliced from the slab on demand.
    """

    layout: Optional[PolarPageLayout] = None
    block_size: int = 0
    head_dim: int = 0
    metadata: dict = field(default_factory=dict)
    page_allocations: int = 0
    bytes_copied_during_growth: int = 0
    total_valid_blocks: int = 0
    _page_pool: List[PolarKPage] = field(default_factory=list, repr=False)
    _pool_prealloc_size: int = field(default=0, repr=False)
    _slab_radii: Optional[mx.array] = field(default=None, repr=False)
    _slab_angle_l1: Optional[mx.array] = field(default=None, repr=False)
    _slab_angle_deep: Optional[mx.array] = field(default=None, repr=False)
    _slab_radii_scales: Optional[mx.array] = field(default=None, repr=False)
    _capacity_blocks: int = field(default=0, repr=False)
    _page_capacity: int = field(default=DEFAULT_PAGE_CAPACITY_BLOCKS, repr=False)

    @property
    def pages(self) -> List[PolarKPage]:
        """Return fresh virtual PolarKPage objects sliced from the slab."""
        if self._slab_radii is None:
            return []
        virtual_pages: List[PolarKPage] = []
        num_pages = (
            self.total_valid_blocks + self._page_capacity - 1
        ) // self._page_capacity
        for i in range(num_pages):
            start = i * self._page_capacity
            end = start + self._page_capacity
            valid_in_page = min(self._page_capacity, self.total_valid_blocks - start)
            page = PolarKPage(
                radii=self._slab_radii[:, :, start:end, :, :],
                angle_codes_l1=self._slab_angle_l1[:, :, start:end, :, :],
                angle_codes_deep=self._slab_angle_deep[:, :, start:end, :, :],
                radii_scales=self._slab_radii_scales[:, :, start:end, :, :]
                if self._slab_radii_scales is not None
                else None,
                valid_blocks=valid_in_page,
                capacity_blocks=self._page_capacity,
            )
            virtual_pages.append(page)
        return virtual_pages

    def preallocate_pool(self, num_pages: int):
        """Pre-allocate ``num_pages`` empty pages from the current layout.

        Must be called after at least one block has been appended so the layout
        is established.  Pages are drawn from this pool before new allocations.
        """
        if self.layout is None:
            raise RuntimeError(
                "PagedPolarKStorage: cannot preallocate pool before layout "
                "is known. Append at least one block first."
            )
        if num_pages <= 0:
            return
        self._pool_prealloc_size = num_pages
        needed = num_pages - len(self._page_pool)
        for _ in range(needed):
            self._page_pool.append(allocate_polar_page(self.layout))
        needed_capacity = num_pages * self._page_capacity
        if self._capacity_blocks < needed_capacity:
            self._grow_slab_to(needed_capacity)

    def _slab_shape(
        self, page_shape: Tuple[int, ...], capacity_blocks: int
    ) -> Tuple[int, ...]:
        return page_shape[:2] + (capacity_blocks,) + page_shape[3:]

    def _init_slabs(self, capacity_blocks: int):
        self._capacity_blocks = capacity_blocks
        self._slab_radii = mx.zeros(
            self._slab_shape(self.layout.radii_shape, capacity_blocks),
            dtype=mx.int8,
        )
        self._slab_angle_l1 = mx.zeros(
            self._slab_shape(self.layout.angle_l1_shape, capacity_blocks),
            dtype=mx.uint8,
        )
        self._slab_angle_deep = mx.zeros(
            self._slab_shape(self.layout.angle_deep_shape, capacity_blocks),
            dtype=mx.uint8,
        )
        self._slab_radii_scales = None
        if self.layout.radii_scales_shape is not None:
            self._slab_radii_scales = mx.zeros(
                self._slab_shape(self.layout.radii_scales_shape, capacity_blocks),
                dtype=mx.float16,
            )

    def _extend_slab(self, slab: mx.array, new_capacity: int) -> mx.array:
        old_capacity = slab.shape[2]
        extension_shape = list(slab.shape)
        extension_shape[2] = new_capacity - old_capacity
        extension = mx.zeros(tuple(extension_shape), dtype=slab.dtype)
        return mx.concatenate([slab, extension], axis=2)

    def _grow_slab_to(self, new_capacity: int):
        if new_capacity <= self._capacity_blocks:
            return
        # Track bytes copied from old slabs
        for arr in (self._slab_radii, self._slab_angle_l1, self._slab_angle_deep):
            self.bytes_copied_during_growth += _nbytes(arr)
        if self._slab_radii_scales is not None:
            self.bytes_copied_during_growth += _nbytes(self._slab_radii_scales)

        self._slab_radii = self._extend_slab(self._slab_radii, new_capacity)
        self._slab_angle_l1 = self._extend_slab(self._slab_angle_l1, new_capacity)
        self._slab_angle_deep = self._extend_slab(self._slab_angle_deep, new_capacity)
        if self._slab_radii_scales is not None:
            self._slab_radii_scales = self._extend_slab(
                self._slab_radii_scales, new_capacity
            )
        self._capacity_blocks = new_capacity

    def _grow_slabs(self):
        if self._capacity_blocks == 0:
            self._init_slabs(self._page_capacity)
        else:
            self._grow_slab_to(self._capacity_blocks * 2)

    def append(self, block: PolarKeyBlock):
        if self._slab_radii is None:
            self.metadata = block.metadata
            self.block_size = block.block_size
            self.head_dim = block.head_dim
            self.layout = compute_polar_page_layout(
                block, DEFAULT_PAGE_CAPACITY_BLOCKS
            )
            self._init_slabs(self._page_capacity)
            self.page_allocations = 1

        if self.total_valid_blocks >= self._capacity_blocks:
            self._grow_slabs()

        if self.total_valid_blocks > 0 and self.total_valid_blocks % self._page_capacity == 0:
            if self._page_pool:
                self._page_pool.pop()
            self.page_allocations += 1

        idx = self.total_valid_blocks
        # block fields are 4-D [B, H, L, ...]; slab fields are 5-D [B, H, C, L, ...]
        self._slab_radii = _set_block(
            self._slab_radii, idx, mx.expand_dims(block.radii, axis=2)
        )
        self._slab_angle_l1 = _set_block(
            self._slab_angle_l1, idx, mx.expand_dims(block.angle_codes_l1, axis=2)
        )
        self._slab_angle_deep = _set_block(
            self._slab_angle_deep,
            idx,
            mx.expand_dims(block.angle_codes_deep, axis=2),
        )
        if block.radii_scales is not None:
            self._slab_radii_scales = _set_block(
                self._slab_radii_scales,
                idx,
                mx.expand_dims(block.radii_scales, axis=2),
            )
        self.total_valid_blocks += 1

    def debug_materialize_all_blocks(self, shape: Tuple[int, ...]) -> PolarKeyBlock:
        """Return a single PolarKeyBlock by slicing the slab.

        This is a debug/export utility only.  Production kernels must process
        pages directly without this concatenation step.
        """
        if self._slab_radii is None or self.total_valid_blocks == 0:
            raise ValueError("No compressed blocks to materialize")

        return PolarKeyBlock(
            radii=self._slab_radii[:, :, : self.total_valid_blocks, :, :],
            angle_codes_l1=self._slab_angle_l1[
                :, :, : self.total_valid_blocks, :, :
            ],
            angle_codes_deep=self._slab_angle_deep[
                :, :, : self.total_valid_blocks, :, :
            ],
            radii_scales=self._slab_radii_scales[
                :, :, : self.total_valid_blocks, :, :
            ]
            if self._slab_radii_scales is not None
            else None,
            shape=shape,
            block_size=self.block_size,
            head_dim=self.head_dim,
            metadata=self.metadata,
        )

    @property
    def page_count(self) -> int:
        if self._slab_radii is None:
            return 0
        return (
            self.total_valid_blocks + self._page_capacity - 1
        ) // self._page_capacity

    def get_memory_stats(self) -> Tuple[int, int]:
        """Return (logical_payload_bytes, allocated_capacity_bytes).

        Logical bytes are computed arithmetically from itemsize and shapes
        rather than by slicing device arrays.  MLX slices produce copies
        rather than views, so slicing solely to count bytes would perturb
        the allocator being measured.
        """
        logical = 0
        allocated = 0
        for arr in (self._slab_radii, self._slab_angle_l1, self._slab_angle_deep):
            arr_bytes = _nbytes(arr)
            allocated += arr_bytes
            if self.total_valid_blocks > 0:
                # Compute valid fraction arithmetically.
                logical += arr_bytes * self.total_valid_blocks // arr.shape[2]
        if self._slab_radii_scales is not None:
            rs_bytes = _nbytes(self._slab_radii_scales)
            allocated += rs_bytes
            if self.total_valid_blocks > 0:
                logical += rs_bytes * self.total_valid_blocks // self._slab_radii_scales.shape[2]
        return logical, allocated

    def get_page_block(self, page_index: int, block_index: int) -> PolarKeyBlock:
        """Return a single PolarKeyBlock from a specific page and block index."""
        flat_idx = page_index * self._page_capacity + block_index
        page_valid = min(
            self._page_capacity,
            self.total_valid_blocks - page_index * self._page_capacity,
        )
        if page_valid <= 0 or block_index >= page_valid:
            raise IndexError(
                f"Block index {block_index} out of range (page has {page_valid} valid blocks)"
            )
        return PolarKeyBlock(
            radii=self._slab_radii[:, :, flat_idx : flat_idx + 1, :, :],
            angle_codes_l1=self._slab_angle_l1[
                :, :, flat_idx : flat_idx + 1, :, :
            ],
            angle_codes_deep=self._slab_angle_deep[
                :, :, flat_idx : flat_idx + 1, :, :
            ],
            radii_scales=self._slab_radii_scales[
                :, :, flat_idx : flat_idx + 1, :, :
            ]
            if self._slab_radii_scales is not None
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
    """Paged storage for quantized value blocks using field-specific slab storage.

    Internally uses one large preallocated array per field that grows by doubling.
    Virtual pages are sliced from the slab on demand.
    """

    layout: Optional[QuantVPageLayout] = None
    group_size: int = 32
    page_allocations: int = 0
    bytes_copied_during_growth: int = 0
    total_valid_blocks: int = 0
    _page_pool: List[QuantVPage] = field(default_factory=list, repr=False)
    _pool_prealloc_size: int = field(default=0, repr=False)
    _slab_codes: Optional[mx.array] = field(default=None, repr=False)
    _slab_scales: Optional[mx.array] = field(default=None, repr=False)
    _slab_zero_points: Optional[mx.array] = field(default=None, repr=False)
    _capacity_blocks: int = field(default=0, repr=False)
    _page_capacity: int = field(default=DEFAULT_PAGE_CAPACITY_BLOCKS, repr=False)

    @property
    def pages(self) -> List[QuantVPage]:
        """Return fresh virtual QuantVPage objects sliced from the slab."""
        if self._slab_codes is None:
            return []
        virtual_pages: List[QuantVPage] = []
        num_pages = (
            self.total_valid_blocks + self._page_capacity - 1
        ) // self._page_capacity
        for i in range(num_pages):
            start = i * self._page_capacity
            end = start + self._page_capacity
            valid_in_page = min(self._page_capacity, self.total_valid_blocks - start)
            page = QuantVPage(
                codes=self._slab_codes[:, :, start:end, :, :],
                scales=self._slab_scales[:, :, start:end, :, :],
                valid_blocks=valid_in_page,
                capacity_blocks=self._page_capacity,
                group_size=self.group_size,
                zero_points=self._slab_zero_points[:, :, start:end, :, :]
                if self._slab_zero_points is not None
                else None,
            )
            virtual_pages.append(page)
        return virtual_pages

    def preallocate_pool(self, num_pages: int):
        """Pre-allocate ``num_pages`` empty pages from the current layout.

        Must be called after at least one block has been appended so the layout
        is established.  Pages are drawn from this pool before new allocations.
        """
        if self.layout is None:
            raise RuntimeError(
                "PagedQuantVStorage: cannot preallocate pool before layout "
                "is known. Append at least one block first."
            )
        if num_pages <= 0:
            return
        self._pool_prealloc_size = num_pages
        needed = num_pages - len(self._page_pool)
        for _ in range(needed):
            self._page_pool.append(allocate_quant_v_page(self.layout))
        needed_capacity = num_pages * self._page_capacity
        if self._capacity_blocks < needed_capacity:
            self._grow_slab_to(needed_capacity)

    def _slab_shape(
        self, page_shape: Tuple[int, ...], capacity_blocks: int
    ) -> Tuple[int, ...]:
        return page_shape[:2] + (capacity_blocks,) + page_shape[3:]

    def _init_slabs(self, capacity_blocks: int):
        self._capacity_blocks = capacity_blocks
        self._slab_codes = mx.zeros(
            self._slab_shape(self.layout.codes_shape, capacity_blocks),
            dtype=mx.int8,
        )
        self._slab_scales = mx.zeros(
            self._slab_shape(self.layout.scales_shape, capacity_blocks),
            dtype=mx.float16,
        )
        self._slab_zero_points = None

    def _extend_slab(self, slab: mx.array, new_capacity: int) -> mx.array:
        old_capacity = slab.shape[2]
        extension_shape = list(slab.shape)
        extension_shape[2] = new_capacity - old_capacity
        extension = mx.zeros(tuple(extension_shape), dtype=slab.dtype)
        return mx.concatenate([slab, extension], axis=2)

    def _grow_slab_to(self, new_capacity: int):
        if new_capacity <= self._capacity_blocks:
            return
        # Track bytes copied from old slabs
        for arr in (self._slab_codes, self._slab_scales):
            self.bytes_copied_during_growth += _nbytes(arr)
        if self._slab_zero_points is not None:
            self.bytes_copied_during_growth += _nbytes(self._slab_zero_points)

        self._slab_codes = self._extend_slab(self._slab_codes, new_capacity)
        self._slab_scales = self._extend_slab(self._slab_scales, new_capacity)
        if self._slab_zero_points is not None:
            self._slab_zero_points = self._extend_slab(
                self._slab_zero_points, new_capacity
            )
        self._capacity_blocks = new_capacity

    def _grow_slabs(self):
        if self._capacity_blocks == 0:
            self._init_slabs(self._page_capacity)
        else:
            self._grow_slab_to(self._capacity_blocks * 2)

    def append(self, block: QuantizedVBlock):
        if self._slab_codes is None:
            self.group_size = block.group_size
            self.layout = compute_quant_v_page_layout(
                block, DEFAULT_PAGE_CAPACITY_BLOCKS
            )
            self._init_slabs(self._page_capacity)
            self.page_allocations = 1

        if self.total_valid_blocks >= self._capacity_blocks:
            self._grow_slabs()

        if self.total_valid_blocks > 0 and self.total_valid_blocks % self._page_capacity == 0:
            if self._page_pool:
                self._page_pool.pop()
            self.page_allocations += 1

        idx = self.total_valid_blocks
        self._slab_codes = _set_block(self._slab_codes, idx, block.codes)
        self._slab_scales = _set_block(self._slab_scales, idx, block.scales)
        if block.zero_points is not None:
            if self._slab_zero_points is None:
                B, H, cap, L, D = self._slab_codes.shape
                num_groups = D // block.group_size
                self._slab_zero_points = mx.zeros(
                    (B, H, self._capacity_blocks, L, num_groups), dtype=mx.float16
                )
            self._slab_zero_points = _set_block(
                self._slab_zero_points, idx, block.zero_points
            )
        self.total_valid_blocks += 1

    def debug_materialize_all_blocks(self) -> QuantizedVBlock:
        """Return a single QuantizedVBlock by slicing the slab.

        This is a debug/export utility only.  Production kernels must process
        pages directly without this concatenation step.
        """
        if self._slab_codes is None or self.total_valid_blocks == 0:
            raise ValueError("No quantized V blocks to materialize")

        has_zp = self._slab_zero_points is not None
        return QuantizedVBlock(
            codes=self._slab_codes[:, :, : self.total_valid_blocks, :, :],
            scales=self._slab_scales[:, :, : self.total_valid_blocks, :, :],
            group_size=self.group_size,
            zero_points=self._slab_zero_points[:, :, : self.total_valid_blocks, :, :]
            if has_zp
            else None,
        )

    @property
    def page_count(self) -> int:
        if self._slab_codes is None:
            return 0
        return (
            self.total_valid_blocks + self._page_capacity - 1
        ) // self._page_capacity

    def get_memory_stats(self) -> Tuple[int, int]:
        """Return (logical_payload_bytes, allocated_capacity_bytes).

        Uses arithmetic on itemsize and shape rather than device slices.
        """
        logical = 0
        allocated = 0
        for arr in (self._slab_codes, self._slab_scales):
            arr_bytes = _nbytes(arr)
            allocated += arr_bytes
            if self.total_valid_blocks > 0:
                logical += arr_bytes * self.total_valid_blocks // arr.shape[2]
        return logical, allocated

    def get_page_block(self, page_index: int, block_index: int) -> QuantizedVBlock:
        """Return a single QuantizedVBlock from a specific page and block index."""
        flat_idx = page_index * self._page_capacity + block_index
        page_valid = min(
            self._page_capacity,
            self.total_valid_blocks - page_index * self._page_capacity,
        )
        if page_valid <= 0 or block_index >= page_valid:
            raise IndexError(
                f"Block index {block_index} out of range (page has {page_valid} valid blocks)"
            )
        zero_points = None
        if self._slab_zero_points is not None:
            zero_points = self._slab_zero_points[
                :, :, flat_idx : flat_idx + 1, :, :
            ]
        return QuantizedVBlock(
            codes=self._slab_codes[:, :, flat_idx : flat_idx + 1, :, :],
            scales=self._slab_scales[:, :, flat_idx : flat_idx + 1, :, :],
            group_size=self.group_size,
            zero_points=zero_points,
        )


def _set_block(dest: mx.array, idx: int, src: mx.array) -> mx.array:
    """Write src into dest[:, :, idx:idx+1, ...] and return the updated array."""
    if idx < 0 or idx >= dest.shape[2]:
        raise IndexError(
            f"Block index {idx} out of range for destination array "
            f"with shape {dest.shape} (axis 2)"
        )
    dest[:, :, idx:idx + 1, ...] = src
    return dest

"""Page rollover tests for PagedPolarKStorage and PagedQuantVStorage.

Every page must be allocated from an explicit immutable layout, never from
another page's current shape.
"""

import mlx.core as mx
import pytest

from rfsn_v11.generation.paged_storage import (
    PagedPolarKStorage,
    PagedQuantVStorage,
    allocate_polar_page,
    allocate_quant_v_page,
    compute_polar_page_layout,
    compute_quant_v_page_layout,
    validate_polar_page_shape,
    validate_quant_v_page_shape,
)
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig


def _make_encoder():
    config = TurboPolarConfig(
        head_dim=128,
        block_size=64,
        num_q_heads=4,
        num_kv_heads=4,
        use_int8_radii=True,
        k_angle_bits_level1=8,
        k_angle_bits_deep=8,
    )
    return PolarQuantEncoder(config)


def _make_block_storage(num_blocks: int):
    """Create a PolarKeyBlock and QuantizedVBlock with num_blocks blocks."""
    encoder = _make_encoder()
    v_quant = GroupedVQuantizer(group_size=32)
    B, H, L, D = 1, 4, 64, 128
    k = mx.random.normal((B, H, num_blocks * L, D)).astype(mx.float16)
    v = mx.random.normal((B, H, num_blocks * L, D)).astype(mx.float16)

    k_blocks = k.reshape(B, H, num_blocks, L, D)
    v_blocks = v.reshape(B, H, num_blocks, L, D)

    polar_blocks = []
    quant_v_blocks = []
    for i in range(num_blocks):
        k_block = k_blocks[:, :, i, :, :]
        v_block = v_blocks[:, :, i, :, :]
        polar_blocks.append(encoder.encode_block(k_block))
        quant_v_blocks.append(v_quant.quantize_block(v_block.reshape(B, H, 1, L, D)))

    return polar_blocks, quant_v_blocks


@pytest.mark.parametrize(
    "block_count", [0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256]
)
def test_polar_page_rollover(block_count):
    """PolarK pages must maintain identical layout across rollovers."""
    polar_blocks, _ = _make_block_storage(max(block_count, 1))
    storage = PagedPolarKStorage()

    if block_count == 0:
        assert storage.page_count == 0
        assert storage.total_valid_blocks == 0
        return

    blocks_to_append = polar_blocks[:block_count]
    for pb in blocks_to_append:
        storage.append(pb)

    expected_pages = (block_count + 15) // 16
    assert (
        storage.page_count == expected_pages
    ), f"page_count mismatch for {block_count} blocks"

    # Verify valid blocks per page
    total_valid = 0
    for page_idx, page in enumerate(storage.pages):
        if page_idx < expected_pages - 1:
            assert page.valid_blocks == 16, f"Page {page_idx} should be full"
        else:
            expected_last = block_count % 16 if block_count % 16 != 0 else 16
            assert page.valid_blocks == expected_last, "Last page valid_blocks mismatch"
        total_valid += page.valid_blocks

    assert total_valid == block_count
    assert storage.total_valid_blocks == block_count

    # Every page must have identical rank (5D) and same layout
    layout = storage.layout
    assert layout is not None
    for page in storage.pages:
        assert page.radii.ndim == 5
        assert page.angle_codes_l1.ndim == 5
        assert page.angle_codes_deep.ndim == 5
        validate_polar_page_shape(page, layout)

    # Total valid tokens
    assert storage.total_valid_blocks * 64 == block_count * 64

    # Retrieve blocks in order and compare to originals
    retrieved_indices = [0]
    if block_count > 1:
        retrieved_indices.append(1)
    if block_count >= 16:
        retrieved_indices.append(15)  # last block of first page
    if block_count > 16:
        retrieved_indices.append(16)  # first block of second page
        retrieved_indices.append(block_count - 1)

    for idx in retrieved_indices:
        page_idx = idx // 16
        block_in_page = idx % 16
        retrieved = storage.get_page_block(page_idx, block_in_page)
        original = blocks_to_append[idx]
        # Retrieved blocks are 5-D [B,H,1,L,...]; originals are 4-D [B,H,L,...]
        assert mx.allclose(mx.squeeze(retrieved.radii, axis=2), original.radii)
        assert mx.allclose(
            mx.squeeze(retrieved.angle_codes_l1, axis=2), original.angle_codes_l1
        )
        assert mx.allclose(
            mx.squeeze(retrieved.angle_codes_deep, axis=2), original.angle_codes_deep
        )


@pytest.mark.parametrize(
    "block_count", [0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256]
)
def test_quant_v_page_rollover(block_count):
    """QuantV pages must maintain identical layout across rollovers."""
    _, quant_v_blocks = _make_block_storage(max(block_count, 1))
    storage = PagedQuantVStorage()

    if block_count == 0:
        assert storage.page_count == 0
        assert storage.total_valid_blocks == 0
        return

    blocks_to_append = quant_v_blocks[:block_count]
    for vb in blocks_to_append:
        storage.append(vb)

    expected_pages = (block_count + 15) // 16
    assert storage.page_count == expected_pages

    total_valid = 0
    for page_idx, page in enumerate(storage.pages):
        if page_idx < expected_pages - 1:
            assert page.valid_blocks == 16
        else:
            expected_last = block_count % 16 if block_count % 16 != 0 else 16
            assert page.valid_blocks == expected_last
        total_valid += page.valid_blocks

    assert total_valid == block_count
    assert storage.total_valid_blocks == block_count

    layout = storage.layout
    assert layout is not None
    for page in storage.pages:
        assert page.codes.ndim == 5
        assert page.scales.ndim == 5
        validate_quant_v_page_shape(page, layout)

    # Retrieve blocks in order
    retrieved_indices = [0]
    if block_count > 1:
        retrieved_indices.append(1)
    if block_count >= 16:
        retrieved_indices.append(15)  # last block of first page
    if block_count > 16:
        retrieved_indices.append(16)  # first block of second page
        retrieved_indices.append(block_count - 1)

    for idx in retrieved_indices:
        page_idx = idx // 16
        block_in_page = idx % 16
        retrieved = storage.get_page_block(page_idx, block_in_page)
        original = blocks_to_append[idx]
        assert mx.allclose(retrieved.codes, original.codes)
        assert mx.allclose(retrieved.scales, original.scales)


def test_allocate_polar_page_from_layout():
    """Allocate_polar_page must produce exact layout shapes."""
    polar_blocks, _ = _make_block_storage(1)
    layout = compute_polar_page_layout(polar_blocks[0], page_capacity_blocks=16)
    page = allocate_polar_page(layout)
    validate_polar_page_shape(page, layout)
    assert page.valid_blocks == 0
    assert page.capacity_blocks == 16


def test_allocate_quant_v_page_from_layout():
    """Allocate_quant_v_page must produce exact layout shapes."""
    _, quant_v_blocks = _make_block_storage(1)
    layout = compute_quant_v_page_layout(quant_v_blocks[0], page_capacity_blocks=16)
    page = allocate_quant_v_page(layout)
    validate_quant_v_page_shape(page, layout)
    assert page.valid_blocks == 0
    assert page.capacity_blocks == 16


def test_17_blocks_exact_assertions():
    """Explicit assertions from the repair plan for 17 blocks."""
    polar_blocks, quant_v_blocks = _make_block_storage(17)
    k_storage = PagedPolarKStorage()
    v_storage = PagedQuantVStorage()
    for pb, vb in zip(polar_blocks, quant_v_blocks):
        k_storage.append(pb)
        v_storage.append(vb)

    assert k_storage.page_count == 2
    assert k_storage.pages[0].valid_blocks == 16
    assert k_storage.pages[1].valid_blocks == 1
    assert k_storage.total_valid_blocks == 17

    assert v_storage.page_count == 2
    assert v_storage.pages[0].valid_blocks == 16
    assert v_storage.pages[1].valid_blocks == 1
    assert v_storage.total_valid_blocks == 17

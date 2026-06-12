"""Tests for page-based online-softmax attention.

These verify that paged attention produces the same output as dense attention
without materializing the full cache.
"""

import mlx.core as mx
import pytest

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge


def _make_config():
    return TurboPolarConfig(
        head_dim=128,
        block_size=64,
        num_q_heads=4,
        num_kv_heads=4,
        use_int8_radii=True,
        k_angle_bits_level1=8,
        k_angle_bits_deep=8,
    )


def _random_kv(B, H, T, D, dtype=mx.float16):
    k = mx.random.normal((B, H, T, D)).astype(dtype)
    v = mx.random.normal((B, H, T, D)).astype(dtype)
    return k, v


def _dense_attention(q, k_hist, v_hist, scale):
    B, H_q, _, D = q.shape
    H_kv = k_hist.shape[1]
    nq = H_q // H_kv
    k_rep = mx.repeat(k_hist, nq, axis=1)
    v_rep = mx.repeat(v_hist, nq, axis=1)
    scores = mx.sum(q * k_rep, axis=-1) * scale
    weights = mx.softmax(scores, axis=-1)
    return mx.sum(weights[:, :, :, None] * v_rep, axis=-2)


@pytest.mark.parametrize("num_tokens", [1, 63, 64, 65, 127, 128, 129, 1024, 2048])
def test_paged_online_attention_matches_dense(num_tokens):
    """Paged online attention must match dense reference."""
    config = _make_config()
    k, v = _random_kv(1, 4, num_tokens, 128)
    scale = 1.0 / (128 ** 0.5)

    # Build cache.
    cache = TurboPolarKVCacheRuntime(config)
    cache.append(k, v)

    view = cache.attention_view()

    # Dense reference from decoded full history.
    from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
    from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
    decoder = PolarQuantDecoder()
    v_dequant = GroupedVQuantizer(group_size=32)

    block, quant_v, tail_k, tail_v, _, actual_len = cache.get_fused_attention_inputs()
    if block is not None:
        k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
        v_dense = v_dequant.dequantize_block(quant_v).reshape(1, 4, -1, 128)[:, :, :actual_len, :]
        if tail_k is not None and tail_k.shape[2] > 0:
            k_dense = mx.concatenate([k_dense, tail_k[:, :, :actual_len, :]], axis=2)
            v_dense = mx.concatenate([v_dense, tail_v[:, :, :actual_len, :]], axis=2)
    else:
        k_dense = tail_k[:, :, :actual_len, :]
        v_dense = tail_v[:, :, :actual_len, :]

    q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    dense_out = _dense_attention(q, k_dense, v_dense, scale)

    # Paged online attention.
    bridge = MetalKernelBridge()
    paged_out, _ = bridge.execute_paged_online_attention(
        q.squeeze(2),
        view.pages,
        view.partial_k,
        view.partial_v,
        config,
        view.total_tokens,
    )

    assert mx.allclose(paged_out, dense_out, atol=2e-3), (
        f"Paged attention mismatch for {num_tokens} tokens"
    )


def test_paged_attention_zero_pages_tail_only():
    """Paged attention with only a partial tail must match dense."""
    config = _make_config()
    k, v = _random_kv(1, 4, 32, 128)
    scale = 1.0 / (128 ** 0.5)

    cache = TurboPolarKVCacheRuntime(config)
    cache.append(k, v)

    view = cache.attention_view()
    assert len(view.pages) == 0
    assert view.partial_length == 32

    q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    dense_out = _dense_attention(q, k, v, scale)

    bridge = MetalKernelBridge()
    paged_out, _ = bridge.execute_paged_online_attention(
        q.squeeze(2),
        view.pages,
        view.partial_k,
        view.partial_v,
        config,
        view.total_tokens,
    )
    assert mx.allclose(paged_out, dense_out, atol=2e-3)


def test_paged_attention_crosses_page_boundary():
    """65 tokens = 1 full block + 1 tail; crosses boundary."""
    config = _make_config()
    k, v = _random_kv(1, 4, 65, 128)
    scale = 1.0 / (128 ** 0.5)

    cache = TurboPolarKVCacheRuntime(config)
    cache.append(k, v)

    view = cache.attention_view()
    assert len(view.pages) == 1
    assert view.pages[0].valid_blocks == 1
    assert view.partial_length == 1

    # Dense reference from decompressed cache.
    from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
    from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
    decoder = PolarQuantDecoder()
    v_dequant = GroupedVQuantizer(group_size=32)
    block, quant_v, tail_k, tail_v, _, actual_len = cache.get_fused_attention_inputs()
    k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
    v_dense = v_dequant.dequantize_block(quant_v).reshape(1, 4, -1, 128)[:, :, :actual_len, :]
    if tail_k is not None and tail_k.shape[2] > 0:
        k_dense = mx.concatenate([k_dense, tail_k[:, :, :actual_len, :]], axis=2)
        v_dense = mx.concatenate([v_dense, tail_v[:, :, :actual_len, :]], axis=2)

    q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    dense_out = _dense_attention(q, k_dense, v_dense, scale)

    bridge = MetalKernelBridge()
    paged_out, _ = bridge.execute_paged_online_attention(
        q.squeeze(2),
        view.pages,
        view.partial_k,
        view.partial_v,
        config,
        view.total_tokens,
    )
    assert mx.allclose(paged_out, dense_out, atol=2e-3)


def test_paged_attention_multiple_pages():
    """256 tokens = 4 full pages."""
    config = _make_config()
    k, v = _random_kv(1, 4, 256, 128)
    scale = 1.0 / (128 ** 0.5)

    cache = TurboPolarKVCacheRuntime(config)
    cache.append(k, v)

    view = cache.attention_view()
    assert len(view.pages) == 1  # 256/64 = 4 blocks, all in one page (capacity 16)
    assert view.pages[0].valid_blocks == 4
    assert view.partial_length == 0

    # Dense reference from decompressed cache.
    from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
    from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
    decoder = PolarQuantDecoder()
    v_dequant = GroupedVQuantizer(group_size=32)
    block, quant_v, tail_k, tail_v, _, actual_len = cache.get_fused_attention_inputs()
    k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
    v_dense = v_dequant.dequantize_block(quant_v).reshape(1, 4, -1, 128)[:, :, :actual_len, :]
    if tail_k is not None and tail_k.shape[2] > 0:
        k_dense = mx.concatenate([k_dense, tail_k[:, :, :actual_len, :]], axis=2)
        v_dense = mx.concatenate([v_dense, tail_v[:, :, :actual_len, :]], axis=2)

    q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    dense_out = _dense_attention(q, k_dense, v_dense, scale)

    bridge = MetalKernelBridge()
    paged_out, _ = bridge.execute_paged_online_attention(
        q.squeeze(2),
        view.pages,
        view.partial_k,
        view.partial_v,
        config,
        view.total_tokens,
    )
    assert mx.allclose(paged_out, dense_out, atol=2e-3)


def test_paged_attention_two_pages_plus_tail():
    """129 tokens = 2 full pages (128) + 1 tail."""
    config = _make_config()
    k, v = _random_kv(1, 4, 129, 128)
    scale = 1.0 / (128 ** 0.5)

    cache = TurboPolarKVCacheRuntime(config)
    cache.append(k, v)

    view = cache.attention_view()
    # 129 tokens = 2 blocks + 1 tail. 2 blocks fit in 1 page (capacity 16).
    assert len(view.pages) == 1
    assert view.pages[0].valid_blocks == 2
    assert view.partial_length == 1

    # Dense reference from decompressed cache.
    from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
    from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
    decoder = PolarQuantDecoder()
    v_dequant = GroupedVQuantizer(group_size=32)
    block, quant_v, tail_k, tail_v, _, actual_len = cache.get_fused_attention_inputs()
    k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
    v_dense = v_dequant.dequantize_block(quant_v).reshape(1, 4, -1, 128)[:, :, :actual_len, :]
    if tail_k is not None and tail_k.shape[2] > 0:
        k_dense = mx.concatenate([k_dense, tail_k[:, :, :actual_len, :]], axis=2)
        v_dense = mx.concatenate([v_dense, tail_v[:, :, :actual_len, :]], axis=2)

    q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    dense_out = _dense_attention(q, k_dense, v_dense, scale)

    bridge = MetalKernelBridge()
    paged_out, _ = bridge.execute_paged_online_attention(
        q.squeeze(2),
        view.pages,
        view.partial_k,
        view.partial_v,
        config,
        view.total_tokens,
    )
    assert mx.allclose(paged_out, dense_out, atol=2e-3)


def test_fast_cache_decode_attention_uses_paged_path():
    """TurboPolarFastCache.decode_attention must produce finite output."""
    config = _make_config()
    fast_cache = TurboPolarFastCache(config)

    k = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    v = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

    # Append 128 tokens so we have 2 full blocks.
    for _ in range(128):
        fast_cache.runtime.append(k, v)

    out = fast_cache.decode_attention(q, k, v, scale=1.0 / (128 ** 0.5))
    assert mx.isfinite(out).all().item()
    assert out.shape == (1, 4, 128)

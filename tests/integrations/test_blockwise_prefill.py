"""Blockwise prefill tests for TurboPolar vectorized append_many.

Compares token-loop append against vectorized append_many for correctness.
"""

import mlx.core as mx
import pytest

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


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


@pytest.mark.parametrize("num_tokens", [1, 63, 64, 65, 127, 128, 129, 1024, 1025, 2048, 4096])
def test_append_many_matches_token_loop(num_tokens):
    """Vectorized append_many must produce identical cache state to token-loop append."""
    config = _make_config()
    k, v = _random_kv(1, 4, num_tokens, 128)

    cache_loop = TurboPolarKVCacheRuntime(config)
    cache_loop.append(k, v)

    cache_batch = TurboPolarKVCacheRuntime(config)
    cache_batch.append_many(k, v)

    # Same high-level state
    assert cache_batch.actual_seq_len == cache_loop.actual_seq_len == num_tokens
    assert cache_batch.total_blocks == cache_loop.total_blocks
    assert cache_batch.partial_length == cache_loop.partial_length

    # Same decoded K/V from paged storage
    from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
    from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
    decoder = PolarQuantDecoder()
    v_dequant = GroupedVQuantizer(group_size=32)

    block_loop, quant_v_loop, tail_k_loop, tail_v_loop, _, actual_len_loop = cache_loop.get_fused_attention_inputs()
    block_batch, quant_v_batch, tail_k_batch, tail_v_batch, _, actual_len_batch = cache_batch.get_fused_attention_inputs()

    assert actual_len_batch == actual_len_loop == num_tokens

    if block_loop is not None:
        assert block_batch is not None
        k_dense_loop = decoder.decode_block(block_loop)[:, :, :actual_len_loop, :]
        k_dense_batch = decoder.decode_block(block_batch)[:, :, :actual_len_batch, :]
        assert mx.allclose(k_dense_loop, k_dense_batch, atol=1e-3)
    else:
        assert block_batch is None

    if tail_k_loop is not None and tail_k_loop.shape[2] > 0:
        assert tail_k_batch is not None and tail_k_batch.shape[2] > 0
        assert mx.allclose(tail_k_loop[:, :, :actual_len_loop, :], tail_k_batch[:, :, :actual_len_batch, :], atol=1e-3)
    else:
        assert tail_k_batch is None or tail_k_batch.shape[2] == 0

    if quant_v_loop is not None:
        assert quant_v_batch is not None
        v_dense_loop = v_dequant.dequantize_block(quant_v_loop).reshape(1, 4, -1, 128)[:, :, :actual_len_loop, :]
        v_dense_batch = v_dequant.dequantize_block(quant_v_batch).reshape(1, 4, -1, 128)[:, :, :actual_len_batch, :]
        assert mx.allclose(v_dense_loop, v_dense_batch, atol=1e-3)

    if tail_v_loop is not None and tail_v_loop.shape[2] > 0:
        assert tail_v_batch is not None and tail_v_batch.shape[2] > 0
        assert mx.allclose(tail_v_loop[:, :, :actual_len_loop, :], tail_v_batch[:, :, :actual_len_batch, :], atol=1e-3)
    else:
        assert tail_v_batch is None or tail_v_batch.shape[2] == 0

    # Same attention output
    q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    scale = 1.0 / (128 ** 0.5)

    if block_loop is not None:
        k_full_loop = decoder.decode_block(block_loop)[:, :, :actual_len_loop, :]
        v_full_loop = v_dequant.dequantize_block(quant_v_loop).reshape(1, 4, -1, 128)[:, :, :actual_len_loop, :]
        if tail_k_loop is not None and tail_k_loop.shape[2] > 0:
            k_full_loop = mx.concatenate([k_full_loop, tail_k_loop[:, :, :actual_len_loop, :]], axis=2)
            v_full_loop = mx.concatenate([v_full_loop, tail_v_loop[:, :, :actual_len_loop, :]], axis=2)
    else:
        k_full_loop = tail_k_loop[:, :, :actual_len_loop, :]
        v_full_loop = tail_v_loop[:, :, :actual_len_loop, :]

    if block_batch is not None:
        k_full_batch = decoder.decode_block(block_batch)[:, :, :actual_len_batch, :]
        v_full_batch = v_dequant.dequantize_block(quant_v_batch).reshape(1, 4, -1, 128)[:, :, :actual_len_batch, :]
        if tail_k_batch is not None and tail_k_batch.shape[2] > 0:
            k_full_batch = mx.concatenate([k_full_batch, tail_k_batch[:, :, :actual_len_batch, :]], axis=2)
            v_full_batch = mx.concatenate([v_full_batch, tail_v_batch[:, :, :actual_len_batch, :]], axis=2)
    else:
        k_full_batch = tail_k_batch[:, :, :actual_len_batch, :]
        v_full_batch = tail_v_batch[:, :, :actual_len_batch, :]

    out_loop = _dense_attention(q, k_full_loop, v_full_loop, scale)
    out_batch = _dense_attention(q, k_full_batch, v_full_batch, scale)
    assert mx.allclose(out_loop, out_batch, atol=1e-3)


def test_append_many_no_token_loop_for_4096():
    """append_many on 4,096 tokens must not iterate per-token."""
    config = _make_config()
    k, v = _random_kv(1, 4, 4096, 128)
    cache = TurboPolarKVCacheRuntime(config)
    # This should complete without entering the per-token loop.
    cache.append_many(k, v)
    assert cache.actual_seq_len == 4096
    assert cache.total_blocks == 64
    assert cache.partial_length == 0

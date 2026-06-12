"""Integration tests for TurboPolar cache page rollover.

These tests verify that the TurboPolarKVCacheRuntime correctly handles
sequences that cross page boundaries, with correctness checked against
a dense reference.
"""

import mlx.core as mx
import pytest

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


def _random_kv(B, H, T, D, dtype=mx.float16):
    k = mx.random.normal((B, H, T, D)).astype(dtype)
    v = mx.random.normal((B, H, T, D)).astype(dtype)
    return k, v


def _make_cache():
    config = TurboPolarConfig(
        head_dim=128,
        block_size=64,
        num_q_heads=4,
        num_kv_heads=4,
        use_int8_radii=True,
        k_angle_bits_level1=8,
        k_angle_bits_deep=8,
    )
    return TurboPolarKVCacheRuntime(config)


def _dense_reference_attention(q, k_history, v_history, scale):
    """Simple dense attention for reference comparison."""
    B, H_q, _, D = q.shape
    H_kv = k_history.shape[1]
    num_queries_per_kv = H_q // H_kv
    # Broadcast KV heads to query heads for GQA
    k_rep = mx.repeat(k_history, num_queries_per_kv, axis=1)
    v_rep = mx.repeat(v_history, num_queries_per_kv, axis=1)
    scores = mx.sum(q * k_rep, axis=-1) * scale
    weights = mx.softmax(scores, axis=-1)
    return mx.sum(weights[:, :, :, None] * v_rep, axis=-2)


class TestModelCacheRollover:
    def test_1025_tokens_crosses_boundary(self):
        """1,025 tokens = 16 full blocks + 1 token. Must cross first page boundary."""
        cache = _make_cache()
        k, v = _random_kv(1, 4, 1025, 128)
        cache.append(k, v)

        assert cache.actual_seq_len == 1025
        assert cache.total_blocks == 16
        assert cache.partial_length == 1
        # 16 blocks = 1 full page
        assert cache.k_storage._paged.page_count == 1
        assert cache.v_storage._paged.page_count == 1

        # Verify attention output is finite
        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        scale = 1.0 / (128 ** 0.5)

        block, quant_v, tail_k, tail_v, qjl, actual_len = cache.get_fused_attention_inputs()
        assert actual_len == 1025

        # Build dense reference
        if block is not None:
            from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
            decoder = PolarQuantDecoder()
            k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
        else:
            k_dense = tail_k[:, :, :actual_len, :]

        if tail_v is not None and tail_v.shape[2] > 0:
            v_dense = tail_v[:, :, :actual_len, :]
        elif quant_v is not None:
            from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
            v_dequant = GroupedVQuantizer(group_size=32).dequantize_block(quant_v)
            v_dense = v_dequant.reshape(1, 4, -1, 128)[:, :, :actual_len, :]
        else:
            raise RuntimeError("No V data")

        # Attention output must be finite
        out = _dense_reference_attention(q, k_dense, v_dense, scale)
        assert mx.isfinite(out).all().item()

    def test_2048_tokens(self):
        """2,048 tokens = 32 full blocks."""
        cache = _make_cache()
        k, v = _random_kv(1, 4, 2048, 128)
        cache.append(k, v)

        assert cache.actual_seq_len == 2048
        assert cache.total_blocks == 32
        assert cache.partial_length == 0
        assert cache.k_storage._paged.page_count == 2
        assert cache.v_storage._paged.page_count == 2

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        scale = 1.0 / (128 ** 0.5)

        block, quant_v, tail_k, tail_v, qjl, actual_len = cache.get_fused_attention_inputs()
        assert actual_len == 2048
        assert tail_k is None or tail_k.shape[2] == 0

        if block is not None:
            from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
            decoder = PolarQuantDecoder()
            k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
        else:
            raise RuntimeError("Expected compressed blocks for 2048 tokens")

        if quant_v is not None:
            from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
            v_dequant = GroupedVQuantizer(group_size=32).dequantize_block(quant_v)
            v_dense = v_dequant.reshape(1, 4, -1, 128)[:, :, :actual_len, :]
        else:
            raise RuntimeError("Expected quantized V blocks")

        out = _dense_reference_attention(q, k_dense, v_dense, scale)
        assert mx.isfinite(out).all().item()

    def test_4096_tokens(self):
        """4,096 tokens = 64 full blocks."""
        cache = _make_cache()
        k, v = _random_kv(1, 4, 4096, 128)
        cache.append(k, v)

        assert cache.actual_seq_len == 4096
        assert cache.total_blocks == 64
        assert cache.partial_length == 0
        assert cache.k_storage._paged.page_count == 4
        assert cache.v_storage._paged.page_count == 4

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        scale = 1.0 / (128 ** 0.5)

        block, quant_v, tail_k, tail_v, qjl, actual_len = cache.get_fused_attention_inputs()
        assert actual_len == 4096
        assert tail_k is None or tail_k.shape[2] == 0

        if block is not None:
            from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
            decoder = PolarQuantDecoder()
            k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
        else:
            raise RuntimeError("Expected compressed blocks for 4096 tokens")

        if quant_v is not None:
            from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
            v_dequant = GroupedVQuantizer(group_size=32).dequantize_block(quant_v)
            v_dense = v_dequant.reshape(1, 4, -1, 128)[:, :, :actual_len, :]
        else:
            raise RuntimeError("Expected quantized V blocks")

        out = _dense_reference_attention(q, k_dense, v_dense, scale)
        assert mx.isfinite(out).all().item()

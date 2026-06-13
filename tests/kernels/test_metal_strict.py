"""Strict no-fallback Metal attention tests.

Every test in this file uses ExecutionMode.METAL_STRICT.
Any unavailable kernel, compilation error, dispatch error, or fallback
is a test failure.
"""

import mlx.core as mx
import pytest

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode
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
        storage_mode="kv_quant",
    )


def _dense_attention(q, k_hist, v_hist, scale):
    """q: [B, Hq, 1, D]; k/v: [B, Hkv, T, D]. GQA expansion handled."""
    B, H_q, _, D = q.shape
    H_kv = k_hist.shape[1]
    nq = H_q // H_kv
    k_rep = mx.repeat(k_hist, nq, axis=1)
    v_rep = mx.repeat(v_hist, nq, axis=1)
    scores = mx.sum(q * k_rep, axis=-1) * scale
    weights = mx.softmax(scores, axis=-1)
    return mx.sum(weights[:, :, :, None] * v_rep, axis=-2).astype(mx.float16)


def _decode_dense_reference(cache, q, scale):
    """Decode the cache to produce a dense K/V reference for comparison."""
    from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
    from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer

    decoder = PolarQuantDecoder()
    v_dequant = GroupedVQuantizer(group_size=32)

    block, quant_v, tail_k, tail_v, _, actual_len = cache.get_fused_attention_inputs()
    if block is not None:
        k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
        num_kv_heads = block.radii.shape[1]
        v_dense = v_dequant.dequantize_block(quant_v).reshape(
            1, num_kv_heads, -1, 128
        )[:, :, :actual_len, :]
        if tail_k is not None and tail_k.shape[2] > 0:
            k_dense = mx.concatenate([k_dense, tail_k[:, :, :actual_len, :]], axis=2)
            v_dense = mx.concatenate([v_dense, tail_v[:, :, :actual_len, :]], axis=2)
    else:
        k_dense = tail_k[:, :, :actual_len, :]
        v_dense = tail_v[:, :, :actual_len, :]

    return _dense_attention(q, k_dense, v_dense, scale)


@pytest.mark.native_metal_required
class TestMetalStrictPagedAttention:
    """Strict Metal paged attention must never fall back."""

    def test_strict_single_page_no_tail(self):
        """64 tokens = 1 full page, no tail."""
        mx.random.seed(4000)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache.append(k, v)

        view = cache.attention_view()
        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

        bridge = MetalKernelBridge()
        out, trace = bridge.execute_paged_online_attention(
            q.squeeze(2),
            view.pages,
            view.partial_k,
            view.partial_v,
            config,
            view.total_tokens,
            mode=ExecutionMode.METAL_STRICT,
        )

        dense_out = _decode_dense_reference(cache, q, config.attention_scale)
        mx.eval(out, dense_out)
        assert mx.allclose(
            out, dense_out, atol=5e-3
        ), f"Max diff: {mx.max(mx.abs(out - dense_out)).item()}"
        assert trace["execution_mode"] == "metal_strict"
        assert trace["fallback_used"] is False
        assert trace["attn_metal_used"] is True
        assert trace["dense_tail_metal"] is False

    def test_strict_single_page_plus_tail(self):
        """65 tokens = 1 full page + 1 tail token."""
        mx.random.seed(4001)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 65, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 65, 128)).astype(mx.float16)
        cache.append(k, v)

        view = cache.attention_view()
        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

        bridge = MetalKernelBridge()
        out, trace = bridge.execute_paged_online_attention(
            q.squeeze(2),
            view.pages,
            view.partial_k,
            view.partial_v,
            config,
            view.total_tokens,
            mode=ExecutionMode.METAL_STRICT,
        )

        dense_out = _decode_dense_reference(cache, q, config.attention_scale)
        mx.eval(out, dense_out)
        assert mx.allclose(
            out, dense_out, atol=5e-3
        ), f"Max diff: {mx.max(mx.abs(out - dense_out)).item()}"
        assert trace["execution_mode"] == "metal_strict"
        assert trace["fallback_used"] is False
        assert trace["attn_metal_used"] is True
        assert trace["dense_tail_metal"] is True

    def _run_strict_assertions(self, cache, q, expected_page_count=None):
        """Helper to run strict attention and assert no fallback with expected page topology."""
        view = cache.attention_view()
        bridge = MetalKernelBridge()
        out, trace = bridge.execute_paged_online_attention(
            q.squeeze(2),
            view.pages,
            view.partial_k,
            view.partial_v,
            cache.config,
            view.total_tokens,
            mode=ExecutionMode.METAL_STRICT,
        )
        dense_out = _decode_dense_reference(cache, q, cache.config.attention_scale)
        mx.eval(out, dense_out)
        assert mx.allclose(
            out, dense_out, atol=5e-3
        ), f"Max diff: {mx.max(mx.abs(out - dense_out)).item()}"
        assert trace["execution_mode"] == "metal_strict"
        assert trace["fallback_used"] is False
        assert trace["attn_metal_used"] is True
        if expected_page_count is not None:
            assert len(trace.get("page_traces", [])) == expected_page_count, (
                f"Expected {expected_page_count} page traces, got {len(trace.get('page_traces', []))}"
            )
        for pt in trace.get("page_traces", []):
            assert pt["metal_used"] is True, f"Page trace reported metal_used=False"
            assert pt["fallback_used"] is False, f"Page trace reported fallback_used=True"
        return out, trace

    def test_strict_multiple_blocks_one_page(self):
        """256 tokens = 4 full blocks, still within a single 1024-token page."""
        mx.random.seed(4002)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 256, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 256, 128)).astype(mx.float16)
        cache.append(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        self._run_strict_assertions(cache, q, expected_page_count=1)

    def test_strict_two_blocks_plus_one_tail(self):
        """129 tokens = 2 blocks + 1 tail, still within a single page."""
        mx.random.seed(4003)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 129, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 129, 128)).astype(mx.float16)
        cache.append(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        out, trace = self._run_strict_assertions(cache, q, expected_page_count=1)
        assert trace["dense_tail_metal"] is True

    def test_strict_exact_one_page_no_tail(self):
        """1024 tokens = exactly one full page (16 blocks), no tail."""
        mx.random.seed(4010)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 1024, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 1024, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        self._run_strict_assertions(cache, q, expected_page_count=1)

    def test_strict_one_page_plus_one_tail(self):
        """1025 tokens = one full page + one tail token."""
        mx.random.seed(4011)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 1025, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 1025, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        out, trace = self._run_strict_assertions(cache, q, expected_page_count=1)
        assert trace["dense_tail_metal"] is True

    def test_strict_two_pages_no_tail(self):
        """2048 tokens = exactly two full pages."""
        mx.random.seed(4012)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 2048, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 2048, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        self._run_strict_assertions(cache, q, expected_page_count=2)

    def test_strict_two_pages_plus_tail(self):
        """2049 tokens = two full pages + one tail token."""
        mx.random.seed(4013)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 2049, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 2049, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        out, trace = self._run_strict_assertions(cache, q, expected_page_count=2)
        assert trace["dense_tail_metal"] is True

    def test_strict_four_pages(self):
        """4096 tokens = four full pages."""
        mx.random.seed(4014)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 4096, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 4096, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        self._run_strict_assertions(cache, q, expected_page_count=4)

    def test_strict_eight_pages(self):
        """8192 tokens = eight full pages."""
        mx.random.seed(4015)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 8192, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 8192, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        self._run_strict_assertions(cache, q, expected_page_count=8)

    def test_strict_sixteen_pages(self):
        """16384 tokens = sixteen full pages."""
        mx.random.seed(4016)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 16384, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 16384, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        self._run_strict_assertions(cache, q, expected_page_count=16)

    def test_strict_tail_only_no_pages(self):
        """63 tokens = no compressed pages, only tail."""
        mx.random.seed(4004)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 63, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 63, 128)).astype(mx.float16)
        cache.append(k, v)

        view = cache.attention_view()
        assert len(view.pages) == 0

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

        bridge = MetalKernelBridge()
        out, trace = bridge.execute_paged_online_attention(
            q.squeeze(2),
            view.pages,
            view.partial_k,
            view.partial_v,
            config,
            view.total_tokens,
            mode=ExecutionMode.METAL_STRICT,
        )

        dense_out = _decode_dense_reference(cache, q, config.attention_scale)
        mx.eval(out, dense_out)
        assert mx.allclose(
            out, dense_out, atol=5e-3
        ), f"Max diff: {mx.max(mx.abs(out - dense_out)).item()}"
        assert trace["execution_mode"] == "metal_strict"
        assert trace["fallback_used"] is False
        assert trace["dense_tail_metal"] is True


class TestMetalStrictGQA:
    """Strict Metal paged attention with GQA ratio > 1 must never fall back."""

    def _run_strict_assertions(self, cache, q, expected_page_count=None):
        view = cache.attention_view()
        bridge = MetalKernelBridge()
        out, trace = bridge.execute_paged_online_attention(
            q.squeeze(2),
            view.pages,
            view.partial_k,
            view.partial_v,
            cache.config,
            view.total_tokens,
            mode=ExecutionMode.METAL_STRICT,
        )
        dense_out = _decode_dense_reference(cache, q, cache.config.attention_scale)
        mx.eval(out, dense_out)
        assert mx.allclose(
            out, dense_out, atol=5e-3
        ), f"Max diff: {mx.max(mx.abs(out - dense_out)).item()}"
        assert trace["execution_mode"] == "metal_strict"
        assert trace["fallback_used"] is False
        assert trace["attn_metal_used"] is True
        if expected_page_count is not None:
            assert len(trace.get("page_traces", [])) == expected_page_count, (
                f"Expected {expected_page_count} page traces, got {len(trace.get('page_traces', []))}"
            )
        for pt in trace.get("page_traces", []):
            assert pt["metal_used"] is True
            assert pt["fallback_used"] is False
        return out, trace

    def test_strict_gqa_two_pages(self):
        """2048 tokens = 2 pages with GQA ratio 4:1 (8 q heads, 2 kv heads)."""
        mx.random.seed(4100)
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=8,
            num_kv_heads=2,
            use_int8_radii=True,
            k_angle_bits_level1=8,
            k_angle_bits_deep=8,
            storage_mode="kv_quant",
        )
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 2, 2048, 128)).astype(mx.float16)
        v = mx.random.normal((1, 2, 2048, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 8, 1, 128)).astype(mx.float16)
        self._run_strict_assertions(cache, q, expected_page_count=2)

    def test_strict_gqa_four_pages(self):
        """4096 tokens = 4 pages with GQA ratio 4:1."""
        mx.random.seed(4101)
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=8,
            num_kv_heads=2,
            use_int8_radii=True,
            k_angle_bits_level1=8,
            k_angle_bits_deep=8,
            storage_mode="kv_quant",
        )
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 2, 4096, 128)).astype(mx.float16)
        v = mx.random.normal((1, 2, 4096, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 8, 1, 128)).astype(mx.float16)
        self._run_strict_assertions(cache, q, expected_page_count=4)

    def test_strict_gqa_sixteen_pages_plus_tail(self):
        """16385 tokens = 16 pages + 1 tail with GQA ratio 4:1."""
        mx.random.seed(4102)
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=8,
            num_kv_heads=2,
            use_int8_radii=True,
            k_angle_bits_level1=8,
            k_angle_bits_deep=8,
            storage_mode="kv_quant",
        )
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 2, 16385, 128)).astype(mx.float16)
        v = mx.random.normal((1, 2, 16385, 128)).astype(mx.float16)
        cache.append_many(k, v)

        q = mx.random.normal((1, 8, 1, 128)).astype(mx.float16)
        out, trace = self._run_strict_assertions(cache, q, expected_page_count=16)
        assert trace["dense_tail_metal"] is True

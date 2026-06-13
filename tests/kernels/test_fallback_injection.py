"""Fallback-injection tests verify graceful degradation when Metal kernels are unavailable.

These tests monkey-patch the MetalKernelBridge to simulate kernel unavailability
and assert that:
- DEVELOPMENT_AUTO falls back to the reference path with correct telemetry.
- METAL_STRICT raises MetalExecutionRequiredError instead of falling back.
"""

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache
from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode
from rfsn_v11.kernels.turbo_polar.metal import (
    MetalExecutionRequiredError,
    MetalKernelBridge,
)


def _decode_dense_reference(cache, q, scale):
    """Decode the cache to produce a dense K/V reference for comparison."""
    from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
    from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer

    decoder = PolarQuantDecoder()
    v_dequant = GroupedVQuantizer(group_size=32)

    block, quant_v, tail_k, tail_v, _, actual_len = cache.get_fused_attention_inputs()
    if block is not None:
        k_dense = decoder.decode_block(block)[:, :, :actual_len, :]
        v_dense = v_dequant.dequantize_block(quant_v).reshape(1, 4, -1, 128)[
            :, :, :actual_len, :
        ]
        if tail_k is not None and tail_k.shape[2] > 0:
            k_dense = mx.concatenate([k_dense, tail_k[:, :, :actual_len, :]], axis=2)
            v_dense = mx.concatenate([v_dense, tail_v[:, :, :actual_len, :]], axis=2)
    else:
        k_dense = tail_k[:, :, :actual_len, :]
        v_dense = tail_v[:, :, :actual_len, :]

    return _dense_attention(q, k_dense, v_dense, scale)


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
    B, H_q, _, D = q.shape
    H_kv = k_hist.shape[1]
    nq = H_q // H_kv
    k_rep = mx.repeat(k_hist, nq, axis=1)
    v_rep = mx.repeat(v_hist, nq, axis=1)
    scores = mx.sum(q * k_rep, axis=-1) * scale
    weights = mx.softmax(scores, axis=-1)
    return mx.sum(weights[:, :, :, None] * v_rep, axis=-2).astype(mx.float16)


class TestFallbackInjection:
    """Inject kernel unavailability and verify fallback behavior."""

    def test_auto_falls_back_when_raw_kernel_unavailable(self):
        """DEVELOPMENT_AUTO must fall back to reference if raw kernel is missing."""
        mx.random.seed(5000)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache.append(k, v)

        view = cache.attention_view()
        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

        bridge = MetalKernelBridge()

        # Inject unavailability by nulling the raw kernel.
        original_kernel = bridge._kernel_attn_quant_raw
        bridge._kernel_attn_quant_raw = None
        try:
            out, trace = bridge.execute_paged_online_attention(
                q.squeeze(2),
                view.pages,
                view.partial_k,
                view.partial_v,
                config,
                view.total_tokens,
                mode=ExecutionMode.DEVELOPMENT_AUTO,
            )
            mx.eval(out)
            assert trace["execution_mode"] == "reference"
            assert trace["fallback_used"] is True
            # Verify output still matches decoded dense reference (fallback path).
            dense_out = _decode_dense_reference(cache, q, config.attention_scale)
            assert mx.allclose(out, dense_out, atol=5e-3)
        finally:
            bridge._kernel_attn_quant_raw = original_kernel

    def test_strict_raises_when_raw_kernel_unavailable(self):
        """METAL_STRICT must raise MetalExecutionRequiredError, never fall back."""
        mx.random.seed(5001)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache.append(k, v)

        view = cache.attention_view()
        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

        bridge = MetalKernelBridge()

        original_kernel = bridge._kernel_attn_quant_raw
        bridge._kernel_attn_quant_raw = None
        try:
            out, trace = bridge.execute_paged_online_attention(
                q.squeeze(2),
                view.pages,
                view.partial_k,
                view.partial_v,
                config,
                view.total_tokens,
                mode=ExecutionMode.METAL_STRICT,
            )
            mx.eval(out)
            raise AssertionError("Expected MetalExecutionRequiredError")
        except MetalExecutionRequiredError:
            pass
        finally:
            bridge._kernel_attn_quant_raw = original_kernel

    def test_auto_falls_back_when_dense_tail_kernel_unavailable(self):
        """DEVELOPMENT_AUTO must fall back if dense-tail raw kernel is missing."""
        mx.random.seed(5002)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 65, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 65, 128)).astype(mx.float16)
        cache.append(k, v)

        view = cache.attention_view()
        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

        bridge = MetalKernelBridge()

        original_kernel = bridge._kernel_dense_tail_raw
        bridge._kernel_dense_tail_raw = None
        try:
            out, trace = bridge.execute_paged_online_attention(
                q.squeeze(2),
                view.pages,
                view.partial_k,
                view.partial_v,
                config,
                view.total_tokens,
                mode=ExecutionMode.DEVELOPMENT_AUTO,
            )
            mx.eval(out)
            assert trace["execution_mode"] == "reference"
            assert trace["fallback_used"] is True
            # Verify output still matches decoded dense reference.
            dense_out = _decode_dense_reference(cache, q, config.attention_scale)
            assert mx.allclose(out, dense_out, atol=5e-3)
        finally:
            bridge._kernel_dense_tail_raw = original_kernel

    def test_strict_raises_when_dense_tail_kernel_unavailable(self):
        """METAL_STRICT must raise when dense-tail raw kernel is missing."""
        mx.random.seed(5003)
        config = _make_config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 65, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 65, 128)).astype(mx.float16)
        cache.append(k, v)

        view = cache.attention_view()
        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

        bridge = MetalKernelBridge()

        original_kernel = bridge._kernel_dense_tail_raw
        bridge._kernel_dense_tail_raw = None
        try:
            out, trace = bridge.execute_paged_online_attention(
                q.squeeze(2),
                view.pages,
                view.partial_k,
                view.partial_v,
                config,
                view.total_tokens,
                mode=ExecutionMode.METAL_STRICT,
            )
            mx.eval(out)
            raise AssertionError("Expected MetalExecutionRequiredError")
        except MetalExecutionRequiredError:
            pass
        finally:
            bridge._kernel_dense_tail_raw = original_kernel

    def test_auto_no_fallback_when_all_kernels_available(self):
        """DEVELOPMENT_AUTO must NOT fall back when all kernels are healthy."""
        mx.random.seed(5004)
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
            mode=ExecutionMode.DEVELOPMENT_AUTO,
        )
        mx.eval(out)
        assert trace["fallback_used"] is False
        assert trace["attn_metal_used"] is True
        assert trace["dense_tail_metal"] is True

    def test_strict_cache_decode_raises_on_kernel_unavailable(self):
        """TurboPolarFastCache.decode_attention in strict mode must raise."""
        mx.random.seed(5005)
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            use_int8_radii=True,
            k_angle_bits_level1=8,
            k_angle_bits_deep=8,
            storage_mode="kv_quant",
            execution_mode=ExecutionMode.METAL_STRICT,
        )
        cache = TurboPolarFastCache(config)

        k = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)

        # Pre-warm the runtime with a few tokens so there is a page to process.
        for _ in range(65):
            cache.decode_attention(q, k, v, config.attention_scale)

        # Inject unavailability by nulling the raw kernel on the bridge.
        original_kernel = cache.bridge._kernel_attn_quant_raw
        cache.bridge._kernel_attn_quant_raw = None
        try:
            try:
                cache.decode_attention(q, k, v, config.attention_scale)
                raise AssertionError("Expected MetalExecutionRequiredError")
            except MetalExecutionRequiredError:
                pass
        finally:
            cache.bridge._kernel_attn_quant_raw = original_kernel

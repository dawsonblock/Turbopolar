"""Telemetry ownership tests.

MetalKernelBridge is a singleton. Stats must be process-global and read once.
"""

import pytest
import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache
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


@pytest.mark.native_metal_required
class TestTelemetryOwnership:
    def test_two_caches_share_one_bridge(self):
        """Two caches must share the same MetalKernelBridge singleton."""
        config = _make_config()
        cache_a = TurboPolarFastCache(config)
        cache_b = TurboPolarFastCache(config)
        assert cache_a.bridge is cache_b.bridge

    def test_each_cache_returns_equal_global_totals(self):
        """Stats from two caches must be identical because they share one bridge."""
        mx.random.seed(6000)
        config = _make_config()
        cache_a = TurboPolarFastCache(config)
        cache_b = TurboPolarFastCache(config)

        # Build state in cache_a.
        k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache_a.runtime.append(k, v)

        view = cache_a.runtime.attention_view()
        q = mx.random.normal((1, 4, 128)).astype(mx.float16)
        _ = cache_a.bridge.execute_paged_online_attention(
            q, view.pages, view.partial_k, view.partial_v,
            config, view.total_tokens, mode=ExecutionMode.METAL_STRICT,
        )

        stats_a = cache_a.execution_stats()
        stats_b = cache_b.execution_stats()
        assert stats_a == stats_b

    def test_reading_from_two_caches_must_not_be_summed(self):
        """Summing stats from two caches would double-count."""
        mx.random.seed(6001)
        config = _make_config()
        cache_a = TurboPolarFastCache(config)
        cache_b = TurboPolarFastCache(config)

        k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache_a.runtime.append(k, v)

        view = cache_a.runtime.attention_view()
        q = mx.random.normal((1, 4, 128)).astype(mx.float16)
        _ = cache_a.bridge.execute_paged_online_attention(
            q, view.pages, view.partial_k, view.partial_v,
            config, view.total_tokens, mode=ExecutionMode.METAL_STRICT,
        )

        stats_a = cache_a.execution_stats()
        stats_b = cache_b.execution_stats()
        # If someone mistakenly summed them, they'd get double.
        assert stats_a.attention_invocations == stats_b.attention_invocations
        assert stats_a.attention_invocations > 0

    def test_reset_through_bridge_clears_all_cache_visible_totals(self):
        """Reset on one cache must zero stats visible from all caches."""
        mx.random.seed(6002)
        config = _make_config()
        cache_a = TurboPolarFastCache(config)
        cache_b = TurboPolarFastCache(config)

        k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache_a.runtime.append(k, v)
        view = cache_a.runtime.attention_view()
        q = mx.random.normal((1, 4, 128)).astype(mx.float16)
        _ = cache_a.bridge.execute_paged_online_attention(
            q, view.pages, view.partial_k, view.partial_v,
            config, view.total_tokens, mode=ExecutionMode.METAL_STRICT,
        )

        assert cache_a.execution_stats().attention_invocations > 0
        cache_a.reset_execution_stats()
        assert cache_a.execution_stats().attention_invocations == 0
        assert cache_b.execution_stats().attention_invocations == 0

    def test_two_sequential_experiments_do_not_contaminate(self):
        """Stats from experiment A must not leak into experiment B."""
        mx.random.seed(6003)
        config = _make_config()
        cache = TurboPolarFastCache(config)

        # Experiment A.
        k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache.runtime.append(k, v)
        view_a = cache.runtime.attention_view()
        q_a = mx.random.normal((1, 4, 128)).astype(mx.float16)
        _ = cache.bridge.execute_paged_online_attention(
            q_a, view_a.pages, view_a.partial_k, view_a.partial_v,
            config, view_a.total_tokens, mode=ExecutionMode.METAL_STRICT,
        )
        stats_a = cache.execution_stats()

        # Reset before experiment B.
        cache.reset_execution_stats()

        # Experiment B: different state, one dispatch.
        k2 = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v2 = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache.runtime.append(k2, v2)
        view_b = cache.runtime.attention_view()
        q_b = mx.random.normal((1, 4, 128)).astype(mx.float16)
        _ = cache.bridge.execute_paged_online_attention(
            q_b, view_b.pages, view_b.partial_k, view_b.partial_v,
            config, view_b.total_tokens, mode=ExecutionMode.METAL_STRICT,
        )
        stats_b = cache.execution_stats()

        # B should have exactly 1 invocation, not A+B.
        assert stats_b.attention_invocations == 1
        assert stats_a.attention_invocations == 1

    def test_expected_dispatch_count_matches_actual(self):
        """For a two-layer model with two compressed pages and N decode steps:
        expected compressed-page dispatches = 2 layers x 2 pages x N steps.
        The reported total must equal that exact number, not twice that number.
        """
        mx.random.seed(6004)
        config = _make_config()
        cache = TurboPolarFastCache(config)
        cache.reset_execution_stats()

        # Build 2 pages worth of state (128 tokens, 2 blocks of 64).
        k = mx.random.normal((1, 4, 128, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 128, 128)).astype(mx.float16)
        cache.runtime.append(k, v)

        view = cache.runtime.attention_view()
        q = mx.random.normal((1, 4, 128)).astype(mx.float16)
        _ = cache.bridge.execute_paged_online_attention(
            q, view.pages, view.partial_k, view.partial_v,
            config, view.total_tokens, mode=ExecutionMode.METAL_STRICT,
        )

        stats = cache.execution_stats()
        # 128 tokens = 2 blocks of 64, but pages hold 16 blocks (1024 tokens) each,
        # so this is still 1 page dispatch.
        assert stats.compressed_page_dispatches == 1
        assert stats.attention_invocations == 1

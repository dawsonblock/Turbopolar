"""Truthful memory accounting tests for TurboPolarKVCacheRuntime."""

import unittest

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


class TestMemoryAccounting(unittest.TestCase):
    @staticmethod
    def _config(num_q_heads: int = 4, num_kv_heads: int = 2):
        return TurboPolarConfig(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=128,
            block_size=64,
            storage_mode="kv_quant",
            use_int8_radii=True,
            k_angle_bits_level1=8,
            k_angle_bits_deep=8,
        )

    def test_empty_cache_reports_zero(self):
        runtime = TurboPolarKVCacheRuntime(self._config())
        stats = runtime.get_memory_stats()
        self.assertEqual(stats.logical_payload_bytes, 0)
        self.assertEqual(stats.allocated_capacity_bytes, 0)
        self.assertEqual(stats.dense_tail_bytes, 0)
        self.assertEqual(stats.metadata_bytes, 0)
        self.assertEqual(stats.dense_equivalent_bytes, 0)
        self.assertEqual(stats.logical_compression_ratio, 0.0)
        self.assertEqual(stats.allocated_compression_ratio, 0.0)

    def test_partial_tail_logical_and_allocated(self):
        runtime = TurboPolarKVCacheRuntime(self._config())
        tokens = 17
        k = mx.random.normal((1, 2, tokens, 128), dtype=mx.float16)
        v = mx.random.normal((1, 2, tokens, 128), dtype=mx.float16)
        runtime.append(k, v)

        stats = runtime.get_memory_stats()
        B, H, T, D = 1, 2, tokens, 128
        bytes_per_token = 2 * D * 2  # fp16 K + fp16 V
        expected_logical_tail = B * H * T * bytes_per_token
        expected_allocated_tail = B * H * 64 * bytes_per_token

        self.assertEqual(stats.logical_payload_bytes, expected_logical_tail)
        self.assertEqual(stats.dense_tail_bytes, expected_logical_tail)
        self.assertEqual(stats.allocated_capacity_bytes, expected_allocated_tail)
        self.assertEqual(
            stats.dense_equivalent_bytes,
            B * H * T * D * 2 * 2,
        )

    def test_full_block_logical_matches_expected_compressed_sizes(self):
        runtime = TurboPolarKVCacheRuntime(self._config())
        L = 64
        B, H, D = 1, 2, 128
        k = mx.random.normal((B, H, L, D), dtype=mx.float16)
        v = mx.random.normal((B, H, L, D), dtype=mx.float16)
        runtime.append(k, v)

        stats = runtime.get_memory_stats()
        cfg = runtime.config
        # split_dim == 0 means all angle capacity goes to the deep bucket.
        half_d = D // 2
        split_half = cfg.split_dim // 2
        l1_dims = split_half
        deep_dims = half_d - split_half
        # K: int8 radii (half_d pairs) + l1 angles + deep angles + fp16 radii_scales.
        expected_k = (
            B * H * L * half_d
            + B * H * L * l1_dims
            + B * H * L * deep_dims
            + B * H * 1 * 1 * 2
        )
        # V: int8 codes + fp16 scales per group.
        num_groups = D // 32
        expected_v = B * H * 1 * L * D + B * H * 1 * L * num_groups * 2
        expected_logical = expected_k + expected_v
        # Buffers for the dense tail are allocated even when empty after a flush.
        allocated_tail_buffers = B * H * L * D * 2 * 2
        self.assertEqual(stats.logical_payload_bytes, expected_logical)
        self.assertEqual(
            stats.allocated_capacity_bytes,
            expected_logical + allocated_tail_buffers,
        )
        self.assertEqual(stats.dense_tail_bytes, 0)
        self.assertEqual(
            stats.dense_equivalent_bytes,
            B * H * L * D * 2 * 2,
        )

    def test_paged_storage_growth(self):
        runtime = TurboPolarKVCacheRuntime(self._config())
        L = 64
        B, H, D = 1, 2, 128
        for block in range(5):
            k = mx.random.normal((B, H, L, D), dtype=mx.float16)
            v = mx.random.normal((B, H, L, D), dtype=mx.float16)
            runtime.append(k, v)
            self.assertEqual(runtime.k_storage.block_count, block + 1)
            self.assertEqual(runtime.v_storage.block_count, block + 1)
            # Paged storage allocates in chunks of 16 blocks per page.
            self.assertEqual(runtime.k_storage.capacity, 16)
            self.assertEqual(runtime.v_storage.capacity, 16)
            # Only one page allocation for the first 16 blocks.
            self.assertEqual(runtime.k_storage.reallocation_count, 1)
            self.assertEqual(runtime.v_storage.reallocation_count, 1)

    def test_peak_memory_probe_reports_at_least_allocated(self):
        runtime = TurboPolarKVCacheRuntime(self._config())
        L = 64
        k = mx.random.normal((1, 2, L, 128), dtype=mx.float16)
        v = mx.random.normal((1, 2, L, 128), dtype=mx.float16)
        peak = runtime.measure_append_peak_memory(k, v)
        stats = runtime.get_memory_stats()
        # The allocator peak must cover the final allocated arrays once materialized.
        self.assertGreaterEqual(peak, stats.allocated_capacity_bytes)
        self.assertGreaterEqual(peak, stats.logical_payload_bytes)

    def test_peak_memory_probe_for_partial_tail(self):
        runtime = TurboPolarKVCacheRuntime(self._config())
        k = mx.random.normal((1, 2, 10, 128), dtype=mx.float16)
        v = mx.random.normal((1, 2, 10, 128), dtype=mx.float16)
        peak = runtime.measure_append_peak_memory(k, v)
        stats = runtime.get_memory_stats()
        self.assertGreaterEqual(peak, stats.allocated_capacity_bytes)


if __name__ == "__main__":
    unittest.main()

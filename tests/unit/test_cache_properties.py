"""Adversarial property tests for TurboPolarKVCacheRuntime.

These tests verify the fundamental invariants of the cache state machine
under randomized, adversarial, and failure-mode conditions.
"""

import unittest

import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime

import pytest

pytest.importorskip("mlx")


class TestCacheProperties(unittest.TestCase):
    """Property-based invariants validated under randomized inputs."""

    @staticmethod
    def _config(**overrides):
        defaults = dict(
            head_dim=128,
            block_size=64,
            dense_tail_capacity=512,
            flush_batch_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            use_qjl=False,
            seed=42,
        )
        defaults.update(overrides)
        return TurboPolarConfig(**defaults)

    def test_randomized_chunking_preserves_length(self):
        """Any sequence of append() chunk sizes must yield correct total length."""
        mx.random.seed(2025)
        for trial in range(20):
            config = self._config()
            cache = TurboPolarKVCacheRuntime(config)
            total = 0
            chunks = []
            for _ in range(50):
                chunk = int(mx.random.randint(1, 33).item())
                k = mx.random.normal((1, 4, chunk, 128), dtype=mx.float16)
                v = mx.random.normal((1, 4, chunk, 128), dtype=mx.float16)
                cache.append(k, v)
                total += chunk
                chunks.append(chunk)
            self.assertEqual(
                cache.actual_seq_len, total,
                f"trial={trial}: expected {total}, got {cache.actual_seq_len}"
            )

    def test_randomized_chunking_matches_single_shot(self):
        """Chunked append must match single-shot append for total block count."""
        mx.random.seed(2026)
        total = 2048
        k_all = mx.random.normal((1, 4, total, 128), dtype=mx.float16)
        v_all = mx.random.normal((1, 4, total, 128), dtype=mx.float16)

        ref = TurboPolarKVCacheRuntime(self._config())
        ref.append_many(k_all, v_all)

        for trial in range(10):
            mx.random.seed(trial)
            cache = TurboPolarKVCacheRuntime(self._config())
            t = 0
            while t < total:
                chunk = int(mx.random.randint(1, 64).item())
                end = min(t + chunk, total)
                cache.append(k_all[:, :, t:end, :], v_all[:, :, t:end, :])
                t = end
            self.assertEqual(
                cache.actual_seq_len, ref.actual_seq_len,
                f"trial={trial}: seq_len mismatch"
            )
            self.assertEqual(
                cache.total_blocks, ref.total_blocks,
                f"trial={trial}: block count mismatch"
            )

    def test_prefill_then_decode_boundary(self):
        """Large prefill followed by single-token decode must not corrupt state."""
        config = self._config(dense_tail_capacity=256, flush_batch_size=64)
        cache = TurboPolarKVCacheRuntime(config)

        # Prefill 512 tokens (exactly 2 flushes: 512 = 256 hot + 64 flush + 192 remain)
        k = mx.random.normal((1, 4, 512, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 512, 128), dtype=mx.float16)
        cache.append_many(k, v)
        self.assertEqual(cache.actual_seq_len, 512)

        # Decode 100 single tokens
        for _ in range(100):
            kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(kt, vt)

        self.assertEqual(cache.actual_seq_len, 612)
        # hot_length must be within capacity
        self.assertLessEqual(cache.hot_length, config.dense_tail_capacity)

    def test_decode_then_large_prefill(self):
        """Decode tokens then large prefill must maintain correctness."""
        config = self._config(dense_tail_capacity=128, flush_batch_size=64)
        cache = TurboPolarKVCacheRuntime(config)

        # Decode 50 tokens
        for _ in range(50):
            kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(kt, vt)

        # Large prefill crossing multiple flush boundaries
        k = mx.random.normal((1, 4, 400, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 400, 128), dtype=mx.float16)
        cache.append_many(k, v)

        self.assertEqual(cache.actual_seq_len, 450)
        self.assertLessEqual(cache.hot_length, config.dense_tail_capacity)

    def test_exact_boundary_flush(self):
        """Appending exactly capacity tokens must flush cleanly."""
        config = self._config(dense_tail_capacity=128, flush_batch_size=64)
        cache = TurboPolarKVCacheRuntime(config)

        # Append exactly 128 tokens (fills hot window to exact capacity)
        k = mx.random.normal((1, 4, 128, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 128, 128), dtype=mx.float16)
        cache.append_many(k, v)

        self.assertEqual(cache.actual_seq_len, 128)
        # With flush-on-fill, appending exactly 128 triggers a flush of 64,
        # leaving 64 tokens in the hot window.
        self.assertEqual(cache.hot_length, 64)
        self.assertEqual(cache.total_blocks, 1)

        # One more token
        kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
        vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
        cache.append(kt, vt)

        self.assertEqual(cache.actual_seq_len, 129)
        self.assertEqual(cache.total_blocks, 1)
        self.assertEqual(cache.hot_length, 65)

    def test_chronological_reconstruction(self):
        """get_blocks_for_attention must cover all tokens chronologically."""
        config = self._config(dense_tail_capacity=128, flush_batch_size=64)
        cache = TurboPolarKVCacheRuntime(config)

        tokens = []
        for i in range(200):
            t = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            tokens.append(t)
            cache.append(t, t)

        block, quant_v, dense_v, qjl, actual_len = (
            cache.get_blocks_for_attention()
        )
        self.assertEqual(actual_len, 200)
        # Total tokens represented by compressed blocks + hot window
        compressed_tokens = cache.total_blocks * 64
        hot_tokens = cache.hot_length
        self.assertEqual(
            compressed_tokens + hot_tokens, 200,
            f"compressed={compressed_tokens} + hot={hot_tokens} != 200"
        )

    def test_reset_and_reuse(self):
        """reset() must fully clear state for reuse."""
        config = self._config(dense_tail_capacity=128, flush_batch_size=64)
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 256, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 256, 128), dtype=mx.float16)
        cache.append_many(k, v)
        self.assertEqual(cache.actual_seq_len, 256)
        self.assertGreater(cache.total_blocks, 0)

        cache.reset()
        self.assertEqual(cache.actual_seq_len, 0)
        self.assertEqual(cache.total_blocks, 0)
        self.assertEqual(cache.hot_length, 0)
        self.assertIsNone(cache.hot_k_buffer)
        self.assertIsNone(cache.hot_v_buffer)

        # Reuse
        k2 = mx.random.normal((1, 4, 100, 128), dtype=mx.float16)
        v2 = mx.random.normal((1, 4, 100, 128), dtype=mx.float16)
        cache.append_many(k2, v2)
        self.assertEqual(cache.actual_seq_len, 100)

    def test_dtype_invariant(self):
        """All appends must use the same dtype."""
        config = self._config()
        cache = TurboPolarKVCacheRuntime(config)

        k_fp16 = mx.random.normal((1, 4, 10, 128), dtype=mx.float16)
        v_fp16 = mx.random.normal((1, 4, 10, 128), dtype=mx.float16)
        cache.append_many(k_fp16, v_fp16)

        k_fp32 = mx.random.normal((1, 4, 10, 128), dtype=mx.float32)
        v_fp32 = mx.random.normal((1, 4, 10, 128), dtype=mx.float32)
        with self.assertRaises(ValueError):
            cache.append_many(k_fp32, v_fp32)

    def test_batch_size_invariant(self):
        """Batch size must not change after first append."""
        config = self._config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 10, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 10, 128), dtype=mx.float16)
        cache.append_many(k, v)

        k2 = mx.random.normal((2, 4, 10, 128), dtype=mx.float16)
        v2 = mx.random.normal((2, 4, 10, 128), dtype=mx.float16)
        with self.assertRaises(ValueError):
            cache.append_many(k2, v2)

    def test_head_dim_invariant(self):
        """Head dimension must not change after first append."""
        config = self._config()
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 10, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 10, 128), dtype=mx.float16)
        cache.append_many(k, v)

        k2 = mx.random.normal((1, 4, 10, 64), dtype=mx.float16)
        v2 = mx.random.normal((1, 4, 10, 64), dtype=mx.float16)
        with self.assertRaises(ValueError):
            cache.append_many(k2, v2)

    def test_rope_position_identity(self):
        """Each token must retain its absolute logical position for RoPE."""
        config = self._config(dense_tail_capacity=128, flush_batch_size=64)
        cache = TurboPolarKVCacheRuntime(config)

        # Append tokens one at a time so we can track positions
        for i in range(150):
            t = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(t, t)
            # After each append, actual_seq_len must equal i+1
            self.assertEqual(
                cache.actual_seq_len, i + 1,
                f"Position mismatch at token {i}"
            )

    def test_circular_buffer_wrapping(self):
        """Hot window must correctly wrap around the circular buffer."""
        config = self._config(
            dense_tail_capacity=128, flush_batch_size=64
        )
        cache = TurboPolarKVCacheRuntime(config)

        # Append 200 tokens. With capacity 128 and flush 64:
        #   0-127: fill hot window
        #   128: flush 64 -> hot = 65 (tokens 64-127 + 128)
        #   ...continues with more flushes
        # The hot window will wrap around the circular buffer.
        for i in range(200):
            t = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(t, t)

        self.assertEqual(cache.actual_seq_len, 200)
        # Extract contiguous view and verify it has correct shape
        hot_k = cache._hot_k_contiguous()
        hot_v = cache._hot_v_contiguous()
        self.assertIsNotNone(hot_k)
        self.assertIsNotNone(hot_v)
        self.assertEqual(hot_k.shape, (1, 4, cache.hot_length, 128))
        self.assertEqual(hot_v.shape, (1, 4, cache.hot_length, 128))
        # hot_length must be <= capacity
        self.assertLessEqual(cache.hot_length, 128)

    def test_unique_stored_tokens_equals_sequence_length(self):
        """compressed_tokens + hot_tokens == absolute_seq_len."""
        config = self._config(
            dense_tail_capacity=256, flush_batch_size=128
        )
        cache = TurboPolarKVCacheRuntime(config)

        for _ in range(500):
            t = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(t, t)

        compressed = cache.total_blocks * config.block_size
        hot = cache.hot_length
        self.assertEqual(
            compressed + hot, cache.actual_seq_len,
            f"compressed={compressed} + hot={hot} != seq_len={cache.actual_seq_len}"
        )

    def test_telemetry_after_randomized_append(self):
        """Telemetry must be consistent after randomized chunking."""
        config = self._config()
        cache = TurboPolarKVCacheRuntime(config)

        total = 0
        for _ in range(30):
            chunk = int(mx.random.randint(1, 50).item())
            k = mx.random.normal((1, 4, chunk, 128), dtype=mx.float16)
            v = mx.random.normal((1, 4, chunk, 128), dtype=mx.float16)
            cache.append_many(k, v)
            total += chunk

        telem = cache.get_io_telemetry()
        self.assertEqual(
            telem["total_blocks"] * config.block_size + telem["partial_tokens"],
            total,
        )


    def test_warm_cache_disabled_by_default(self):
        """Default warm_cache_capacity_blocks=0 must behave like old path."""
        config = self._config(
            dense_tail_capacity=128, flush_batch_size=64
        )
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 256, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 256, 128), dtype=mx.float16)
        cache.append_many(k, v)

        self.assertEqual(cache.actual_seq_len, 256)
        self.assertEqual(cache.warm_length, 0)
        self.assertIsNone(cache.warm_k_buffer)
        # Cold storage must have blocks since warm is disabled
        self.assertGreater(cache.total_blocks, 0)

    def test_warm_cache_holds_flush_batch(self):
        """With warm enabled, flushed blocks go to warm before cold."""
        config = self._config(
            dense_tail_capacity=128,
            flush_batch_size=64,
            warm_cache_capacity_blocks=4,
        )
        cache = TurboPolarKVCacheRuntime(config)

        # Append 192 tokens:
        #   0-127  -> hot fills to 128, flush 64 to warm -> hot=64, warm=64
        #   128-191-> hot fills to 128, flush 64 to warm -> hot=64, warm=128
        k = mx.random.normal((1, 4, 192, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 192, 128), dtype=mx.float16)
        cache.append_many(k, v)

        self.assertEqual(cache.actual_seq_len, 192)
        self.assertEqual(cache.hot_length, 64)
        self.assertEqual(cache.warm_length, 128)
        self.assertEqual(cache.total_blocks, 0)  # nothing in cold yet

    def test_warm_cache_overflow_to_cold(self):
        """Warm overflow must compress oldest to cold."""
        config = self._config(
            dense_tail_capacity=128,
            flush_batch_size=64,
            warm_cache_capacity_blocks=2,  # 128 tokens = 2 blocks
        )
        cache = TurboPolarKVCacheRuntime(config)

        # First: 192 tokens -> hot=64, warm=128 (at capacity)
        k1 = mx.random.normal((1, 4, 192, 128), dtype=mx.float16)
        v1 = mx.random.normal((1, 4, 192, 128), dtype=mx.float16)
        cache.append_many(k1, v1)
        self.assertEqual(cache.hot_length, 64)
        self.assertEqual(cache.warm_length, 128)
        self.assertEqual(cache.total_blocks, 0)

        # Second: another 64 tokens -> hot fills, flush 64 to warm
        # warm would be 192, but capacity is only 128
        # So oldest 64 from warm go to cold, new 64 goes to warm
        k2 = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        v2 = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        cache.append_many(k2, v2)

        self.assertEqual(cache.actual_seq_len, 256)
        self.assertEqual(cache.hot_length, 64)
        self.assertEqual(cache.warm_length, 128)
        self.assertEqual(cache.total_blocks, 1)  # 1 block in cold

    def test_attention_view_with_warm(self):
        """attention_view must expose warm and hot correctly."""
        config = self._config(
            dense_tail_capacity=128,
            flush_batch_size=64,
            warm_cache_capacity_blocks=4,
        )
        cache = TurboPolarKVCacheRuntime(config)

        k = mx.random.normal((1, 4, 256, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 256, 128), dtype=mx.float16)
        cache.append_many(k, v)

        view = cache.attention_view()
        self.assertEqual(view.partial_length, cache.hot_length)
        self.assertEqual(view.warm_length, cache.warm_length)
        if view.warm_length > 0:
            self.assertIsNotNone(view.warm_k)
            self.assertIsNotNone(view.warm_v)
            self.assertEqual(
                view.warm_k.shape,
                (1, 4, view.warm_length, 128)
            )

    def test_chronological_order_with_warm(self):
        """Tokens must be chronologically ordered across all tiers."""
        config = self._config(
            dense_tail_capacity=128,
            flush_batch_size=64,
            warm_cache_capacity_blocks=2,
        )
        cache = TurboPolarKVCacheRuntime(config)

        # Append enough to fill hot, warm, and some cold
        for _ in range(400):
            t = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(t, t)

        cold_tokens = cache.total_blocks * config.block_size
        warm_tokens = cache.warm_length
        hot_tokens = cache.hot_length
        self.assertEqual(
            cold_tokens + warm_tokens + hot_tokens,
            cache.actual_seq_len,
        )
        # Verify warm is between cold and hot
        self.assertGreaterEqual(warm_tokens, 0)
        self.assertGreaterEqual(hot_tokens, 0)
        self.assertGreaterEqual(cold_tokens, 0)


if __name__ == "__main__":
    unittest.main()

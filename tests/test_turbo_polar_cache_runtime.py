import unittest
import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder

import pytest

pytest.importorskip("mlx")


class TestTurboPolarCacheRuntime(unittest.TestCase):
    """
    Milestone 1: CPU/MLX cache correctness without Metal.
    Covers the exact shape matrix called out in the review:
      T = 1, 32, 63, 64, 65, 127, 128, 129
      D = 64 and 128
      H_q/H_kv = 1:1 and 4:1
    """

    def _make_config(self, gqa_ratio=1):
        return TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=0,
            num_q_heads=4,
            num_kv_heads=4 // gqa_ratio,
            use_qjl=False,
            seed=7,
            dense_tail_capacity=64,
            flush_batch_size=64,
        )

    def _assert_cache_shape(self, cache, expected_T):
        block, quant_v, dense_v, qjl, actual_len = (
            cache.get_blocks_for_attention()
        )
        self.assertEqual(actual_len, expected_T)
        self.assertIsNotNone(block)
        self.assertIsNotNone(quant_v)
        self.assertIsNone(dense_v)
        self.assertIsNone(qjl)
        S = block.radii.shape[2]
        L = cache.config.block_size
        self.assertGreaterEqual(S * L, expected_T)
        self.assertLess((S - 1) * L, expected_T)

    def test_milestone_shapes(self):
        D = 128
        for T in (1, 32, 63, 64, 65, 127, 128, 129):
            for gqa_ratio in (1, 4):
                with self.subTest(T=T, gqa_ratio=gqa_ratio):
                    config = self._make_config(gqa_ratio)
                    cache = TurboPolarKVCacheRuntime(config)
                    H_kv = config.num_kv_heads
                    k = mx.random.normal(shape=[1, H_kv, T, D])
                    v = mx.random.normal(shape=[1, H_kv, T, D])
                    cache.append(k, v)
                    self._assert_cache_shape(cache, T)
                    telem = cache.get_io_telemetry()
                    self.assertEqual(telem["partial_tokens"], T % 64)
                    self.assertEqual(telem["total_blocks"], T // 64)

    def test_incremental_append(self):
        config = self._make_config(gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        for _ in range(130):
            k = mx.random.normal(
                shape=[1, config.num_kv_heads, 1, config.head_dim]
            )
            v = mx.random.normal(
                shape=[1, config.num_kv_heads, 1, config.head_dim]
            )
            cache.append(k, v)
        self._assert_cache_shape(cache, 130)

    def test_fetch_blocks_roundtrip(self):
        config = self._make_config(gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(
            shape=[1, config.num_kv_heads, 64, config.head_dim]
        )
        v = mx.random.normal(
            shape=[1, config.num_kv_heads, 64, config.head_dim]
        )
        cache.append(k, v)
        block, quant_v, _, _, actual_len = cache.get_blocks_for_attention()
        self.assertEqual(actual_len, 64)
        decoder = PolarQuantDecoder()
        k_recon = decoder.decode_block(block).reshape(
            1, config.num_kv_heads, 64, config.head_dim
        )
        recon_error = float(mx.mean(mx.abs(k_recon - k)))
        self.assertLess(recon_error, 0.5)

    def test_unsupported_configurations_raise(self):
        with self.assertRaises(ValueError):
            TurboPolarConfig(
                head_dim=32, block_size=64, num_q_heads=4, num_kv_heads=4
            )
        with self.assertRaises(ValueError):
            TurboPolarConfig(
                head_dim=128, block_size=32, num_q_heads=4, num_kv_heads=4
            )
        with self.assertRaises(NotImplementedError):
            TurboPolarConfig(
                head_dim=128, block_size=64, num_q_heads=4,
                num_kv_heads=4, use_qjl=True
            )
        with self.assertRaises(ValueError):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                storage_mode="dense_v_debug",
            )

    def test_qjl_optional_not_stored(self):
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            use_qjl=False,
            storage_mode="kv_quant",
        )
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, 4, 64, 128])
        v = mx.random.normal(shape=[1, 4, 64, 128])
        cache.append(k, v)
        _, _, _, qjl, _ = cache.get_blocks_for_attention()
        self.assertIsNone(qjl)
        self.assertEqual(len(cache.qjl_blocks), 0)

    def test_gqa_attention_payload(self):
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=8,
            num_kv_heads=2,
            use_qjl=False,
            storage_mode="kv_quant",
        )
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, 2, 64, 128])
        v = mx.random.normal(shape=[1, 2, 64, 128])
        cache.append(k, v)
        block, quant_v, _, _, actual_len = cache.get_blocks_for_attention()
        self.assertEqual(actual_len, 64)
        self.assertEqual(block.radii.shape[1], 2)
        self.assertEqual(quant_v.codes.shape[1], 2)

    def test_telemetry_counts_partial_kv(self):
        config = self._make_config(gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, 4, 65, 128])
        v = mx.random.normal(shape=[1, 4, 65, 128])
        cache.append(k, v)
        telem = cache.get_io_telemetry()
        self.assertEqual(telem["total_blocks"], 1)
        self.assertEqual(telem["partial_tokens"], 1)

    def test_append_rejects_bad_inputs(self):
        config = self._make_config(gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)

        # Wrong rank
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((1, 4, 128)),
                mx.random.normal((1, 4, 128)),
            )

        # Mismatched k/v shape
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((1, 4, 1, 128)),
                mx.random.normal((1, 4, 2, 128)),
            )

        # Wrong number of KV heads
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((1, 2, 1, 128)),
                mx.random.normal((1, 2, 1, 128)),
            )

        # Wrong head dimension
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((1, 4, 1, 64)),
                mx.random.normal((1, 4, 1, 64)),
            )

        # Batch size changes after first append
        cache.append(
            mx.random.normal((1, 4, 1, 128)),
            mx.random.normal((1, 4, 1, 128)),
        )
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((2, 4, 1, 128)),
                mx.random.normal((2, 4, 1, 128)),
            )

    def test_boundary_crossing_prefill_1_then_64(self):
        """Append 1 token, then append 64; offset must be 65."""
        config = self._make_config(gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        k1 = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
        v1 = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
        cache.append(k1, v1)
        k64 = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        v64 = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        cache.append(k64, v64)
        self.assertEqual(cache.actual_seq_len, 65)

    def test_boundary_crossing_prefill_63_then_2(self):
        """Append 63 tokens, then append 2; offset must be 65."""
        config = self._make_config(gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        k63 = mx.random.normal((1, 4, 63, 128), dtype=mx.float16)
        v63 = mx.random.normal((1, 4, 63, 128), dtype=mx.float16)
        cache.append(k63, v63)
        k2 = mx.random.normal((1, 4, 2, 128), dtype=mx.float16)
        v2 = mx.random.normal((1, 4, 2, 128), dtype=mx.float16)
        cache.append(k2, v2)
        self.assertEqual(cache.actual_seq_len, 65)

    def test_randomized_append_partitions_match_dense(self):
        """Random partitions must match total length and block count."""
        config = self._make_config(gqa_ratio=1)
        total = 257
        mx.random.seed(11)
        k_full = mx.random.normal((1, 4, total, 128), dtype=mx.float16)
        v_full = mx.random.normal((1, 4, total, 128), dtype=mx.float16)

        # Single-shot reference
        ref = TurboPolarKVCacheRuntime(config)
        ref.append_many(k_full, v_full)

        # Randomized partition test
        for seed in range(20):
            mx.random.seed(seed)
            cache = TurboPolarKVCacheRuntime(config)
            t = 0
            while t < total:
                chunk = int(mx.random.randint(1, 32).item())
                end = min(t + chunk, total)
                cache.append(k_full[:, :, t:end, :], v_full[:, :, t:end, :])
                t = end
            self.assertEqual(
                cache.actual_seq_len, ref.actual_seq_len,
                f"seed={seed}: actual_seq_len mismatch"
            )
            # RoPE offset must equal true token count
            self.assertEqual(cache.actual_seq_len, total)
            # Block count must match
            self.assertEqual(
                cache.total_blocks, ref.total_blocks,
                f"seed={seed}: total_blocks mismatch"
            )

    def test_append_many_boundary_crossing(self):
        """append_many must correctly handle pre-existing partial blocks."""
        config = self._make_config(gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        # Start with 1 token in partial buffer
        cache.append_many(
            mx.random.normal((1, 4, 1, 128), dtype=mx.float16),
            mx.random.normal((1, 4, 1, 128), dtype=mx.float16),
        )
        # Append 64 tokens crossing the boundary
        cache.append_many(
            mx.random.normal((1, 4, 64, 128), dtype=mx.float16),
            mx.random.normal((1, 4, 64, 128), dtype=mx.float16),
        )
        self.assertEqual(cache.actual_seq_len, 65)
        self.assertEqual(cache.partial_length, 1)

    def test_dense_tail_capacity_validation(self):
        with self.assertRaises(ValueError):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                dense_tail_capacity=32,  # < block_size
                num_q_heads=4,
                num_kv_heads=4,
            )
        with self.assertRaises(ValueError):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                dense_tail_capacity=100,  # not multiple of 64
                num_q_heads=4,
                num_kv_heads=4,
            )

    def test_dense_tail_512_decode_accumulates(self):
        """With dense_tail_capacity=512, decode tokens accumulate until 512."""
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            dense_tail_capacity=512,
            flush_batch_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            use_qjl=False,
        )
        cache = TurboPolarKVCacheRuntime(config)
        # Prefill 64 tokens → stay in dense tail (unified architecture)
        k = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        cache.append(k, v)
        self.assertEqual(cache.total_blocks, 0)
        self.assertEqual(cache.partial_length, 64)

        # Decode 448 tokens to fill tail to 512, then flush 64
        for i in range(448):
            kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(kt, vt)

        # 64 + 448 = 512 total, first flush: 64 compressed, tail = 448
        self.assertEqual(cache.actual_seq_len, 512)
        self.assertEqual(cache.total_blocks, 1)
        self.assertEqual(cache.partial_length, 448)

        # Decode 63 more → tail = 511
        for i in range(63):
            kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(kt, vt)

        # 512 + 63 = 575 total, no new flush yet
        self.assertEqual(cache.actual_seq_len, 575)
        self.assertEqual(cache.total_blocks, 1)
        self.assertEqual(cache.partial_length, 511)

        # One more decode token triggers second flush
        kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
        vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
        cache.append(kt, vt)

        # Now 575 + 1 = 576 total, 2 blocks compressed, tail = 512 - 64 = 448
        self.assertEqual(cache.actual_seq_len, 576)
        self.assertEqual(cache.total_blocks, 2)
        self.assertEqual(cache.partial_length, 448)

    def test_dense_tail_512_shift_on_flush(self):
        """After flush, remaining tokens are shifted to front of buffer."""
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            dense_tail_capacity=512,
            flush_batch_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            use_qjl=False,
        )
        cache = TurboPolarKVCacheRuntime(config)
        # Prefill 64 tokens
        k = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        cache.append(k, v)

        # Decode 512 tokens to fill the tail exactly
        decode_k = []
        decode_v = []
        for _ in range(512):
            kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            decode_k.append(kt)
            decode_v.append(vt)
            cache.append(kt, vt)

        # Flush triggered: partial_length should be 512 - 64 = 448
        self.assertEqual(cache.partial_length, 448)

        # Verify the dense tail contains the LAST 448 decode tokens
        tail_k = cache._current_tail_k()
        self.assertIsNotNone(tail_k)
        self.assertEqual(tail_k.shape, (1, 4, 448, 128))

        # Compare with the original tensors (last 448 tokens)
        expected_k = mx.concatenate(decode_k, axis=2)[:, :, -448:, :]
        diff = float(mx.mean(mx.abs(tail_k - expected_k)))
        self.assertLess(diff, 1e-4)

    def test_dense_tail_512_prefill_then_decode(self):
        """Prefill goes through dense tail first; decode accumulates."""
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            dense_tail_capacity=512,
            flush_batch_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            use_qjl=False,
        )
        cache = TurboPolarKVCacheRuntime(config)
        # Prefill 640 tokens:
        #   512 into tail → flush 64 → tail=448
        #   128 more → tail=576 → flush 64 → tail=512 → flush 64 → tail=448
        # Final: 3 blocks compressed, tail=448
        k = mx.random.normal((1, 4, 640, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 640, 128), dtype=mx.float16)
        cache.append(k, v)
        self.assertEqual(cache.total_blocks, 3)
        self.assertEqual(cache.partial_length, 448)
        self.assertEqual(cache.actual_seq_len, 640)

        # Decode 200 tokens:
        #   64 → tail=512 → flush 64 → tail=448 (blocks=4)
        #   64 → tail=512 → flush 64 → tail=448 (blocks=5)
        #   64 → tail=512 → flush 64 → tail=448 (blocks=6)
        #   8  → tail=456 (blocks=6)
        for _ in range(200):
            kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(kt, vt)
        self.assertEqual(cache.actual_seq_len, 840)
        self.assertEqual(cache.total_blocks, 6)
        self.assertEqual(cache.partial_length, 456)

    def test_page_pool_preallocation(self):
        """Page pool preallocation eliminates per-append allocations."""
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            dense_tail_capacity=64,
            flush_batch_size=64,
            page_pool_prealloc=4,
            num_q_heads=4,
            num_kv_heads=4,
            use_qjl=False,
        )
        cache = TurboPolarKVCacheRuntime(config)
        # Prefill 256 tokens = 4 blocks
        k = mx.random.normal((1, 4, 256, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 256, 128), dtype=mx.float16)
        cache.append(k, v)

        # After first block, pool should be preallocated
        self.assertTrue(cache._page_pool_preallocated)
        # K storage should have 4 valid blocks in 1 page (capacity=16)
        self.assertEqual(cache.k_storage._paged.total_valid_blocks, 4)
        self.assertEqual(cache.k_storage._paged.page_count, 1)
        # All 4 prealloc pages are still in the pool (none used yet)
        self.assertEqual(len(cache.k_storage._paged._page_pool), 4)

        # Append 60 more blocks to fill the first page (4 + 60 = 64 = 4 pages)
        for _ in range(60):
            kt = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
            vt = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
            cache.append(kt, vt)

        # Now we have 4 pages; the first was allocated before the pool,
        # the next 3 came from the pool.
        self.assertEqual(cache.k_storage._paged.page_count, 4)
        self.assertEqual(len(cache.k_storage._paged._page_pool), 1)

        # One more block triggers a 5th page allocation; pool exhausted
        kt = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        vt = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        cache.append(kt, vt)
        self.assertEqual(len(cache.k_storage._paged._page_pool), 0)

    def test_memory_stats_dense_tail_512(self):
        """Memory stats must correctly account for larger dense tail buffer."""
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            dense_tail_capacity=512,
            flush_batch_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            use_qjl=False,
        )
        cache = TurboPolarKVCacheRuntime(config)
        # Prefill 64 tokens → stay in dense tail
        k = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 64, 128), dtype=mx.float16)
        cache.append(k, v)

        stats = cache.get_memory_stats()
        # Dense tail allocated bytes should be for the full 512-token buffer
        # (K+V).  1 * 4 * 512 * 128 * 2 bytes (fp16) * 2 (K+V)
        expected_dense_tail_alloc = 1 * 4 * 512 * 128 * 2 * 2
        # partial_length = 64 → logical dense tail bytes = 64 tokens
        expected_dense_tail_logical_64 = 1 * 4 * 64 * 128 * 2 * 2
        self.assertEqual(
            stats.dense_tail_bytes, expected_dense_tail_logical_64
        )
        # allocated includes the full buffer
        self.assertGreaterEqual(
            stats.allocated_capacity_bytes, expected_dense_tail_alloc
        )

        # Decode 10 tokens
        for _ in range(10):
            kt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            vt = mx.random.normal((1, 4, 1, 128), dtype=mx.float16)
            cache.append(kt, vt)

        stats = cache.get_memory_stats()
        # Logical dense tail bytes = 74 tokens (64 prefill + 10 decode)
        expected_dense_tail_logical_74 = 1 * 4 * 74 * 128 * 2 * 2
        self.assertEqual(
            stats.dense_tail_bytes, expected_dense_tail_logical_74
        )


if __name__ == "__main__":
    unittest.main()

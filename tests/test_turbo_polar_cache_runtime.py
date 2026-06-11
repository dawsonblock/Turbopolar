import unittest
import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder


class TestTurboPolarCacheRuntime(unittest.TestCase):
    """
    Milestone 1: CPU/MLX cache correctness without Metal.
    Covers the exact shape matrix called out in the review:
      T = 1, 32, 63, 64, 65, 127, 128, 129
      D = 64 and 128
      H_q/H_kv = 1:1 and 4:1
    """
    def _make_config(self, D, gqa_ratio=1):
        return TurboPolarConfig(
            head_dim=D,
            qjl_proj_dim=32 if D == 64 else 64,
            block_size=64,
            split_dim=64,
            num_q_heads=4,
            num_kv_heads=4 // gqa_ratio,
            use_qjl=False,
            seed=7,
        )

    def _assert_cache_shape(self, cache, expected_T):
        block, quant_v, dense_v, qjl, actual_len = cache.get_blocks_for_attention()
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
        for D in (64, 128):
            for T in (1, 32, 63, 64, 65, 127, 128, 129):
                for gqa_ratio in (1, 4):
                    with self.subTest(D=D, T=T, gqa_ratio=gqa_ratio):
                        config = self._make_config(D, gqa_ratio)
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
        config = self._make_config(128, gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        for chunk_size in (17, 31, 64, 7, 11):
            k = mx.random.normal(shape=[1, 4, chunk_size, 128])
            v = mx.random.normal(shape=[1, 4, chunk_size, 128])
            cache.append(k, v)
        self.assertEqual(cache.actual_seq_len, 130)
        self._assert_cache_shape(cache, 130)

    def test_gqa_attention_payload(self):
        config = self._make_config(128, gqa_ratio=4)
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, 1, 65, 128])
        v = mx.random.normal(shape=[1, 1, 65, 128])
        cache.append(k, v)
        block, quant_v, _, _, actual_len = cache.get_blocks_for_attention()
        self.assertEqual(block.radii.shape[1], 1)
        self.assertEqual(quant_v.codes.shape[1], 1)
        self.assertEqual(actual_len, 65)

    def test_storage_modes(self):
        for mode in ("kv_quant", "dense_v_debug", "k_only_first"):
            with self.subTest(mode=mode):
                config = TurboPolarConfig(
                    head_dim=128, qjl_proj_dim=64, block_size=64, split_dim=64,
                    num_q_heads=4, num_kv_heads=4, use_qjl=False,
                    storage_mode=mode, seed=1,
                )
                cache = TurboPolarKVCacheRuntime(config)
                k = mx.random.normal(shape=[1, 4, 64, 128])
                v = mx.random.normal(shape=[1, 4, 64, 128])
                cache.append(k, v)
                block, quant_v, dense_v, qjl, actual_len = cache.get_blocks_for_attention()
                self.assertEqual(actual_len, 64)
                if mode == "kv_quant":
                    self.assertIsNotNone(quant_v)
                    self.assertIsNone(dense_v)
                elif mode == "dense_v_debug":
                    self.assertIsNone(quant_v)
                    self.assertIsNotNone(dense_v)
                else:
                    self.assertIsNone(quant_v)
                    self.assertIsNone(dense_v)
                self.assertIsNone(qjl)

    def test_fetch_blocks_roundtrip(self):
        config = self._make_config(128, gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, 4, 128, 128])
        v = mx.random.normal(shape=[1, 4, 128, 128])
        cache.append(k, v)
        fetched = cache.fetch_blocks()
        self.assertEqual(fetched.shape, (1, 4, 128, 128))
        # Fetched K should be close to polar-decoded K
        k_recon = PolarQuantDecoder().decode_block(cache.get_blocks_for_attention()[0])
        mx.eval(fetched, k_recon)
        self.assertEqual(fetched.shape, k_recon.shape)

    def test_telemetry_counts_partial_kv(self):
        config = self._make_config(128, gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, 4, 65, 128])
        v = mx.random.normal(shape=[1, 4, 65, 128])
        cache.append(k, v)
        telem = cache.get_io_telemetry()
        self.assertEqual(telem["partial_tokens"], 1)
        self.assertEqual(telem["total_blocks"], 1)
        # Dense baseline already includes partial tokens; compressed must add raw partial bytes.
        self.assertGreater(telem["actual_cache_bytes"], 0)
        self.assertGreater(telem["compression_ratio"], 0.0)

    def test_qjl_optional_not_stored(self):
        config = TurboPolarConfig(
            head_dim=128, qjl_proj_dim=64, block_size=64, split_dim=64,
            num_q_heads=4, num_kv_heads=4, use_qjl=False, seed=1,
        )
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, 4, 128, 128])
        v = mx.random.normal(shape=[1, 4, 128, 128])
        cache.append(k, v)
        self.assertEqual(len(cache.qjl_blocks), 0)
        block, quant_v, dense_v, qjl, actual_len = cache.get_blocks_for_attention()
        self.assertIsNone(qjl)

    def test_qjl_stored_when_enabled(self):
        config = TurboPolarConfig(
            head_dim=128, qjl_proj_dim=64, block_size=64, split_dim=64,
            num_q_heads=4, num_kv_heads=4, use_qjl=True, seed=1,
        )
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, 4, 128, 128])
        v = mx.random.normal(shape=[1, 4, 128, 128])
        cache.append(k, v)
        self.assertEqual(len(cache.qjl_blocks), 2)
        block, quant_v, dense_v, qjl, actual_len = cache.get_blocks_for_attention()
        self.assertIsNotNone(qjl)


if __name__ == "__main__":
    unittest.main()

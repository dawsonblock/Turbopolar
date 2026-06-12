import unittest
import mlx.core as mx

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
            k = mx.random.normal(shape=[1, config.num_kv_heads, 1, config.head_dim])
            v = mx.random.normal(shape=[1, config.num_kv_heads, 1, config.head_dim])
            cache.append(k, v)
        self._assert_cache_shape(cache, 130)

    def test_fetch_blocks_roundtrip(self):
        config = self._make_config(gqa_ratio=1)
        cache = TurboPolarKVCacheRuntime(config)
        k = mx.random.normal(shape=[1, config.num_kv_heads, 64, config.head_dim])
        v = mx.random.normal(shape=[1, config.num_kv_heads, 64, config.head_dim])
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
            TurboPolarConfig(head_dim=64, block_size=64, num_q_heads=4, num_kv_heads=4)
        with self.assertRaises(ValueError):
            TurboPolarConfig(head_dim=128, block_size=32, num_q_heads=4, num_kv_heads=4)
        with self.assertRaises(NotImplementedError):
            TurboPolarConfig(
                head_dim=128, block_size=64, num_q_heads=4, num_kv_heads=4, use_qjl=True
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
            cache.append(mx.random.normal((1, 4, 128)), mx.random.normal((1, 4, 128)))

        # Mismatched k/v shape
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((1, 4, 1, 128)), mx.random.normal((1, 4, 2, 128))
            )

        # Wrong number of KV heads
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((1, 2, 1, 128)), mx.random.normal((1, 2, 1, 128))
            )

        # Wrong head dimension
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((1, 4, 1, 64)), mx.random.normal((1, 4, 1, 64))
            )

        # Batch size changes after first append
        cache.append(mx.random.normal((1, 4, 1, 128)), mx.random.normal((1, 4, 1, 128)))
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((2, 4, 1, 128)), mx.random.normal((2, 4, 1, 128))
            )


if __name__ == "__main__":
    unittest.main()

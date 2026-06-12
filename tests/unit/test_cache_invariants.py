"""Tests that TurboPolarKVCacheRuntime persists invariants across block boundaries."""

import unittest

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


class TestCacheInvariants(unittest.TestCase):
    """Invariants must survive exact 64-token flushes where the tail is reset."""

    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=4,
        )

    def _valid_kv(self, B=1, H_kv=4, T=1, D=128, dtype=mx.float16):
        k = mx.random.normal((B, H_kv, T, D)).astype(dtype)
        v = mx.random.normal((B, H_kv, T, D)).astype(dtype)
        return k, v

    def test_batch_size_change_after_full_flush_rejected(self):
        cache = TurboPolarKVCacheRuntime(self.config)
        k, v = self._valid_kv(B=1, T=64)
        cache.append(k, v)
        self.assertEqual(cache.partial_length, 0)
        k2, v2 = self._valid_kv(B=2, T=1)
        with self.assertRaisesRegex(ValueError, "batch size changed"):
            cache.append(k2, v2)

    def test_dtype_change_after_full_flush_rejected(self):
        cache = TurboPolarKVCacheRuntime(self.config)
        k, v = self._valid_kv(T=64, dtype=mx.float16)
        cache.append(k, v)
        self.assertEqual(cache.partial_length, 0)
        k2, v2 = self._valid_kv(T=1, dtype=mx.float32)
        with self.assertRaisesRegex(ValueError, "dtype changed"):
            cache.append(k2, v2)

    def test_kv_head_count_change_rejected(self):
        cache = TurboPolarKVCacheRuntime(self.config)
        k, v = self._valid_kv(H_kv=4, T=1)
        cache.append(k, v)
        k2, v2 = self._valid_kv(H_kv=2, T=1)
        with self.assertRaisesRegex(ValueError, "KV head count changed"):
            cache.append(k2, v2)

    def test_head_dim_change_rejected(self):
        cache = TurboPolarKVCacheRuntime(self.config)
        k, v = self._valid_kv(D=128, T=1)
        cache.append(k, v)
        k2 = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        v2 = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        # head_dim is also rejected by config, but invariant check catches it after first append.
        with self.assertRaises(ValueError):
            cache.append(k2, v2)

    def test_reset_then_initialize_with_new_shape(self):
        cache = TurboPolarKVCacheRuntime(self.config)
        k, v = self._valid_kv(B=1, T=64)
        cache.append(k, v)
        cache.reset()
        k2, v2 = self._valid_kv(B=2, T=1)
        cache.append(k2, v2)
        self.assertEqual(cache._batch_size, 2)

    def test_append_malformed_shapes_rejected(self):
        cache = TurboPolarKVCacheRuntime(self.config)
        with self.assertRaises(ValueError):
            cache.append(mx.random.normal((1, 4, 128)), mx.random.normal((1, 4, 128)))
        with self.assertRaises(ValueError):
            cache.append(
                mx.random.normal((1, 4, 1, 128)).astype(mx.float16),
                mx.random.normal((1, 4, 2, 128)).astype(mx.float16),
            )

    def test_nonfinite_values_rejected_when_validation_enabled(self):
        config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            validate_finite_inputs=True,
        )
        cache = TurboPolarKVCacheRuntime(config)
        k, v = self._valid_kv(T=2)
        # Inject one inf by concatenating a small tensor containing it.
        k_bad = mx.concatenate(
            [mx.full((1, 4, 1, 128), float("inf"), k.dtype), k[:, :, 1:, :]], axis=2
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            cache.append(k_bad, v)

    def test_finite_validation_disabled_by_default(self):
        cache = TurboPolarKVCacheRuntime(self.config)
        self.assertFalse(cache.config.validate_finite_inputs)
        self.assertEqual(cache.config.finite_audit_interval, 0)


if __name__ == "__main__":
    unittest.main()

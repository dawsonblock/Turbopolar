"""Tests for the Cartesian int8 KV cache baseline."""

import unittest

import mlx.core as mx

from benchmarks.cartesian_int8_cache import CartesianInt8Cache


class TestCartesianInt8Cache(unittest.TestCase):
    def test_empty_cache(self):
        cache = CartesianInt8Cache()
        self.assertTrue(cache.empty())
        self.assertEqual(cache.nbytes, 0)

    def test_update_and_fetch_preserves_shape(self):
        cache = CartesianInt8Cache()
        B, H, T, D = 1, 2, 4, 128
        keys = mx.random.normal((B, H, T, D), dtype=mx.float16)
        values = mx.random.normal((B, H, T, D), dtype=mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        self.assertEqual(k_out.shape, keys.shape)
        self.assertEqual(v_out.shape, values.shape)
        self.assertEqual(cache.size(), T)
        self.assertFalse(cache.empty())

    def test_memory_smaller_than_dense(self):
        cache = CartesianInt8Cache()
        B, H, T, D = 1, 2, 64, 128
        keys = mx.random.normal((B, H, T, D), dtype=mx.float16)
        values = mx.random.normal((B, H, T, D), dtype=mx.float16)
        cache.update_and_fetch(keys, values)
        dense_bytes = B * H * T * D * 2 * 2  # fp16 K + V
        self.assertLess(cache.nbytes, dense_bytes)

    def test_incremental_append(self):
        cache = CartesianInt8Cache()
        B, H, D = 1, 2, 128
        for t in range(5):
            k = mx.random.normal((B, H, 1, D), dtype=mx.float16)
            v = mx.random.normal((B, H, 1, D), dtype=mx.float16)
            k_out, v_out = cache.update_and_fetch(k, v)
            self.assertEqual(k_out.shape[2], t + 1)


if __name__ == "__main__":
    unittest.main()

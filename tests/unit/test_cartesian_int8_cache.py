"""Tests for the Cartesian int8 KV cache baseline."""

import unittest

import mlx.core as mx

from benchmarks.cartesian_int8_cache import CartesianInt8Cache
from rfsn_v11.generation.cartesian_int8_paged_cache import PagedCartesianInt8KVCache


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


class TestPagedCartesianInt8KVCache(unittest.TestCase):
    """Test the MLX-LM-compatible paged Cartesian cache."""
    
    def test_empty_cache(self):
        cache = PagedCartesianInt8KVCache(block_size=64)
        self.assertTrue(cache.empty())
        self.assertEqual(cache.offset, 0)
        self.assertEqual(cache.size(), 0)
    
    def test_update_and_fetch_interface(self):
        """Test that update_and_fetch provides MLX-LM-compatible interface."""
        cache = PagedCartesianInt8KVCache(block_size=64)
        B, H, T, D = 1, 2, 4, 128
        keys = mx.random.normal((B, H, T, D), dtype=mx.float16)
        values = mx.random.normal((B, H, T, D), dtype=mx.float16)
        
        # Test update_and_fetch returns correct shapes
        k_out, v_out = cache.update_and_fetch(keys, values)
        self.assertEqual(k_out.shape, keys.shape)
        self.assertEqual(v_out.shape, values.shape)
        self.assertEqual(cache.offset, T)
        self.assertFalse(cache.empty())
    
    def test_update_and_fetch_preserves_dtype(self):
        """Test that update_and_fetch preserves original dtype."""
        cache = PagedCartesianInt8KVCache(block_size=64)
        B, H, T, D = 1, 2, 4, 128
        
        # Test with float32 input
        keys_f32 = mx.random.normal((B, H, T, D), dtype=mx.float32)
        values_f32 = mx.random.normal((B, H, T, D), dtype=mx.float32)
        k_out, v_out = cache.update_and_fetch(keys_f32, values_f32)
        self.assertEqual(k_out.dtype, mx.float32)
        self.assertEqual(v_out.dtype, mx.float32)
    
    def test_block_boundary_handling(self):
        """Test that cache correctly handles block boundaries."""
        cache = PagedCartesianInt8KVCache(block_size=64)
        B, H, D = 1, 2, 128
        
        # Add exactly one block of tokens
        keys = mx.random.normal((B, H, 64, D), dtype=mx.float16)
        values = mx.random.normal((B, H, 64, D), dtype=mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        
        self.assertEqual(cache.offset, 64)
        self.assertEqual(cache.storage.block_count, 1)
        self.assertEqual(cache.partial_length, 0)  # Should be flushed to storage
        
        # Add one more token to start a new partial block
        keys = mx.random.normal((B, H, 1, D), dtype=mx.float16)
        values = mx.random.normal((B, H, 1, D), dtype=mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        
        self.assertEqual(cache.offset, 65)
        self.assertEqual(cache.storage.block_count, 1)  # Still 1 full block
        self.assertEqual(cache.partial_length, 1)  # One token in partial buffer
    
    def test_memory_accounting(self):
        """Test that logical and allocated memory are reported correctly."""
        cache = PagedCartesianInt8KVCache(block_size=64)
        B, H, D = 1, 2, 128
        
        # Add less than one block
        keys = mx.random.normal((B, H, 32, D), dtype=mx.float16)
        values = mx.random.normal((B, H, 32, D), dtype=mx.float16)
        cache.update_and_fetch(keys, values)
        
        # Logical should only count actual data
        logical = cache.nbytes
        # Allocated should count full page capacity
        allocated = cache.allocated_bytes
        
        # With partial data, allocated >= logical
        self.assertGreaterEqual(allocated, logical)
        
        # Add more data to trigger page allocation
        keys = mx.random.normal((B, H, 64, D), dtype=mx.float16)
        values = mx.random.normal((B, H, 64, D), dtype=mx.float16)
        cache.update_and_fetch(keys, values)
        
        # Now we should have allocated pages
        self.assertGreater(cache.allocated_bytes, 0)
    
    def test_mask_generation(self):
        """Test that make_mask provides MLX-LM-compatible interface."""
        cache = PagedCartesianInt8KVCache(block_size=64)
        B, H, T, D = 1, 2, 10, 128
        keys = mx.random.normal((B, H, T, D), dtype=mx.float16)
        values = mx.random.normal((B, H, T, D), dtype=mx.float16)
        cache.update_and_fetch(keys, values)
        
        # Test mask generation
        mask = cache.make_mask(N=5, return_array=True)
        self.assertIsNotNone(mask)
    
    def test_get_history_decompression(self):
        """Test that get_history correctly decompresses stored data."""
        cache = PagedCartesianInt8KVCache(block_size=64)
        B, H, D = 1, 2, 128
        
        # Add data that will be compressed
        original_keys = mx.random.normal((B, H, 64, D), dtype=mx.float16)
        original_values = mx.random.normal((B, H, 64, D), dtype=mx.float16)
        cache.update_and_fetch(original_keys, original_values)
        
        # Retrieve decompressed history
        k_hist, v_hist = cache.get_history()
        
        # Shapes should match
        self.assertEqual(k_hist.shape, original_keys.shape)
        self.assertEqual(v_hist.shape, original_values.shape)
        
        # Data type should be float16
        self.assertEqual(k_hist.dtype, mx.float16)
        self.assertEqual(v_hist.dtype, mx.float16)


if __name__ == "__main__":
    unittest.main()

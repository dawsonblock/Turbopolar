"""Tests that the fused attention adapter rejects unsupported attention modes."""

import unittest

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from benchmarks.turbopolar_fast_attention import (
    TurboPolarFastCache,
    _is_standard_causal_mask,
)


class TestUnsupportedAttentionRejection(unittest.TestCase):
    """Unsupported attention semantics must raise NotImplementedError or ValueError."""

    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=8,
            num_kv_heads=4,
        )
        self.cache = TurboPolarFastCache(self.config)

    def _valid_qkv(self, B=1, H_q=8, H_kv=4, T=1, D=128):
        q = mx.random.normal((B, H_q, T, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, T, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, T, D)).astype(mx.float16)
        return q, k, v

    def test_batch_size_greater_than_one_rejected(self):
        q, k, v = self._valid_qkv(B=2)
        with self.assertRaisesRegex(NotImplementedError, "batch size 1"):
            self.cache.decode_attention(q, k, v, scale=self.config.attention_scale)

    def test_query_length_greater_than_one_rejected(self):
        q, k, v = self._valid_qkv(T=5)
        with self.assertRaisesRegex(ValueError, "single query"):
            self.cache.decode_attention(q, k, v, scale=self.config.attention_scale)

    def test_key_value_length_greater_than_one_rejected(self):
        q, _, _ = self._valid_qkv(T=1)
        k = mx.random.normal((1, 4, 5, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 5, 128)).astype(mx.float16)
        with self.assertRaisesRegex(ValueError, "single query/key/value token"):
            self.cache.decode_attention(q, k, v, scale=self.config.attention_scale)

    def test_unsupported_head_dim_rejected(self):
        q = mx.random.normal((1, 8, 1, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        with self.assertRaisesRegex(NotImplementedError, "head_dim == 128"):
            self.cache.decode_attention(q, k, v, scale=self.config.attention_scale)

    def test_gqa_ratio_must_divide(self):
        q = mx.random.normal((1, 8, 1, 128)).astype(mx.float16)
        k = mx.random.normal((1, 3, 1, 128)).astype(mx.float16)
        v = mx.random.normal((1, 3, 1, 128)).astype(mx.float16)
        with self.assertRaisesRegex(ValueError, "GQA ratio"):
            self.cache.decode_attention(q, k, v, scale=self.config.attention_scale)

    def test_none_mask_accepted(self):
        self.assertTrue(_is_standard_causal_mask(None, 1, 10))

    def test_boolean_all_true_mask_rejected(self):
        mask = mx.ones((1, 1, 1, 10), dtype=mx.float16)
        self.assertFalse(_is_standard_causal_mask(mask, 1, 10))
        with self.assertRaisesRegex(NotImplementedError, "mask=None"):
            self.cache.decode_attention(
                *self._valid_qkv(), scale=self.config.attention_scale, mask=mask
            )

    def test_triangular_mask_rejected(self):
        mask = mx.tril(mx.ones((1, 1, 1, 10), dtype=mx.float16))
        self.assertFalse(_is_standard_causal_mask(mask, 1, 10))
        with self.assertRaisesRegex(NotImplementedError, "mask=None"):
            self.cache.decode_attention(
                *self._valid_qkv(), scale=self.config.attention_scale, mask=mask
            )

    def test_additive_zero_mask_rejected(self):
        mask = mx.zeros((1, 1, 1, 10), dtype=mx.float16)
        self.assertFalse(_is_standard_causal_mask(mask, 1, 10))
        with self.assertRaisesRegex(NotImplementedError, "mask=None"):
            self.cache.decode_attention(
                *self._valid_qkv(), scale=self.config.attention_scale, mask=mask
            )

    def test_sliding_window_mask_rejected(self):
        mask = mx.zeros((1, 1, 1, 10), dtype=mx.float16)
        mask[..., -4:] = 1.0
        self.assertFalse(_is_standard_causal_mask(mask, 1, 10))
        with self.assertRaisesRegex(NotImplementedError, "mask=None"):
            self.cache.decode_attention(
                *self._valid_qkv(), scale=self.config.attention_scale, mask=mask
            )


if __name__ == "__main__":
    unittest.main()

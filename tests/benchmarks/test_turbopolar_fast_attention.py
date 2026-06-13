import unittest

import mlx.core as mx
import numpy as np
import pytest

from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig


@pytest.mark.native_metal_required
class TestTurboPolarFastAttention(unittest.TestCase):
    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=2,
            storage_mode="kv_quant",
            use_int8_radii=True,
        )
        self.cache = TurboPolarFastCache(self.config)

    def test_decode_attention_with_dense_tail(self):
        """decode_attention is correct before a block is flushed to compressed storage."""
        B, H_q, H_kv, D = 1, 4, 2, 128
        scale = self.config.attention_scale
        mx.random.seed(2026)

        # Append 37 tokens; since block_size=64, all stay in the dense tail.
        total_tokens = 37
        k_tokens = []
        v_tokens = []
        for i in range(total_tokens):
            q = mx.random.normal(shape=[B, H_q, 1, D])
            k_new = mx.random.normal(shape=[B, H_kv, 1, D])
            v_new = mx.random.normal(shape=[B, H_kv, 1, D])
            k_tokens.append(k_new)
            v_tokens.append(v_new)
            out = self.cache.decode_attention(q, k_new, v_new, scale)
            self.assertEqual(out.shape, (B, H_q, D))
            mx.eval(out)
            self.assertFalse(np.isnan(np.array(out)).any())

        # Reference: standard causal attention over the full dense history.
        full_k = mx.concatenate(k_tokens, axis=2)
        full_v = mx.concatenate(v_tokens, axis=2)
        full_k = mx.repeat(full_k, H_q // H_kv, axis=1)
        full_v = mx.repeat(full_v, H_q // H_kv, axis=1)
        q_last = mx.random.normal(shape=[B, H_q, 1, D])
        scores = mx.sum(q_last * full_k, axis=-1) * scale
        weights = mx.softmax(scores, axis=-1)
        ref_out = mx.sum(weights[:, :, :, None] * full_v, axis=-2)

        k_new = mx.random.normal(shape=[B, H_kv, 1, D])
        v_new = mx.random.normal(shape=[B, H_kv, 1, D])
        gpu_out = self.cache.decode_attention(q_last, k_new, v_new, scale)
        mx.eval(ref_out, gpu_out)
        ref_np = np.array(ref_out)
        gpu_np = np.array(gpu_out)
        cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
        )
        self.assertGreaterEqual(cosine, 0.99)

    def test_decode_attention_mixed_compressed_and_tail(self):
        """decode_attention is correct after at least one block is compressed."""
        B, H_q, H_kv, D = 1, 4, 2, 128
        scale = self.config.attention_scale
        mx.random.seed(2027)

        total_tokens = 100  # one full compressed block (64) + 36 tail tokens
        k_tokens = []
        v_tokens = []
        for i in range(total_tokens):
            q = mx.random.normal(shape=[B, H_q, 1, D])
            k_new = mx.random.normal(shape=[B, H_kv, 1, D])
            v_new = mx.random.normal(shape=[B, H_kv, 1, D])
            k_tokens.append(k_new)
            v_tokens.append(v_new)
            out = self.cache.decode_attention(q, k_new, v_new, scale)
            self.assertEqual(out.shape, (B, H_q, D))
            mx.eval(out)
            self.assertFalse(np.isnan(np.array(out)).any())

        # Reference over the full dense history (the cache compresses, but the
        # ground-truth attention is over the original tokens).
        full_k = mx.concatenate(k_tokens, axis=2)
        full_v = mx.concatenate(v_tokens, axis=2)
        full_k = mx.repeat(full_k, H_q // H_kv, axis=1)
        full_v = mx.repeat(full_v, H_q // H_kv, axis=1)
        q_last = mx.random.normal(shape=[B, H_q, 1, D])
        scores = mx.sum(q_last * full_k, axis=-1) * scale
        weights = mx.softmax(scores, axis=-1)
        ref_out = mx.sum(weights[:, :, :, None] * full_v, axis=-2)

        k_new = mx.random.normal(shape=[B, H_kv, 1, D])
        v_new = mx.random.normal(shape=[B, H_kv, 1, D])
        gpu_out = self.cache.decode_attention(q_last, k_new, v_new, scale)
        mx.eval(ref_out, gpu_out)
        ref_np = np.array(ref_out)
        gpu_np = np.array(gpu_out)
        cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
        )
        self.assertGreaterEqual(cosine, 0.95)

    def test_decode_attention_at_block_boundary(self):
        """decode_attention is correct when the cache ends exactly on a block boundary."""
        B, H_q, H_kv, D = 1, 4, 2, 128
        scale = self.config.attention_scale
        mx.random.seed(2028)

        total_tokens = 64
        k_tokens = []
        v_tokens = []
        for i in range(total_tokens):
            q = mx.random.normal(shape=[B, H_q, 1, D])
            k_new = mx.random.normal(shape=[B, H_kv, 1, D])
            v_new = mx.random.normal(shape=[B, H_kv, 1, D])
            k_tokens.append(k_new)
            v_tokens.append(v_new)
            out = self.cache.decode_attention(q, k_new, v_new, scale)
            self.assertEqual(out.shape, (B, H_q, D))

        full_k = mx.concatenate(k_tokens, axis=2)
        full_v = mx.concatenate(v_tokens, axis=2)
        full_k = mx.repeat(full_k, H_q // H_kv, axis=1)
        full_v = mx.repeat(full_v, H_q // H_kv, axis=1)
        q_last = mx.random.normal(shape=[B, H_q, 1, D])
        scores = mx.sum(q_last * full_k, axis=-1) * scale
        weights = mx.softmax(scores, axis=-1)
        ref_out = mx.sum(weights[:, :, :, None] * full_v, axis=-2)

        k_new = mx.random.normal(shape=[B, H_kv, 1, D])
        v_new = mx.random.normal(shape=[B, H_kv, 1, D])
        gpu_out = self.cache.decode_attention(q_last, k_new, v_new, scale)
        mx.eval(ref_out, gpu_out)
        ref_np = np.array(ref_out)
        gpu_np = np.array(gpu_out)
        cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
        )
        self.assertGreaterEqual(cosine, 0.95)


if __name__ == "__main__":
    unittest.main()

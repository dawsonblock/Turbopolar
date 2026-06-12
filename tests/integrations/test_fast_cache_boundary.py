"""Same-history correctness test at exact block boundaries and transitions."""

import unittest

import mlx.core as mx
import numpy as np

from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig


def _dense_attention(q, k, v, scale):
    """Reference causal attention over dense K/V history.

    q: [B, H_q, D]
    k, v: [B, H_q, T, D]
    """
    scores = mx.sum(q[:, :, None, :] * k, axis=-1) * scale
    weights = mx.softmax(scores, axis=-1)
    return mx.sum(weights[:, :, :, None] * v, axis=-2)


class TestFastCacheBoundary(unittest.TestCase):
    """Correctness when the cache ends exactly on, or just after, a block boundary."""

    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=8,
            num_kv_heads=4,
            use_int8_radii=True,
            k_angle_bits_deep=8,
            split_dim=0,
        )
        self.cache = TurboPolarFastCache(self.config)
        self.cache.reset_execution_stats()
        self.B = 1
        self.H_q = 8
        self.H_kv = 4
        self.D = 128

    def test_exact_64_flush_and_65_transition(self):
        mx.random.seed(2032)
        prefill_tokens = 65  # exactly one block flushed, then one tail token

        k_tokens = []
        v_tokens = []
        for _ in range(prefill_tokens):
            q = mx.random.normal((self.B, self.H_q, 1, self.D)).astype(mx.float16)
            k_new = mx.random.normal((self.B, self.H_kv, 1, self.D)).astype(mx.float16)
            v_new = mx.random.normal((self.B, self.H_kv, 1, self.D)).astype(mx.float16)
            k_tokens.append(k_new)
            v_tokens.append(v_new)
            out = self.cache.decode_attention(q, k_new, v_new, self.config.attention_scale)
            self.assertEqual(out.shape, (self.B, self.H_q, self.D))
            mx.eval(out)
            self.assertFalse(np.isnan(np.array(out)).any())
            self.assertFalse(np.isinf(np.array(out)).any())

        # After 64 tokens the tail buffer should have been flushed and reset.
        self.assertEqual(self.cache.runtime.total_blocks, 1)
        self.assertEqual(self.cache.runtime.partial_length, 1)

        q_last = mx.random.normal((self.B, self.H_q, 1, self.D)).astype(mx.float16)
        k_new = mx.random.normal((self.B, self.H_kv, 1, self.D)).astype(mx.float16)
        v_new = mx.random.normal((self.B, self.H_kv, 1, self.D)).astype(mx.float16)

        k_history = mx.concatenate(k_tokens + [k_new], axis=2)
        v_history = mx.concatenate(v_tokens + [v_new], axis=2)
        k_history = mx.repeat(k_history, self.H_q // self.H_kv, axis=1)
        v_history = mx.repeat(v_history, self.H_q // self.H_kv, axis=1)

        turbo_out = self.cache.decode_attention(
            q_last, k_new, v_new, self.config.attention_scale
        )
        self.assertEqual(self.cache.runtime.actual_seq_len, prefill_tokens + 1)
        self.assertEqual(k_history.shape[2], self.cache.runtime.actual_seq_len)

        ref_out = _dense_attention(
            q_last.squeeze(2), k_history, v_history, self.config.attention_scale
        )

        mx.eval(ref_out, turbo_out)
        ref_np = np.array(ref_out)
        turbo_np = np.array(turbo_out)

        self.assertFalse(np.isnan(ref_np).any() or np.isnan(turbo_np).any())
        self.assertFalse(np.isinf(ref_np).any() or np.isinf(turbo_np).any())

        cosine = np.dot(ref_np.flatten(), turbo_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(turbo_np) + 1e-12
        )
        mae = float(np.mean(np.abs(ref_np - turbo_np)))
        max_err = float(np.max(np.abs(ref_np - turbo_np)))

        self.assertGreaterEqual(cosine, 0.97, f"cosine too low: {cosine}")
        self.assertLessEqual(mae, 0.10, f"MAE too high: {mae}")
        self.assertLessEqual(max_err, 0.35, f"max error too high: {max_err}")

        stats = self.cache.execution_stats()
        # The 65th token uses compressed + one tail token; the comparison step
        # appends another token and therefore also uses the dense-tail kernel.
        self.assertGreater(stats.dense_tail_calls, 0)
        self.assertEqual(stats.fallback_calls, 0)
        self.assertGreater(stats.online_attention_calls, 0)

    def test_multiple_flushes(self):
        mx.random.seed(2033)
        prefill_tokens = 192  # three full blocks, no tail

        k_tokens = []
        v_tokens = []
        for _ in range(prefill_tokens):
            q = mx.random.normal((self.B, self.H_q, 1, self.D)).astype(mx.float16)
            k_new = mx.random.normal((self.B, self.H_kv, 1, self.D)).astype(mx.float16)
            v_new = mx.random.normal((self.B, self.H_kv, 1, self.D)).astype(mx.float16)
            k_tokens.append(k_new)
            v_tokens.append(v_new)
            self.cache.decode_attention(q, k_new, v_new, self.config.attention_scale)

        self.assertEqual(self.cache.runtime.total_blocks, 3)
        self.assertEqual(self.cache.runtime.partial_length, 0)

        q_last = mx.random.normal((self.B, self.H_q, 1, self.D)).astype(mx.float16)
        k_new = mx.random.normal((self.B, self.H_kv, 1, self.D)).astype(mx.float16)
        v_new = mx.random.normal((self.B, self.H_kv, 1, self.D)).astype(mx.float16)

        k_history = mx.concatenate(k_tokens + [k_new], axis=2)
        v_history = mx.concatenate(v_tokens + [v_new], axis=2)
        turbo_out = self.cache.decode_attention(
            q_last, k_new, v_new, self.config.attention_scale
        )
        self.assertEqual(self.cache.runtime.actual_seq_len, prefill_tokens + 1)
        ref_out = _dense_attention(
            q_last.squeeze(2),
            mx.repeat(k_history, self.H_q // self.H_kv, axis=1),
            mx.repeat(v_history, self.H_q // self.H_kv, axis=1),
            self.config.attention_scale,
        )

        mx.eval(ref_out, turbo_out)
        ref_np = np.array(ref_out)
        turbo_np = np.array(turbo_out)
        cosine = np.dot(ref_np.flatten(), turbo_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(turbo_np) + 1e-12
        )
        self.assertGreaterEqual(cosine, 0.97)
        self.assertEqual(self.cache.execution_stats().fallback_calls, 0)


if __name__ == "__main__":
    unittest.main()

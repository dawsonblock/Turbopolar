"""Unit tests for the speed-matrix benchmark script."""

import unittest

import mlx.core as mx
import mlx_lm.models.llama as llama

from benchmarks.run_speed_matrix import (
    _device_side_next_token,
    _make_turbo_config,
    benchmark_length,
    _model_cache_config,
)
from rfsn_v11.integrations.mlx_lm.llama_adapter import TurboPolarLlamaAdapter


class TestRunSpeedMatrix(unittest.TestCase):
    @staticmethod
    def _tiny_model():
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=512,
            num_hidden_layers=2,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=100,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=False,
        )
        return llama.Model(args)

    def test_device_side_next_token(self):
        logits = mx.random.normal((1, 3, 100))
        next_token = _device_side_next_token(logits)
        self.assertEqual(next_token.shape, (1,))
        self.assertIn(str(next_token.dtype), ("mlx.core.int32", "mlx.core.uint32"))

    def test_benchmark_length_alternating_order(self):
        model = self._tiny_model()
        nq, nkv, hd = _model_cache_config(model)
        adapter = TurboPolarLlamaAdapter(_make_turbo_config(nq, nkv, hd))
        tokens = list(range(10, 74))  # 64 tokens, within tiny vocab

        dense_a, turbo_a = benchmark_length(
            model, tokens, 2, nq, nkv, hd, adapter, turbo_first=False
        )
        dense_b, turbo_b = benchmark_length(
            model, tokens, 2, nq, nkv, hd, adapter, turbo_first=True
        )

        self.assertGreater(dense_a, 0.0)
        self.assertGreater(turbo_a, 0.0)
        self.assertGreater(dense_b, 0.0)
        self.assertGreater(turbo_b, 0.0)
        # Adapter must always be uninstalled after the benchmark.
        self.assertFalse(adapter._installed)


if __name__ == "__main__":
    unittest.main()

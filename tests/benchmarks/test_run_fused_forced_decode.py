"""Unit tests for the fused forced-decode benchmark script."""

import unittest
from pathlib import Path

import mlx.core as mx
import mlx_lm.models.llama as llama

from benchmarks.prompt_fixtures import load_token_fixtures, normalize_prompts
from benchmarks.run_fused_forced_decode import (
    _aggregate_execution_stats,
    _make_dense_cache,
    _make_turbo_config,
    _measure_decode_speed,
    _model_cache_config,
    benchmark_prompt,
)
from rfsn_v11.integrations.mlx_lm.llama_adapter import TurboPolarLlamaAdapter


class _FakeTokenizer:
    def encode(self, text: str):
        return [ord(c) % 100 for c in text]

    def decode(self, ids):
        return ""


class TestRunFusedForcedDecode(unittest.TestCase):
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

    def test_model_cache_config_inference(self):
        model = self._tiny_model()
        n_q, n_kv, d = _model_cache_config(model)
        self.assertEqual((n_q, n_kv, d), (4, 2, 128))

    def test_benchmark_prompt_runs_and_returns_similar_logits(self):
        model = self._tiny_model()
        tokenizer = _FakeTokenizer()
        adapter = TurboPolarLlamaAdapter(_make_turbo_config(4, 2, 128))
        result = benchmark_prompt(model, tokenizer, "hello world", 20, adapter)

        self.assertEqual(result.prompt_tokens, len(tokenizer.encode("hello world")))
        self.assertEqual(result.dense_logits_shape, result.turbo_logits_shape)
        self.assertGreater(result.logit_cosine, 0.95)
        self.assertGreaterEqual(result.top5_overlap, 0.0)
        self.assertLessEqual(result.top5_overlap, 1.0)
        self.assertGreater(result.peak_kv_bytes_dense, 0)
        self.assertGreater(result.peak_kv_bytes_turbo, 0)
        # Ensure adapter is cleaned up even if the model is reused.
        self.assertFalse(adapter._installed)

    def test_decode_speed_and_kernel_stats(self):
        model = self._tiny_model()
        tokenizer = _FakeTokenizer()
        tokens = tokenizer.encode("the quick brown fox jumps")

        dense_cache = _make_dense_cache(2)
        dense_tok_s = _measure_decode_speed(model, tokenizer, dense_cache, tokens, 4)
        self.assertGreater(dense_tok_s, 0.0)

        from benchmarks.turbopolar_fast_attention import make_turbo_caches

        turbo_cache = make_turbo_caches(2, 4, 2, 128, use_qjl=False)
        for c in turbo_cache:
            c.reset_execution_stats()

        adapter = TurboPolarLlamaAdapter(_make_turbo_config(4, 2, 128))
        adapter.install(model)
        try:
            turbo_tok_s = _measure_decode_speed(model, tokenizer, turbo_cache, tokens, 4)
        finally:
            adapter.uninstall()

        self.assertGreater(turbo_tok_s, 0.0)
        stats = _aggregate_execution_stats(turbo_cache)
        self.assertIn("fused_qk_calls", stats)
        self.assertIn("online_attention_calls", stats)
        self.assertIn("dense_tail_calls", stats)
        self.assertIn("fallback_calls", stats)

    def test_normalize_text_prompts(self):
        suite_path = Path(__file__).resolve().parents[2] / "benchmarks" / "prompt_suite.jsonl"
        normalized = normalize_prompts(_FakeTokenizer(), suite_path)
        self.assertIsInstance(normalized, list)
        self.assertTrue(all("tokens" in entry for entry in normalized))

    def test_load_token_fixtures_has_exact_lengths(self):
        fixtures_path = (
            Path(__file__).resolve().parents[2] / "benchmarks" / "exact_token_fixtures.jsonl"
        )
        fixtures = load_token_fixtures(fixtures_path)
        self.assertTrue(len(fixtures) > 0)
        for fx in fixtures:
            self.assertEqual(len(fx["tokens"]), fx["length"])
            self.assertIn(fx["category"], ("short", "boundary", "medium", "long", "stress"))


if __name__ == "__main__":
    unittest.main()

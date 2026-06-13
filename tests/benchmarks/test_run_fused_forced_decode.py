"""Unit tests for the fused forced-decode benchmark script."""

import unittest
from pathlib import Path

import mlx_lm.models.llama as llama
import numpy as np

from benchmarks.prompt_fixtures import load_token_fixtures, normalize_prompts
from benchmarks.run_fused_forced_decode import (
    _aggregate_execution_stats,
    _compute_step_metrics,
    _model_cache_config,
    benchmark_forced_decode_fixture,
)
from rfsn_v11.integrations.mlx_lm.adapter import TurboPolarLlamaAdapter
from rfsn_v11.integrations.mlx_lm.cache import make_turbo_caches


class _FakeTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text: str):
        return [ord(c) % self.vocab_size for c in text]

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

    def test_compute_step_metrics_basic(self):
        dense = np.array([1.0, 2.0, 3.0, 4.0])
        turbo = np.array([1.1, 1.9, 3.1, 3.9])
        step = _compute_step_metrics(dense, turbo, forced_token=2, position=0)
        self.assertEqual(step.position, 0)
        self.assertGreater(step.logit_cosine, 0.99)
        self.assertTrue(step.top1_agreement)
        self.assertGreater(step.top5_overlap, 0.0)
        self.assertFalse(step.any_nan_or_inf)

    def test_compute_step_metrics_detects_nan(self):
        dense = np.array([1.0, np.nan, 3.0, 4.0])
        turbo = np.array([1.0, 2.0, 3.0, 4.0])
        step = _compute_step_metrics(dense, turbo, forced_token=0, position=0)
        self.assertTrue(step.any_nan_or_inf)
        self.assertEqual(step.logit_cosine, 0.0)

    def test_compute_step_metrics_top1_disagreement(self):
        dense = np.array([1.0, 5.0, 3.0, 4.0])
        turbo = np.array([1.0, 2.0, 6.0, 4.0])
        step = _compute_step_metrics(dense, turbo, forced_token=1, position=0)
        self.assertFalse(step.top1_agreement)
        self.assertEqual(step.dense_argmax_rank_in_turbo, 2)

    def test_aggregate_execution_stats_empty(self):
        turbo_cache = make_turbo_caches(2, 4, 2, 128, use_qjl=False)
        for cache in turbo_cache:
            cache.reset_execution_stats()
        stats = _aggregate_execution_stats(turbo_cache)
        self.assertEqual(stats["online_attention_calls"], 0)
        self.assertEqual(stats["fallback_calls"], 0)

    def test_benchmark_forced_decode_fixture_runs(self):
        model = self._tiny_model()
        tokenizer = _FakeTokenizer()
        context_tokens = tokenizer.encode("hello world this is a test")
        continuation_tokens = tokenizer.encode(" forced continuation tokens ")
        from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig

        adapter = TurboPolarLlamaAdapter(
            TurboPolarConfig(
                num_q_heads=4,
                num_kv_heads=2,
                head_dim=128,
                block_size=64,
                storage_mode="kv_quant",
                use_int8_radii=True,
                k_angle_bits_level1=8,
                k_angle_bits_deep=8,
                split_dim=0,
            )
        )
        result = benchmark_forced_decode_fixture(
            model, tokenizer, context_tokens, continuation_tokens, adapter
        )
        self.assertEqual(result.context_length, len(context_tokens))
        self.assertEqual(result.continuation_length, len(continuation_tokens))
        self.assertEqual(len(result.steps), len(continuation_tokens))
        self.assertGreater(result.kernel_stats["online_attention_calls"], 0)
        # The paged attention path is currently a fallback reference
        # implementation; a true fused Metal kernel is not yet available.
        self.assertGreaterEqual(result.kernel_stats["fallback_calls"], 0)
        for step in result.steps:
            self.assertFalse(step.any_nan_or_inf)

    def test_normalize_text_prompts(self):
        suite_path = (
            Path(__file__).resolve().parents[2] / "benchmarks" / "prompt_suite.jsonl"
        )
        normalized = normalize_prompts(_FakeTokenizer(), suite_path)
        self.assertIsInstance(normalized, list)
        self.assertTrue(all("tokens" in entry for entry in normalized))

    def test_load_token_fixtures_has_exact_lengths(self):
        fixtures_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "exact_token_fixtures.jsonl"
        )
        fixtures = load_token_fixtures(fixtures_path)
        self.assertTrue(len(fixtures) > 0)
        for fx in fixtures:
            self.assertEqual(len(fx["tokens"]), fx["length"])
            self.assertIn(
                fx["category"], ("short", "boundary", "medium", "long", "stress")
            )


if __name__ == "__main__":
    unittest.main()

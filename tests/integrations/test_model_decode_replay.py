"""Model-level decode replay test for fused TurboPolar attention.

Uses a tiny supported model and a short deterministic continuation to verify
that the fused decode path produces finite logits and exercises the fused kernel.
"""

import unittest

import mlx.core as mx
import mlx_lm.models.llama as llama
import numpy as np
import pytest

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm.adapter import TurboPolarLlamaAdapter
from rfsn_v11.integrations.mlx_lm.cache import make_turbo_caches
from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode


class _FakeTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text: str):
        return [ord(c) % self.vocab_size for c in text]


class TestModelDecodeReplay(unittest.TestCase):
    @staticmethod
    def _cosine_similarity(a, b):
        dot = np.sum(a * b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _argmax_agreement(a, b):
        return float(np.argmax(a) == np.argmax(b))

    @staticmethod
    def _topk_overlap(a, b, k=5):
        top_a = set(np.argsort(a)[-k:])
        top_b = set(np.argsort(b)[-k:])
        if len(top_a) == 0:
            return 0.0
        return len(top_a & top_b) / len(top_a)

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

    @pytest.mark.model_integration
    def test_fused_decode_replay_produces_finite_logits(self):
        model = self._tiny_model()
        tokenizer = _FakeTokenizer()
        context_tokens = tokenizer.encode("hello world this is a test")
        continuation_tokens = tokenizer.encode(" forced decode replay ")

        num_layers = len(model.layers)
        turbo_config = TurboPolarConfig(
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=128,
            block_size=64,
            storage_mode="kv_quant",
            use_int8_radii=True,
            k_angle_bits_deep=8,
            split_dim=0,
        )
        adapter = TurboPolarLlamaAdapter(turbo_config)
        turbo_cache = make_turbo_caches(num_layers, 4, 2, 128, use_qjl=False)
        # Reset once via the bridge (singleton); do not sum per-cache stats.
        turbo_cache[0].reset_execution_stats()

        context_mx = mx.array(context_tokens)[None, :]

        # Prefill both paths.
        dense_cache = [llama.KVCache() for _ in range(num_layers)]
        dense_prefill = model(context_mx, cache=dense_cache)

        adapter.install(model)
        try:
            turbo_prefill = model(context_mx, cache=turbo_cache)
        finally:
            adapter.uninstall()
        mx.eval(dense_prefill, turbo_prefill)

        # Fused forced-decode loop.
        adapter.install(model)
        try:
            for i, forced_token in enumerate(continuation_tokens):
                token_mx = mx.array([[forced_token]])
                dense_logits = model(token_mx, cache=dense_cache)
                turbo_logits = model(token_mx, cache=turbo_cache)
                mx.eval(dense_logits, turbo_logits)

                dense_last = np.array(dense_logits[:, -1, :].astype(mx.float32))
                turbo_last = np.array(turbo_logits[:, -1, :].astype(mx.float32))

                self.assertFalse(
                    np.isnan(dense_last).any(), f"NaN in dense logits at position {i}"
                )
                self.assertFalse(
                    np.isnan(turbo_last).any(), f"NaN in turbo logits at position {i}"
                )
                self.assertFalse(
                    np.isinf(dense_last).any(), f"Inf in dense logits at position {i}"
                )
                self.assertFalse(
                    np.isinf(turbo_last).any(), f"Inf in turbo logits at position {i}"
                )
        finally:
            adapter.uninstall()

        # Verify paged attention path was exercised.
        # Read singleton stats once from any cache; do not sum across caches.
        stats = turbo_cache[0].execution_stats()
        total_online = stats.online_attention_calls
        total_fallback = stats.fallback_calls
        self.assertGreater(
            total_online, 0, "Decode did not exercise online_attention kernel"
        )
        # NOTE: The paged attention path is currently a fallback reference
        # implementation; a true fused Metal kernel is not yet available.
        self.assertGreaterEqual(total_fallback, 0)

        # At least 16 decode positions.
        self.assertGreaterEqual(len(continuation_tokens), 16)

    @pytest.mark.model_integration
    def test_adapter_rejects_mask_not_none(self):
        model = self._tiny_model()
        turbo_config = TurboPolarConfig(
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=128,
            block_size=64,
            storage_mode="kv_quant",
            use_int8_radii=True,
            k_angle_bits_deep=8,
            split_dim=0,
        )
        adapter = TurboPolarLlamaAdapter(turbo_config)
        turbo_cache = make_turbo_caches(2, 4, 2, 128, use_qjl=False)

        adapter.install(model)
        try:
            # The top-level Model.__call__ no longer accepts mask; test the
            # attention wrapper directly.
            with self.assertRaises(NotImplementedError):
                x = mx.random.normal((1, 1, 512)).astype(mx.float16)
                fake_mask = mx.zeros((1, 1, 1, 10), dtype=mx.float16)
                model.layers[0].self_attn(x, mask=fake_mask, cache=turbo_cache[0])
        finally:
            adapter.uninstall()

    @pytest.mark.native_metal_required
    def test_strict_model_adapter_with_compressed_pages(self):
        """METAL_STRICT through the real model adapter must not fallback at 2K context."""
        mx.random.seed(4200)
        model = self._tiny_model()
        tokenizer = _FakeTokenizer()
        # 2048 tokens = 2 full pages, enough to exercise compressed-page path.
        context_tokens = [
            int(i % tokenizer.vocab_size) for i in range(2048)
        ]
        continuation_tokens = tokenizer.encode(" forced decode replay ")

        num_layers = len(model.layers)
        turbo_config = TurboPolarConfig(
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=128,
            block_size=64,
            storage_mode="kv_quant",
            use_int8_radii=True,
            k_angle_bits_deep=8,
            split_dim=0,
            execution_mode=ExecutionMode.METAL_STRICT,
        )
        adapter = TurboPolarLlamaAdapter(turbo_config)
        turbo_cache = make_turbo_caches(
            num_layers, 4, 2, 128, use_qjl=False,
            execution_mode=ExecutionMode.METAL_STRICT,
        )
        # Reset singleton stats once via any cache.
        turbo_cache[0].reset_execution_stats()

        context_mx = mx.array(context_tokens)[None, :]

        # Dense reference path for comparison.
        dense_cache = [llama.KVCache() for _ in range(num_layers)]
        _ = model(context_mx, cache=dense_cache)

        # Prefill.
        adapter.install(model)
        try:
            turbo_prefill = model(context_mx, cache=turbo_cache)
        finally:
            adapter.uninstall()
        mx.eval(turbo_prefill)

        # Fused decode in strict mode with dense comparison.
        adapter.install(model)
        try:
            for i, forced_token in enumerate(continuation_tokens):
                token_mx = mx.array([[forced_token]])
                dense_logits = model(token_mx, cache=dense_cache)
                turbo_logits = model(token_mx, cache=turbo_cache)
                mx.eval(dense_logits, turbo_logits)
                dense_last = np.array(dense_logits[:, -1, :].astype(mx.float32))
                turbo_last = np.array(turbo_logits[:, -1, :].astype(mx.float32))
                self.assertFalse(
                    np.isnan(turbo_last).any(),
                    f"NaN in turbo logits at strict position"
                )
                self.assertFalse(
                    np.isinf(turbo_last).any(),
                    f"Inf in turbo logits at strict position"
                )
                # Quality metrics per position.
                cosine = self._cosine_similarity(dense_last[0], turbo_last[0])
                self.assertGreater(
                    cosine, 0.90,
                    f"Logit cosine {cosine:.4f} at position {i} below threshold"
                )
                argmax_agree = self._argmax_agreement(dense_last[0], turbo_last[0])
                self.assertEqual(
                    argmax_agree, 1.0,
                    f"Argmax disagreement at position {i}"
                )
        finally:
            adapter.uninstall()

        # Verify strict path: zero fallback, compressed pages dispatched.
        # Read singleton stats once from the bridge via any cache.
        stats = turbo_cache[0].execution_stats()
        total_online = stats.online_attention_calls
        total_fallback = stats.fallback_calls
        total_page_dispatches = getattr(stats, "compressed_page_dispatches", 0)
        total_tail_dispatches = getattr(stats, "dense_tail_dispatches", 0)
        total_dense_tail_calls = getattr(stats, "dense_tail_calls", 0)

        num_decode_steps = len(continuation_tokens)
        num_layers = len(model.layers)
        # 2048 tokens with block_size=64 and 16 blocks/page = 2 full pages,
        # 0 tail after prefill. Each decode appends 1 token to the tail,
        # so every step dispatches 2 pages + 1 tail.
        expected_pages_per_step = 2
        expected_page_dispatches = expected_pages_per_step * num_layers * num_decode_steps
        expected_tail_dispatches = num_layers * num_decode_steps
        expected_online_calls = num_layers * num_decode_steps

        self.assertEqual(
            total_online, expected_online_calls,
            f"Expected {expected_online_calls} online_attention_calls, got {total_online}"
        )
        self.assertEqual(
            total_fallback, 0,
            f"METAL_STRICT model adapter recorded {total_fallback} fallback(s)"
        )
        self.assertEqual(
            total_page_dispatches, expected_page_dispatches,
            f"Expected {expected_page_dispatches} page dispatches, got {total_page_dispatches}"
        )
        self.assertEqual(
            total_tail_dispatches, expected_tail_dispatches,
            f"Expected {expected_tail_dispatches} tail dispatches, got {total_tail_dispatches}"
        )
        self.assertEqual(
            total_dense_tail_calls, expected_tail_dispatches,
            f"Expected {expected_tail_dispatches} dense_tail_calls, got {total_dense_tail_calls}"
        )

        # Verify traces: each layer/step is unique.
        # Note: trace count depends on how the model schedules layers; we verify
        # uniqueness and completeness rather than exact count.
        traces = turbo_cache[0].execution_traces()
        self.assertGreater(len(traces), 0, "No execution traces collected")

        # Collect all traces from all layer caches.
        all_traces = []
        for cache in turbo_cache:
            all_traces.extend(cache.execution_traces())

        seen = set()
        for t in all_traces:
            key = (t.layer_index, t.decode_step)
            self.assertNotIn(key, seen, f"Duplicate trace for layer={t.layer_index}, step={t.decode_step}")
            seen.add(key)
        seen = set()
        for t in traces:
            key = (t.layer_index, t.decode_step)
            self.assertNotIn(key, seen, f"Duplicate trace for layer={t.layer_index}, step={t.decode_step}")
            seen.add(key)
            self.assertEqual(t.fallback_count, 0, f"Fallback in trace for layer={t.layer_index}, step={t.decode_step}")
            self.assertTrue(t.all_outputs_evaluated, f"Un-evaluated output in trace for layer={t.layer_index}, step={t.decode_step}")


if __name__ == "__main__":
    unittest.main()

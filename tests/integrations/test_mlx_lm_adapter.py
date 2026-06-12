"""Tests for the instance-level mlx_lm Llama adapter."""

import unittest

import mlx.core as mx
from mlx_lm.models import llama

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm import TurboPolarLlamaAdapter


def _tiny_llama_args(**overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return llama.ModelArgs(**defaults)


def _make_tiny_llama(**overrides):
    return llama.Model(_tiny_llama_args(**overrides))


class TestMLXLMAdapter(unittest.TestCase):
    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=2,
        )

    def test_install_wraps_all_layers(self):
        model = _make_tiny_llama()
        adapter = TurboPolarLlamaAdapter(self.config)
        adapter.install(model)
        self.assertEqual(adapter.wrapped_layer_count, len(model.layers))
        for layer in model.layers:
            self.assertEqual(type(layer.self_attn).__name__, "TurboPolarLlamaAttention")
        adapter.uninstall()
        for layer in model.layers:
            self.assertIsInstance(layer.self_attn, llama.Attention)

    def test_uninstall_restores_originals(self):
        model = _make_tiny_llama()
        adapter = TurboPolarLlamaAdapter(self.config)
        adapter.install(model)
        originals = [layer.self_attn for layer in model.layers]
        adapter.uninstall()
        for i, layer in enumerate(model.layers):
            self.assertIs(layer.self_attn, originals[i].original_attention)

    def test_two_models_only_one_patched(self):
        model_a = _make_tiny_llama()
        model_b = _make_tiny_llama()
        adapter = TurboPolarLlamaAdapter(self.config)
        adapter.install(model_a)
        for layer in model_a.layers:
            self.assertEqual(type(layer.self_attn).__name__, "TurboPolarLlamaAttention")
        for layer in model_b.layers:
            self.assertIsInstance(layer.self_attn, llama.Attention)
        adapter.uninstall()

    def test_unsupported_model_rejected(self):
        adapter = TurboPolarLlamaAdapter(self.config)
        with self.assertRaises(ValueError):
            adapter.install(object())

    def test_partial_installation_rolls_back(self):
        model = _make_tiny_llama(num_hidden_layers=3)
        # Corrupt the middle layer so installation fails partway through.
        corrupted = object()
        model.layers[1].self_attn = corrupted
        adapter = TurboPolarLlamaAdapter(self.config)
        with self.assertRaises(ValueError):
            adapter.install(model)
        # Rollback must restore layers 0 and 2; layer 1 stays as we corrupted it.
        self.assertIsInstance(model.layers[0].self_attn, llama.Attention)
        self.assertIs(model.layers[1].self_attn, corrupted)
        self.assertIsInstance(model.layers[2].self_attn, llama.Attention)
        self.assertFalse(adapter._installed)

    def test_double_install_prevented(self):
        model = _make_tiny_llama()
        adapter = TurboPolarLlamaAdapter(self.config)
        adapter.install(model)
        with self.assertRaises(RuntimeError):
            adapter.install(model)
        adapter.uninstall()

    def test_wrapped_count_equals_expected(self):
        model = _make_tiny_llama(num_hidden_layers=5)
        adapter = TurboPolarLlamaAdapter(self.config)
        adapter.install(model)
        self.assertEqual(adapter.wrapped_layer_count, 5)
        adapter.uninstall()

    def test_unsupported_head_dim_rejected(self):
        model = _make_tiny_llama(hidden_size=128, num_attention_heads=2, num_key_value_heads=1)
        adapter = TurboPolarLlamaAdapter(self.config)
        with self.assertRaises(ValueError):
            adapter.install(model)

    def test_config_q_heads_mismatch_rejected(self):
        model = _make_tiny_llama(num_attention_heads=8)
        adapter = TurboPolarLlamaAdapter(self.config)
        with self.assertRaises(ValueError):
            adapter.install(model)

    def test_config_kv_heads_mismatch_rejected(self):
        model = _make_tiny_llama(num_key_value_heads=1)
        adapter = TurboPolarLlamaAdapter(self.config)
        with self.assertRaises(ValueError):
            adapter.install(model)

    def test_parameter_tree_unchanged_after_install(self):
        model = _make_tiny_llama()
        before = list(model.parameters().keys())
        adapter = TurboPolarLlamaAdapter(self.config)
        adapter.install(model)
        after = list(model.parameters().keys())
        self.assertEqual(before, after)
        adapter.uninstall()

    def test_state_dict_unchanged_after_install(self):
        model = _make_tiny_llama()
        before = set(model.state_dict().keys())
        adapter = TurboPolarLlamaAdapter(self.config)
        adapter.install(model)
        after = set(model.state_dict().keys())
        self.assertEqual(before, after)
        adapter.uninstall()

    def test_uninstall_restores_exact_classes(self):
        model = _make_tiny_llama()
        adapter = TurboPolarLlamaAdapter(self.config)
        adapter.install(model)
        adapter.uninstall()
        for layer in model.layers:
            self.assertIs(type(layer.self_attn), llama.Attention)


if __name__ == "__main__":
    unittest.main()

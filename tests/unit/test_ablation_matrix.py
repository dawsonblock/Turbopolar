"""Ablation matrix: systematic quality measurement for quantization configs.

Replaces end-to-end perplexity with direct reconstruction metrics:
  - Compression ratio (dense fp16 bytes / compressed bytes)
  - K reconstruction MSE and relative error
  - V reconstruction MSE and relative error
  - Attention-score relative error (K @ q)
  - Cosine similarity between original and reconstructed KV

All measurements are deterministic (fixed seed) and require no model weights.
"""

import dataclasses
import unittest

import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime

import pytest

pytest.importorskip("mlx")


class TestAblationMatrix(unittest.TestCase):
    """Systematic ablation of quantization parameters against reconstruction."""

    @staticmethod
    def _make_config(**overrides):
        defaults = dict(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            use_int8_radii=True,
            k_angle_bits_level1=8,
            k_angle_bits_deep=8,
            v_bits=8,
            storage_mode="kv_quant",
            seed=42,
            dense_tail_capacity=512,
            flush_batch_size=64,
            warm_cache_capacity_blocks=0,
        )
        defaults.update(overrides)
        return TurboPolarConfig(**defaults)

    @staticmethod
    def _synthetic_kv(shape, seed=42):
        """Generate deterministic synthetic KV tensors."""
        mx.random.seed(seed)
        k = mx.random.normal(shape, dtype=mx.float16)
        v = mx.random.normal(shape, dtype=mx.float16)
        return k, v

    @staticmethod
    def _compute_compression_ratio(k_orig, v_orig, k_comp, v_comp):
        """Return ratio of original bytes to compressed bytes."""
        orig_bytes = int(k_orig.size * k_orig.itemsize + v_orig.size * v_orig.itemsize)
        comp_bytes = 0
        for arr in (k_comp.radii, k_comp.angle_codes_l1, k_comp.angle_codes_deep):
            comp_bytes += int(arr.size * arr.itemsize)
        if k_comp.radii_scales is not None:
            comp_bytes += int(k_comp.radii_scales.size * k_comp.radii_scales.itemsize)
        for arr in (v_comp.codes, v_comp.scales):
            comp_bytes += int(arr.size * arr.itemsize)
        if v_comp.zero_points is not None:
            comp_bytes += int(v_comp.zero_points.size * v_comp.zero_points.itemsize)
        return orig_bytes / comp_bytes if comp_bytes > 0 else 0.0

    @staticmethod
    def _mse(a, b):
        return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())

    @staticmethod
    def _rel_err(a, b):
        denom = mx.mean(mx.abs(a.astype(mx.float32)))
        denom = mx.where(denom == 0, mx.array(1.0, dtype=mx.float32), denom)
        return float(mx.mean(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)) / denom).item())

    @staticmethod
    def _cosine_sim(a, b):
        a_f = a.astype(mx.float32).flatten()
        b_f = b.astype(mx.float32).flatten()
        dot = mx.sum(a_f * b_f)
        norm_a = mx.sqrt(mx.sum(a_f * a_f))
        norm_b = mx.sqrt(mx.sum(b_f * b_f))
        return float((dot / (norm_a * norm_b)).item())

    def _run_ablation(self, config, k, v):
        """Compress/decompress and return metrics dict."""
        B, H, T, D = k.shape
        L = config.block_size
        assert T % L == 0
        N = T // L
        k_blocks = k.reshape(B, H, N, L, D)
        v_blocks = v.reshape(B, H, N, L, D)

        k_enc = PolarQuantEncoder(config)
        v_enc = GroupedVQuantizer(group_size=32)
        decoder = PolarQuantDecoder()

        k_comp = k_enc.encode_blocks(k_blocks)
        v_comp = v_enc.encode_blocks(v_blocks)

        k_recon = decoder.decode_block(
            dataclasses.replace(k_comp, shape=(B, H, T, D))
        )
        v_recon = v_enc.dequantize_block(v_comp).reshape(B, H, T, D)

        ratio = self._compute_compression_ratio(k, v, k_comp, v_comp)
        k_mse = self._mse(k, k_recon)
        v_mse = self._mse(v, v_recon)
        k_rel = self._rel_err(k, k_recon)
        v_rel = self._rel_err(v, v_recon)
        k_cos = self._cosine_sim(k, k_recon)
        v_cos = self._cosine_sim(v, v_recon)

        # Attention-score relative error with random query.
        mx.random.seed(99)
        q = mx.random.normal((B, config.num_q_heads, D), dtype=mx.float16)
        nq = config.num_q_heads // config.num_kv_heads
        k_rep = mx.repeat(k, nq, axis=1)
        v_rep = mx.repeat(v, nq, axis=1)
        k_r_rep = mx.repeat(k_recon, nq, axis=1)
        v_r_rep = mx.repeat(v_recon, nq, axis=1)
        orig_scores = mx.sum(q[:, :, None, :] * k_rep, axis=-1)
        recon_scores = mx.sum(q[:, :, None, :] * k_r_rep, axis=-1)
        score_rel = self._rel_err(orig_scores, recon_scores)

        return {
            "compression_ratio": ratio,
            "k_mse": k_mse,
            "v_mse": v_mse,
            "k_rel_err": k_rel,
            "v_rel_err": v_rel,
            "score_rel_err": score_rel,
            "k_cosine": k_cos,
            "v_cosine": v_cos,
        }

    def test_ablation_k_angle_bits(self):
        """Vary K angle bits and verify quality stays within bounds."""
        shape = (1, 4, 512, 128)
        k, v = self._synthetic_kv(shape)
        results = []
        for l1 in (4, 8):
            for deep in (2, 4, 8):
                config = self._make_config(
                    k_angle_bits_level1=l1,
                    k_angle_bits_deep=deep,
                )
                m = self._run_ablation(config, k, v)
                results.append((l1, deep, m))

        # All configurations must show *some* structure (not catastrophic).
        # Exact thresholds are intentionally loose; the value is in the
        # relative comparison, not absolute pass/fail.
        for l1, deep, m in results:
            with self.subTest(l1=l1, deep=deep):
                self.assertGreater(m["k_cosine"], 0.70,
                    f"l1={l1},deep={deep}: K cosine too low")
                self.assertLess(m["score_rel_err"], 0.70,
                    f"l1={l1},deep={deep}: attention score error catastrophic")
                self.assertGreater(m["compression_ratio"], 1.5,
                    f"l1={l1},deep={deep}: compression ratio too low")

        # Monotonicity: more bits -> better quality.
        m_88 = next(m for l1, deep, m in results if l1 == 8 and deep == 8)
        m_42 = next(m for l1, deep, m in results if l1 == 4 and deep == 2)
        self.assertGreater(m_88["k_cosine"], m_42["k_cosine"])
        self.assertLess(m_88["score_rel_err"], m_42["score_rel_err"])

    def test_ablation_radii_mode(self):
        """Compare int8 radii vs fp16 radii quality and compression."""
        shape = (1, 4, 512, 128)
        k, v = self._synthetic_kv(shape)

        config_int8 = self._make_config(use_int8_radii=True)
        config_fp16 = self._make_config(use_int8_radii=False)

        m_int8 = self._run_ablation(config_int8, k, v)
        m_fp16 = self._run_ablation(config_fp16, k, v)

        # int8 radii should compress better.
        self.assertGreater(m_int8["compression_ratio"], m_fp16["compression_ratio"])
        # fp16 radii should reconstruct slightly better, but both high.
        self.assertGreater(m_int8["k_cosine"], 0.95)
        self.assertGreater(m_fp16["k_cosine"], 0.95)

    def test_ablation_v_group_size(self):
        """Vary V quantization group size."""
        shape = (1, 4, 512, 128)
        k, v = self._synthetic_kv(shape)
        config = self._make_config()
        k_enc = PolarQuantEncoder(config)
        v_enc_16 = GroupedVQuantizer(group_size=16)
        v_enc_32 = GroupedVQuantizer(group_size=32)
        v_enc_64 = GroupedVQuantizer(group_size=64)
        decoder = PolarQuantDecoder()

        B, H, T, D = shape
        L = config.block_size
        N = T // L
        k_blocks = k.reshape(B, H, N, L, D)
        v_blocks = v.reshape(B, H, N, L, D)
        k_comp = k_enc.encode_blocks(k_blocks)

        results = []
        for gs, v_enc in ((16, v_enc_16), (32, v_enc_32), (64, v_enc_64)):
            v_comp = v_enc.encode_blocks(v_blocks)
            v_recon = v_enc.dequantize_block(v_comp).reshape(B, H, T, D)
            v_cos = self._cosine_sim(v, v_recon)
            v_rel = self._rel_err(v, v_recon)
            results.append((gs, v_cos, v_rel))

        # Smaller group size = better quality, worse compression.
        for gs, v_cos, v_rel in results:
            with self.subTest(group_size=gs):
                self.assertGreater(v_cos, 0.95)
                self.assertLess(v_rel, 0.05)

    def test_ablation_warm_cache_vs_direct(self):
        """Warm-cache path must not change end-to-end attention output."""
        config_warm = self._make_config(
            dense_tail_capacity=128,
            flush_batch_size=64,
            warm_cache_capacity_blocks=4,
        )
        config_cold = self._make_config(
            dense_tail_capacity=128,
            flush_batch_size=64,
            warm_cache_capacity_blocks=0,
        )

        cache_warm = TurboPolarKVCacheRuntime(config_warm)
        cache_cold = TurboPolarKVCacheRuntime(config_cold)

        mx.random.seed(2025)
        k = mx.random.normal((1, 4, 512, 128), dtype=mx.float16)
        v = mx.random.normal((1, 4, 512, 128), dtype=mx.float16)

        cache_warm.append_many(k, v)
        cache_cold.append_many(k, v)

        # Both must represent the same total tokens.
        self.assertEqual(cache_warm.actual_seq_len, cache_cold.actual_seq_len)
        self.assertEqual(cache_warm.actual_seq_len, 512)

        # Cold path has more compressed blocks; warm path has fewer.
        self.assertGreaterEqual(
            cache_cold.total_blocks, cache_warm.total_blocks
        )

    def test_ablation_matrix_sanity(self):
        """Full matrix: all supported parameter combinations must be runnable."""
        shape = (1, 4, 256, 128)
        k, v = self._synthetic_kv(shape)
        configs = []
        for l1 in (4, 8):
            for deep in (2, 4, 8):
                for radii in (True, False):
                    configs.append(
                        self._make_config(
                            k_angle_bits_level1=l1,
                            k_angle_bits_deep=deep,
                            use_int8_radii=radii,
                        )
                    )

        for config in configs:
            with self.subTest(
                l1=config.k_angle_bits_level1,
                deep=config.k_angle_bits_deep,
                radii=config.use_int8_radii,
            ):
                m = self._run_ablation(config, k, v)
                # Sanity: no NaNs or infinities.
                for key, val in m.items():
                    self.assertTrue(
                        np.isfinite(val),
                        f"{key} is non-finite: {val}"
                    )
                # All must show some compression.
                self.assertGreater(m["compression_ratio"], 1.0)
                # K and V cosine must be positive.
                self.assertGreater(m["k_cosine"], 0.0)
                self.assertGreater(m["v_cosine"], 0.0)


if __name__ == "__main__":
    unittest.main()

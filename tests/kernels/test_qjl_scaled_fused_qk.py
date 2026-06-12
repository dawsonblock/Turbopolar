"""Tests that the fused QK kernel applies attention_scale to the QJL correction."""

import unittest

import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge


class TestQJLScaledFusedQK(unittest.TestCase):
    """CPU and Metal QJL score paths must use the same scale convention."""

    def setUp(self):
        if not mx.metal.is_available():
            self.skipTest("Metal not available.")
        self.config = TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=0,
            num_q_heads=4,
            num_kv_heads=4,
            seed=42,
        )
        self.encoder = PolarQuantEncoder(self.config)
        self.decoder = PolarQuantDecoder()
        self.qjl_encoder = QJLResidualEncoder(self.config)
        self.bridge = MetalKernelBridge()

    def _encode_unified(self, k_original):
        B, H, T, D = k_original.shape
        S = T // self.config.block_size
        k_blocked = k_original.reshape(B, H, S, self.config.block_size, D)
        blocks = [self.encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        return blocks[0].__class__(
            radii=mx.stack([b.radii for b in blocks], axis=2),
            angle_codes_l1=mx.stack([b.angle_codes_l1 for b in blocks], axis=2),
            angle_codes_deep=mx.stack([b.angle_codes_deep for b in blocks], axis=2),
            shape=(B, H, T, D),
            block_size=self.config.block_size,
            head_dim=D,
            metadata=blocks[0].metadata,
        )

    def _pack_q_signs(self, q_proj):
        B, H, P = q_proj.shape
        signs = q_proj >= 0
        reshaped = signs.reshape(B, H, P // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        return mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)

    def test_qjl_contribution_scales_with_attention_scale(self):
        """Difference between QJL-on and QJL-off scores must scale with attention_scale."""
        B, H, S, L, D = 1, 4, 1, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal((B, H, S * L, D))
        q = mx.random.normal((B, H, D))
        unified = self._encode_unified(k_original)
        k_recon = self.decoder.decode_block(unified)
        residual = k_original - k_recon
        qjl_payload = self.qjl_encoder.compute_residual_sketch(
            residual.reshape(B, H, S, L, D)
        )
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_packed = self._pack_q_signs(q_proj)

        for scale in (1.0, 0.5, 0.25):
            cfg = TurboPolarConfig(
                head_dim=128,
                qjl_proj_dim=64,
                block_size=64,
                split_dim=0,
                num_q_heads=4,
                num_kv_heads=4,
                seed=42,
                attention_scale=scale,
            )
            scores_no_qjl = self.bridge.execute_fused_qk(q, unified, cfg)
            scores_qjl = self.bridge.execute_fused_qk_qjl(
                q, unified, qjl_payload, q_packed, cfg
            )
            mx.eval(scores_no_qjl, scores_qjl)

            qjl_contribution = np.mean(
                np.abs(np.array(scores_qjl) - np.array(scores_no_qjl))
            )
            self.assertGreater(
                qjl_contribution, 0.0, "QJL contribution should be nonzero"
            )
            if scale != 1.0:
                # We cannot compare across runs directly because the base scores also change,
                # but the ratio of contribution to scale should be constant.
                self.assertAlmostEqual(
                    qjl_contribution / scale,
                    self._reference_contribution(q, qjl_payload, q_packed) / 1.0,
                    places=3,
                )

    def _reference_contribution(self, q, qjl_payload, q_packed):
        """Mean absolute unscaled QJL correction from the estimator."""
        from rfsn_v11.quant.qjl.score_estimate import qjl_dot_estimate

        corr = qjl_dot_estimate(q, qjl_payload, q_packed)
        mx.eval(corr)
        return float(np.mean(np.abs(np.array(corr))))

    def test_cpu_metal_qjl_scale_convention_match(self):
        """CPU fallback and Metal kernel must agree on scaled QJL addition."""
        B, H, S, L, D = 1, 4, 1, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal((B, H, S * L, D))
        q = mx.random.normal((B, H, D))
        unified = self._encode_unified(k_original)
        k_recon = self.decoder.decode_block(unified)
        residual = k_original - k_recon
        qjl_payload = self.qjl_encoder.compute_residual_sketch(
            residual.reshape(B, H, S, L, D)
        )
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_packed = self._pack_q_signs(q_proj)

        cfg = TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=0,
            num_q_heads=4,
            num_kv_heads=4,
            seed=42,
            attention_scale=0.25,
        )

        metal_scores = self.bridge.execute_fused_qk_qjl(
            q, unified, qjl_payload, q_packed, cfg
        )
        cpu_scores = self.bridge._cpu_fused_qk_qjl(
            q, unified, qjl_payload, q_packed, cfg
        )
        mx.eval(metal_scores, cpu_scores)

        # Metal accumulates in fp16; CPU uses fp32. Allow small absolute differences
        # and ignore relative differences near zero.
        np.testing.assert_allclose(
            np.array(metal_scores), np.array(cpu_scores), rtol=5e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()

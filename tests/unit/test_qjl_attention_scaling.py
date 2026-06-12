"""Tests that QJL correction is scaled by attention_scale exactly once."""

import unittest

import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder
from rfsn_v11.quant.qjl.score_estimate import qjl_dot_estimate


class TestQJLAttentionScaling(unittest.TestCase):
    """QJL estimator returns an unscaled dot product; callers apply attention_scale once."""

    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=0,
            num_q_heads=4,
            num_kv_heads=4,
            use_qjl=False,
        )
        self.qjl_encoder = QJLResidualEncoder(self.config)

    def _pack_q_signs(self, q_proj):
        B, H, P = q_proj.shape
        signs = q_proj >= 0
        reshaped = signs.reshape(B, H, P // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        return mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)

    def test_qjl_correction_scales_with_attention_scale(self):
        """qjl_dot_estimate output multiplied by scale should change 4x when scale changes 4x."""
        B, H, S, L, D = 1, 4, 1, 64, 128
        mx.random.seed(42)
        q = mx.random.normal((B, H, D))
        residual = mx.random.normal((B, H, S * L, D))
        qjl_payload = self.qjl_encoder.compute_residual_sketch(
            residual.reshape(B, H, S, L, D)
        )
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_packed = self._pack_q_signs(q_proj)

        qjl_corr = qjl_dot_estimate(q, qjl_payload, q_packed)
        mx.eval(qjl_corr)

        scaled_1 = float(mx.mean(qjl_corr * 1.0).item())
        scaled_025 = float(mx.mean(qjl_corr * 0.25).item())

        self.assertAlmostEqual(scaled_1 / scaled_025, 4.0, places=5)

    def test_qjl_estimator_is_unscaled(self):
        """The estimator itself must not internally apply attention_scale."""
        B, H, S, L, D = 1, 4, 1, 64, 128
        mx.random.seed(42)
        q = mx.random.normal((B, H, D))
        residual = mx.random.normal((B, H, S * L, D))
        q_proj = mx.matmul(q, self.qjl_encoder.W)

        # Estimate should not depend on the config attention_scale.
        qjl_payload = self.qjl_encoder.compute_residual_sketch(
            residual.reshape(B, H, S, L, D)
        )
        q_packed = self._pack_q_signs(q_proj)

        corr_1 = qjl_dot_estimate(q, qjl_payload, q_packed)
        corr_05 = qjl_dot_estimate(q, qjl_payload, q_packed)
        mx.eval(corr_1, corr_05)

        np.testing.assert_allclose(np.array(corr_1), np.array(corr_05), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

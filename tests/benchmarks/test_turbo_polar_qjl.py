import unittest
import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder
from rfsn_v11.quant.qjl.score_estimate import qjl_dot_estimate


class TestTurboPolarQJL(unittest.TestCase):
    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128, qjl_proj_dim=64, block_size=64, split_dim=128,
            num_q_heads=2, num_kv_heads=2, seed=42,
        )
        self.polar_encoder = PolarQuantEncoder(self.config)
        self.polar_decoder = PolarQuantDecoder()
        self.qjl_encoder = QJLResidualEncoder(self.config)

    def _encode_unified(self, k_original):
        B, H, T, D = k_original.shape
        S = T // self.config.block_size
        k_blocked = k_original.reshape(B, H, S, self.config.block_size, D)
        blocks = [self.polar_encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        radii = mx.stack([b.radii for b in blocks], axis=2)
        angle_l1 = mx.stack([b.angle_codes_l1 for b in blocks], axis=2)
        angle_deep = mx.stack([b.angle_codes_deep for b in blocks], axis=2)
        return blocks[0].__class__(
            radii=radii, angle_codes_l1=angle_l1, angle_codes_deep=angle_deep,
            shape=(B, H, T, D), block_size=self.config.block_size, head_dim=D,
            metadata=blocks[0].metadata
        )

    def _pack_q_signs(self, q_proj):
        B, H, P = q_proj.shape
        signs = q_proj >= 0
        reshaped = signs.reshape(B, H, P // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        return mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)

    def test_qjl_residual_correlation(self):
        """QJL correction should be positively correlated with the dense residual dot product."""
        B, H, S, L, D = 1, 2, 2, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        q = mx.random.normal(shape=[B, H, D])
        unified = self._encode_unified(k_original)
        k_recon = self.polar_decoder.decode_block(unified)
        residual_E = k_original - k_recon
        qjl_payload = self.qjl_encoder.compute_residual_sketch(residual_E.reshape(B, H, S, L, D))
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_packed = self._pack_q_signs(q_proj)
        qjl_corr = qjl_dot_estimate(q, qjl_payload, q_packed)

        dense_corr = mx.sum(q[:, :, None, :] * residual_E, axis=-1).reshape(B, H, S * L)
        mx.eval(dense_corr, qjl_corr)

        dc = np.array(dense_corr).flatten()
        qc = np.array(qjl_corr).flatten()
        corr = np.dot(dc, qc) / (np.linalg.norm(dc) * np.linalg.norm(qc) + 1e-12)
        self.assertGreater(corr, 0.0, "QJL correction is anti-correlated with dense residual dot product.")

    def test_qjl_calibrated_estimator_matches_cosine(self):
        """
        On synthetic data with known cosine similarity, the calibrated estimator
        (sin((pi/2) * sign_corr)) should recover the cosine more accurately than
        the raw sign correlation.
        """
        D = self.config.head_dim
        proj_dim = self.config.qjl_proj_dim
        rng = np.random.default_rng(self.config.seed)
        n_pairs = 200

        true_cosines = []
        linear_estimates = []
        calibrated_estimates = []

        for _ in range(n_pairs):
            # Random unit vector u
            u = rng.standard_normal(D).astype(np.float32)
            u /= np.linalg.norm(u) + 1e-12
            # Random unit vector v with controlled cosine w.r.t. u
            target_cos = rng.uniform(-0.9, 0.9)
            v_perp = rng.standard_normal(D).astype(np.float32)
            v_perp -= u * np.dot(u, v_perp)
            v_perp /= np.linalg.norm(v_perp) + 1e-12
            v = target_cos * u + np.sqrt(max(0.0, 1.0 - target_cos**2)) * v_perp
            v /= np.linalg.norm(v) + 1e-12

            # Random projection signs
            proj_u = u @ np.array(self.qjl_encoder.W)
            proj_v = v @ np.array(self.qjl_encoder.W)
            signs_u = proj_u >= 0
            signs_v = proj_v >= 0
            agreement = np.mean(signs_u == signs_v)
            sign_corr = 2.0 * agreement - 1.0

            linear_estimates.append(sign_corr)
            calibrated_estimates.append(np.sin((np.pi / 2.0) * sign_corr))
            true_cosines.append(target_cos)

        tc = np.array(true_cosines)
        le = np.array(linear_estimates)
        ce = np.array(calibrated_estimates)

        linear_mse = np.mean((tc - le) ** 2)
        calibrated_mse = np.mean((tc - ce) ** 2)
        self.assertLess(calibrated_mse, linear_mse,
                        "Calibrated cosine estimator should be closer to true cosine than raw sign correlation.")
        # The calibrated estimator should be essentially unbiased on this synthetic data.
        self.assertLess(np.abs(np.mean(ce - tc)), 0.05,
                        "Calibrated estimator is biased on synthetic cosine targets.")

    def test_qjl_sign_packing_bitorder(self):
        """Sign packing and unpacking must round-trip with little-endian bit order."""
        B, H, S, L, D = 1, 2, 1, 8, 128
        mx.random.seed(self.config.seed)
        k = mx.random.normal(shape=[B, H, S * L, D])
        qjl_payload = self.qjl_encoder.compute_residual_sketch(k.reshape(B, H, S, L, D))
        packed = np.array(qjl_payload.packed_signs)
        proj_dim = self.config.qjl_proj_dim
        flat = packed.reshape(-1, proj_dim // 8)
        bits = np.unpackbits(flat, axis=1, bitorder="little")[:, -proj_dim:]
        # Reconstruct signs and compare to raw projection signs
        flat_k = k.reshape(-1, D)
        proj = np.array(mx.matmul(mx.array(flat_k), self.qjl_encoder.W))
        expected_signs = (proj >= 0).reshape(-1, proj_dim)
        reconstructed_signs = bits.astype(bool)
        self.assertTrue(np.array_equal(expected_signs, reconstructed_signs))


if __name__ == "__main__":
    unittest.main()

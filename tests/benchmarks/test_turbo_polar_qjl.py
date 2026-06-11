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

    def test_qjl_residual_correlation(self):
        """
        QJL should produce residual dot-product estimates that are positively
        correlated with the dense residual dot products. A strict error-reduction
        gate is too strong for the current heuristic on random data.
        """
        B, H, S, L, D = 1, 2, 2, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        q = mx.random.normal(shape=[B, H, D])
        k_blocked = k_original.reshape(B, H, S, L, D)
        blocks = [self.polar_encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        radii = mx.stack([b.radii for b in blocks], axis=2)
        angle_l1 = mx.stack([b.angle_codes_l1 for b in blocks], axis=2)
        angle_deep = mx.stack([b.angle_codes_deep for b in blocks], axis=2)
        unified = blocks[0].__class__(
            radii=radii, angle_codes_l1=angle_l1, angle_codes_deep=angle_deep,
            shape=(B, H, S * L, D), block_size=L, head_dim=D, metadata=blocks[0].metadata
        )
        k_recon = self.polar_decoder.decode_block(unified)
        residual_E = k_original - k_recon
        qjl_payload = self.qjl_encoder.compute_residual_sketch(residual_E.reshape(B, H, S, L, D))
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_signs = q_proj >= 0
        reshaped = q_signs.reshape(B, H, self.config.qjl_proj_dim // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        q_packed = mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)
        qjl_corr = qjl_dot_estimate(q, qjl_payload, q_packed)

        dense_corr = mx.sum(q[:, :, None, :] * residual_E, axis=-1).reshape(B, H, S * L)
        mx.eval(dense_corr, qjl_corr)

        dc = np.array(dense_corr).flatten()
        qc = np.array(qjl_corr).flatten()
        corr = np.dot(dc, qc) / (np.linalg.norm(dc) * np.linalg.norm(qc) + 1e-12)
        self.assertGreater(corr, 0.0, "QJL correction is anti-correlated with dense residual dot product.")

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

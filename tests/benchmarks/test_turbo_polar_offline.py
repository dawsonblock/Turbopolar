import unittest
import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder


class TestTurboPolarOffline(unittest.TestCase):
    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128, qjl_proj_dim=64, block_size=64, split_dim=64,
            num_q_heads=4, num_kv_heads=4, seed=42,
        )
        self.encoder = PolarQuantEncoder(self.config)
        self.decoder = PolarQuantDecoder()

    def test_offline_reconstruction_gate(self):
        B, H, S, L, D = 1, 4, 2, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        k_blocked = k_original.reshape(B, H, S, L, D)
        blocks = [self.encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]

        # Verify bit-packed shapes
        self.assertEqual(blocks[0].radii.shape, (B, H, L, D // 2))
        l1_packed_len = (D // 4 + 1) // 2  # ceil(split_half/2) = ceil(32/2) = 16
        deep_packed_len = (D // 4 + 3) // 4  # ceil(32/4) = 8
        self.assertEqual(blocks[0].angle_codes_l1.shape, (B, H, L, l1_packed_len))
        self.assertEqual(blocks[0].angle_codes_deep.shape, (B, H, L, deep_packed_len))

        radii = mx.stack([b.radii for b in blocks], axis=2)
        angle_l1 = mx.stack([b.angle_codes_l1 for b in blocks], axis=2)
        angle_deep = mx.stack([b.angle_codes_deep for b in blocks], axis=2)

        unified = blocks[0].__class__(
            radii=radii, angle_codes_l1=angle_l1, angle_codes_deep=angle_deep,
            shape=(B, H, S * L, D), block_size=L, head_dim=D, metadata=blocks[0].metadata
        )

        self.assertEqual(unified.radii.shape, (B, H, S, L, D // 2))
        self.assertEqual(unified.angle_codes_l1.shape, (B, H, S, L, l1_packed_len))

        k_recon = self.decoder.decode_block(unified)
        mx.eval(k_original, k_recon)
        self.assertEqual(k_recon.shape, (B, H, S * L, D))

        orig_np = np.array(k_original)
        recon_np = np.array(k_recon)

        cosine = np.dot(orig_np.flatten(), recon_np.flatten()) / (
            np.linalg.norm(orig_np) * np.linalg.norm(recon_np) + 1e-12
        )
        mse = np.mean((orig_np - recon_np) ** 2)

        # Default split_dim=64 uses 4-bit angles for the first half of pairs and
        # 2-bit angles for the second half, so reconstruction is lossy.
        self.assertGreaterEqual(cosine, 0.90)
        self.assertLessEqual(mse, 0.5)

        q = mx.random.normal(shape=[B, H, D])
        scores_ref = mx.sum(q[:, :, None, :] * k_original, axis=-1)
        scores_recon = mx.sum(q[:, :, None, :] * k_recon, axis=-1)
        mx.eval(scores_ref, scores_recon)

        score_cosine = np.dot(np.array(scores_ref).flatten(), np.array(scores_recon).flatten()) / (
            np.linalg.norm(np.array(scores_ref)) * np.linalg.norm(np.array(scores_recon)) + 1e-12
        )
        self.assertGreaterEqual(score_cosine, 0.92)

    def test_bit_packing_roundtrip(self):
        B, H, S, L, D = 1, 2, 1, 64, 128
        mx.random.seed(123)
        k = mx.random.normal(shape=[B, H, S * L, D])
        k_blocked = k.reshape(B, H, S, L, D)
        block = self.encoder.encode_block(k_blocked[:, :, 0, :, :])

        # Verify metadata marks packed
        self.assertTrue(block.metadata.get("l1_packed"))
        self.assertTrue(block.metadata.get("deep_packed"))

        # Decode and compare
        unified = block.__class__(
            radii=mx.expand_dims(block.radii, axis=2),
            angle_codes_l1=mx.expand_dims(block.angle_codes_l1, axis=2),
            angle_codes_deep=mx.expand_dims(block.angle_codes_deep, axis=2),
            shape=(B, H, L, D), block_size=L, head_dim=D, metadata=block.metadata
        )
        k_recon = self.decoder.decode_block(unified)
        mx.eval(k, k_recon)

        max_err = float(mx.max(mx.abs(k.reshape(B, H, L, D) - k_recon[:, :, :L, :])))
        self.assertLess(max_err, 4.0)  # Loose bound for quantized reconstruction


if __name__ == "__main__":
    unittest.main()

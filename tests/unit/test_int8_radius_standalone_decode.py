"""Tests that 4D single-block decoding preserves int8 radius scales."""

import unittest

import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder


class TestInt8RadiusStandaloneDecode(unittest.TestCase):
    """A single-block 4D payload must decode identically to a unified 5D payload."""

    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            split_dim=0,
            num_q_heads=4,
            num_kv_heads=4,
            use_int8_radii=True,
            k_angle_bits_deep=8,
        )
        self.encoder = PolarQuantEncoder(self.config)
        self.decoder = PolarQuantDecoder()

    def test_4d_decode_preserves_radii_scales(self):
        B, H, L, D = 1, 4, 64, 128
        mx.random.seed(42)
        k = mx.random.normal((B, H, L, D))
        block = self.encoder.encode_block(k)

        self.assertEqual(block.radii.dtype, mx.int8)
        self.assertIsNotNone(block.radii_scales)

        # Decode the 4D single-block payload directly.
        decoded_4d = self.decoder.decode_block(block)
        self.assertTrue(mx.isfinite(decoded_4d).all().item())
        self.assertEqual(decoded_4d.shape, (B, H, L, D))

        # Decode the same block as a unified 5D payload.
        unified = block.__class__(
            radii=mx.expand_dims(block.radii, axis=2),
            angle_codes_l1=mx.expand_dims(block.angle_codes_l1, axis=2),
            angle_codes_deep=mx.expand_dims(block.angle_codes_deep, axis=2),
            radii_scales=mx.expand_dims(block.radii_scales, axis=2),
            shape=(B, H, 1, L, D),
            block_size=L,
            head_dim=D,
            metadata=block.metadata,
        )
        decoded_5d = self.decoder.decode_block(unified).reshape(B, H, L, D)

        np.testing.assert_allclose(
            np.array(decoded_4d), np.array(decoded_5d), rtol=1e-4, atol=1e-4
        )

        # Reconstruction quality should be reasonable.
        cosine = float(
            mx.sum(decoded_4d * k)
            / (mx.sqrt(mx.sum(decoded_4d**2)) * mx.sqrt(mx.sum(k**2)))
        )
        self.assertGreaterEqual(cosine, 0.99)
        mae = float(mx.mean(mx.abs(decoded_4d - k)))
        self.assertLess(mae, 0.1)


if __name__ == "__main__":
    unittest.main()

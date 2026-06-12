"""Tests for the supported configuration contract."""

import unittest

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig


class TestSupportedConfigContract(unittest.TestCase):
    """Unsupported configurations must raise clear errors."""

    def test_supported_config_constructs(self):
        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=8,
            num_kv_heads=4,
            use_qjl=False,
            storage_mode="kv_quant",
        )
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.block_size, 64)
        self.assertFalse(cfg.use_qjl)

    def test_head_dim_64_rejected(self):
        with self.assertRaisesRegex(ValueError, "head_dim == 128"):
            TurboPolarConfig(head_dim=64, block_size=64, num_q_heads=4, num_kv_heads=4)

    def test_head_dim_256_rejected(self):
        with self.assertRaisesRegex(ValueError, "head_dim == 128"):
            TurboPolarConfig(head_dim=256, block_size=64, num_q_heads=4, num_kv_heads=4)

    def test_block_size_not_64_rejected(self):
        with self.assertRaisesRegex(ValueError, "block_size == 64"):
            TurboPolarConfig(head_dim=128, block_size=32, num_q_heads=4, num_kv_heads=4)

    def test_qjl_rejected(self):
        with self.assertRaisesRegex(ValueError, "use_qjl=False"):
            TurboPolarConfig(
                head_dim=128, block_size=64, num_q_heads=4, num_kv_heads=4, use_qjl=True
            )

    def test_non_kv_quant_storage_rejected(self):
        with self.assertRaisesRegex(ValueError, "storage_mode='kv_quant'"):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                storage_mode="dense_v_debug",
            )

    def test_v_bits_not_8_rejected(self):
        with self.assertRaisesRegex(ValueError, "v_bits must be 8"):
            TurboPolarConfig(
                head_dim=128, block_size=64, num_q_heads=4, num_kv_heads=4, v_bits=4
            )

    def test_gqa_ratio_must_divide(self):
        with self.assertRaisesRegex(ValueError, "num_q_heads must be divisible"):
            TurboPolarConfig(
                head_dim=128, block_size=64, num_q_heads=7, num_kv_heads=4
            )


if __name__ == "__main__":
    unittest.main()

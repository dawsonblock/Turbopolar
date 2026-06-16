"""Tests for the supported configuration contract."""

import unittest

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig, validate_supported_configuration


class TestSupportedConfigContract(unittest.TestCase):
    """Unsupported configurations must raise clear errors."""

    def test_supported_config_constructs(self):
        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            page_capacity_blocks=16,
            num_q_heads=8,
            num_kv_heads=4,
            use_qjl=False,
            storage_mode="kv_quant",
        )
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.block_size, 64)
        self.assertEqual(cfg.page_capacity_blocks, 16)
        self.assertFalse(cfg.use_qjl)
        self.assertFalse(cfg.validate_finite_inputs)
        self.assertEqual(cfg.finite_audit_interval, 0)

    def test_head_dim_64_accepted(self):
        cfg = TurboPolarConfig(
            head_dim=64, block_size=64, num_q_heads=4, num_kv_heads=4
        )
        self.assertEqual(cfg.head_dim, 64)

    def test_head_dim_256_rejected(self):
        with self.assertRaisesRegex(ValueError, "head_dim=64 or 128"):
            TurboPolarConfig(head_dim=256, block_size=64, num_q_heads=4, num_kv_heads=4)

    def test_block_size_not_64_rejected(self):
        with self.assertRaisesRegex(ValueError, "block_size=64"):
            TurboPolarConfig(head_dim=128, block_size=32, num_q_heads=4, num_kv_heads=4)

    def test_qjl_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "QJL is disabled"):
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
            TurboPolarConfig(head_dim=128, block_size=64, num_q_heads=7, num_kv_heads=4)

    def test_zero_q_heads_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_q_heads must be positive"):
            TurboPolarConfig(head_dim=128, block_size=64, num_q_heads=0, num_kv_heads=4)

    def test_zero_kv_heads_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_kv_heads must be positive"):
            TurboPolarConfig(head_dim=128, block_size=64, num_q_heads=4, num_kv_heads=0)

    def test_non_positive_attention_scale_rejected(self):
        with self.assertRaisesRegex(ValueError, "attention_scale must be positive"):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                attention_scale=-0.1,
            )

    def test_unsupported_angle_bits_rejected(self):
        with self.assertRaisesRegex(ValueError, "k_angle_bits_level1 must be 4 or 8"):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                k_angle_bits_level1=2,
            )
        with self.assertRaisesRegex(ValueError, "k_angle_bits_deep must be 2, 4, or 8"):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                k_angle_bits_deep=6,
            )

    def test_finite_audit_interval_non_negative(self):
        with self.assertRaisesRegex(
            ValueError, "finite_audit_interval must be non-negative"
        ):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                finite_audit_interval=-1,
            )

    def test_string_execution_mode_normalized_to_enum(self):
        from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode

        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            execution_mode="metal_strict",
        )
        self.assertIsInstance(cfg.execution_mode, ExecutionMode)
        self.assertIs(cfg.execution_mode, ExecutionMode.METAL_STRICT)

    def test_enum_execution_mode_preserved(self):
        from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode

        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            execution_mode=ExecutionMode.METAL_STRICT,
        )
        self.assertIs(cfg.execution_mode, ExecutionMode.METAL_STRICT)

    def test_invalid_execution_mode_type_rejected(self):
        with self.assertRaisesRegex(TypeError, "execution_mode must be an ExecutionMode"):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                execution_mode=12345,
            )

    def test_invalid_execution_mode_string_rejected(self):
        with self.assertRaisesRegex(ValueError, "'not_a_mode' is not a valid ExecutionMode"):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                execution_mode="not_a_mode",
            )

    def test_page_capacity_not_16_rejected(self):
        with self.assertRaisesRegex(ValueError, "page_capacity_blocks=16"):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                page_capacity_blocks=8,
            )

    def test_trace_validation_mode_string_normalized(self):
        from rfsn_v11.kernels.turbo_polar.execution import TraceValidationMode

        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=4,
            num_kv_heads=4,
            trace_validation_mode="synchronous_evidence",
        )
        self.assertIsInstance(cfg.trace_validation_mode, TraceValidationMode)
        self.assertIs(cfg.trace_validation_mode, TraceValidationMode.SYNCHRONOUS_EVIDENCE)

    def test_invalid_trace_validation_mode_rejected(self):
        with self.assertRaisesRegex(TypeError, "trace_validation_mode must be a TraceValidationMode"):
            TurboPolarConfig(
                head_dim=128,
                block_size=64,
                num_q_heads=4,
                num_kv_heads=4,
                trace_validation_mode=12345,
            )

    def test_default_config_has_8bit_angles(self):
        """Default configuration must use 8/8-bit angles for supported mode."""
        cfg = TurboPolarConfig()
        self.assertEqual(cfg.k_angle_bits_level1, 8)
        self.assertEqual(cfg.k_angle_bits_deep, 8)

    def test_validate_supported_configuration_accepts_8bit_angles(self):
        """Supported configuration validator must accept 8/8-bit angles."""
        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            page_capacity_blocks=16,
            num_q_heads=8,
            num_kv_heads=4,
            k_angle_bits_level1=8,
            k_angle_bits_deep=8,
            use_qjl=False,
            storage_mode="kv_quant",
        )
        # Should not raise
        validate_supported_configuration(cfg)

    def test_validate_supported_configuration_rejects_4bit_deep_angles(self):
        """Supported configuration validator must reject 4-bit deep angles."""
        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            page_capacity_blocks=16,
            num_q_heads=8,
            num_kv_heads=4,
            k_angle_bits_level1=8,
            k_angle_bits_deep=4,  # Not supported for promotion
            use_qjl=False,
            storage_mode="kv_quant",
        )
        with self.assertRaisesRegex(ValueError, "Supported configuration requires 8-bit deep angles"):
            validate_supported_configuration(cfg)

    def test_validate_supported_configuration_rejects_2bit_deep_angles(self):
        """Supported configuration validator must reject 2-bit deep angles."""
        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            page_capacity_blocks=16,
            num_q_heads=8,
            num_kv_heads=4,
            k_angle_bits_level1=8,
            k_angle_bits_deep=2,  # Not supported for promotion
            use_qjl=False,
            storage_mode="kv_quant",
        )
        with self.assertRaisesRegex(ValueError, "Supported configuration requires 8-bit deep angles"):
            validate_supported_configuration(cfg)

    def test_validate_supported_configuration_rejects_4bit_level1_angles(self):
        """Supported configuration validator must reject 4-bit level-1 angles."""
        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            page_capacity_blocks=16,
            num_q_heads=8,
            num_kv_heads=4,
            k_angle_bits_level1=4,  # Not supported for promotion
            k_angle_bits_deep=8,
            use_qjl=False,
            storage_mode="kv_quant",
        )
        with self.assertRaisesRegex(ValueError, "Supported configuration requires 8-bit level-1 angles"):
            validate_supported_configuration(cfg)

    def test_experimental_4bit_mode_constructs_but_fails_validation(self):
        """Experimental 4-bit mode can construct but fails supported config validation."""
        cfg = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            page_capacity_blocks=16,
            num_q_heads=8,
            num_kv_heads=4,
            k_angle_bits_level1=8,
            k_angle_bits_deep=4,
            use_qjl=False,
            storage_mode="kv_quant",
        )
        # Construction should succeed (experimental mode)
        self.assertEqual(cfg.k_angle_bits_deep, 4)
        # But validation should fail
        with self.assertRaisesRegex(ValueError, "Supported configuration requires 8-bit deep angles"):
            validate_supported_configuration(cfg)


if __name__ == "__main__":
    unittest.main()

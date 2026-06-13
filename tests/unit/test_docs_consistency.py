"""Documentation consistency tests.

Verify that README, STATUS, and code agree on promotion state, supported
configuration, and required contexts.
"""

import ast
import pathlib

from rfsn_v11.candidates.turbo_polar_config import validate_supported_configuration, TurboPolarConfig
from rfsn_v11.promotion.gate import PromotionGate


class TestDocsConsistency:
    def test_promotion_lock_value_matches_status(self):
        """PROMOTION_LOCKED in gate.py must be True."""
        assert PromotionGate.PROMOTION_LOCKED is True

    def test_supported_configuration_matches_config_validation(self):
        """validate_supported_configuration must reject the same scope docs claim."""
        from dataclasses import replace

        # Valid narrow config should pass.
        valid = TurboPolarConfig(
            head_dim=128,
            block_size=64,
            num_q_heads=32,
            num_kv_heads=8,
            storage_mode="kv_quant",
            use_qjl=False,
        )
        validate_supported_configuration(valid)  # should not raise

        # Invalid configs must raise.
        import pytest
        with pytest.raises(ValueError):
            validate_supported_configuration(replace(valid, head_dim=64))
        with pytest.raises(ValueError):
            validate_supported_configuration(replace(valid, block_size=32))
        with pytest.raises(NotImplementedError):
            validate_supported_configuration(replace(valid, use_qjl=True))
        with pytest.raises(ValueError):
            validate_supported_configuration(replace(valid, storage_mode="fp16"))

    def test_required_contexts_match_promotion_constants(self):
        """REQUIRED_CONTEXTS in gate.py must match documented requirements."""
        assert PromotionGate.REQUIRED_CONTEXTS == {512, 2048, 4096, 8192, 16384}

    def test_readme_does_not_hardcode_test_count(self):
        """README must not claim a specific test count; it should reference generation."""
        readme_path = pathlib.Path(__file__).parents[2] / "README.md"
        readme_text = readme_path.read_text()
        # Reject hardcoded counts like "~208" or "208 tests".
        assert "~208" not in readme_text, "README hardcodes test count ~208"
        assert "208 test" not in readme_text, "README hardcodes test count 208"
        # Should reference generation instead.
        assert "test report" in readme_text.lower() or "generate" in readme_text.lower()

    def test_readme_does_not_claim_production_ready(self):
        """README must not claim production readiness."""
        readme_path = pathlib.Path(__file__).parents[2] / "README.md"
        readme_text = readme_path.read_text().lower()
        assert "production ready" not in readme_text
        assert "production-ready" not in readme_text

    def test_status_does_not_claim_full_metal_attention(self):
        """STATUS must not claim single fused kernel or full Metal attention."""
        status_path = pathlib.Path(__file__).parents[2] / "STATUS.md"
        status_text = status_path.read_text().lower()
        assert "single fused kernel" not in status_text
        assert "full metal attention" not in status_text

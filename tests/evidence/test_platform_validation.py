"""Tests for platform validation."""

import unittest

from rfsn_v11.evidence.platform_validation import (
    validate_apple_silicon_platform,
    validate_current_platform_is_apple_silicon,
    validate_metal_execution_mode,
)
from rfsn_v11.evidence.provenance import ProvenanceEvidence


class TestPlatformValidation(unittest.TestCase):
    """Test platform validation logic."""

    def test_validate_apple_silicon_complete(self):
        """Complete Apple Silicon provenance should pass."""
        provenance = ProvenanceEvidence(
            macos_version="14.0",
            chip_model="M2",
            metal_kernel_source_hash="a" * 64,
            kernel_binding_hash="b" * 64,
            execution_mode="metal_strict",
        )
        errors = validate_apple_silicon_platform(provenance)
        self.assertEqual(errors, [])

    def test_validate_missing_macos_version(self):
        """Missing macOS version should fail."""
        provenance = ProvenanceEvidence(
            chip_model="M2",
            metal_kernel_source_hash="a" * 64,
            kernel_binding_hash="b" * 64,
        )
        errors = validate_apple_silicon_platform(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("macOS" in errors[0])

    def test_validate_missing_chip_model(self):
        """Missing chip model should fail."""
        provenance = ProvenanceEvidence(
            macos_version="14.0",
            metal_kernel_source_hash="a" * 64,
            kernel_binding_hash="b" * 64,
        )
        errors = validate_apple_silicon_platform(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Chip model" in errors[0])

    def test_validate_invalid_chip_model(self):
        """Invalid chip model should fail."""
        provenance = ProvenanceEvidence(
            macos_version="14.0",
            chip_model="Intel i7",  # Not Apple Silicon
            metal_kernel_source_hash="a" * 64,
            kernel_binding_hash="b" * 64,
        )
        errors = validate_apple_silicon_platform(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Apple Silicon" in errors[0])

    def test_validate_valid_chip_models(self):
        """All valid Apple Silicon chip models should pass."""
        valid_chips = [
            "M1", "M1 Pro", "M1 Max", "M1 Ultra",
            "M2", "M2 Pro", "M2 Max", "M2 Ultra",
            "M3", "M3 Pro", "M3 Max", "M3 Ultra",
            "M4", "M4 Pro", "M4 Max", "M4 Ultra",
            "Apple M1", "Apple M2 Pro", "Apple M4 Max",
        ]
        for chip in valid_chips:
            provenance = ProvenanceEvidence(
                macos_version="14.0",
                chip_model=chip,
                metal_kernel_source_hash="a" * 64,
                kernel_binding_hash="b" * 64,
            )
            errors = validate_apple_silicon_platform(provenance)
            self.assertEqual(errors, [], f"Chip {chip} should be valid")

    def test_validate_missing_metal_hash(self):
        """Missing Metal kernel source hash should fail."""
        provenance = ProvenanceEvidence(
            macos_version="14.0",
            chip_model="M2",
            kernel_binding_hash="b" * 64,
        )
        errors = validate_apple_silicon_platform(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Metal kernel" in errors[0])

    def test_validate_missing_kernel_binding_hash(self):
        """Missing kernel binding hash should fail."""
        provenance = ProvenanceEvidence(
            macos_version="14.0",
            chip_model="M2",
            metal_kernel_source_hash="a" * 64,
        )
        errors = validate_apple_silicon_platform(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Kernel binding" in errors[0])

    def test_validate_metal_execution_mode_strict(self):
        """Metal strict execution mode should pass."""
        provenance = ProvenanceEvidence(
            execution_mode="metal_strict",
            chip_model="M2",
        )
        errors = validate_metal_execution_mode(provenance)
        self.assertEqual(errors, [])

    def test_validate_metal_execution_mode_non_strict_on_apple_silicon(self):
        """Non-strict mode on Apple Silicon should fail."""
        provenance = ProvenanceEvidence(
            execution_mode="development_auto",
            chip_model="M2",
        )
        errors = validate_metal_execution_mode(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("metal_strict" in errors[0])

    def test_validate_invalid_execution_mode(self):
        """Invalid execution mode should fail."""
        provenance = ProvenanceEvidence(
            execution_mode="invalid_mode",
        )
        errors = validate_metal_execution_mode(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("invalid" in errors[0])

    def test_validate_missing_execution_mode(self):
        """Missing execution mode should fail."""
        provenance = ProvenanceEvidence()
        errors = validate_metal_execution_mode(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Execution mode" in errors[0])


if __name__ == "__main__":
    unittest.main()
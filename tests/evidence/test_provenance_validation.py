"""Tests for provenance validation."""

import unittest

from rfsn_v11.evidence.provenance import (
    ProvenanceEvidence,
    validate_provenance_immutable_fields,
    validate_provenance_evidence_kind,
)


class TestProvenanceValidation(unittest.TestCase):
    """Test provenance validation logic."""

    def test_validate_complete_provenance(self):
        """Complete provenance should pass validation."""
        provenance = ProvenanceEvidence(
            run_id="test_run_123",
            git_commit="a" * 64,
            python_version="3.12.0",
            mlx_version="0.31.2",
            model_repo_id="mlx-community/Llama-3.2-1B-Instruct",
            model_revision="abc123",
            turbopolar_config_hash="d" * 64,
            execution_mode="metal_strict",
        )
        errors = validate_provenance_immutable_fields(provenance)
        self.assertEqual(errors, [])

    def test_validate_missing_required_fields(self):
        """Missing required fields should fail validation."""
        provenance = ProvenanceEvidence()
        errors = validate_provenance_immutable_fields(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("run_id" in errors[0])

    def test_validate_invalid_hash_length(self):
        """Hash fields with wrong length should fail."""
        provenance = ProvenanceEvidence(
            run_id="test_run_123",
            git_commit="abc",  # Too short
            python_version="3.12.0",
            mlx_version="0.31.2",
            model_repo_id="mlx-community/Llama-3.2-1B-Instruct",
            model_revision="abc123",
            turbopolar_config_hash="d" * 64,
            execution_mode="metal_strict",
        )
        errors = validate_provenance_immutable_fields(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("git_commit" in errors[0])

    def test_validate_invalid_tree_state(self):
        """Invalid git tree state should fail."""
        provenance = ProvenanceEvidence(
            run_id="test_run_123",
            git_commit="a" * 64,
            git_tree_state="INVALID_STATE",
            python_version="3.12.0",
            mlx_version="0.31.2",
            model_repo_id="mlx-community/Llama-3.2-1B-Instruct",
            model_revision="abc123",
            turbopolar_config_hash="d" * 64,
            execution_mode="metal_strict",
        )
        errors = validate_provenance_immutable_fields(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("git_tree_state" in errors[0])

    def test_validate_dirty_tree_without_diff_hash(self):
        """Dirty tree without diff hash should fail."""
        provenance = ProvenanceEvidence(
            run_id="test_run_123",
            git_commit="a" * 64,
            git_tree_state="DIRTY",
            git_diff_hash="",  # Missing
            python_version="3.12.0",
            mlx_version="0.31.2",
            model_repo_id="mlx-community/Llama-3.2-1B-Instruct",
            model_revision="abc123",
            turbopolar_config_hash="d" * 64,
            execution_mode="metal_strict",
        )
        errors = validate_provenance_immutable_fields(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("git_diff_hash" in errors[0])

    def test_validate_config_hash_mismatch(self):
        """Config hash that doesn't match computed hash should fail."""
        import hashlib
        import json
        
        config = {"key": "value"}
        config_json = json.dumps(config, sort_keys=True)
        computed_hash = hashlib.sha256(config_json.encode()).hexdigest()
        
        provenance = ProvenanceEvidence(
            run_id="test_run_123",
            git_commit="a" * 64,
            python_version="3.12.0",
            mlx_version="0.31.2",
            model_repo_id="mlx-community/Llama-3.2-1B-Instruct",
            model_revision="abc123",
            turbopolar_config_hash="wrong_hash_" + "0" * 52,  # Wrong hash
            turbopolar_config=config,
            execution_mode="metal_strict",
        )
        errors = validate_provenance_immutable_fields(provenance)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("config_hash" in errors[0])

    def test_validate_config_hash_match(self):
        """Config hash that matches computed hash should pass."""
        import hashlib
        import json
        
        config = {"key": "value"}
        config_json = json.dumps(config, sort_keys=True)
        computed_hash = hashlib.sha256(config_json.encode()).hexdigest()
        
        provenance = ProvenanceEvidence(
            run_id="test_run_123",
            git_commit="a" * 64,
            python_version="3.12.0",
            mlx_version="0.31.2",
            model_repo_id="mlx-community/Llama-3.2-1B-Instruct",
            model_revision="abc123",
            turbopolar_config_hash=computed_hash,
            turbopolar_config=config,
            execution_mode="metal_strict",
        )
        errors = validate_provenance_immutable_fields(provenance)
        self.assertEqual(errors, [])

    def test_validate_evidence_kind_match(self):
        """Matching evidence kind should pass."""
        provenance = ProvenanceEvidence(
            run_id="test_run_123",
            git_commit="a" * 64,
            python_version="3.12.0",
            mlx_version="0.31.2",
            model_repo_id="mlx-community/Llama-3.2-1B-Instruct",
            model_revision="abc123",
            turbopolar_config_hash="d" * 64,
            execution_mode="metal_strict",
            notes=["evidence_kind: experimental"],
        )
        errors = validate_provenance_evidence_kind(provenance, "experimental")
        self.assertEqual(errors, [])

    def test_validate_evidence_kind_mismatch(self):
        """Mismatched evidence kind should fail."""
        provenance = ProvenanceEvidence(
            run_id="test_run_123",
            git_commit="a" * 64,
            python_version="3.12.0",
            mlx_version="0.31.2",
            model_repo_id="mlx-community/Llama-3.2-1B-Instruct",
            model_revision="abc123",
            turbopolar_config_hash="d" * 64,
            execution_mode="metal_strict",
            notes=["evidence_kind: synthetic_dry_run"],
        )
        errors = validate_provenance_evidence_kind(provenance, "experimental")
        self.assertTrue(len(errors) > 0)
        self.assertTrue("evidence kind" in errors[0].lower())


if __name__ == "__main__":
    unittest.main()
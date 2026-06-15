"""Tests for capture_provenance() function.

These tests verify that capture_provenance() produces valid provenance
that passes platform and execution validation.
"""

import unittest
from pathlib import Path
from unittest import mock

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.evidence.platform_validation import (
    validate_apple_silicon_platform,
    validate_metal_execution_mode,
)
from rfsn_v11.evidence.provenance import validate_provenance_immutable_fields
from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode
from rfsn_v11.promotion.provenance import capture_provenance


class TestCaptureProvenance(unittest.TestCase):
    """Test that capture_provenance() produces valid provenance."""

    def test_capture_provenance_populates_required_fields(self):
        """P0: capture_provenance() must populate kernel_binding_hash
        and execution_mode."""
        config = TurboPolarConfig(
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=128,
            block_size=64,
            execution_mode=ExecutionMode.METAL_STRICT,
        )

        mock_mlx = mock.MagicMock()
        mock_mlx.core = mock.MagicMock(__version__="0.18.0")
        mock_mlx_lm = mock.MagicMock(__version__="0.19.0")
        with (
            mock.patch(
                "rfsn_v11.promotion.provenance._file_sha256"
            ) as mock_file_hash,
            mock.patch(
                "rfsn_v11.promotion.provenance._dir_sha256"
            ) as mock_dir_hash,
            mock.patch(
                "rfsn_v11.promotion.provenance._run"
            ) as mock_run,
            mock.patch(
                "rfsn_v11.promotion.provenance.platform"
            ) as mock_platform,
            mock.patch(
                "rfsn_v11.promotion.provenance.uuid"
            ) as mock_uuid,
            mock.patch(
                "rfsn_v11.promotion.provenance.datetime"
            ) as mock_datetime,
            mock.patch.dict("sys.modules", {
                "mlx": mock_mlx,
                "mlx.core": mock_mlx.core,
                "mlx_lm": mock_mlx_lm,
            }),
        ):

            mock_platform.python_version.return_value = "3.12.0"
            mock_platform.mac_ver.return_value = (
                "14.0", ("", "", ""), ""
            )
            mock_platform.machine.return_value = "arm64"
            mock_platform.system.return_value = "Darwin"

            cmd_responses = {
                ("git", "rev-parse", "HEAD"): "a" * 40,
                ("git", "status", "--porcelain"): "",
                ("git", "rev-parse", "--show-toplevel"): "/test/repo",
                (
                    "sysctl", "-n",
                    "machdep.cpu.brand_string"
                ): "Apple M3 Max",
                (
                    "sysctl", "-n", "hw.memsize"
                ): "34359738368",
            }
            mock_run.side_effect = (
                lambda cmd: cmd_responses.get(tuple(cmd), "")
            )

            mock_file_hash.return_value = "f" * 64
            mock_dir_hash.return_value = "d" * 64

            mock_uuid.uuid4.return_value = "test-run-id-12345"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2026-01-01T00:00:00Z"
            )

            provenance = capture_provenance(
                model_repo_id="test/model",
                model_revision="abc123",
                tokenizer_revision="def456",
                turbopolar_config=config,
                prompt_suite_path=Path("/test/prompts.jsonl"),
                benchmark_command="python run_test.py",
                warmup_count=1,
                trial_count=5,
                context_lengths=[512, 2048, 4096],
                decode_token_count=128,
                qjl_enabled=False,
                evidence_kind="experimental",
            )

        self.assertTrue(
            provenance.kernel_binding_hash,
            "kernel_binding_hash must not be empty"
        )
        self.assertEqual(
            len(provenance.kernel_binding_hash), 64,
            "kernel_binding_hash must be 64 characters (full SHA-256)"
        )
        self.assertTrue(
            provenance.execution_mode,
            "execution_mode must not be empty"
        )
        self.assertEqual(
            provenance.execution_mode, "metal_strict",
            "execution_mode must match config"
        )

    def test_captured_provenance_passes_validation(self):
        """P0: Captured provenance must pass platform and execution
        validation."""
        config = TurboPolarConfig(
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=128,
            block_size=64,
            execution_mode=ExecutionMode.METAL_STRICT,
        )

        mock_mlx = mock.MagicMock()
        mock_mlx.core = mock.MagicMock(__version__="0.18.0")
        mock_mlx_lm = mock.MagicMock(__version__="0.19.0")
        with (
            mock.patch(
                "rfsn_v11.promotion.provenance._file_sha256"
            ) as mock_file_hash,
            mock.patch(
                "rfsn_v11.promotion.provenance._dir_sha256"
            ) as mock_dir_hash,
            mock.patch(
                "rfsn_v11.promotion.provenance._run"
            ) as mock_run,
            mock.patch(
                "rfsn_v11.promotion.provenance.platform"
            ) as mock_platform,
            mock.patch(
                "rfsn_v11.promotion.provenance.uuid"
            ) as mock_uuid,
            mock.patch(
                "rfsn_v11.promotion.provenance.datetime"
            ) as mock_datetime,
            mock.patch.dict("sys.modules", {
                "mlx": mock_mlx,
                "mlx.core": mock_mlx.core,
                "mlx_lm": mock_mlx_lm,
            }),
        ):

            mock_platform.python_version.return_value = "3.12.0"
            mock_platform.mac_ver.return_value = (
                "14.0", ("", "", ""), ""
            )
            mock_platform.machine.return_value = "arm64"
            mock_platform.system.return_value = "Darwin"

            cmd_responses = {
                ("git", "rev-parse", "HEAD"): "a" * 40,
                ("git", "status", "--porcelain"): "",  # Clean tree
                ("git", "rev-parse", "--show-toplevel"): "/test/repo",
                (
                    "sysctl", "-n",
                    "machdep.cpu.brand_string"
                ): "Apple M3 Max",
                (
                    "sysctl", "-n", "hw.memsize"
                ): "34359738368",
            }
            mock_run.side_effect = (
                lambda cmd: cmd_responses.get(tuple(cmd), "")
            )

            mock_file_hash.return_value = "f" * 64
            mock_dir_hash.return_value = "d" * 64

            mock_uuid.uuid4.return_value = "test-run-id-12345"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2026-01-01T00:00:00Z"
            )

            provenance = capture_provenance(
                model_repo_id="test/model",
                model_revision="abc123",
                tokenizer_revision="def456",
                turbopolar_config=config,
                prompt_suite_path=Path("/test/prompts.jsonl"),
                benchmark_command="python run_test.py",
                warmup_count=1,
                trial_count=5,
                context_lengths=[512, 2048, 4096],
                decode_token_count=128,
                qjl_enabled=False,
                evidence_kind="experimental",
            )

        from rfsn_v11.evidence.provenance import ProvenanceEvidence

        pv = ProvenanceEvidence(
            run_id=provenance.run_id,
            timestamp_utc=provenance.timestamp_utc,
            git_commit=provenance.git_commit,
            git_tree_state=provenance.git_tree_state.value,
            git_diff_hash=provenance.git_diff_hash,
            python_version=provenance.python_version,
            mlx_version=provenance.mlx_version,
            mlx_lm_version=provenance.mlx_lm_version,
            macos_version=provenance.macos_version,
            chip_model=provenance.chip_model,
            system_memory_gb=provenance.system_memory_gb,
            model_repo_id=provenance.model_repo_id,
            model_revision=provenance.model_revision,
            tokenizer_repo_id=provenance.model_repo_id,
            tokenizer_revision=provenance.tokenizer_revision,
            prompt_suite_hash=provenance.prompt_suite_hash,
            continuation_hashes={},
            context_hashes={},
            turbopolar_config_hash=provenance.turbopolar_config_hash,
            turbopolar_config=provenance.turbopolar_config,
            execution_mode=provenance.execution_mode,
            trace_validation_mode=provenance.turbopolar_config.get(
                "trace_validation_mode", ""
            ),
            page_capacity=provenance.turbopolar_config.get(
                "page_capacity_blocks", 16
            ),
            block_size=provenance.turbopolar_config.get(
                "block_size", 64
            ),
            k_bit_widths=(
                f"{provenance.turbopolar_config.get('k_angle_bits_level1', 8)}"
                f"/{provenance.turbopolar_config.get('k_angle_bits_deep', 8)}"
            ),
            v_bit_width=provenance.turbopolar_config.get("v_bits", 8),
            v_group_size=32,
            num_q_heads=provenance.turbopolar_config.get(
                "num_q_heads", 8
            ),
            num_kv_heads=provenance.turbopolar_config.get(
                "num_kv_heads", 8
            ),
            head_dim=provenance.turbopolar_config.get("head_dim", 64),
            attention_scale=provenance.turbopolar_config.get(
                "attention_scale", 1.0
            ),
            benchmark_command=provenance.benchmark_command,
            warmup_count=provenance.warmup_count,
            trial_count=provenance.trial_count,
            context_lengths=provenance.context_lengths,
            decode_token_count=provenance.decode_token_count,
            qjl_enabled=provenance.qjl_enabled,
            metal_kernel_source_hash=provenance.metal_kernel_source_hash,
            kernel_binding_hash=provenance.kernel_binding_hash,
        )

        immutable_errors = validate_provenance_immutable_fields(pv)
        self.assertEqual(
            immutable_errors, [],
            f"Provenance failed immutable field validation: "
            f"{immutable_errors}"
        )

        platform_errors = validate_apple_silicon_platform(pv)
        self.assertEqual(
            platform_errors, [],
            f"Provenance failed platform validation: {platform_errors}"
        )

        execution_errors = validate_metal_execution_mode(pv)
        self.assertEqual(
            execution_errors, [],
            f"Provenance failed execution mode validation: "
            f"{execution_errors}"
        )

    def test_captured_provenance_round_trips(self):
        """P0: Captured provenance must round-trip through serialization
        without data loss."""
        from dataclasses import asdict

        config = TurboPolarConfig(
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=128,
            block_size=64,
            execution_mode=ExecutionMode.METAL_STRICT,
        )

        with (
            mock.patch(
                "rfsn_v11.promotion.provenance._file_sha256"
            ) as mock_file_hash,
            mock.patch(
                "rfsn_v11.promotion.provenance._dir_sha256"
            ) as mock_dir_hash,
            mock.patch(
                "rfsn_v11.promotion.provenance._run"
            ) as mock_run,
            mock.patch(
                "rfsn_v11.promotion.provenance.platform"
            ) as mock_platform,
            mock.patch(
                "rfsn_v11.promotion.provenance.uuid"
            ) as mock_uuid,
            mock.patch(
                "rfsn_v11.promotion.provenance.datetime"
            ) as mock_datetime,
        ):

            mock_platform.python_version.return_value = "3.12.0"
            mock_platform.mac_ver.return_value = (
                "14.0", ("", "", ""), ""
            )
            mock_platform.machine.return_value = "arm64"
            mock_platform.system.return_value = "Darwin"

            cmd_responses = {
                ("git", "rev-parse", "HEAD"): "a" * 40,
                ("git", "status", "--porcelain"): "",
                ("git", "rev-parse", "--show-toplevel"): "/test/repo",
                (
                    "sysctl", "-n",
                    "machdep.cpu.brand_string"
                ): "Apple M3 Max",
                (
                    "sysctl", "-n", "hw.memsize"
                ): "34359738368",
            }
            mock_run.side_effect = (
                lambda cmd: cmd_responses.get(tuple(cmd), "")
            )

            mock_file_hash.return_value = "f" * 64
            mock_dir_hash.return_value = "d" * 64
            mock_uuid.uuid4.return_value = "test-run-id-12345"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2026-01-01T00:00:00Z"
            )

            provenance = capture_provenance(
                model_repo_id="test/model",
                model_revision="abc123",
                tokenizer_revision="def456",
                turbopolar_config=config,
                prompt_suite_path=Path("/test/prompts.jsonl"),
                benchmark_command="python run_test.py",
                warmup_count=1,
                trial_count=5,
                context_lengths=[512, 2048, 4096],
                decode_token_count=128,
                qjl_enabled=False,
                evidence_kind="experimental",
            )

        from rfsn_v11.promotion.schema import BenchmarkProvenance

        d = asdict(provenance)
        restored = BenchmarkProvenance.from_dict(d)

        self.assertEqual(
            restored.kernel_binding_hash,
            provenance.kernel_binding_hash
        )
        self.assertEqual(
            restored.execution_mode, provenance.execution_mode
        )
        self.assertEqual(restored.run_id, provenance.run_id)
        self.assertEqual(restored.git_commit, provenance.git_commit)
        self.assertEqual(
            restored.turbopolar_config_hash,
            provenance.turbopolar_config_hash
        )


if __name__ == "__main__":
    unittest.main()

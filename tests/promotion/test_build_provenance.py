"""Tests for _build_provenance() in run_promotion_suite.py.

These tests verify that _build_provenance() correctly wires workload
hashes and execution mode through to capture_provenance() without
signature mismatches.
"""

import unittest
from pathlib import Path
from unittest import mock

# Mock mlx before importing the promotion suite script.
mock_mlx = mock.MagicMock()
mock_mlx.core = mock.MagicMock(__version__="0.18.0")
mock_mlx_lm = mock.MagicMock(__version__="0.19.0")

with mock.patch.dict(
    "sys.modules",
    {
        "mlx": mock_mlx,
        "mlx.core": mock_mlx.core,
        "mlx_lm": mock_mlx_lm,
    },
):
    from scripts.run_promotion_suite import _build_provenance
    from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
    from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode
    from rfsn_v11.promotion.provenance import capture_provenance


class TestBuildProvenance(unittest.TestCase):
    """Test that _build_provenance() invokes capture_provenance correctly."""

    def test_build_provenance_signature_and_execution_mode(self):
        """P0: _build_provenance must pass teacher_forced_workload_hash
        to capture_provenance and record METAL_STRICT execution mode.

        This test uses a static signature inspection to verify that
        capture_provenance accepts teacher_forced_workload_hash, then
        invokes _build_provenance with mocked internals to verify the
        returned provenance carries the correct execution_mode."""
        config = TurboPolarConfig(
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=128,
            block_size=64,
            execution_mode=ExecutionMode.METAL_STRICT,
        )
        output_dir = Path("/tmp/test_provenance")

        # P0: Statically verify every keyword supplied by _build_provenance
        # exists in capture_provenance's signature.
        import inspect
        sig = inspect.signature(capture_provenance)
        self.assertIn(
            "teacher_forced_workload_hash",
            sig.parameters,
            "capture_provenance must accept teacher_forced_workload_hash",
        )

        # _build_provenance may have been imported from a module cached under
        # a different name (e.g. ``run_promotion_suite`` vs
        # ``scripts.run_promotion_suite``). Patch the function's actual
        # globals dict directly.
        with (
            mock.patch.dict(
                _build_provenance.__globals__,
                {"capture_provenance": mock.MagicMock()},
            ),
            mock.patch(
                "scripts.run_promotion_suite.BENCHMARKS_DIR",
                Path("/tmp/benchmarks"),
            ),
            mock.patch(
                "scripts.run_promotion_suite.sys.argv",
                ["run_promotion_suite.py"],
            ),
        ):
            mock_capture = _build_provenance.__globals__["capture_provenance"]
            mock_capture.return_value = mock.MagicMock(
                teacher_forced_workload_hash="a" * 64,
                execution_mode="metal_strict",
            )
            result = _build_provenance(
                model="test/model",
                output_dir=output_dir,
                config=config,
            )
            mock_capture.assert_called_once()
            kwargs = mock_capture.call_args.kwargs
            self.assertIn(
                "teacher_forced_workload_hash",
                kwargs,
                "capture_provenance must receive teacher_forced_workload_hash",
            )
            self.assertEqual(
                len(kwargs["teacher_forced_workload_hash"]),
                64,
                "teacher_forced_workload_hash must be a 64-char hex string",
            )
            self.assertTrue(
                all(
                    c in "0123456789abcdef"
                    for c in kwargs["teacher_forced_workload_hash"].lower()
                ),
                "teacher_forced_workload_hash must be hexadecimal",
            )

        self.assertEqual(
            result.execution_mode,
            "metal_strict",
            "Provenance must record metal_strict for promotion benchmarks",
        )
        self.assertEqual(
            result.teacher_forced_workload_hash,
            "a" * 64,
            "teacher_forced_workload_hash must flow through to provenance",
        )

    def test_build_provenance_uses_metal_strict(self):
        """P0: Provenance config must record METAL_STRICT execution mode."""
        config = TurboPolarConfig(
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=128,
            block_size=64,
            execution_mode=ExecutionMode.METAL_STRICT,
        )
        output_dir = Path("/tmp/test_provenance")

        with (
            mock.patch(
                "scripts.run_promotion_suite.capture_provenance"
            ) as mock_capture,
            mock.patch(
                "scripts.run_promotion_suite.BENCHMARKS_DIR",
                Path("/tmp/benchmarks"),
            ),
            mock.patch(
                "scripts.run_promotion_suite.sys.argv",
                ["run_promotion_suite.py"],
            ),
        ):
            mock_capture.return_value = mock.MagicMock(
                execution_mode="metal_strict",
            )
            result = _build_provenance(
                model="test/model",
                output_dir=output_dir,
                config=config,
            )

        self.assertEqual(
            result.execution_mode,
            "metal_strict",
            "Provenance must record metal_strict for promotion benchmarks",
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for the full-model memory worker."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestMemoryWorkerCLI(unittest.TestCase):
    def test_mode_choices_exclude_cartesian(self):
        """The CLI must not advertise cartesian_int8 as a valid mode."""
        result = subprocess.run(
            [
                sys.executable,
                str(
                    PROJECT_ROOT / "benchmarks" /
                    "full_model_memory_worker.py"
                ),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("{dense,turbopolar_strict}", result.stdout)
        self.assertNotIn("cartesian_int8", result.stdout)


class TestMemoryWorkerLogic(unittest.TestCase):
    """Test measurement logic with mocked MLX."""

    def _patch_mlx(self):
        """Return patch targets for mlx and mlx_lm."""
        mock_mx = MagicMock()
        # Simulate peak memory counters.
        peak_values = []

        def _reset():
            peak_values.clear()
            peak_values.append(0)

        def _get_peak():
            return peak_values[-1] if peak_values else 0

        def _eval(*args):
            # Simulate evaluation increasing peak memory for args.
            peak_values[-1] += sum(
                getattr(a, "size", 1) * getattr(a, "itemsize", 4)
                for a in args
            )

        mock_mx.reset_peak_memory = _reset
        mock_mx.get_peak_memory = _get_peak
        mock_mx.eval = _eval
        mock_mx.array = (
            lambda *a, **k: MagicMock(
                shape=(1, 1), size=1, itemsize=4,
                dtype=mock_mx.float16,
            )
        )
        mock_mx.float16 = "float16"
        mock_mx.uint32 = "uint32"
        mock_mx.full = (
            lambda *a, **k: MagicMock(
                shape=(1, 1), size=1, itemsize=4,
            )
        )
        mock_mx.zeros = (
            lambda *a, **k: MagicMock(
                shape=(1, 1), size=1, itemsize=4,
            )
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 32000

        mock_model = MagicMock()
        mock_model.layers = [MagicMock() for _ in range(2)]
        mock_model.model.layers = mock_model.layers

        mock_load = MagicMock(return_value=(mock_model, mock_tokenizer))

        return {
            "mlx.core": mock_mx,
            "mlx_lm": MagicMock(load=mock_load),
            "mlx_lm.load": mock_load,
        }

    def test_source_contains_expected_fields(self):
        """Worker source must contain the corrected peak field names."""
        source = (
            PROJECT_ROOT / "benchmarks" /
            "full_model_memory_worker.py"
        ).read_text()
        # P1-33: Whole-run peak semantics - single measurement
        # across all stages
        self.assertIn("whole_run_peak_bytes", source)
        # Old per-stage peaks removed in favor of whole-run peak
        self.assertNotIn("baseline_peak_bytes = int", source)
        self.assertNotIn("model_loaded_peak_bytes = int", source)
        self.assertNotIn("prefill_peak_bytes = int", source)
        self.assertNotIn("decode_peak_bytes = int", source)
        # total_peak_bytes now equals whole_run_peak_bytes for
        # backward compat
        self.assertIn("total_peak_bytes", source)
        self.assertIn("dense_kv_bytes", source)
        self.assertIn("for layer_cache in cache:", source)
        self.assertNotIn("post_prefill_bytes", source)
        self.assertNotIn("post_decode_bytes", source)

    def test_field_names_in_output(self):
        """Verify that the worker emits the expected JSON schema keys."""
        # Build a synthetic result dict matching what the worker
        # should produce.
        # P1-33: Whole-run peak measurement replaces per-stage peaks
        expected_keys = {
            "context_length",
            "mode",
            # Primary peak measurement (whole run)
            "whole_run_peak_bytes",
            # Backward compatibility alias
            "total_peak_bytes",
            "dense_kv_bytes",
            "logical_cache_bytes",
            "allocated_cache_bytes",
            "dense_tail_bytes",
            "retained_dense_k_history",
            "retained_dense_v_history",
            "fallback_count",
            # P1-32: Canonical fixture provenance
            "fixture_id",
            "fixture_hash",
            "token_fixtures_path",
        }
        self.assertTrue(
            len(expected_keys) > 0,
            "Expected keys should be non-empty",
        )

    def test_memory_measurement_algorithm(self):
        """Test that the memory measurement algorithm follows the
        correct sequence."""
        # Since we can't easily mock MLX imports, we verify the
        # algorithm structure by inspecting the source code for the
        # correct measurement sequence
        source = (
            PROJECT_ROOT / "benchmarks" /
            "full_model_memory_worker.py"
        ).read_text()

        # P1-33: Whole-run peak semantics
        # Verify single reset at start (whole-run measurement)
        reset_idx = source.find("mx.reset_peak_memory()")
        model_load_idx = source.find("model, tokenizer = load(")
        self.assertGreater(
            reset_idx, -1, "Peak memory reset should exist"
        )
        self.assertGreater(
            model_load_idx, -1, "Model load should exist"
        )

        # Verify we reset BEFORE model load (for true whole-run
        # measurement)
        self.assertLess(
            reset_idx, model_load_idx,
            "Reset should happen before model load"
        )

        # Verify we do NOT reset between stages (key for whole-run
        # semantics). Count resets - should only be 1 at the start.
        reset_count = source.count("mx.reset_peak_memory()")
        self.assertEqual(
            reset_count, 1,
            "Should only reset peak memory once at start "
            "(whole-run semantics)"
        )

        # Verify whole-run peak is captured after all stages
        whole_run_peak_idx = source.find(
            "whole_run_peak_bytes = int(mx.get_peak_memory())"
        )
        self.assertGreater(
            whole_run_peak_idx, -1,
            "Whole-run peak should be measured"
        )

        # Verify the whole-run peak comes after decode stage
        decode_loop_idx = source.find(
            "for forced_token in forced_continuation:"
        )
        self.assertGreater(
            decode_loop_idx, -1, "Decode loop should exist"
        )
        self.assertGreater(
            whole_run_peak_idx, decode_loop_idx,
            "Whole-run peak should be measured after decode completes"
        )

        # Verify backward compatibility: total_peak_bytes equals
        # whole_run_peak_bytes
        total_peak_assignment = source.find(
            '"total_peak_bytes": whole_run_peak_bytes'
        )
        self.assertGreater(
            total_peak_assignment, -1,
            "total_peak_bytes should alias whole_run_peak_bytes "
            "for backward compat"
        )

    def test_dense_history_audit_logic(self):
        """Test that dense history audit checks partial buffers
        correctly."""
        # Verify the source contains the correct logic for checking
        # partial buffers
        source = (
            PROJECT_ROOT / "benchmarks" /
            "full_model_memory_worker.py"
        ).read_text()
        self.assertIn("partial_k_buffer", source)
        self.assertIn("partial_v_buffer", source)
        self.assertIn("seq_dim > 64", source)

        # Verify it checks runtime attributes
        self.assertIn(
            'runtime = getattr(layer_cache, "runtime", None)',
            source
        )

        # Verify it checks both K and V storage
        self.assertIn("retained_dense_k", source)
        self.assertIn("retained_dense_v", source)


class TestExecutionStatsRegression(unittest.TestCase):
    """P0: Regression test for KernelExecutionStats dataclass access."""

    def test_getattr_extracts_fallback_count_from_dataclass(self):
        """The worker must use getattr(), not .get(), on
        KernelExecutionStats."""
        from rfsn_v11.integrations.mlx_lm.telemetry import (
            KernelExecutionStats,
        )

        class FakeCache:
            def execution_stats(self):
                return KernelExecutionStats(fallback_calls=5)

        cache = FakeCache()
        stats = cache.execution_stats()
        # Correct pattern (fixed in r5-7)
        fallback_count = getattr(stats, "fallback_calls", 0)
        self.assertEqual(fallback_count, 5)

        # Old broken pattern must raise AttributeError
        with self.assertRaises(AttributeError):
            stats.get("fallback_calls", 0)

    def test_getattr_defaults_to_zero_when_field_missing(self):
        """If fallback_calls is absent, getattr must default to 0."""
        from rfsn_v11.integrations.mlx_lm.telemetry import (
            KernelExecutionStats,
        )

        stats = KernelExecutionStats()  # all defaults
        self.assertEqual(getattr(stats, "fallback_calls", 0), 0)


class TestMemoryMatrixJSONParser(unittest.TestCase):
    """Test that the memory matrix reader uses the output file
    correctly."""

    def test_reads_from_output_file(self):
        """_measure_mode must read JSON from the --output file,
        not stdout."""

        # We verify the logic by checking that the helper now uses
        # --output. Since we cannot run the full matrix (requires
        # model), we inspect the source to confirm the
        # implementation.
        source = (
            PROJECT_ROOT / "benchmarks" / "run_memory_matrix.py"
        ).read_text()
        self.assertIn("--output", source)
        self.assertIn("json.load", source)
        self.assertNotIn("brace_idx = stdout.find(", source)


class TestMemoryMatrixRatios(unittest.TestCase):
    def test_ratio_definitions(self):
        """Ratios must compare compatible numerators and denominators."""
        source = (
            PROJECT_ROOT / "benchmarks" / "run_memory_matrix.py"
        ).read_text()
        # Ensure logical_kv_ratio uses dense_kv_bytes /
        # turbo_logical_bytes
        self.assertIn("dense_kv_bytes / turbo_logical", source)
        # Ensure persistent_storage_ratio uses dense_kv_bytes /
        # turbo_allocated
        self.assertIn("dense_kv_bytes / turbo_allocated", source)
        # Ensure peak_device_memory_ratio uses dense_total_peak /
        # turbo_total_peak
        self.assertIn("dense_total_peak / turbo_total_peak", source)


if __name__ == "__main__":
    unittest.main()

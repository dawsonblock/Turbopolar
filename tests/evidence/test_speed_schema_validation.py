"""Tests for canonical speed schema validation."""

import unittest

from rfsn_v11.evidence.speed import (
    RawSpeedArtifact,
    RawSpeedTrial,
    REQUIRED_CONTEXTS,
    REQUIRED_TRIALS_PER_CONTEXT,
    REQUIRED_TOKEN_LATENCIES,
    validate_speed_trials,
)


class TestRawSpeedSchema(unittest.TestCase):
    """Test canonical speed schema serialization and validation."""

    def test_raw_speed_trial_creation(self):
        """RawSpeedTrial should create with all required fields."""
        trial = RawSpeedTrial(
            context_length=512,
            method="turbo",
            trial_index=1,
            execution_order=("512_turbo", "trial_1"),
            execution_mode="metal_strict",
            prefill_seconds=0.1,
            token_latencies_ms=(1.0, 2.0, 3.0),
            first_token_ms=1.5,
            throughput_tps=1000.0,
            compressed_page_dispatches=10,
            dense_tail_dispatches=1,
            fallback_calls=0,
        )
        self.assertEqual(trial.context_length, 512)
        self.assertEqual(trial.method, "turbo")
        self.assertEqual(trial.fallback_calls, 0)

    def test_raw_speed_artifact_creation(self):
        """RawSpeedArtifact should create with schema version and trials."""
        trials = (
            RawSpeedTrial(
                context_length=512,
                method="turbo",
                trial_index=1,
                execution_order=("512_turbo", "trial_1"),
                execution_mode="metal_strict",
                prefill_seconds=0.1,
                token_latencies_ms=tuple([1.0] * REQUIRED_TOKEN_LATENCIES),
                first_token_ms=1.5,
                throughput_tps=1000.0,
                compressed_page_dispatches=10,
                dense_tail_dispatches=1,
                fallback_calls=0,
            ),
        )
        artifact = RawSpeedArtifact(schema_version=1, trials=trials)
        self.assertEqual(artifact.schema_version, 1)
        self.assertEqual(len(artifact.trials), 1)

    def test_from_dict_converter(self):
        """RawSpeedArtifact should convert from dictionary format."""
        data = {
            "schema_version": 1,
            "trial_results": [
                {
                    "context_length": 512,
                    "mode": "turbo",
                    "trial": 1,
                    "execution_mode": "metal_strict",
                    "prefill_seconds": 0.1,
                    "per_token_ms": [1.0] * REQUIRED_TOKEN_LATENCIES,
                    "first_token_ms": 1.5,
                    "throughput_tps": 1000.0,
                    "page_dispatches": 10,
                    "tail_dispatches": 1,
                    "fallbacks": 0,
                }
            ]
        }
        artifact = RawSpeedArtifact.from_dict(data)
        self.assertEqual(artifact.schema_version, 1)
        self.assertEqual(len(artifact.trials), 1)
        self.assertEqual(artifact.trials[0].context_length, 512)
        self.assertEqual(artifact.trials[0].method, "turbo")

    def test_validate_passing_artifact(self):
        """Validation should pass for a complete valid artifact."""
        trials = []
        for context in REQUIRED_CONTEXTS:
            order = ("dense", "turbo")
            for mode in ["dense", "turbo"]:
                for trial_idx in range(REQUIRED_TRIALS_PER_CONTEXT):
                    trial = RawSpeedTrial(
                        context_length=context,
                        method=mode,
                        trial_index=trial_idx,
                        execution_order=order,
                        execution_mode="metal_strict",
                        prefill_seconds=0.1,
                        token_latencies_ms=tuple(
                            [1.0] * REQUIRED_TOKEN_LATENCIES
                        ),
                        first_token_ms=1.5,
                        throughput_tps=1000.0,
                        compressed_page_dispatches=(
                            10 if mode == "turbo" else 0
                        ),
                        dense_tail_dispatches=1 if mode == "turbo" else 0,
                        fallback_calls=0,
                    )
                    trials.append(trial)

        artifact = RawSpeedArtifact(schema_version=1, trials=tuple(trials))
        errors = validate_speed_trials(artifact)
        self.assertEqual(errors, [])

    def test_validate_missing_context(self):
        """Validation should fail when required context is missing."""
        trials = [
            RawSpeedTrial(
                context_length=512,  # Only one context, missing others
                method="turbo",
                trial_index=1,
                execution_order=("512_turbo", "trial_1"),
                execution_mode="metal_strict",
                prefill_seconds=0.1,
                token_latencies_ms=tuple(
                    [1.0] * REQUIRED_TOKEN_LATENCIES
                ),
                first_token_ms=1.5,
                throughput_tps=1000.0,
                compressed_page_dispatches=10,
                dense_tail_dispatches=1,
                fallback_calls=0,
            )
        ]
        artifact = RawSpeedArtifact(schema_version=1, trials=tuple(trials))
        errors = validate_speed_trials(artifact)
        self.assertTrue(
            any("Missing required context" in error for error in errors)
        )

    def test_validate_insufficient_trials(self):
        """Validation should fail when insufficient trials per context."""
        trials = []
        for context in REQUIRED_CONTEXTS:
            for mode in ["dense", "turbo"]:
                # Only 3 trials instead of required 5
                for trial_idx in range(3):
                    trial = RawSpeedTrial(
                        context_length=context,
                        method=mode,
                        trial_index=trial_idx + 1,
                        execution_order=(
                            f"{context}_{mode}",
                            f"trial_{trial_idx + 1}",
                        ),
                        execution_mode=(
                            "metal_strict" if mode == "turbo" else "reference"
                        ),
                        prefill_seconds=0.1,
                        token_latencies_ms=tuple(
                            [1.0] * REQUIRED_TOKEN_LATENCIES
                        ),
                        first_token_ms=1.5,
                        throughput_tps=1000.0,
                        compressed_page_dispatches=(
                            10 if mode == "turbo" else 0
                        ),
                        dense_tail_dispatches=1 if mode == "turbo" else 0,
                        fallback_calls=0,
                    )
                    trials.append(trial)

        artifact = RawSpeedArtifact(schema_version=1, trials=tuple(trials))
        errors = validate_speed_trials(artifact)
        self.assertTrue(any("required 5" in error for error in errors))

    def test_validate_wrong_execution_mode(self):
        """Validation should fail when turbo trials don't use metal_strict."""
        trials = [
            RawSpeedTrial(
                context_length=512,
                method="turbo",
                trial_index=1,
                execution_order=("512_turbo", "trial_1"),
                execution_mode="development_auto",  # Wrong mode
                prefill_seconds=0.1,
                token_latencies_ms=tuple([1.0] * REQUIRED_TOKEN_LATENCIES),
                first_token_ms=1.5,
                throughput_tps=1000.0,
                compressed_page_dispatches=10,
                dense_tail_dispatches=1,
                fallback_calls=0,
            )
        ]
        artifact = RawSpeedArtifact(schema_version=1, trials=tuple(trials))
        errors = validate_speed_trials(artifact)
        self.assertTrue(any("execution_mode" in error for error in errors))

    def test_validate_fallback_calls(self):
        """Validation should fail when turbo trials have fallback calls."""
        trials = [
            RawSpeedTrial(
                context_length=512,
                method="turbo",
                trial_index=1,
                execution_order=("512_turbo", "trial_1"),
                execution_mode="metal_strict",
                prefill_seconds=0.1,
                token_latencies_ms=tuple([1.0] * REQUIRED_TOKEN_LATENCIES),
                first_token_ms=1.5,
                throughput_tps=1000.0,
                compressed_page_dispatches=10,
                dense_tail_dispatches=1,
                fallback_calls=1,  # Has fallback
            )
        ]
        artifact = RawSpeedArtifact(schema_version=1, trials=tuple(trials))
        errors = validate_speed_trials(artifact)
        self.assertTrue(any("fallback calls" in error for error in errors))

    def test_validate_wrong_latency_count(self):
        """Validation should fail when token latencies count is wrong."""
        trials = [
            RawSpeedTrial(
                context_length=512,
                method="turbo",
                trial_index=1,
                execution_order=("512_turbo", "trial_1"),
                execution_mode="metal_strict",
                prefill_seconds=0.1,
                token_latencies_ms=tuple([1.0] * 127),  # Wrong count
                first_token_ms=1.5,
                throughput_tps=1000.0,
                compressed_page_dispatches=10,
                dense_tail_dispatches=1,
                fallback_calls=0,
            )
        ]
        artifact = RawSpeedArtifact(schema_version=1, trials=tuple(trials))
        errors = validate_speed_trials(artifact)
        self.assertTrue(any("latencies" in error for error in errors))

    def test_validate_invalid_timing_values(self):
        """Validation should fail when timing values are invalid."""
        trials = [
            RawSpeedTrial(
                context_length=512,
                method="turbo",
                trial_index=1,
                execution_order=("512_turbo", "trial_1"),
                execution_mode="metal_strict",
                prefill_seconds=-0.1,  # Invalid: negative
                token_latencies_ms=tuple([1.0] * REQUIRED_TOKEN_LATENCIES),
                first_token_ms=1.5,
                throughput_tps=1000.0,
                compressed_page_dispatches=10,
                dense_tail_dispatches=1,
                fallback_calls=0,
            )
        ]
        artifact = RawSpeedArtifact(schema_version=1, trials=tuple(trials))
        errors = validate_speed_trials(artifact)
        self.assertTrue(any("prefill_seconds" in error for error in errors))


    def test_canonical_round_trip(self):
        """Canonical dataclass → dict → from_dict must preserve all fields."""
        from dataclasses import asdict

        original = RawSpeedArtifact(
            schema_version=1,
            trials=(
                RawSpeedTrial(
                    context_length=512,
                    method="turbo",
                    trial_index=2,
                    execution_order=("512_turbo", "trial_2"),
                    execution_mode="metal_strict",
                    prefill_seconds=0.1,
                    token_latencies_ms=(1.0,) * REQUIRED_TOKEN_LATENCIES,
                    first_token_ms=1.5,
                    throughput_tps=100.0,
                    compressed_page_dispatches=10,
                    dense_tail_dispatches=1,
                    fallback_calls=0,
                ),
            ),
        )
        # Serialize via asdict (what run_speed_matrix.py uses)
        d = asdict(original)
        # Deserialize via canonical from_dict
        restored = RawSpeedArtifact.from_dict(d)

        self.assertEqual(restored.schema_version, original.schema_version)
        self.assertEqual(len(restored.trials), len(original.trials))
        rt = restored.trials[0]
        ot = original.trials[0]
        self.assertEqual(rt.context_length, ot.context_length)
        self.assertEqual(rt.method, ot.method)
        self.assertEqual(rt.trial_index, ot.trial_index)
        self.assertEqual(rt.execution_mode, ot.execution_mode)
        self.assertEqual(rt.prefill_seconds, ot.prefill_seconds)
        self.assertEqual(rt.first_token_ms, ot.first_token_ms)
        self.assertEqual(rt.throughput_tps, ot.throughput_tps)
        self.assertEqual(rt.compressed_page_dispatches, ot.compressed_page_dispatches)
        self.assertEqual(rt.dense_tail_dispatches, ot.dense_tail_dispatches)
        self.assertEqual(rt.fallback_calls, ot.fallback_calls)

    def test_legacy_format_still_parses(self):
        """Legacy benchmark format with 'mode', 'trial', 'per_token_ms' still works."""
        legacy = {
            "schema_version": 1,
            "trial_results": [
                {
                    "context_length": 512,
                    "mode": "turbo",
                    "trial": 2,
                    "execution_mode": "metal_strict",
                    "prefill_seconds": 0.1,
                    "per_token_ms": [1.0] * REQUIRED_TOKEN_LATENCIES,
                    "first_token_ms": 1.5,
                    "throughput_tps": 100.0,
                    "page_dispatches": 10,
                    "tail_dispatches": 1,
                    "fallbacks": 0,
                },
            ],
        }
        artifact = RawSpeedArtifact.from_dict(legacy)
        self.assertEqual(len(artifact.trials), 1)
        t = artifact.trials[0]
        self.assertEqual(t.method, "turbo")
        self.assertEqual(t.trial_index, 2)
        self.assertEqual(t.execution_mode, "metal_strict")
        self.assertEqual(t.compressed_page_dispatches, 10)
        self.assertEqual(t.dense_tail_dispatches, 1)
        self.assertEqual(t.fallback_calls, 0)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Run the full TurboPolar promotion suite and render a decision.

This script is the single entry point for producing ``PromotionEvidence``. It:
  1. Runs the test suite and records kernel/integration correctness.
  2. Runs real-model benchmarks (teacher-forced, fused decode, speed matrix,
     memory bench) and converts their JSON reports into evidence reports.
  3. Captures immutable provenance (git, kernel source hash, config hash).
  4. Calls ``PromotionGate.evaluate()`` and writes ``evidence.json`` and
     ``decision.json``.

Use ``--dry-run`` to synthesise evidence without a model (for CI smoke).
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from xml.etree import ElementTree as ET

import mlx.core as mx

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import (  # noqa: E402
    TurboPolarConfig,
)
from rfsn_v11.kernels.turbo_polar.execution import (  # noqa: E402
    ExecutionMode,
)
from rfsn_v11.promotion import (  # noqa: E402
    BaselineComparisonReport,
    BenchmarkProvenance,
    FusedDecodeReport,
    GitTreeState,
    KernelReport,
    MemoryReport,
    PromotionEvidence,
    PromotionGate,
    SpeedReport,
    TeacherForcedReport,
)
from rfsn_v11.promotion.provenance import (  # noqa: E402
    capture_provenance,
    _hash_jsonable,
    compute_speed_workload_hash,
    compute_memory_workload_hash,
    compute_fused_decode_workload_hash,
    compute_cartesian_workload_hash,
    compute_teacher_forced_workload_hash,
)


BENCHMARKS_DIR = project_root / "benchmarks"


def _run(
    cmd: List[str], cwd: Path = project_root, timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


REQUIRED_METAL_TESTS = {
    "tests.kernels.test_paged_online_attention",
    "tests.kernels.test_qjl_scaled_fused_qk",
    "tests.kernels.test_qjl_scaled_online_attention",
    "tests.kernels.test_metal_strict",
    "tests.kernels.test_fallback_injection",
    "tests.benchmarks.test_turbopolar_fast_attention",
    "tests.benchmarks.test_turbo_polar_online_attention",
}


def _parse_junit_xml(path: Path) -> Dict[str, Any]:
    """Parse pytest JUnit XML into a structured dict.

    P1-35: Uses explicit markers when available, falling back to
    classname inference.

    Explicit markers are preferred over classname inference because:
    1. They survive test refactoring (moving tests between files)
    2. They clearly indicate intent in the test source
    3. They don't depend on module naming conventions

    To mark a test as a metal test, use:
        @pytest.mark.metal

    The marker will appear in JUnit XML as a
    <property name="markers" value="metal"/>.
    """
    if not path.exists():
        return {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "metal_tests_required": list(REQUIRED_METAL_TESTS),
            "metal_tests_present": [],
            "metal_tests_passed": [],
        }
    tree = ET.parse(path)
    root = tree.getroot()
    testsuite = root if root.tag == "testsuite" else root.find("testsuite")
    if testsuite is None:
        return {}
    collected = int(testsuite.get("tests", 0))
    failures = int(testsuite.get("failures", 0))
    errors = int(testsuite.get("errors", 0))
    skipped = int(testsuite.get("skipped", 0))
    passed = collected - failures - errors - skipped

    metal_present: Set[str] = set()
    metal_passed: Set[str] = set()
    metal_skipped: Set[str] = set()
    testcases: List[Dict[str, Any]] = []

    for testcase in testsuite.findall("testcase"):
        cls = testcase.get("classname", "")
        name = testcase.get("name", "")
        failed = any(child.tag in ("failure", "error") for child in testcase)
        skipped_tc = any(child.tag == "skipped" for child in testcase)

        # P1-35: Check for explicit markers in properties
        has_metal_marker = False
        properties_elem = testcase.find("properties")
        if properties_elem is not None:
            for prop in properties_elem.findall("property"):
                prop_name = prop.get("name", "")
                prop_value = prop.get("value", "")
                if (
                    prop_name == "markers"
                    and "metal" in prop_value
                ):
                    has_metal_marker = True

        # Also check for marker in test name
        # (pytest can include markers in test id)
        if "[metal]" in name or "metal_" in name.lower():
            has_metal_marker = True

        testcases.append(
            {
                "classname": cls,
                "name": name,
                "failed": failed,
                "skipped": skipped_tc,
                "has_metal_marker": has_metal_marker,
            }
        )

        # P1-35: Determine if this is a metal test
        # Priority: explicit marker > classname inference
        is_metal_test = False
        module_prefix = ""

        if has_metal_marker:
            is_metal_test = True
            # Use module prefix from classname.
            # For module-level tests, the classname IS the module name and
            # must not be stripped.
            if cls in REQUIRED_METAL_TESTS:
                module_prefix = cls
            else:
                parts = cls.split(".")
                module_prefix = ".".join(parts[:-1]) if len(parts) >= 2 else cls
        else:
            # Fallback to classname inference (legacy method).
            # For module-level tests, the classname IS the module name.
            if cls in REQUIRED_METAL_TESTS:
                module_prefix = cls
                is_metal_test = True
            else:
                parts = cls.split(".")
                if len(parts) >= 2:
                    module_prefix = ".".join(parts[:-1])
                else:
                    module_prefix = cls
                if any(
                    module_prefix.startswith(req)
                    for req in REQUIRED_METAL_TESTS
                ):
                    is_metal_test = True

        if is_metal_test and module_prefix:
            metal_present.add(module_prefix)
            if not failed and not skipped_tc:
                metal_passed.add(module_prefix)
            if skipped_tc:
                metal_skipped.add(module_prefix)

    return {
        "collected": collected,
        "passed": passed,
        "failed": failures + errors,
        "skipped": skipped,
        "testcases": testcases,
        "metal_tests_required": sorted(REQUIRED_METAL_TESTS),
        "metal_tests_present": sorted(metal_present),
        "metal_tests_passed": sorted(metal_passed),
        "metal_tests_skipped": sorted(metal_skipped),
        # P1-35: Indicates we support explicit markers
        "marker_based_detection": True,
    }


def _run_pytest(artifact_dir: Path) -> KernelReport:
    """Run pytest and record pass/fail per category from JUnit XML."""
    junit_path = artifact_dir / "pytest.xml"
    result = _run(
        [
            sys.executable, "-m", "pytest", "tests/", "-q",
            f"--junitxml={junit_path}",
        ],
        timeout=600,
    )
    junit = _parse_junit_xml(junit_path)

    # Categorize by module prefix for independent booleans.
    unit_prefixes = ("tests.unit.",)
    kernel_prefixes = ("tests.kernels.", "tests.benchmarks.")
    integration_prefixes = ("tests.integrations.",)

    def _prefix_match(cls: str, prefixes: tuple) -> bool:
        return any(cls.startswith(p) for p in prefixes)

    def _all_passed(prefixes: tuple) -> bool:
        for testcase in junit.get("testcases", []):
            cls = testcase.get("classname", "")
            if _prefix_match(cls, prefixes):
                if testcase.get("failed") or testcase.get("skipped"):
                    return False
        return True

    all_passed = result.returncode == 0
    return KernelReport(
        all_unit_tests_passed=(
            all_passed and _all_passed(unit_prefixes)
        ),
        all_kernel_tests_passed=(
            all_passed and _all_passed(kernel_prefixes)
        ),
        all_integration_tests_passed=(
            all_passed and _all_passed(integration_prefixes)
        ),
        cpu_metal_agreement_verified=all_passed,
        notes=[
            "pytest exit code: " + str(result.returncode),
            "stdout lines: " + str(len(result.stdout.splitlines())),
        ],
        required_metal_tests=junit.get("metal_tests_required", []),
        metal_tests_present=junit.get("metal_tests_present", []),
        metal_tests_passed=junit.get("metal_tests_passed", []),
        metal_tests_skipped=junit.get("metal_tests_skipped", []),
    )


def _run_benchmark(
    script: str, *args: str, timeout: Optional[int] = None
) -> Dict[str, Any]:
    cmd = [sys.executable, str(BENCHMARKS_DIR / script), *args]
    result = _run(cmd, timeout=timeout)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Benchmark {script} failed with exit code {result.returncode}"
        )
    return result


def _teacher_forced_report(
    model: str, output_dir: Path
) -> TeacherForcedReport:
    _run_benchmark(
        "run_dense_vs_turbopolar.py",
        "--model",
        model,
        "--output-dir",
        str(output_dir / "teacher_forced"),
        "--context-lengths",
        "512",
        "2048",
        "4096",
        "8192",
        "16384",
        "--max-tokens",
        "16384",
        "--num-decode",
        "128",
        "--skip-decode-speed",
        timeout=7200,
    )
    report = _load_json(output_dir / "teacher_forced" / "report.json")
    agg = report.get("aggregate", {})
    
    # Calculate total_positions from nested prompt position_metrics
    # BenchmarkReport doesn't have a top-level total_positions field
    total_positions = sum(
        len(prompt.get("position_metrics", []))
        for prompt in report.get("prompts", [])
    )

    # Calculate full SHA-256 of the raw metrics file
    raw_metrics_path = output_dir / "teacher_forced" / "report.json"
    import hashlib
    with open(raw_metrics_path, "rb") as f:
        raw_metrics_hash = hashlib.sha256(f.read()).hexdigest()

    # Also write a dedicated raw_metrics.json file for the gate to parse
    raw_metrics_dedicated_path = (
        output_dir / "teacher_forced" / "raw_metrics.json"
    )
    with open(
        raw_metrics_dedicated_path, "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, sort_keys=True, indent=2, allow_nan=False)
    with open(raw_metrics_dedicated_path, "rb") as f:
        raw_metrics_hash = hashlib.sha256(f.read()).hexdigest()
    
    return TeacherForcedReport(
        model=model,
        evaluated_contexts=list(report.get("evaluated_contexts", [])),
        total_positions=total_positions,
        mean_logit_cosine=agg.get("mean_logit_cosine"),
        p05_logit_cosine=agg.get("p05_logit_cosine"),
        min_logit_cosine=agg.get("min_logit_cosine"),
        argmax_agreement=agg.get("argmax_agreement"),
        mean_top5_overlap=agg.get("mean_top5_overlap"),
        mean_top10_overlap=agg.get("mean_top10_overlap"),
        mean_perplexity_delta=agg.get("mean_perplexity_delta"),
        any_nans_or_infs=bool(agg.get("any_nans_or_infs", True)),
        raw_metrics_path=str(raw_metrics_dedicated_path),
        raw_metrics_hash=raw_metrics_hash,
        notes=list(report.get("notes", [])),
    )


def _teacher_forced_report_quick(model: str, output_dir: Path) -> TeacherForcedReport:
    """Quick-mode teacher-forced: skip very long contexts to keep runtime reasonable."""
    _run_benchmark(
        "run_dense_vs_turbopolar.py",
        "--model",
        model,
        "--output-dir",
        str(output_dir / "teacher_forced"),
        "--context-lengths",
        "512",
        "2048",
        "4096",
        "--max-tokens",
        "4096",
        "--num-decode",
        "32",
        "--skip-decode-speed",
        timeout=1800,
    )
    report = _load_json(output_dir / "teacher_forced" / "report.json")
    agg = report.get("aggregate", {})
    
    # Calculate total_positions from nested prompt position_metrics
    # BenchmarkReport doesn't have a top-level total_positions field
    total_positions = sum(
        len(prompt.get("position_metrics", []))
        for prompt in report.get("prompts", [])
    )

    # Calculate full SHA-256 of the raw metrics file
    raw_metrics_path = output_dir / "teacher_forced" / "report.json"
    import hashlib
    with open(raw_metrics_path, "rb") as f:
        raw_metrics_hash = hashlib.sha256(f.read()).hexdigest()

    # Also write a dedicated raw_metrics.json file for the gate to parse
    raw_metrics_dedicated_path = (
        output_dir / "teacher_forced" / "raw_metrics.json"
    )
    with open(
        raw_metrics_dedicated_path, "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, sort_keys=True, indent=2, allow_nan=False)
    with open(raw_metrics_dedicated_path, "rb") as f:
        raw_metrics_hash = hashlib.sha256(f.read()).hexdigest()
    
    return TeacherForcedReport(
        model=model,
        evaluated_contexts=list(report.get("evaluated_contexts", [])),
        total_positions=total_positions,
        mean_logit_cosine=agg.get("mean_logit_cosine"),
        p05_logit_cosine=agg.get("p05_logit_cosine"),
        min_logit_cosine=agg.get("min_logit_cosine"),
        argmax_agreement=agg.get("argmax_agreement"),
        mean_top5_overlap=agg.get("mean_top5_overlap"),
        mean_top10_overlap=agg.get("mean_top10_overlap"),
        mean_perplexity_delta=agg.get("mean_perplexity_delta"),
        any_nans_or_infs=bool(agg.get("any_nans_or_infs", True)),
        raw_metrics_path=str(raw_metrics_dedicated_path),
        raw_metrics_hash=raw_metrics_hash,
        notes=list(report.get("notes", [])) + ["Quick mode: 512-4096 contexts only, 32 decode tokens."],
    )


def _fused_decode_report(model: str, output_dir: Path, token_fixtures: Optional[Path] = None) -> FusedDecodeReport:
    args = [
        "--contexts",
        "512",
        "2048",
        "4096",
        "8192",
        "16384",
        "--forced-decode-tokens",
        "129",
        "--execution-mode",
        "metal_strict",
    ]
    # P1-28: Pass exact fixtures to fused decode
    if token_fixtures:
        args.extend(["--token-fixtures", str(token_fixtures)])
    
    _run_benchmark("run_fused_forced_decode.py", "--model", model, "--output-dir", str(output_dir / "fused_decode"), *args, timeout=3600)
    report = _load_json(output_dir / "fused_decode" / "report.json")
    agg = report.get("aggregate", {})
    contexts = report.get("contexts_evaluated", [])
    return FusedDecodeReport(
        model=model,
        model_layer_count=int(report.get("num_layers", 0)),
        contexts_evaluated=contexts,
        requested_fused_positions_per_context=agg.get("requested_fused_positions", 0),
        positions_per_context=dict(agg.get("positions_per_context", {})),
        failed_positions_per_context=dict(agg.get("failed_positions_per_context", {})),
        compressed_page_dispatches_per_context=dict(agg.get("compressed_page_dispatches_per_context", {})),
        dense_tail_dispatches_per_context=dict(agg.get("dense_tail_dispatches_per_context", {})),
        fallback_calls_per_context=dict(agg.get("fallback_calls_per_context", {})),
        trace_artifact_path=report.get("trace_artifact_path", ""),
        trace_artifact_hash=report.get("trace_artifact_hash", ""),
        mean_logit_cosine=agg.get("mean_logit_cosine"),
        p05_logit_cosine=agg.get("p05_logit_cosine"),
        min_logit_cosine=agg.get("min_logit_cosine"),
        mean_top5_overlap=agg.get("mean_top5_overlap"),
        mean_top10_overlap=agg.get("mean_top10_overlap"),
        argmax_agreement=agg.get("mean_top1_agreement"),
        mean_perplexity_delta=agg.get("mean_perplexity_delta"),
        any_nans_or_infs=agg.get("any_nans_or_infs", True),
        execution_mode=agg.get("execution_mode"),
        compressed_page_metal_calls=agg.get("compressed_page_metal_calls"),
        dense_tail_metal_calls=agg.get("dense_tail_metal_calls"),
        merge_metal_calls=agg.get("merge_metal_calls"),
        finalization_metal_calls=agg.get("finalization_metal_calls"),
        compressed_page_fallback_calls=agg.get("compressed_page_fallback_calls"),
        dense_tail_fallback_calls=agg.get("dense_tail_fallback_calls"),
        full_attention_fallback_calls=agg.get("full_attention_fallback_calls"),
        fallback_reasons=agg.get("fallback_reasons"),
        fallback_calls=agg.get("fallback_calls", 0),
        first_argmax_divergence_step=agg.get("first_argmax_divergence_position"),
        actual_fused_positions=agg.get("actual_fused_positions"),
    )


def _fused_decode_report_quick(model: str, output_dir: Path, token_fixtures: Optional[Path] = None) -> FusedDecodeReport:
    args = [
        "--contexts",
        "512",
        "2048",
        "4096",
        "--execution-mode",
        "metal_strict",
        "--forced-decode-tokens",
        "32",
    ]
    # P1-28: Pass exact fixtures to fused decode
    if token_fixtures:
        args.extend(["--token-fixtures", str(token_fixtures)])
    
    _run_benchmark("run_fused_forced_decode.py", "--model", model, "--output-dir", str(output_dir / "fused_decode"), *args, timeout=1800)
    report = _load_json(output_dir / "fused_decode" / "report.json")
    agg = report.get("aggregate", {})
    contexts = report.get("contexts_evaluated", [])
    return FusedDecodeReport(
        model=model,
        model_layer_count=int(report.get("num_layers", 0)),
        contexts_evaluated=contexts,
        requested_fused_positions_per_context=agg.get("requested_fused_positions", 0),
        positions_per_context=dict(agg.get("positions_per_context", {})),
        failed_positions_per_context=dict(agg.get("failed_positions_per_context", {})),
        compressed_page_dispatches_per_context=dict(agg.get("compressed_page_dispatches_per_context", {})),
        dense_tail_dispatches_per_context=dict(agg.get("dense_tail_dispatches_per_context", {})),
        fallback_calls_per_context=dict(agg.get("fallback_calls_per_context", {})),
        trace_artifact_path=report.get("trace_artifact_path", ""),
        trace_artifact_hash=report.get("trace_artifact_hash", ""),
        mean_logit_cosine=agg.get("mean_logit_cosine"),
        p05_logit_cosine=agg.get("p05_logit_cosine"),
        min_logit_cosine=agg.get("min_logit_cosine"),
        mean_top5_overlap=agg.get("mean_top5_overlap"),
        mean_top10_overlap=agg.get("mean_top10_overlap"),
        argmax_agreement=agg.get("mean_top1_agreement"),
        mean_perplexity_delta=agg.get("mean_perplexity_delta"),
        any_nans_or_infs=agg.get("any_nans_or_infs", True),
        execution_mode=agg.get("execution_mode"),
        compressed_page_metal_calls=agg.get("compressed_page_metal_calls"),
        dense_tail_metal_calls=agg.get("dense_tail_metal_calls"),
        merge_metal_calls=agg.get("merge_metal_calls"),
        finalization_metal_calls=agg.get("finalization_metal_calls"),
        compressed_page_fallback_calls=agg.get("compressed_page_fallback_calls"),
        dense_tail_fallback_calls=agg.get("dense_tail_fallback_calls"),
        full_attention_fallback_calls=agg.get("full_attention_fallback_calls"),
        fallback_reasons=agg.get("fallback_reasons"),
        fallback_calls=agg.get("fallback_calls", 0),
        first_argmax_divergence_step=agg.get("first_argmax_divergence_position"),
        actual_fused_positions=agg.get("actual_fused_positions"),
        notes=["Quick mode: 512-4096 contexts only."],
    )


def _speed_report(
    model: str, output_dir: Path, token_fixtures: Optional[Path] = None
) -> SpeedReport:
    cmd = [
        "run_speed_matrix.py",
        "--model",
        model,
        "--output-dir",
        str(output_dir / "speed_matrix"),
        "--lengths",
        "64",
        "128",
        "256",
        "512",
        "1024",
        "2048",
        "4096",
        "8192",
        "16384",
        "--num-decode",
        "128",
        "--trials",
        "5",
        "--execution-mode",
        "metal_strict",
    ]
    if token_fixtures:
        cmd.extend(["--token-fixtures", str(token_fixtures)])
    _run_benchmark(*cmd, timeout=3600)
    report = _load_json(output_dir / "speed_matrix" / "speed_matrix.json")
    records = report.get("records", [])
    contexts = [r["length"] for r in records]
    speedups = [r.get("speedup") for r in records if r.get("speedup") is not None]

    def _ratios_at(min_len: int):
        vals = [
            r["speedup"]
            for r in records
            if r["length"] >= min_len and r.get("speedup") is not None
        ]
        if not vals:
            return None, None, None
        return min(vals), max(vals), float(sorted(vals)[len(vals) // 2])

    min_4096, max_4096, _ = _ratios_at(4096)
    _, _, median_8192 = _ratios_at(8192)

    # Use minimum valid_trials across all contexts, not requested trial count.
    min_valid_trials = min(
        (r.get("valid_trials", 0) for r in records),
        default=0,
    )
    
    # Calculate total fallback count from trial records
    trial_results = report.get("trial_results", [])
    total_fallbacks = sum(
        tr.get("fallbacks", 0)
        for tr in trial_results
        if tr.get("mode") == "turbo"
    )
    
    # Write raw timing data to dedicated file and calculate full SHA-256
    # Use the unified schema format for the raw timing file
    import hashlib
    import json
    from dataclasses import asdict, is_dataclass

    raw_timing_path = output_dir / "speed_matrix" / "raw_timing.json"

    # Extract the speed_evidence if available, otherwise use trial_results
    if "speed_evidence" in report:
        # Convert SpeedEvidence dataclass to dict for JSON serialization
        # After JSON deserialization, speed_evidence is already a dict
        speed_evidence = report["speed_evidence"]
        if is_dataclass(speed_evidence):
            speed_evidence_dict = asdict(speed_evidence)
        else:
            speed_evidence_dict = speed_evidence
        raw_timing_data = {
            "schema_version": 1,
            "speed_evidence": speed_evidence_dict,
        }
    else:
        # Fallback to legacy format
        raw_timing_data = {
            "schema_version": 1,
            "trial_results": trial_results,
        }

    with open(raw_timing_path, "w") as f:
        json.dump(raw_timing_data, f, sort_keys=True, indent=2, allow_nan=False)
    with open(raw_timing_path, "rb") as f:
        raw_timing_hash = hashlib.sha256(f.read()).hexdigest()

    return SpeedReport(
        model=model,
        contexts_evaluated=contexts,
        trials_per_context=min_valid_trials,
        median_ratio=float(sorted(speedups)[len(speedups) // 2]) if speedups else None,
        min_ratio_at_4096_plus=min_4096,
        max_ratio_at_4096_plus=max_4096,
        median_ratio_at_8192_plus=median_8192,
        execution_mode=report.get("execution_mode"),
        fallback_calls=total_fallbacks,
        raw_timing_path=str(raw_timing_path),
        raw_timing_hash=raw_timing_hash,
    )


def _speed_report_quick(
    model: str, output_dir: Path, token_fixtures: Optional[Path] = None
) -> SpeedReport:
    cmd = [
        "run_speed_matrix.py",
        "--model",
        model,
        "--output-dir",
        str(output_dir / "speed_matrix"),
        "--lengths",
        "512",
        "1024",
        "2048",
        "4096",
        "--num-decode",
        "64",
        "--trials",
        "3",
        "--execution-mode",
        "metal_strict",
    ]
    if token_fixtures:
        cmd.extend(["--token-fixtures", str(token_fixtures)])
    _run_benchmark(*cmd, timeout=1800)
    report = _load_json(output_dir / "speed_matrix" / "speed_matrix.json")
    records = report.get("records", [])
    contexts = [r["length"] for r in records]
    speedups = [r.get("speedup") for r in records if r.get("speedup") is not None]

    def _ratios_at(min_len: int):
        vals = [
            r["speedup"]
            for r in records
            if r["length"] >= min_len and r.get("speedup") is not None
        ]
        if not vals:
            return None, None, None
        return min(vals), max(vals), float(sorted(vals)[len(vals) // 2])

    min_4096, max_4096, _ = _ratios_at(4096)
    _, _, median_8192 = _ratios_at(8192)

    min_valid_trials = min(
        (r.get("valid_trials", 0) for r in records),
        default=0,
    )
    
    # Calculate total fallback count from trial records
    trial_results = report.get("trial_results", [])
    total_fallbacks = sum(
        tr.get("fallbacks", 0)
        for tr in trial_results
        if tr.get("mode") == "turbo"
    )
    
    # Write raw timing data to dedicated file and calculate full SHA-256
    # Use the unified schema format for the raw timing file
    import hashlib
    import json
    from dataclasses import asdict, is_dataclass

    raw_timing_path = output_dir / "speed_matrix" / "raw_timing.json"

    # Extract the speed_evidence if available, otherwise use trial_results
    if "speed_evidence" in report:
        # Convert SpeedEvidence dataclass to dict for JSON serialization
        # After JSON deserialization, speed_evidence is already a dict
        speed_evidence = report["speed_evidence"]
        if is_dataclass(speed_evidence):
            speed_evidence_dict = asdict(speed_evidence)
        else:
            speed_evidence_dict = speed_evidence
        raw_timing_data = {
            "schema_version": 1,
            "speed_evidence": speed_evidence_dict,
        }
    else:
        # Fallback to legacy format
        raw_timing_data = {
            "schema_version": 1,
            "trial_results": trial_results,
        }

    with open(raw_timing_path, "w") as f:
        json.dump(raw_timing_data, f, sort_keys=True, indent=2, allow_nan=False)
    with open(raw_timing_path, "rb") as f:
        raw_timing_hash = hashlib.sha256(f.read()).hexdigest()

    return SpeedReport(
        model=model,
        contexts_evaluated=contexts,
        trials_per_context=min_valid_trials,
        median_ratio=float(sorted(speedups)[len(speedups) // 2]) if speedups else None,
        min_ratio_at_4096_plus=min_4096,
        max_ratio_at_4096_plus=max_4096,
        median_ratio_at_8192_plus=median_8192,
        execution_mode=report.get("execution_mode"),
        fallback_calls=total_fallbacks,
        raw_timing_path=str(raw_timing_path),
        raw_timing_hash=raw_timing_hash,
    )


def _memory_report(
    model: str,
    output_dir: Path,
    token_fixtures: Optional[Path] = None,
    strict: bool = False,
) -> MemoryReport:
    cmd = [
        "run_memory_matrix.py",
        "--model",
        model,
        "--lengths",
        "64",
        "128",
        "256",
        "512",
        "1024",
        "2048",
        "4096",
        "8192",
        "16384",
        "--output-dir",
        str(output_dir / "memory_matrix"),
    ]
    if token_fixtures:
        cmd.extend(["--token-fixtures", str(token_fixtures)])
    if strict:
        cmd.append("--strict")
    _run_benchmark(*cmd, timeout=1800)

    raw_memory_path = output_dir / "memory_matrix" / "memory_matrix.json"
    with open(raw_memory_path, "rb") as f:
        raw_memory_hash = hashlib.sha256(f.read()).hexdigest()

    report = _load_json(raw_memory_path)
    records = report.get("records", [])
    contexts = [r["length"] for r in records]

    # P0: Use the worst (minimum) ratio across all long contexts
    long_records = [r for r in records if r["length"] >= 8192]
    if long_records:
        logical_kv_ratio = min(
            r["logical_kv_ratio"] for r in long_records
            if r.get("logical_kv_ratio") is not None
        )
        persistent_storage_ratio = min(
            r["persistent_storage_ratio"] for r in long_records
            if r.get("persistent_storage_ratio") is not None
        )
        peak_device_memory_ratio = min(
            r["peak_device_memory_ratio"] for r in long_records
            if r.get("peak_device_memory_ratio") is not None
        )
    else:
        last = records[-1] if records else {}
        logical_kv_ratio = last.get("logical_kv_ratio")
        persistent_storage_ratio = last.get("persistent_storage_ratio")
        peak_device_memory_ratio = last.get("peak_device_memory_ratio")

    hidden_dense = any(r.get("hidden_dense_cache_detected", True) for r in records)
    total_fallbacks = sum(r.get("fallback_count", 0) for r in records)

    return MemoryReport(
        model=model,
        contexts_evaluated=contexts,
        logical_kv_ratio=logical_kv_ratio,
        persistent_storage_ratio=persistent_storage_ratio,
        peak_device_memory_ratio_at_8192_plus=peak_device_memory_ratio,
        hidden_dense_cache_detected=hidden_dense,
        fallback_calls=total_fallbacks,
        raw_memory_path=str(raw_memory_path),
        raw_memory_hash=raw_memory_hash,
    )


def _memory_report_quick(
    model: str,
    output_dir: Path,
    token_fixtures: Optional[Path] = None,
    strict: bool = False,
) -> MemoryReport:
    cmd = [
        "run_memory_matrix.py",
        "--model",
        model,
        "--lengths",
        "512",
        "2048",
        "4096",
        "--output-dir",
        str(output_dir / "memory_matrix"),
    ]
    if token_fixtures:
        cmd.extend(["--token-fixtures", str(token_fixtures)])
    if strict:
        cmd.append("--strict")
    _run_benchmark(*cmd, timeout=900)

    raw_memory_path = output_dir / "memory_matrix" / "memory_matrix.json"
    with open(raw_memory_path, "rb") as f:
        raw_memory_hash = hashlib.sha256(f.read()).hexdigest()

    report = _load_json(raw_memory_path)
    records = report.get("records", [])
    contexts = [r["length"] for r in records]

    # P0: Use the worst (minimum) ratio across all long contexts
    long_records = [r for r in records if r["length"] >= 4096]
    if long_records:
        logical_kv_ratio = min(
            r["logical_kv_ratio"] for r in long_records
            if r.get("logical_kv_ratio") is not None
        )
        persistent_storage_ratio = min(
            r["persistent_storage_ratio"] for r in long_records
            if r.get("persistent_storage_ratio") is not None
        )
        peak_device_memory_ratio = min(
            r["peak_device_memory_ratio"] for r in long_records
            if r.get("peak_device_memory_ratio") is not None
        )
    else:
        last = records[-1] if records else {}
        logical_kv_ratio = last.get("logical_kv_ratio")
        persistent_storage_ratio = last.get("persistent_storage_ratio")
        peak_device_memory_ratio = last.get("peak_device_memory_ratio")

    hidden_dense = any(r.get("hidden_dense_cache_detected", True) for r in records)
    total_fallbacks = sum(r.get("fallback_count", 0) for r in records)

    return MemoryReport(
        model=model,
        contexts_evaluated=contexts,
        logical_kv_ratio=logical_kv_ratio,
        persistent_storage_ratio=persistent_storage_ratio,
        peak_device_memory_ratio_at_8192_plus=peak_device_memory_ratio,
        hidden_dense_cache_detected=hidden_dense,
        fallback_calls=total_fallbacks,
        raw_memory_path=str(raw_memory_path),
        raw_memory_hash=raw_memory_hash,
    )


def _baseline_comparison_report(
    model: str, output_dir: Path
) -> BaselineComparisonReport:
    _run_benchmark(
        "run_cartesian_int8_baseline.py",
        "--model",
        model,
        "--lengths",
        "64",
        "128",
        "256",
        "512",
        "1024",
        "2048",
        "4096",
        "8192",
        "16384",
        "--num-decode",
        "128",
        "--execution-mode",
        "metal_strict",
        "--output-dir",
        str(output_dir / "cartesian_baseline"),
        timeout=3600,
    )
    report = _load_json(output_dir / "cartesian_baseline" / "report.json")
    agg = report.get("baseline_comparison_report", {})
    return BaselineComparisonReport(
        model=model,
        contexts_evaluated=report.get("contexts_evaluated", []),
        cartesian_int8_baseline_implemented=bool(
            agg.get("cartesian_int8_baseline_implemented", False)
        ),
        turbo_polar_wins_on_quality=agg.get("turbo_polar_wins_on_quality"),
        turbo_polar_wins_on_memory=agg.get("turbo_polar_wins_on_memory"),
        turbo_polar_wins_on_speed=agg.get("turbo_polar_wins_on_speed"),
        recommendation=agg.get("recommendation", ""),
        notes=list(agg.get("notes", [])),
    )


def _baseline_comparison_report_quick(
    model: str, output_dir: Path
) -> BaselineComparisonReport:
    _run_benchmark(
        "run_cartesian_int8_baseline.py",
        "--model",
        model,
        "--lengths",
        "512",
        "1024",
        "2048",
        "4096",
        "--num-decode",
        "32",
        "--execution-mode",
        "metal_strict",
        "--output-dir",
        str(output_dir / "cartesian_baseline"),
        timeout=1800,
    )
    report = _load_json(output_dir / "cartesian_baseline" / "report.json")
    agg = report.get("baseline_comparison_report", {})
    return BaselineComparisonReport(
        model=model,
        contexts_evaluated=report.get("contexts_evaluated", []),
        cartesian_int8_baseline_implemented=bool(
            agg.get("cartesian_int8_baseline_implemented", False)
        ),
        turbo_polar_wins_on_quality=agg.get("turbo_polar_wins_on_quality"),
        turbo_polar_wins_on_memory=agg.get("turbo_polar_wins_on_memory"),
        turbo_polar_wins_on_speed=agg.get("turbo_polar_wins_on_speed"),
        recommendation=agg.get("recommendation", ""),
        notes=list(agg.get("notes", [])) + ["Quick mode: 512-4096 contexts, 32 decode tokens."],
    )


def _placeholder_baseline_report() -> BaselineComparisonReport:
    return BaselineComparisonReport(
        cartesian_int8_baseline_implemented=False,
        recommendation="Cartesian int8 baseline pending.",
    )


def _extract_model_revision(model_path: str) -> str:
    """Extract immutable revision from model path.

    Handles:
    - HuggingFace repo IDs (e.g., "mlx-community/phi-3")
    - Local git repositories
    - Local file paths

    Returns commit SHA if available, empty string otherwise.
    """
    from pathlib import Path

    model_path = Path(model_path)

    # Try to get git revision if it's a git repository
    if model_path.is_dir():
        # Check if it's a git repository
        try:
            git_dir = model_path / ".git"
            if git_dir.exists() or (model_path / ".." / ".git").exists():
                # It's a git repository, get the commit SHA
                import subprocess
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=model_path if git_dir.exists() else model_path.parent,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception:
            pass

    # For HuggingFace IDs, we can't easily get the commit SHA without huggingface_hub
    # Return empty string to indicate it should be provided manually
    return ""


def _build_provenance(
    model: str,
    output_dir: Path,
    config: TurboPolarConfig,
    token_fixtures: Optional[Path] = None,
    model_revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    quick: bool = False,
) -> BenchmarkProvenance:
    prompt_suite = BENCHMARKS_DIR / "exact_token_fixtures.jsonl"

    # P0: Use provided revisions or try to extract automatically
    model_rev = model_revision if model_revision else _extract_model_revision(model)
    tokenizer_rev = tokenizer_revision if tokenizer_revision else model_rev  # Usually same as model

    # If still empty, this will be caught by the gate as a failure
    if not model_rev:
        model_rev = ""  # Empty string triggers gate failure
    if not tokenizer_rev:
        tokenizer_rev = ""  # Empty string triggers gate failure

    # P0: Compute workload hashes for each benchmark family
    # Hash actual fixture file contents, not the path string
    token_fixtures_hash = (
        hashlib.sha256(token_fixtures.read_bytes()).hexdigest()
        if token_fixtures and token_fixtures.exists()
        else ""
    )

    # Use actual executed parameters, distinguishing full vs quick mode
    if quick:
        speed_contexts = [512, 1024, 2048, 4096]
        speed_trials = 3
        speed_decode = 64
        memory_contexts = [512, 2048, 4096]
        memory_decode = 128
        fused_contexts = [512, 2048, 4096]
        fused_decode = 32
        cartesian_contexts = [512, 1024, 2048, 4096]
        cartesian_decode = 32
        teacher_contexts = [512, 2048, 4096]
        teacher_decode = 32
    else:
        speed_contexts = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        speed_trials = 5
        speed_decode = 128
        memory_contexts = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        memory_decode = 128
        fused_contexts = [512, 2048, 4096, 8192, 16384]
        fused_decode = 128
        cartesian_contexts = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        cartesian_decode = 128
        teacher_contexts = [512, 2048, 4096, 8192, 16384]
        teacher_decode = 128

    speed_workload_hash = compute_speed_workload_hash(
        context_lengths=speed_contexts,
        trial_count=speed_trials,
        decode_token_count=speed_decode,
        token_fixtures_hash=token_fixtures_hash,
    )
    memory_workload_hash = compute_memory_workload_hash(
        context_lengths=memory_contexts,
        forced_decode_count=memory_decode,
        token_fixtures_hash=token_fixtures_hash,
    )
    fused_decode_workload_hash = compute_fused_decode_workload_hash(
        context_lengths=fused_contexts,
        continuation_token_count=fused_decode,
        token_fixtures_hash=token_fixtures_hash,
    )
    cartesian_workload_hash = compute_cartesian_workload_hash(
        context_lengths=cartesian_contexts,
        forced_decode_count=cartesian_decode,
        token_fixtures_hash=token_fixtures_hash,
    )
    teacher_forced_workload_hash = compute_teacher_forced_workload_hash(
        context_lengths=teacher_contexts,
        forced_decode_count=teacher_decode,
        token_fixtures_hash=token_fixtures_hash,
    )

    return capture_provenance(
        model_repo_id=model,
        model_revision=model_rev,
        tokenizer_revision=tokenizer_rev,
        turbopolar_config=config,
        prompt_suite_path=prompt_suite,
        benchmark_command=" ".join(sys.argv),
        warmup_count=2,
        trial_count=speed_trials,
        context_lengths=speed_contexts,
        decode_token_count=speed_decode,
        qjl_enabled=False,
        token_fixtures_path=token_fixtures,
        # P0: Pass computed workload hashes
        speed_workload_hash=speed_workload_hash,
        memory_workload_hash=memory_workload_hash,
        fused_decode_workload_hash=fused_decode_workload_hash,
        cartesian_workload_hash=cartesian_workload_hash,
        teacher_forced_workload_hash=teacher_forced_workload_hash,
    )


def _synthetic_evidence() -> PromotionEvidence:
    """Create structurally-complete evidence for --dry-run smoke tests.

    Dry-run verifies command wiring, schema compatibility, artifact writing,
    and gate execution. Values are explicitly synthetic so the gate still
    returns REVIEW_REQUIRED due to PROMOTION_LOCKED.
    """
    required_contexts = [512, 2048, 4096, 8192, 16384]
    per_context = {ctx: 128 for ctx in required_contexts}
    return PromotionEvidence(
        kernel_report=KernelReport(
            all_unit_tests_passed=True,
            all_kernel_tests_passed=True,
            all_integration_tests_passed=True,
            cpu_metal_agreement_verified=True,
            required_metal_tests=list(REQUIRED_METAL_TESTS),
            metal_tests_present=list(REQUIRED_METAL_TESTS),
            metal_tests_passed=list(REQUIRED_METAL_TESTS),
            metal_tests_skipped=[],
        ),
        teacher_forced_report=TeacherForcedReport(
            model="dry-run/model",
            evaluated_contexts=required_contexts,
            total_positions=640,
            mean_logit_cosine=1.0,
            p05_logit_cosine=1.0,
            min_logit_cosine=1.0,
            argmax_agreement=1.0,
            mean_top5_overlap=1.0,
            mean_top10_overlap=1.0,
            mean_perplexity_delta=0.0,
            any_nans_or_infs=False,
        ),
        fused_decode_report=FusedDecodeReport(
            model="dry-run/model",
            contexts_evaluated=required_contexts,
            requested_fused_positions_per_context=128,
            positions_per_context=per_context,
            failed_positions_per_context={},
            compressed_page_dispatches_per_context=per_context,
            dense_tail_dispatches_per_context=per_context,
            fallback_calls_per_context={},
            execution_mode="metal_strict",
            mean_logit_cosine=1.0,
            p05_logit_cosine=1.0,
            min_logit_cosine=1.0,
            argmax_agreement=1.0,
            mean_top5_overlap=1.0,
            mean_top10_overlap=1.0,
            mean_perplexity_delta=0.0,
            any_nans_or_infs=False,
            compressed_page_metal_calls=1,
            dense_tail_metal_calls=1,
            compressed_page_fallback_calls=0,
            dense_tail_fallback_calls=0,
            full_attention_fallback_calls=0,
            fallback_reasons=[],
            actual_fused_positions=128,
            trace_artifact_path="",
            trace_artifact_hash="",
        ),
        speed_report=SpeedReport(
            model="dry-run/model",
            contexts_evaluated=required_contexts,
            trials_per_context=5,
            median_ratio=1.1,
            min_ratio_at_4096_plus=0.98,
            max_ratio_at_4096_plus=1.10,
            median_ratio_at_8192_plus=1.05,
            execution_mode="metal_strict",
            fallback_calls=0,
            raw_timing_hash="dry_run_hash",
        ),
        memory_report=MemoryReport(
            contexts_evaluated=required_contexts,
            logical_kv_ratio=2.0,
            persistent_storage_ratio=2.0,
            peak_device_memory_ratio_at_8192_plus=1.5,
            hidden_dense_cache_detected=False,
        ),
        baseline_comparison_report=BaselineComparisonReport(
            model="dry-run/model",
            contexts_evaluated=required_contexts,
            cartesian_int8_baseline_implemented=True,
            turbo_polar_wins_on_quality=True,
            turbo_polar_wins_on_memory=True,
            turbo_polar_wins_on_speed=True,
            recommendation="Dry-run baseline.",
        ),
        provenance=BenchmarkProvenance(
            git_tree_state=GitTreeState.CLEAN,
            model_repo_id="dry-run/model",
            # Empty string triggers gate failure for dry-run
            model_revision="",
            # Empty string triggers gate failure for dry-run
            tokenizer_revision="",
            turbopolar_config_hash="dry-run",
            evidence_kind="synthetic_dry_run",
        ),
    )


def _clean_dict(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        return {k: _clean_dict(v) for k, v in d.items()}
    if isinstance(obj, dict):
        return {k: _clean_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_dict(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Run the full TurboPolar promotion suite"
    )
    parser.add_argument("--model", default=None, help="MLX model path or HF identifier")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts" / "promotion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic evidence instead of running real benchmarks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced benchmark set for fast pipeline validation (fewer contexts, fewer trials)",
    )
    parser.add_argument(
        "--token-fixtures",
        type=Path,
        default=BENCHMARKS_DIR / "exact_token_fixtures.jsonl",
        help="Path to exact token fixtures JSONL file (P1-28)",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Immutable model revision (git commit SHA or HF commit SHA) - required for promotion. If not provided, will attempt to extract from git repository.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Immutable tokenizer revision (git commit SHA or HF commit SHA) - required for promotion. If not provided, defaults to model-revision.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)

    if not args.dry_run and not args.model:
        parser.error("--model is required unless --dry-run is set")

    # Immutable artifact directory: timestamp + short commit + config hash
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    commit = (
        _run(["git", "rev-parse", "--short", "HEAD"], cwd=project_root).stdout.strip()
        or "unknown"
    )
    config = TurboPolarConfig(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=64,
        storage_mode="kv_quant",
        use_int8_radii=True,
        k_angle_bits_deep=8,
        split_dim=0,
        execution_mode=ExecutionMode.METAL_STRICT,
    )
    config_hash = "dry-run" if args.dry_run else _hash_jsonable(config.__dict__)
    artifact_dir = args.output_dir / f"{timestamp}_{commit}_{config_hash}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts: {artifact_dir}")

    print("Step 1/5: running test suite...")
    kernel_report = _run_pytest(artifact_dir)

    if args.dry_run:
        print("Step 2-5: --dry-run, using placeholder benchmark evidence...")
        evidence = _synthetic_evidence()
        evidence.provenance.git_tree_state = GitTreeState.CLEAN
    else:
        print("Step 2/5: teacher-forced benchmark...")
        if args.quick:
            teacher_report = _teacher_forced_report_quick(args.model, artifact_dir)
        else:
            teacher_report = _teacher_forced_report(args.model, artifact_dir)

        print("Step 3/5: fused decode benchmark...")
        if args.quick:
            fused_report = _fused_decode_report_quick(args.model, artifact_dir, args.token_fixtures)
        else:
            fused_report = _fused_decode_report(args.model, artifact_dir, args.token_fixtures)

        print("Step 4/5: speed matrix benchmark...")
        if args.quick:
            speed_report = _speed_report_quick(
                args.model, artifact_dir, args.token_fixtures
            )
        else:
            speed_report = _speed_report(
                args.model, artifact_dir, args.token_fixtures
            )

        print("Step 5/5: memory benchmark...")
        if args.quick:
            memory_report = _memory_report_quick(
                args.model, artifact_dir,
                token_fixtures=args.token_fixtures,
                strict=True,
            )
        else:
            memory_report = _memory_report(
                args.model, artifact_dir,
                token_fixtures=args.token_fixtures,
                strict=True,
            )

        print("Step 6/5: Cartesian int8 baseline comparison...")
        if args.quick:
            baseline_report = _baseline_comparison_report_quick(args.model, artifact_dir)
        else:
            baseline_report = _baseline_comparison_report(args.model, artifact_dir)

        provenance = _build_provenance(
            args.model, artifact_dir, config,
            args.token_fixtures, args.model_revision,
            args.tokenizer_revision, quick=args.quick,
        )

        evidence = PromotionEvidence(
            kernel_report=kernel_report,
            teacher_forced_report=teacher_report,
            fused_decode_report=fused_report,
            speed_report=speed_report,
            memory_report=memory_report,
            baseline_comparison_report=baseline_report,
            provenance=provenance,
        )

    print("Evaluating promotion gate...")
    decision = PromotionGate().evaluate(evidence)

    evidence_path = artifact_dir / "evidence.json"
    decision_path = artifact_dir / "promotion_decision.json"
    provenance_path = artifact_dir / "provenance.json"

    with open(evidence_path, "w") as f:
        json.dump(_clean_dict(evidence), f, indent=2, allow_nan=False)
    with open(decision_path, "w") as f:
        json.dump(
            {
                "state": decision.state.value,
                "reasons": decision.reasons,
            },
            f,
            indent=2,
            allow_nan=False,
        )
    with open(provenance_path, "w") as f:
        json.dump(_clean_dict(evidence.provenance), f, indent=2, allow_nan=False)

    print(f"Evidence written to {evidence_path}")
    print(f"Decision written to {decision_path}")
    print(f"Provenance written to {provenance_path}")
    print(f"Promotion state: {decision.state.value}")
    if decision.reasons:
        print("Reasons:")
        for reason in decision.reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()

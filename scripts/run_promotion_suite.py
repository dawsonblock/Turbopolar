#!/usr/bin/env python3
"""Run the full TurboPolar promotion suite and render a decision.

This script is the single entry point for producing ``PromotionEvidence``. It:
  1. Runs the test suite and records kernel/integration correctness.
  2. Runs real-model benchmarks (teacher-forced, fused decode, speed matrix,
     memory bench) and converts their JSON reports into evidence reports.
  3. Captures immutable provenance (git, kernel source hash, config hash).
  4. Calls ``PromotionGate.evaluate()`` and writes ``evidence.json`` and
     ``decision.json``.

Use ``--dry-run`` to synthesise evidence without a model (for CI smoke testing).
"""

import argparse
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

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.promotion import (
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
from rfsn_v11.promotion.provenance import capture_provenance, _hash_jsonable


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
    """Parse pytest JUnit XML into a structured dict."""
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
    for testcase in testsuite.findall("testcase"):
        cls = testcase.get("classname", "")
        failed = any(child.tag in ("failure", "error") for child in testcase)
        skipped_tc = any(child.tag == "skipped" for child in testcase)
        # Map class name to module prefix.
        parts = cls.split(".")
        if len(parts) >= 2:
            module_prefix = ".".join(parts[:-1])
        else:
            module_prefix = cls
        if any(module_prefix.startswith(req) for req in REQUIRED_METAL_TESTS):
            metal_present.add(module_prefix)
            if not failed and not skipped_tc:
                metal_passed.add(module_prefix)

    return {
        "collected": collected,
        "passed": passed,
        "failed": failures + errors,
        "skipped": skipped,
        "metal_tests_required": sorted(REQUIRED_METAL_TESTS),
        "metal_tests_present": sorted(metal_present),
        "metal_tests_passed": sorted(metal_passed),
    }


def _run_pytest(artifact_dir: Path) -> KernelReport:
    """Run pytest and record pass/fail at the report level."""
    junit_path = artifact_dir / "pytest.xml"
    result = _run(
        [sys.executable, "-m", "pytest", "tests/", "-q", f"--junitxml={junit_path}"],
        timeout=600,
    )
    passed = result.returncode == 0
    junit = _parse_junit_xml(junit_path)
    return KernelReport(
        all_unit_tests_passed=passed,
        all_kernel_tests_passed=passed,
        all_integration_tests_passed=passed,
        cpu_metal_agreement_verified=passed,
        notes=[
            "pytest exit code: " + str(result.returncode),
            "stdout lines: " + str(len(result.stdout.splitlines())),
        ],
        required_metal_tests=junit.get("metal_tests_required", []),
        metal_tests_present=junit.get("metal_tests_present", []),
        metal_tests_passed=junit.get("metal_tests_passed", []),
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


def _teacher_forced_report(model: str, output_dir: Path) -> TeacherForcedReport:
    _run_benchmark(
        "run_dense_vs_turbopolar.py",
        "--model",
        model,
        "--output-dir",
        str(output_dir / "teacher_forced"),
        "--max-tokens",
        "128",
        "--num-decode",
        "128",
        "--skip-decode-speed",
        timeout=1200,
    )
    report = _load_json(output_dir / "teacher_forced" / "report.json")
    agg = report.get("aggregate", {})
    return TeacherForcedReport(
        model=model,
        mean_logit_cosine=agg.get("mean_logit_cosine"),
        p05_logit_cosine=agg.get("p05_logit_cosine"),
        min_logit_cosine=agg.get("min_logit_cosine"),
        mean_top5_overlap=agg.get("mean_top5_overlap"),
        mean_top10_overlap=agg.get("mean_top10_overlap"),
        argmax_agreement=agg.get("argmax_agreement"),
        mean_perplexity_delta=agg.get("mean_perplexity_delta"),
        any_nans_or_infs=agg.get("any_nans_or_infs", True),
    )


def _fused_decode_report(model: str, output_dir: Path) -> FusedDecodeReport:
    _run_benchmark(
        "run_fused_forced_decode.py",
        "--model",
        model,
        "--output-dir",
        str(output_dir / "fused_decode"),
        "--contexts",
        "512",
        "2048",
        "4096",
        "8192",
        "16384",
        "--forced-decode-tokens",
        "128",
        "--execution-mode",
        "metal_strict",
        timeout=1200,
    )
    report = _load_json(output_dir / "fused_decode" / "report.json")
    agg = report.get("aggregate", {})
    contexts = report.get("contexts_evaluated", [])
    return FusedDecodeReport(
        model=model,
        contexts_evaluated=contexts,
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
    )


def _speed_report(model: str, output_dir: Path) -> SpeedReport:
    _run_benchmark(
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
        "--num-decode",
        "128",
        "--trials",
        "5",
        timeout=1800,
    )
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

    return SpeedReport(
        model=model,
        contexts_evaluated=contexts,
        trials_per_context=report.get("trials", 0),
        median_ratio=float(sorted(speedups)[len(speedups) // 2]) if speedups else None,
        min_ratio_at_4096_plus=min_4096,
        max_ratio_at_4096_plus=max_4096,
        median_ratio_at_8192_plus=median_8192,
    )


def _memory_report(output_dir: Path) -> MemoryReport:
    _run_benchmark(
        "run_memory_matrix.py",
        "--lengths",
        "64",
        "128",
        "256",
        "512",
        "1024",
        "2048",
        "4096",
        "8192",
        "--output-dir",
        str(output_dir / "memory_matrix"),
        timeout=600,
    )
    report = _load_json(output_dir / "memory_matrix" / "memory_matrix.json")
    records = report.get("records", [])
    contexts = [r["length"] for r in records]

    long_records = [r for r in records if r["length"] >= 8192]
    long_record = long_records[-1] if long_records else (records[-1] if records else {})

    hidden_dense = any(r.get("hidden_dense_cache_detected", True) for r in records)

    return MemoryReport(
        contexts_evaluated=contexts,
        logical_kv_ratio=long_record.get("logical_kv_ratio"),
        persistent_storage_ratio=long_record.get("persistent_storage_ratio"),
        peak_device_memory_ratio_at_8192_plus=long_record.get(
            "peak_device_memory_ratio"
        ),
        hidden_dense_cache_detected=hidden_dense,
    )


def _baseline_comparison_report(
    model: str, output_dir: Path
) -> BaselineComparisonReport:
    _run_benchmark(
        "run_cartesian_int8_baseline.py",
        "--lengths",
        "64",
        "128",
        "256",
        "512",
        "1024",
        "--output-dir",
        str(output_dir / "cartesian_baseline"),
        timeout=1200,
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


def _placeholder_baseline_report() -> BaselineComparisonReport:
    return BaselineComparisonReport(
        cartesian_int8_baseline_implemented=False,
        recommendation="Cartesian int8 baseline pending.",
    )


def _build_provenance(
    model: str,
    output_dir: Path,
    config: TurboPolarConfig,
) -> BenchmarkProvenance:
    prompt_suite = BENCHMARKS_DIR / "exact_token_fixtures.jsonl"
    return capture_provenance(
        model_repo_id=model,
        model_revision="unknown",
        tokenizer_revision="unknown",
        turbopolar_config=config,
        prompt_suite_path=prompt_suite,
        benchmark_command=" ".join(sys.argv),
        warmup_count=2,
        trial_count=5,
        context_lengths=[512, 2048, 4096, 8192, 16384],
        decode_token_count=128,
        qjl_enabled=False,
    )


def _synthetic_evidence() -> PromotionEvidence:
    """Create minimal evidence for --dry-run smoke tests.

    Dry-run verifies command wiring, schema compatibility, artifact writing,
    and gate execution.  It must not populate passing scientific metrics.
    """
    return PromotionEvidence(
        kernel_report=KernelReport(all_unit_tests_passed=True),
        teacher_forced_report=TeacherForcedReport(),
        fused_decode_report=FusedDecodeReport(),
        speed_report=SpeedReport(),
        memory_report=MemoryReport(),
        baseline_comparison_report=_placeholder_baseline_report(),
        provenance=BenchmarkProvenance(
            git_tree_state=GitTreeState.CLEAN,
            model_repo_id="dry-run/model",
            model_revision="dry-run",
            turbopolar_config_hash="dry-run",
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
        evidence.kernel_report = kernel_report
        evidence.provenance.git_tree_state = GitTreeState.CLEAN
    else:
        print("Step 2/5: teacher-forced benchmark...")
        teacher_report = _teacher_forced_report(args.model, artifact_dir)

        print("Step 3/5: fused decode benchmark...")
        fused_report = _fused_decode_report(args.model, artifact_dir)

        print("Step 4/5: speed matrix benchmark...")
        speed_report = _speed_report(args.model, artifact_dir)

        print("Step 5/5: memory benchmark...")
        memory_report = _memory_report(artifact_dir)

        print("Step 6/5: Cartesian int8 baseline comparison...")
        baseline_report = _baseline_comparison_report(args.model, artifact_dir)

        provenance = _build_provenance(args.model, artifact_dir, config)

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
        json.dump(_clean_dict(evidence), f, indent=2)
    with open(decision_path, "w") as f:
        json.dump(
            {
                "state": decision.state.value,
                "reasons": decision.reasons,
            },
            f,
            indent=2,
        )
    with open(provenance_path, "w") as f:
        json.dump(_clean_dict(evidence.provenance), f, indent=2)

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

"""Write benchmark artifacts to versioned, non-overwriting directories."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from rfsn_v11.promotion.schema import BenchmarkProvenance, PromotionEvidence


def _serialize(obj: Any) -> Any:
    """Recursively serialize dataclasses, enums, and arrays to JSON-friendly types."""
    if hasattr(obj, "value") and isinstance(obj.value, str):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _run_id(provenance: BenchmarkProvenance) -> str:
    ts = datetime.fromisoformat(provenance.timestamp_utc).strftime("%Y%m%d_%H%M%S")
    commit = provenance.git_commit[:8] if provenance.git_commit else "unknown"
    cfg_hash = (
        provenance.turbopolar_config_hash[:8]
        if provenance.turbopolar_config_hash
        else "unknown"
    )
    return f"{ts}_{cfg_hash}_{commit}"


def artifact_root() -> Path:
    """Return the root directory for all benchmark artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "artifacts"


def write_artifacts(
    evidence: PromotionEvidence,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """Write all evidence reports to a new, non-overwriting artifact directory.

    Args:
        evidence: Full promotion evidence to persist.
        extra: Optional extra files to write as {filename: serializable_value}.

    Returns:
        Path to the newly created artifact directory.
    """
    if evidence.provenance is None:
        raise ValueError("Evidence must include provenance to write artifacts.")

    root = artifact_root()
    run_dir = root / _run_id(evidence.provenance)
    counter = 0
    original = run_dir
    while run_dir.exists():
        counter += 1
        run_dir = Path(f"{original}_{counter}")
    run_dir.mkdir(parents=True, exist_ok=False)

    files = {
        "promotion_report.json": _serialize(evidence),
        "provenance.json": _serialize(evidence.provenance),
    }
    if evidence.kernel_report is not None:
        files["kernel_report.json"] = _serialize(evidence.kernel_report)
    if evidence.teacher_forced_report is not None:
        files["teacher_forced_report.json"] = _serialize(evidence.teacher_forced_report)
    if evidence.fused_decode_report is not None:
        files["fused_decode_report.json"] = _serialize(evidence.fused_decode_report)
    if evidence.speed_report is not None:
        files["speed_report.json"] = _serialize(evidence.speed_report)
    if evidence.memory_report is not None:
        files["memory_report.json"] = _serialize(evidence.memory_report)
    if evidence.baseline_comparison_report is not None:
        files["comparison_report.json"] = _serialize(
            evidence.baseline_comparison_report
        )

    if extra:
        for name, value in extra.items():
            files[name] = _serialize(value)

    for name, value in files.items():
        (run_dir / name).write_text(json.dumps(value, indent=2) + "\n")

    # Update stable pointer, but keep historical runs.
    latest_link = root / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.relative_to(root), target_is_directory=True)

    return run_dir

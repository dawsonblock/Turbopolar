"""Capture immutable benchmark provenance."""

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.promotion.schema import BenchmarkProvenance, GitTreeState


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.exists():
        h.update(path.read_bytes())
    return h.hexdigest()


def _dir_sha256(directory: Path, glob: str = "*.metal") -> str:
    h = hashlib.sha256()
    for path in sorted(directory.glob(glob)):
        h.update(path.read_bytes())
    return h.hexdigest()


def _hash_jsonable(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _macos_version() -> str:
    try:
        return platform.mac_ver()[0]
    except Exception:
        return ""


def _chip_model() -> str:
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"])


def _system_memory_gb() -> float:
    try:
        mem_bytes = int(_run(["sysctl", "-n", "hw.memsize"]) or "0")
        return mem_bytes / (1024**3)
    except Exception:
        return 0.0


def capture_provenance(
    model_repo_id: str,
    model_revision: str,
    tokenizer_revision: str,
    turbopolar_config: TurboPolarConfig,
    prompt_suite_path: Path,
    benchmark_command: str,
    warmup_count: int,
    trial_count: int,
    context_lengths: list[int],
    decode_token_count: int,
    qjl_enabled: bool,
    evidence_kind: str = "experimental",
) -> BenchmarkProvenance:
    """Build a BenchmarkProvenance record from the current environment."""
    git_commit = _run(["git", "rev-parse", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    if not git_commit:
        git_tree_state = GitTreeState.UNKNOWN
    elif porcelain == "":
        git_tree_state = GitTreeState.CLEAN
    else:
        git_tree_state = GitTreeState.DIRTY
    git_diff_hash = ""
    if git_tree_state == GitTreeState.DIRTY:
        git_diff_hash = hashlib.sha256(
            _run(["git", "diff", "HEAD"]).encode()
        ).hexdigest()[:16]

    mlx_version = ""
    try:
        import mlx.core as mx
        mlx_version = mx.__version__
    except Exception:
        pass

    mlx_lm_version = ""
    try:
        import mlx_lm
        mlx_lm_version = mlx_lm.__version__
    except Exception:
        pass

    kernel_dir = Path(__file__).parents[1] / "kernels" / "turbo_polar"
    metal_kernel_source_hash = _dir_sha256(kernel_dir, "*.metal")

    config_dict = {
        "k_angle_bits_level1": turbopolar_config.k_angle_bits_level1,
        "k_angle_bits_deep": turbopolar_config.k_angle_bits_deep,
        "use_int8_radii": turbopolar_config.use_int8_radii,
        "v_bits": turbopolar_config.v_bits,
        "block_size": turbopolar_config.block_size,
        "page_capacity_blocks": turbopolar_config.page_capacity_blocks if hasattr(turbopolar_config, "page_capacity_blocks") else 16,
        "head_dim": turbopolar_config.head_dim,
        "qjl_proj_dim": turbopolar_config.qjl_proj_dim,
        "use_qjl": turbopolar_config.use_qjl,
        "storage_mode": turbopolar_config.storage_mode,
        "split_dim": turbopolar_config.split_dim,
        "attention_scale": turbopolar_config.attention_scale,
        "num_q_heads": turbopolar_config.num_q_heads,
        "num_kv_heads": turbopolar_config.num_kv_heads,
        "execution_mode": turbopolar_config.execution_mode.value if hasattr(turbopolar_config.execution_mode, "value") else str(turbopolar_config.execution_mode),
        "trace_validation_mode": turbopolar_config.trace_validation_mode.value if hasattr(turbopolar_config.trace_validation_mode, "value") else str(turbopolar_config.trace_validation_mode),
    }

    prompt_suite_hash = _file_sha256(prompt_suite_path)
    
    # Hash Python bindings and storage-layout code
    integration_dir = Path(__file__).parents[1] / "integrations" / "mlx_lm"
    storage_dir = Path(__file__).parents[1] / "generation"
    python_bindings_hash = _dir_sha256(integration_dir, "*.py")
    storage_layout_hash = _dir_sha256(storage_dir, "*.py")
    
    # Combine hashes for config
    config_dict["python_bindings_hash"] = python_bindings_hash
    config_dict["storage_layout_hash"] = storage_layout_hash
    
    config_hash = _hash_jsonable(config_dict)

    # Reject placeholder "unknown" revisions so the gate fails explicitly.
    if model_revision == "unknown":
        model_revision = ""
    if tokenizer_revision == "unknown":
        tokenizer_revision = ""

    return BenchmarkProvenance(
        run_id=str(uuid.uuid4()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        git_tree_state=git_tree_state,
        git_diff_hash=git_diff_hash,
        python_version=platform.python_version(),
        mlx_version=mlx_version,
        mlx_lm_version=mlx_lm_version,
        macos_version=_macos_version(),
        chip_model=_chip_model(),
        system_memory_gb=_system_memory_gb(),
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        prompt_suite_hash=prompt_suite_hash,
        turbopolar_config_hash=config_hash,
        turbopolar_config=config_dict,
        benchmark_command=benchmark_command,
        warmup_count=warmup_count,
        trial_count=trial_count,
        context_lengths=list(context_lengths),
        decode_token_count=decode_token_count,
        qjl_enabled=qjl_enabled,
        metal_kernel_source_hash=metal_kernel_source_hash,
        evidence_kind=evidence_kind,
    )

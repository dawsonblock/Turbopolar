"""Typed provenance evidence schema for TurboPolar promotion."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProvenanceEvidence:
    """Immutable provenance for a benchmark run."""

    run_id: str = ""
    timestamp_utc: str = ""
    git_commit: str = ""
    git_tree_state: str = "UNKNOWN"
    git_diff_hash: str = ""
    python_version: str = ""
    mlx_version: str = ""
    mlx_lm_version: str = ""
    macos_version: str = ""
    chip_model: str = ""
    system_memory_gb: Optional[float] = None
    model_repo_id: str = ""
    model_revision: str = ""
    tokenizer_repo_id: str = ""
    tokenizer_revision: str = ""
    prompt_suite_hash: str = ""
    context_hashes: Dict[str, str] = field(default_factory=dict)
    continuation_hashes: Dict[str, str] = field(default_factory=dict)
    turbopolar_config_hash: str = ""
    turbopolar_config: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = ""
    trace_validation_mode: str = ""
    page_capacity: int = 0
    block_size: int = 0
    k_bit_widths: str = ""
    v_bit_width: int = 0
    v_group_size: int = 0
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    attention_scale: float = 0.0
    benchmark_command: str = ""
    warmup_count: int = 0
    trial_count: int = 0
    context_lengths: List[int] = field(default_factory=list)
    decode_token_count: int = 0
    qjl_enabled: bool = False
    metal_kernel_source_hash: str = ""
    kernel_binding_hash: str = ""
    git_source_hash: str = ""
    notes: List[str] = field(default_factory=list)


def validate_provenance_immutable_fields(
    provenance: ProvenanceEvidence,
) -> List[str]:
    """Validate that all immutable provenance fields are populated.

    Args:
        provenance: Provenance evidence to validate

    Returns:
        List of validation error messages. Empty list means validation passed.
    """
    errors = []

    # Critical immutable fields that must be present
    required_fields = [
        ("run_id", provenance.run_id),
        ("git_commit", provenance.git_commit),
        ("python_version", provenance.python_version),
        ("mlx_version", provenance.mlx_version),
        ("model_repo_id", provenance.model_repo_id),
        ("model_revision", provenance.model_revision),
        ("turbopolar_config_hash", provenance.turbopolar_config_hash),
        ("execution_mode", provenance.execution_mode),
    ]

    for field_name, value in required_fields:
        if not value:
            errors.append(f"Required provenance field '{field_name}' is empty")

    # Hash fields should be valid hex lengths
    # git_commit can be 40 (SHA-1) or 64 (SHA-256)
    if provenance.git_commit and len(provenance.git_commit) not in (40, 64):
        errors.append(
            f"git_commit must be 40 or 64 characters, "
            f"got {len(provenance.git_commit)}"
        )

    # Other hashes must be SHA-256 (64 characters) only
    sha256_only_fields = [
        ("turbopolar_config_hash", provenance.turbopolar_config_hash),
        ("metal_kernel_source_hash", provenance.metal_kernel_source_hash),
        ("kernel_binding_hash", provenance.kernel_binding_hash),
    ]

    for field_name, value in sha256_only_fields:
        if value and len(value) != 64:
            errors.append(
                f"Hash field '{field_name}' must be 64 characters "
                f"(SHA-256), got {len(value)}"
            )

    # Git tree state must be valid
    valid_tree_states = ["CLEAN", "DIRTY", "UNKNOWN"]
    if provenance.git_tree_state not in valid_tree_states:
        errors.append(
            f"git_tree_state must be one of {valid_tree_states}, "
            f"got '{provenance.git_tree_state}'"
        )

    # If tree is dirty, diff_hash must be present
    if provenance.git_tree_state == "DIRTY" and not provenance.git_diff_hash:
        errors.append("git_diff_hash is required when git_tree_state is DIRTY")

    # Config hash should match computed hash from config dict
    if provenance.turbopolar_config and provenance.turbopolar_config_hash:
        import hashlib
        import json

        config_json = json.dumps(
            provenance.turbopolar_config, sort_keys=True, allow_nan=False
        )
        computed_hash = hashlib.sha256(config_json.encode()).hexdigest()
        if computed_hash != provenance.turbopolar_config_hash:
            errors.append(
                "turbopolar_config_hash does not match computed "
                "hash from config dict"
            )

    return errors


def validate_provenance_evidence_kind(
    provenance: ProvenanceEvidence, expected_kind: str
) -> List[str]:
    """Validate that evidence kind matches expectation.

    Args:
        provenance: Provenance evidence to validate
        expected_kind: Expected evidence kind (e.g., "experimental",
            "synthetic_dry_run")

    Returns:
        List of validation error messages. Empty list means validation passed.
    """
    errors = []

    # This would be added to ProvenanceEvidence in a full implementation
    # For now, we validate via notes if present
    for note in provenance.notes:
        if "evidence_kind" in note.lower():
            # Extract kind from note
            if expected_kind not in note:
                errors.append(
                    f"Evidence kind mismatch: expected '{expected_kind}', "
                    f"found '{note}'"
                )

    return errors

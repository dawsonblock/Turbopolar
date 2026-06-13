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

"""Schema for dense-vs-TurboPolar benchmark reports.

Promotion decisions are owned exclusively by rfsn_v11.promotion.gate.
This module only defines report shapes used by benchmark scripts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class PromptResult:
    prompt: str
    prompt_tokens: int
    dense_logits_shape: Tuple[int, ...]
    turbo_logits_shape: Tuple[int, ...]
    logit_cosine: float
    top5_overlap: float
    top10_overlap: float
    kl_divergence: float
    perplexity_delta: float
    compression_ratio: float
    peak_kv_bytes_turbo: int
    peak_kv_bytes_dense: int


@dataclass
class BenchmarkReport:
    model: str
    mlx_version: str
    mlx_lm_version: str
    dtype: str
    seed: int
    num_layers: int
    num_prompts: int
    aggregate: Dict[str, Any] = field(default_factory=dict)
    prompts: List[PromptResult] = field(default_factory=list)

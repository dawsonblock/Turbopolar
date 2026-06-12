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


@dataclass
class DecodeStepMetrics:
    """Per-token metrics from a single forced-decode position."""

    position: int
    logit_cosine: float
    top1_agreement: bool
    top5_overlap: float
    top10_overlap: float
    kl_divergence: float
    js_divergence: float
    dense_argmax_rank_in_turbo: int
    dense_argmax_prob_delta: float
    nll_delta: float
    any_nan_or_inf: bool


@dataclass
class ForcedDecodeFixtureResult:
    """Result for one fixture (context + continuation) in forced-decode mode."""

    fixture_id: str
    context_length: int
    continuation_length: int
    steps: List[DecodeStepMetrics] = field(default_factory=list)
    kernel_stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class ForcedDecodeAggregate:
    """Aggregated statistics across all fixtures and decode positions."""

    mean_logit_cosine: float
    median_logit_cosine: float
    p05_logit_cosine: float
    p95_logit_cosine: float
    min_logit_cosine: float
    max_logit_cosine: float
    mean_top1_agreement: float
    mean_top5_overlap: float
    mean_top10_overlap: float
    mean_kl_divergence: float
    mean_js_divergence: float
    mean_perplexity_delta: float
    min_dense_argmax_rank: int
    max_dense_argmax_rank: int
    mean_dense_argmax_prob_delta: float
    worst_fixture_id: str = ""
    worst_position: int = -1
    first_argmax_divergence_position: int = -1
    any_nans_or_infs: bool = False
    online_attention_calls: int = 0
    dense_tail_calls: int = 0
    fallback_calls: int = 0


@dataclass
class ForcedDecodeReport:
    """Full report for the fused forced-decode benchmark."""

    model: str
    mlx_version: str
    mlx_lm_version: str
    dtype: str
    seed: int
    num_layers: int
    forced_decode_tokens: int
    aggregate: ForcedDecodeAggregate
    fixtures: List[ForcedDecodeFixtureResult] = field(default_factory=list)

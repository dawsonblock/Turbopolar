"""Schema and promotion gates for dense-vs-TurboPolar benchmark reports."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple


PROMOTION_GATES = {
    "kv_compression_ratio": 1.7,
    "logit_cosine": 0.995,
    "top5_overlap": 0.95,
    "top10_overlap": 0.95,
    "perplexity_delta": 0.02,
    "decode_tokens_per_sec": 1.0,  # relative to dense baseline
}


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
    promotion_allowed: bool = False
    gate_passed: Dict[str, bool] = field(default_factory=dict)
    aggregate: Dict[str, Any] = field(default_factory=dict)
    prompts: List[PromptResult] = field(default_factory=list)
    visible_drift: bool = False

    def evaluate_gates(self) -> None:
        agg = self.aggregate
        self.gate_passed = {
            "kv_compression_ratio": agg.get("compression_ratio", 0.0) >= PROMOTION_GATES["kv_compression_ratio"],
            "logit_cosine": agg.get("logit_cosine", 0.0) >= PROMOTION_GATES["logit_cosine"],
            "top5_overlap": agg.get("top5_overlap", 0.0) >= PROMOTION_GATES["top5_overlap"],
            "top10_overlap": agg.get("top10_overlap", 0.0) >= PROMOTION_GATES["top10_overlap"],
            "perplexity_delta": abs(agg.get("perplexity_delta", float("inf"))) <= PROMOTION_GATES["perplexity_delta"],
            "decode_speed": (
                True
                if agg.get("decode_speed_ratio") is None
                else agg.get("decode_speed_ratio", 0.0) >= PROMOTION_GATES["decode_tokens_per_sec"]
            ),
        }
        self.promotion_allowed = all(self.gate_passed.values())
        self.visible_drift = not all([
            self.gate_passed["logit_cosine"],
            self.gate_passed["top5_overlap"],
            self.gate_passed["top10_overlap"],
            self.gate_passed["perplexity_delta"],
        ])

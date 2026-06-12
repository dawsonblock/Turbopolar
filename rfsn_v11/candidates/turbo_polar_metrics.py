import mlx.core as mx
import numpy as np
from typing import Dict


def _log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    x_max = mx.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True))


def mean_token_kl(baseline_logits: mx.array, candidate_logits: mx.array) -> float:
    log_p = _log_softmax(baseline_logits, axis=-1)
    log_q = _log_softmax(candidate_logits, axis=-1)
    p = mx.exp(log_p)
    kl_per_token = mx.sum(p * (log_p - log_q), axis=-1)
    return float(mx.mean(kl_per_token))


def topk_set_overlap_np(base_logits: mx.array, cand_logits: mx.array, k: int) -> float:
    base = np.array(base_logits)
    cand = np.array(cand_logits)
    base_top = np.argpartition(base, -k, axis=-1)[..., -k:]
    cand_top = np.argpartition(cand, -k, axis=-1)[..., -k:]
    base_flat = base_top.reshape(-1, k)
    cand_flat = cand_top.reshape(-1, k)
    overlaps = []
    for b, c in zip(base_flat, cand_flat):
        overlaps.append(len(set(b.tolist()) & set(c.tolist())) / k)
    return float(np.mean(overlaps))


def calculate_logit_deltas(
    baseline_logits: mx.array, candidate_logits: mx.array
) -> Dict[str, float]:
    abs_diff = mx.abs(baseline_logits - candidate_logits)
    flat_diff = np.array(abs_diff).flatten()
    return {
        "mean_abs_logit_delta": float(np.mean(flat_diff)),
        "p99_abs_logit_delta": float(np.percentile(flat_diff, 99)),
        "max_logit_delta": float(np.max(flat_diff)),
    }

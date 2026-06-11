import mlx.core as mx
import numpy as np
from rfsn_v11.quant.qjl.encoder import QJLPayload


def qjl_dot_estimate(q: mx.array, qjl_payload: QJLPayload, q_proj_signs: mx.array) -> mx.array:
    """
    Estimate q · E (the residual dot-product) from packed QJL sign sketches.

    Handles GQA: query-head signs are compared against the KV-head signs that
    serve them, using num_queries_per_kv inferred from the shapes.

    NOTE: This currently uses a sign-agreement heuristic. A calibrated estimator
    (hamming -> angular -> cosine -> dot) is desirable, but the finite sketch
    dimension and arbitrary q/E geometry make variance high on random data.
    """
    B, H_kv, S, L, _ = qjl_payload.shape
    H_q = q_proj_signs.shape[1]
    proj_dim = qjl_payload.proj_dim
    num_queries_per_kv = H_q // H_kv

    # Unpack K residual sign sketch [B, H_kv, S, L, proj_dim]
    packed = np.array(qjl_payload.packed_signs)
    flat_packed = packed.reshape(-1, proj_dim // 8)
    flat_bits = np.unpackbits(flat_packed, axis=1, bitorder="little")[:, -proj_dim:]
    k_signs = flat_bits.reshape(B, H_kv, S, L, proj_dim).astype(np.int8) * 2 - 1
    k_signs = mx.array(k_signs)
    # Broadcast KV-head signs to query heads
    k_signs = mx.repeat(k_signs, num_queries_per_kv, axis=1)

    # Unpack Q sign sketch [B, H_q, proj_dim]
    q_packed = np.array(q_proj_signs)
    q_flat = q_packed.reshape(-1, proj_dim // 8)
    q_bits = np.unpackbits(q_flat, axis=1, bitorder="little")[:, -proj_dim:]
    q_signs = q_bits.reshape(B, H_q, proj_dim).astype(np.int8) * 2 - 1
    q_signs = mx.array(q_signs)

    match_score = mx.sum(q_signs[:, :, None, None, :] * k_signs, axis=-1)
    q_norm = mx.sqrt(mx.sum(q * q, axis=-1))
    norm_E = qjl_payload.norms
    # Broadcast residual norms from KV heads to query heads
    norm_E = mx.repeat(norm_E, num_queries_per_kv, axis=1)
    correction = (norm_E * q_norm[:, :, None, None]) * (match_score / proj_dim)
    return correction.reshape(B, H_q, S * L)

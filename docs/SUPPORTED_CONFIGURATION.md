# TurboPolar Supported Configuration

This document defines the narrow configuration contract that TurboPolar currently supports. Anything outside this scope is unsupported and must raise a clear error.

## Supported runtime

- **Framework:** MLX + mlx-lm
- **Platform:** Apple Silicon with Metal
- **Python:** ≥3.10
- **MLX:** ≥0.31.2

## Supported model

- **Architecture:** Llama-style GQA
- **Head dimension:** 128 only
- **Query heads:** any value divisible by the KV-head count
- **KV heads:** any value ≤ query heads that divides evenly
- **Block size:** 64 only

## Supported mode

- **Batch size:** 1
- **Operation:** single-batch autoregressive decode
- **Attention:** standard full causal GQA over full history
- **Value storage:** grouped int8 (`storage_mode="kv_quant"`)

## Unsupported features

The following are explicitly unsupported. Attempting to use them must raise a `NotImplementedError` or `ValueError`:

- Sliding-window attention
- Local attention
- Padding masks with unequal sequence lengths
- Custom attention bias
- Query length != 1 in fused decode
- Batch size != 1
- `head_dim` other than 128
- `block_size` other than 64
- `v_bits` other than 8
- `storage_mode` other than `"kv_quant"`
- QJL (`use_qjl=True`)
- Speculative decoding
- Continuous batching
- Multi-model serving

## Status

TurboPolar is an end-to-end research alpha. Promotion to `PROMOTED_EXPERIMENTAL` is locked until the following are proven by reproducible artifacts:

1. Fused compressed attention produces logits acceptably close to dense attention during autoregressive decode.
2. Measured device memory is reduced, not merely logical payload bytes.
3. Steady-state decode is non-regressive or faster at long context under a sound benchmark.

See `STATUS.md` for current progress.

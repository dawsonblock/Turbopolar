# Supported Configuration

TurboPolar operates under a narrow, explicitly validated contract.
Any deviation from this contract is unsupported and will raise an error.

## Platform

- **Hardware:** Apple Silicon (M-series GPUs)
- **OS:** macOS

## Runtime

- **Framework:** MLX >= 0.31.2
- **LLM library:** mlx-lm (exact verified revision only)

## Model

- **Architecture:** Llama-style GQA
- **Verified implementation:** one exact `mlx_lm` Llama class only
- **Head dimension:** 128
- **KV block size:** 64
- **Batch size:** 1
- **Attention:** full-history causal GQA
- **Decode query length:** 1 token
- **Mask:** None only

## Quantization formats

- **Key (K):** log-int8 radius + 8-bit angle codes
- **Value (V):** grouped int8 (`storage_mode="kv_quant"`)
- **QJL:** disabled

## Unsupported features

The following are explicitly unsupported in this release:

- `head_dim` other than 128
- `block_size` other than 64
- Batch size != 1
- Sliding-window attention
- Continuous batching
- Speculative decoding
- QJL
- Multi-user serving
- Non-Apple-Silicon platforms

## Validation

All entry points must call:

```python
from rfsn_v11.candidates.turbo_polar_config import validate_supported_configuration
validate_supported_configuration(config)
```

This function raises `ValueError` or `NotImplementedError` for any unsupported configuration.

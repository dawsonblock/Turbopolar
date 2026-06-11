# TurboPolar

A compressed KV cache for Llama-style models on MLX / Apple Silicon.

This is **alpha software**. It passes unit tests and runs Metal kernels natively on M2, but it has **not yet been validated on a real model**. Do not use it in production.

## What it does

- Compresses the key cache using polar quantization (magnitude + angle codes).
- Stores values in grouped int8 by default, with an optional dense float path.
- Implements custom Metal kernels for QK score and attention so the compressed cache can stay on the GPU.
- Targets Llama-style GQA models with `head_dim == 128` and `block_size == 64`.

## What it does not do

- Run arbitrary models. Only Llama-style GQA with the constraints above.
- Guarantee >2× KV compression yet. Honest reduction today is ~1.66–1.78×.
- Prove real-model quality. We have not measured logits, perplexity, or generation against dense baseline.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥3.10 and MLX ≥0.31.2.

## Run tests

```bash
make test
```

All 24 tests should pass. If a Metal kernel fails to compile, the bridge falls back to CPU and reports it.

## Quick example

```python
import mlx.core as mx
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime

cfg = TurboPolarConfig(
    num_q_heads=8,
    num_kv_heads=4,
    head_dim=128,
    block_size=64,
)
cache = TurboPolarKVCacheRuntime(cfg)

for i in range(100):
    k = mx.random.normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(mx.float16)
    v = mx.random.normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(mx.float16)
    cache.append(k, v)

print(cache.get_io_telemetry())
```

## Architecture

```
rfsn_v11/
├── candidates/          # Config and policy definitions
├── generation/          # Runtime KV cache
├── kernels/             # Metal kernels and CPU fallbacks
│   └── turbo_polar/
├── quant/
│   ├── polar/           # Key polar quantization encode/decode
│   ├── qjl/             # QJL score estimator (experimental, off)
│   └── v_quant/         # Value int8 quantization
└── tests/               # Unit tests
```

## Current limitations

- `head_dim` must be 64 or 128 and divisible by 32.
- `block_size` is fixed at 64.
- Only `v_bits == 8` is allowed.
- QJL is disabled by default because its score estimator is uncalibrated.
- Promotion to wider use is blocked until real-model gates pass.

## Promotion gates

TurboPolar stays behind `promotion_allowed=False` until all of these are true:

1. KV memory reduction ≥ 1.7×.
2. Logit cosine similarity vs dense ≥ 0.995.
3. Top-5 token overlap vs dense ≥ 0.95.
4. Perplexity delta vs dense ≤ 0.02.
5. Long-context decode speed ≥ dense baseline.
6. No tail / partial-block crashes.

See `STATUS.md` for the latest progress.

## License

MIT. See `LICENSE`.

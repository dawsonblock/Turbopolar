# TurboPolar

A **Memory-Tiering Cache for Extreme Contexts** on MLX / Apple Silicon.

TurboPolar v0.4.0.dev0 is an **Adaptive Tiered KV Cache**. It keeps the active working context in a fast dense hot window while archiving historical tokens into compressed polar pages. For contexts under 4K tokens, decode uses pure dense attention with zero overhead. At 16K+ tokens, the compressed cold tier activates, trading a controlled amount of speed for massive memory savings.

This is **research alpha software**. Real-model long-context evidence, speed benchmarking, and comparative-value analysis remain incomplete. Promotion remains locked. Do not use it in production.

## What it does

- **Adaptive Tiered Architecture** (Milestone 1): A dense hot window (up to 4K tokens) plus compressed polar cold pages for historical context. Warm cache tier planned for Milestone 3.
- **Polar quantization** for keys (magnitude + angle codes) and grouped int8 for values.
- **True Paged Dispatch**: Each compressed page is dispatched independently with lazy MLX fusion into a single GPU command buffer. No monolithic arena copy, no per-page Python state merge.
- **Batched flushing**: When the dense window fills, the oldest 2K tokens are compressed in one batch rather than one block at a time.
- Targets Llama-style GQA models with `head_dim == 64 or 128` and `block_size == 64`.

## Supported configuration

TurboPolar currently supports a narrow configuration:

- **Framework:** MLX + mlx-lm on Apple Silicon
- **Model:** Llama-style GQA
- **Head dimension:** 64 or 128
- **Block size:** 64
- **Value storage:** grouped int8 (`storage_mode="kv_quant"`)
- **Mode:** single-batch autoregressive decode
- **Attention:** standard full causal GQA
- **QJL:** disabled
- **Dense tail capacity:** 4096 tokens (configurable)
- **Flush batch size:** 2048 tokens (configurable)

Anything outside this scope is unsupported and will raise an error. See `docs/SUPPORTED_CONFIGURATION.md` for the full contract.

## What it does not do

- Run arbitrary models or architectures.
- Support `head_dim` other than 64 or 128.
- Support `block_size` other than 64.
- Support QJL, sliding-window attention, continuous batching, speculative decoding, or multi-user serving.
- Guarantee production readiness.

## Install

```bash
pip install -e ".[dev]"        # core + test dependencies
pip install -e ".[all]"        # includes mlx-lm for real-model benchmarks
```

Requires Python ≥3.10 and MLX ≥0.31.2. Real-model benchmarks also require `mlx-lm`.

## Run tests

```bash
make test
```

Three execution modes are supported: `REFERENCE` (CPU/Python reference), `METAL_STRICT` (Metal required, no fallback), and `DEVELOPMENT_AUTO` (try Metal, fall back to reference on failure). Promotion benchmarks use `METAL_STRICT`. Test count is generated from the test report; do not hardcode it.

## Run the real-model benchmarks

```bash
make bench MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make fused-bench MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make speed-matrix MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make memory-bench MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make cartesian-bench MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make promote MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

These compare dense KV-cache logits against TurboPolar on a real MLX model, measure decode speed across sequence lengths, probe actual allocator memory, compare against a Cartesian int8 baseline, and produce promotion evidence. Real-model long-context evidence remains incomplete. Promotion remains locked. See `benchmarks/README.md` for details.

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

```text
rfsn_v11/
├── candidates/          # Config and policy definitions
├── generation/          # Runtime KV cache
├── integrations/        # mlx_lm model adapters
├── kernels/             # Metal kernels and CPU fallbacks
│   └── turbo_polar/
├── quant/
│   ├── polar/           # Key polar quantization encode/decode
│   ├── qjl/             # QJL score estimator (experimental, disabled)
│   └── v_quant/         # Value int8 quantization
├── tests/               # Unit, kernel, integration, and governance tests
└── benchmarks/          # Real-model validation harnesses and baselines
```

## Current limitations

- `head_dim` must be 64 or 128.
- `block_size` is fixed at 64.
- Only `v_bits == 8` is allowed (asymmetric zero-point quantization is implemented in Python but requires a Metal kernel update to activate).
- Only `storage_mode == "kv_quant"` is allowed.
- QJL is disabled and unsupported.
- Single-batch decode only.
- Only one Llama implementation has been validated.
- Promotion is locked until Phases 1-4 pass the full evidence matrix.

## Development status

See `STATUS.md` for the latest progress and promotion criteria.

## License

MIT. See `LICENSE`.

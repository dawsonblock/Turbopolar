# TurboPolar Alpha 9

> Polar-quantized KV-cache runtime for efficient transformer inference on Apple Silicon.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-0.31.2+-orange.svg)](https://ml-explore.github.io/mlx/)
[![Metal](https://img.shields.io/badge/GPU-Apple%20Metal-purple.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TurboPolar is an experimental **key-value cache compression runtime** for decoder-only transformers. It represents key vectors in polar coordinates (radii + angles), bit-packs the angular codes, optionally stores a QJL residual sketch, and supports grouped int8 value quantization. The runtime is built on [MLX](https://ml-explore.github.io/mlx/) and ships with custom Metal kernels that run natively on Apple Silicon.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Storage Modes](#storage-modes)
- [Metal GPU Support](#metal-gpu-support)
- [Testing](#testing)
- [Compression & Accuracy](#compression--accuracy)
- [Project Structure](#project-structure)
- [Roadmap & Known Limitations](#roadmap--known-limitations)
- [License](#license)

---

## Overview

Standard transformer attention materialises full-precision key (K) and value (V) tensors for every token in the context. For long sequences this becomes the dominant memory cost. TurboPolar attacks this by:

1. **Polar key quantization**: each key vector is split into consecutive pairs, converted to polar `(radius, angle)`, and the angles are scalar-quantized with small bit budgets.
2. **Bit packing**: the per-pair angle codes are packed into bytes (4-bit level-1 + 2-bit deep angles).
3. **Grouped int8 V quantization**: values are compressed per 32-element group.
4. **Optional QJL residual sketch**: a random-projection sign sketch of the reconstruction residual can be stored and used as a dot-product correction term.
5. **Custom Metal kernels**: fused dequantization + QK score kernels and online softmax attention kernels avoid materialising full-precision K/V during generation.

The result is a ~1.7× honest KV-cache compression ratio at small accuracy cost, with a path toward 2×+ as lower-precision radii and packed 4-bit V are added.

---

## Key Features

| Feature | Status |
|---|---|
| Polar K quantization (radii + 4-bit/2-bit angles) | ✅ |
| Bit-packed angle codes | ✅ |
| Stateful incremental KV cache | ✅ |
| Grouped Query Attention (GQA) | ✅ |
| Optional QJL residual sketch | ✅ |
| 8-bit grouped V quantization | ✅ |
| Dense-V debug storage mode | ✅ |
| K-only storage mode | ✅ |
| Custom Metal kernels (QK + online attention) | ✅ M2 verified |
| CPU fallback for non-Metal environments | ✅ |
| Honest byte-level compression telemetry | ✅ |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/dawsonblock/Turbopolar.git
cd Turbopolar

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install MLX
pip install mlx==0.31.2

# Optional: install as an editable package
pip install -e .
```

No build step is required; the Metal kernels are compiled on first use by MLX.

---

## Quick Start

### 1. Create a configuration

```python
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig

config = TurboPolarConfig(
    head_dim=128,
    qjl_proj_dim=64,
    block_size=64,
    split_dim=64,
    num_q_heads=32,
    num_kv_heads=8,   # GQA: 4 query heads per KV head
    use_qjl=True,
    storage_mode="kv_quant",
)
```

### 2. Build the incremental cache

```python
import mlx.core as mx
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime

cache = TurboPolarKVCacheRuntime(config)

# Append one or more tokens
k_new = mx.random.normal(shape=[1, config.num_kv_heads, 1, config.head_dim])
v_new = mx.random.normal(shape=[1, config.num_kv_heads, 1, config.head_dim])
cache.append(k_new, v_new)

# Get the unified attention payload
block, quant_v, dense_v, qjl, actual_len = cache.get_blocks_for_attention()
print(f"Attendable tokens: {actual_len}")
```

### 3. Run fused attention on Metal

```python
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder

bridge = MetalKernelBridge()

# Compute packed query signs for QJL
qjl_encoder = QJLResidualEncoder(config)
q_proj = mx.matmul(q, qjl_encoder.W)
q_signs = q_proj >= 0
q_packed = q_signs.reshape(
    q.shape[0], q.shape[1], config.qjl_proj_dim // 8, 8
).astype(mx.uint8) * mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
q_packed = q_packed.sum(axis=-1).astype(mx.uint8)

output, trace = bridge.execute_online_attention_quant_v(
    q=q,
    block=block,
    quant_v=quant_v,
    qjl_payload=qjl,
    q_proj_signs=q_packed,
    config=config,
    actual_seq_len=actual_len,
    use_qjl=True,
)

print(trace)
# {'kernel_name': 'tqpolar_online_attention_quant_v',
#  'metal_used': True,
#  'fallback_used': False,
#  ...}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TurboPolar Runtime                       │
├─────────────────────────────────────────────────────────────┤
│  TurboPolarConfig                                            │
│  • validates dims, block_size, GQA, storage_mode             │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ PolarQuant   │ │ Grouped  │ │ QJLResidual  │
│ Encoder      │ │ VQuant   │ │ Encoder      │
│ (K)          │ │ (V)      │ │ (residual)   │
└──────┬───────┘ └────┬─────┘ └──────┬───────┘
       │              │              │
       ▼              ▼              ▼
┌─────────────────────────────────────────────┐
│     TurboPolarKVCacheRuntime                │
│  • incremental append / flush               │
│  • partial-tail padding                     │
│  • honest byte accounting                   │
└──────────────────────┬──────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌─────────────────┐      ┌──────────────────────┐
│ PolarQuantDecoder│      │ MetalKernelBridge    │
│ (4D/5D payloads) │      │ • fused QK kernels   │
└─────────────────┘      │ • online attention   │
                         │   (dense/quant V)    │
                         │ • CPU fallback       │
                         └──────────────────────┘
```

### Polar key encoding

For each key vector `x ∈ ℝ^D`:

1. Split into consecutive pairs `(x[2j], x[2j+1])`.
2. Convert to polar: `r_j = sqrt(x[2j]^2 + x[2j+1]^2)`, `θ_j = atan2(x[2j+1], x[2j])`.
3. Normalize `θ_j` to `[0, 1)` and quantize:
   - First `split_dim/2` pairs with `k_angle_bits_level1` bits (default 4).
   - Remaining pairs with `k_angle_bits_deep` bits (default 2).
4. Bit-pack the codes: 2 × 4-bit nibbles per byte, 4 × 2-bit pairs per byte.

At decode time the angles are reconstructed and the Cartesian pairs are regenerated.

### QJL residual sketch

If `use_qjl=True`, the residual `E = K_true − K_recon` is projected onto a random Gaussian matrix, the sign pattern is packed into bits, and the per-token residual norm is stored. At attention time a heuristic sign-agreement term is added to the QK score.

---

## Storage Modes

| Mode | K | V | QJL | Use case |
|---|---|---|---|---|
| `kv_quant` (default) | polar | int8 grouped | optional | production target |
| `dense_v_debug` | polar | fp16 dense | optional | debug / quality baseline |
| `k_only_first` | polar | none | none | K-only experiments |

Set via `TurboPolarConfig(storage_mode="...")`.

---

## Metal GPU Support

TurboPolar includes hand-written Metal kernels for:

- `tqpolar_fused_dequant_qk` — fused polar dequantization + QK dot-product.
- `tqpolar_fused_dequant_qk_qjl` — same + QJL correction.
- `tqpolar_online_attention_dense_v` — online softmax attention with dense V.
- `tqpolar_online_attention_quant_v` — online softmax attention with grouped int8 V.

The bridge auto-detects whether the local MLX/Metal runtime treats `grid` as threadgroups or as total threads (MLX 0.31.2 uses the latter) and scales launch dimensions accordingly. On Apple Silicon with a working Metal runtime the kernels run without CPU fallback.

Verified on **Mac M2** with MLX 0.31.2.

---

## Testing

```bash
python3 -m unittest discover -s tests -v
```

Expected output on Metal hardware:

```text
Ran 24 tests in ~0.4s
OK
```

The suite covers:
- Polar bit-packing round-trip
- Offline reconstruction gates
- Fused QK precision (with fp16 tolerance)
- QJL sign packing and correlation
- Online attention with dense/quantized V
- GQA correctness
- Incremental cache shapes and partial-tail attention
- Storage modes
- Byte-level telemetry
- Promotion gate evaluation

---

## Compression & Accuracy

Honest compression ratios (including radii, packed angles, int8 V, and optional QJL):

| head_dim | QJL | ratio |
|---|---|---|
| 128 | on | ~1.72× |
| 128 | off | ~1.78× |
| 64 | on | ~1.66× |
| 64 | off | ~1.73× |

The promotion gate currently requires ≥1.65× and remains hardcoded as **not promoted** while further quality gates are being validated.

Accuracy:
- Metal QK scores vs dense fp32 reference: cosine similarity ≥0.999, max absolute error ~5e-3 (fp16 accumulation).
- Quantized-V attention vs dequantized reference: cosine similarity ≥0.999.
- QJL correction is a heuristic; it is structurally validated and improves correlation but is not yet a calibrated residual estimator.

---

## Project Structure

```text
.
├── rfsn_v11/
│   ├── candidates/          # Config, adapter, metrics, trace
│   ├── generation/          # Incremental KV cache runtime
│   ├── kernels/
│   │   └── turbo_polar/     # Metal shaders + Python bridge
│   ├── quant/
│   │   ├── polar/           # Polar K encoder/decoder/payload
│   │   ├── qjl/             # QJL encoder + dot-product estimate
│   │   └── v_quant/         # Grouped int8 V quantizer
│   └── __init__.py
├── tests/
│   ├── benchmarks/          # Kernel-level benchmarks
│   ├── test_promotion_gate.py
│   └── test_turbo_polar_cache_runtime.py
├── LICENSE
└── README.md
```

---

## Roadmap & Known Limitations

- **True >2× compression** requires lower-precision radii, packed 4-bit V, or dropping stored QJL.
- **QJL correction** is a heuristic sign-agreement estimator, not yet a calibrated residual dot-product estimator.
- **Metal kernel portability** has been verified on M2 + MLX 0.31.2; other MLX versions may need dispatch-tuning.
- **Promotion gate** is intentionally locked (`promotion_allowed=False`) until all required gates pass.

---

## License

MIT License — see [LICENSE](LICENSE).

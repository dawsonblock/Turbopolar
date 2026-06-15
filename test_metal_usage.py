"""Test to check if Metal kernels are actually being used."""

import mlx.core as mx
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge, KernelExecutionStats
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime

# Reset stats
stats = KernelExecutionStats()
bridge = MetalKernelBridge()

# Create a simple cache and append some data
config = TurboPolarConfig(
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    block_size=64,
    storage_mode="kv_quant",
    use_int8_radii=True,
    k_angle_bits_deep=8,
    execution_mode="metal_strict",
)

cache = TurboPolarKVCacheRuntime(config)

# Generate some test data
B, H, T, D = 1, 8, 64, 128
k_data = mx.random.normal((B, H, T, D), dtype=mx.float16)
v_data = mx.random.normal((B, H, T, D), dtype=mx.float16)

# Append data
cache.append(k_data, v_data)

# Check stats
print("Metal Kernel Execution Statistics:")
print(f"  Attention invocations: {stats.attention_invocations}")
print(f"  Compressed page dispatches: {stats.compressed_page_dispatches}")
print(f"  Compressed page failures: {stats.compressed_page_failures}")
print(f"  Compressed page fallbacks: {stats.compressed_page_fallbacks}")
print(f"  Dense tail dispatches: {stats.dense_tail_dispatches}")
print(f"  Dense tail failures: {stats.dense_tail_failures}")
print(f"  Dense tail fallbacks: {stats.dense_tail_fallbacks}")
print(f"  Full attention fallbacks: {stats.full_attention_fallbacks}")
print(f"  Total fallback calls: {stats.fallback_calls}")

# Check threadgroup support
print(f"\nThreadgroup supported: {bridge.threadgroup_supported}")
print(f"Grid semantics: {bridge.grid_semantics}")

if stats.fallback_calls > 0:
    print("\n⚠️  WARNING: Fallbacks detected! Metal kernels may not be working.")
    print("This could explain the severe performance degradation.")
else:
    print("\n✅ No fallbacks detected. Metal kernels are being used.")
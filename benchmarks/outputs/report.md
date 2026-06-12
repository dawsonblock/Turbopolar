# TurboPolar Benchmark Report

**Model:** `/Users/dawsonblock/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`  
**MLX:** 0.31.2  
**mlx_lm:** 0.31.3  
**dtype:** mlx.core.bfloat16  
**seed:** 42  
**Prompts:** 4  

## Promotion Gates

| Gate | Threshold | Actual | Passed |
|------|-----------|--------|--------|
| KV compression ratio | ≥ 1.70× | 1.000× | ❌ |
| Logit cosine | ≥ 0.9950 | 0.0000 | ❌ |
| Top-5 overlap | ≥ 0.95 | 0.0000 | ❌ |
| Top-10 overlap | ≥ 0.95 | 0.0000 | ❌ |
| Perplexity delta | ≤ 0.020 | inf | ❌ |
| Decode speed ratio | ≥ 1.00× | 0.000× | ❌ |

**Promotion allowed:** NO
**Visible drift:** YES

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| logit_cosine | 0.000000 |
| top5_overlap | 0.000000 |
| top10_overlap | 0.000000 |
| kl_divergence | inf |
| perplexity_delta | inf |
| compression_ratio | 1.000000 |
| peak_kv_bytes_dense | 3145728 |
| peak_kv_bytes_turbo | 135168 |
| decode_speed_dense_tok_per_sec | 0.000000 |
| decode_speed_turbo_tok_per_sec | 0.000000 |
| decode_speed_ratio | 0.000000 |

## Per-Prompt Results

| Prompt tokens | Cosine | Top-5 | Top-10 | PPL Δ | Compression |
|---------------|--------|-------|--------|-------|-------------|
| 5 | 0.0000 | 0.0000 | 0.0000 | inf | 1.000× |
| 7 | 0.0000 | 0.0000 | 0.0000 | inf | 1.000× |
| 11 | 0.0000 | 0.0000 | 0.0000 | inf | 1.000× |
| 10 | 0.0000 | 0.0000 | 0.0000 | inf | 1.000× |

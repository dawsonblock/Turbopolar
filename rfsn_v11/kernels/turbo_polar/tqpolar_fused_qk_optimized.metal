#include <metal_stdlib>
using namespace metal;

// OPTIMIZATION: Fast math approximations for better performance
// Use fast_exp instead of exp for better performance on Apple Silicon
inline half fast_exp_approx(half x) {
    return exp(x); // Metal compiler may optimize this
}

// OPTIMIZATION: Precomputed constants to avoid repeated calculations
constant half TWO_PI = 2.0h * M_PI_H;
constant half PI = M_PI_H;

kernel void tqpolar_fused_dequant_qk_optimized(
    device const half* q                     [[buffer(0)]],
    device const half* polar_radii           [[buffer(1)]],
    device const int8_t* polar_radii_i8      [[buffer(2)]],
    device const half* radii_scales          [[buffer(3)]],
    device const uchar* angle_codes_l1       [[buffer(4)]],
    device const uchar* angle_codes_deep     [[buffer(5)]],
    device half* scores                      [[buffer(6)]],
    constant uint& head_dim                  [[buffer(7)]],
    constant uint& split_dim                 [[buffer(8)]],
    constant uint& block_size                [[buffer(9)]],
    constant half& l1_scale                  [[buffer(10)]],
    constant half& deep_scale                [[buffer(11)]],
    constant half& attention_scale           [[buffer(12)]],
    constant uint& num_queries_per_kv        [[buffer(13)]],
    constant uint& int8_radii                [[buffer(14)]],
    constant uint& log_radii                 [[buffer(15)]],
    constant uint& l1_bits                   [[buffer(16)]],
    constant uint& deep_bits                 [[buffer(17)]],
    device const uint* strides               [[buffer(18)]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_index_in_threadgroup]])
{
    uint b = tgid.x;
    uint q_head = tgid.y;
    uint kv_head = q_head / num_queries_per_kv;
    uint s = tgid.z;
    uint half_d = head_dim / 2;
    uint split_half_d = split_dim / 2;

    // OPTIMIZATION: Precompute stride calculations outside loops
    uint stride_q_b = strides[0],  stride_q_h = strides[1];
    uint stride_r_b = strides[2],  stride_r_h = strides[3],  stride_r_s = strides[4],  stride_r_l = strides[5];
    uint stride_rs_b = strides[6], stride_rs_h = strides[7], stride_rs_s = strides[8];
    uint stride_c1_b = strides[9],  stride_c1_h = strides[10], stride_c1_s = strides[11], stride_c1_l = strides[12];
    uint stride_cd_b = strides[13], stride_cd_h = strides[14], stride_cd_s = strides[15], stride_cd_l = strides[16];
    uint stride_s_b = strides[17], stride_s_h = strides[18], stride_s_tok = strides[19];

    // OPTIMIZATION: Precompute base offsets to reduce repeated calculations
    uint base_offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s;
    uint base_offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s;
    uint base_offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s;
    uint base_offset_q = b * stride_q_b + q_head * stride_q_h;
    uint base_offset_rs = b * stride_rs_b + kv_head * stride_rs_h + s * stride_rs_s;
    uint base_offset_s = b * stride_s_b + q_head * stride_s_h + s * block_size * stride_s_tok;

    // OPTIMIZATION: Move branch conditions outside the inner loop
    bool use_int8_radii = (int8_radii != 0);
    bool use_log_radii = (log_radii != 0);
    bool l1_is_8bit = (l1_bits == 8);
    bool deep_is_8bit = (deep_bits == 8);
    bool deep_is_4bit = (deep_bits == 4);

    for (uint l = 0; l < block_size; l++) {
        half private_sum = 0.0h;
        
        // OPTIMIZATION: Precompute per-loop offsets
        uint offset_r_l = base_offset_r + l * stride_r_l;
        uint offset_c1_l = base_offset_c1 + l * stride_c1_l;
        uint offset_cd_l = base_offset_cd + l * stride_cd_l;
        uint dest_idx = base_offset_s + l * stride_s_tok;

        for (uint j = tid; j < half_d; j += 32) {
            uint offset_r = offset_r_l + j;

            // OPTIMIZATION: Branchless radii extraction
            half r;
            if (use_int8_radii) {
                int8_t code = polar_radii_i8[offset_r];
                half scale = radii_scales[base_offset_rs];
                half value = static_cast<half>(code) * scale;
                r = use_log_radii ? fast_exp_approx(value) : value;
            } else {
                r = polar_radii[offset_r];
            }

            // OPTIMIZATION: Branchless angle code extraction
            half norm_angle;
            bool in_l1 = (j < split_half_d);
            
            if (in_l1) {
                uint offset_c1 = offset_c1_l;
                uchar code = l1_is_8bit ? angle_codes_l1[offset_c1 + j] :
                              ((j & 1) == 0) ? (angle_codes_l1[offset_c1 + (j >> 1)] & 0x0F) :
                                                  ((angle_codes_l1[offset_c1 + (j >> 1)] >> 4) & 0x0F);
                norm_angle = static_cast<half>(code) / l1_scale;
            } else {
                uint rel_j = j - split_half_d;
                uint offset_cd = offset_cd_l;
                uchar code;
                if (deep_is_8bit) {
                    code = angle_codes_deep[offset_cd + rel_j];
                } else if (deep_is_4bit) {
                    uchar byte = angle_codes_deep[offset_cd + (rel_j >> 1)];
                    code = ((rel_j & 1) == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                } else {
                    uchar byte = angle_codes_deep[offset_cd + (rel_j >> 2)];
                    uint shift = (rel_j & 3) * 2;
                    code = (byte >> shift) & 0x03;
                }
                norm_angle = static_cast<half>(code) / deep_scale;
            }

            // OPTIMIZATION: Use precomputed constants
            half angle = (norm_angle * TWO_PI) - PI;
            
            // OPTIMIZATION: Use sincos for better performance (single call)
            half sin_a, cos_a;
            sincos(angle, sin_a, cos_a);
            half k_x = r * cos_a;
            half k_y = r * sin_a;

            half q_x = q[base_offset_q + j * 2];
            half q_y = q[base_offset_q + j * 2 + 1];

            private_sum += (q_x * k_x + q_y * k_y) * attention_scale;
        }

        half total_score = simd_sum(private_sum);
        if (tid == 0) {
            scores[dest_idx] = total_score;
        }
    }
}
#include <metal_stdlib>
using namespace metal;

kernel void tqpolar_fused_dequant_qk(
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

    uint stride_q_b = strides[0],  stride_q_h = strides[1];
    uint stride_r_b = strides[2],  stride_r_h = strides[3],  stride_r_s = strides[4],  stride_r_l = strides[5];
    uint stride_rs_b = strides[6], stride_rs_h = strides[7], stride_rs_s = strides[8];
    uint stride_c1_b = strides[9],  stride_c1_h = strides[10],  stride_c1_s = strides[11],  stride_c1_l = strides[12];
    uint stride_cd_b = strides[13], stride_cd_h = strides[14], stride_cd_s = strides[15], stride_cd_l = strides[16];
    uint stride_s_b = strides[17], stride_s_h = strides[18], stride_s_tok = strides[19];

    for (uint l = 0; l < block_size; l++) {
        half private_sum = 0.0h;
        for (uint j = tid; j < half_d; j += 32) {
            uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;

            half r;
            if (int8_radii == 0) {
                r = polar_radii[offset_r];
            } else {
                int8_t code = polar_radii_i8[offset_r];
                half scale = radii_scales[b * stride_rs_b + kv_head * stride_rs_h + s * stride_rs_s];
                half value = static_cast<half>(code) * scale;
                r = (log_radii != 0) ? exp(value) : value;
            }

            half norm_angle;
            if (j < split_half_d) {
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l;
                uchar code;
                if (l1_bits == 8) {
                    code = angle_codes_l1[offset_c1 + j];
                } else {
                    uchar byte = angle_codes_l1[offset_c1 + j / 2];
                    code = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                }
                norm_angle = static_cast<half>(code) / l1_scale;
            } else {
                uint rel_j = j - split_half_d;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l;
                uchar code;
                if (deep_bits == 8) {
                    code = angle_codes_deep[offset_cd + rel_j];
                } else if (deep_bits == 4) {
                    uchar byte = angle_codes_deep[offset_cd + rel_j / 2];
                    code = (rel_j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                } else {
                    uchar byte = angle_codes_deep[offset_cd + rel_j / 4];
                    uint shift = (rel_j % 4) * 2;
                    code = (byte >> shift) & 0x03;
                }
                norm_angle = static_cast<half>(code) / deep_scale;
            }

            half angle = (norm_angle * 2.0h * M_PI_H) - M_PI_H;
            half k_x = r * cos(angle);
            half k_y = r * sin(angle);

            half q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
            half q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];

            private_sum += (q_x * k_x + q_y * k_y) * attention_scale;
        }

        half total_score = simd_sum(private_sum);
        if (tid == 0) {
            uint dest_idx = b * stride_s_b + q_head * stride_s_h + (s * block_size + l) * stride_s_tok;
            scores[dest_idx] = total_score;
        }
    }
}

kernel void tqpolar_fused_dequant_qk_qjl(
    device const half* q                     [[buffer(0)]],
    device const half* polar_radii           [[buffer(1)]],
    device const int8_t* polar_radii_i8      [[buffer(2)]],
    device const half* radii_scales          [[buffer(3)]],
    device const uchar* angle_codes_l1       [[buffer(4)]],
    device const uchar* angle_codes_deep     [[buffer(5)]],
    device const uchar* qjl_packed_signs     [[buffer(6)]],
    device const half* qjl_norms             [[buffer(7)]],
    device const uchar* q_proj_signs         [[buffer(8)]],
    device half* scores                      [[buffer(9)]],
    constant uint& head_dim                  [[buffer(10)]],
    constant uint& split_dim                 [[buffer(11)]],
    constant uint& block_size                [[buffer(12)]],
    constant uint& qjl_proj_dim              [[buffer(13)]],
    constant half& l1_scale                  [[buffer(14)]],
    constant half& deep_scale                [[buffer(15)]],
    constant half& attention_scale           [[buffer(16)]],
    constant uint& num_queries_per_kv        [[buffer(17)]],
    constant uint& int8_radii                [[buffer(18)]],
    constant uint& log_radii                 [[buffer(19)]],
    constant uint& l1_bits                   [[buffer(20)]],
    constant uint& deep_bits                 [[buffer(21)]],
    device const uint* strides               [[buffer(22)]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_index_in_threadgroup]])
{
    uint b = tgid.x;
    uint q_head = tgid.y;
    uint kv_head = q_head / num_queries_per_kv;
    uint s = tgid.z;
    uint half_d = head_dim / 2;
    uint split_half_d = split_dim / 2;
    uint qjl_bytes = qjl_proj_dim / 8;

    uint stride_q_b   = strides[0];  uint stride_q_h   = strides[1];
    uint stride_r_b   = strides[2];  uint stride_r_h   = strides[3];  uint stride_r_s   = strides[4];  uint stride_r_l = strides[5];
    uint stride_rs_b  = strides[6];  uint stride_rs_h  = strides[7];  uint stride_rs_s  = strides[8];
    uint stride_c1_b  = strides[9];  uint stride_c1_h  = strides[10]; uint stride_c1_s  = strides[11]; uint stride_c1_l = strides[12];
    uint stride_cd_b  = strides[13]; uint stride_cd_h  = strides[14]; uint stride_cd_s  = strides[15]; uint stride_cd_l = strides[16];
    uint stride_qjl_b = strides[17]; uint stride_qjl_h = strides[18]; uint stride_qjl_s = strides[19]; uint stride_qjl_l = strides[20];
    uint stride_qn_b  = strides[21]; uint stride_qn_h  = strides[22]; uint stride_qn_s  = strides[23]; uint stride_qn_l = strides[24];
    uint stride_qp_b  = strides[25]; uint stride_qp_h  = strides[26];
    uint stride_s_b   = strides[27]; uint stride_s_h   = strides[28]; uint stride_s_tok = strides[29];

    threadgroup half shared_q_norm[1];
    if (tid == 0) {
        half q_sum = 0.0h;
        for (uint d = 0; d < head_dim; d++) {
            half val = q[b * stride_q_b + q_head * stride_q_h + d];
            q_sum += val * val;
        }
        shared_q_norm[0] = sqrt(q_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half q_norm = shared_q_norm[0];

    for (uint l = 0; l < block_size; l++) {
        half private_sum = 0.0h;
        for (uint j = tid; j < half_d; j += 32) {
            uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;

            half r;
            if (int8_radii == 0) {
                r = polar_radii[offset_r];
            } else {
                int8_t code = polar_radii_i8[offset_r];
                half scale = radii_scales[b * stride_rs_b + kv_head * stride_rs_h + s * stride_rs_s];
                half value = static_cast<half>(code) * scale;
                r = (log_radii != 0) ? exp(value) : value;
            }

            half norm_angle;
            if (j < split_half_d) {
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l;
                uchar code;
                if (l1_bits == 8) {
                    code = angle_codes_l1[offset_c1 + j];
                } else {
                    uchar byte = angle_codes_l1[offset_c1 + j / 2];
                    code = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                }
                norm_angle = static_cast<half>(code) / l1_scale;
            } else {
                uint rel_j = j - split_half_d;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l;
                uchar code;
                if (deep_bits == 8) {
                    code = angle_codes_deep[offset_cd + rel_j];
                } else if (deep_bits == 4) {
                    uchar byte = angle_codes_deep[offset_cd + rel_j / 2];
                    code = (rel_j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                } else {
                    uchar byte = angle_codes_deep[offset_cd + rel_j / 4];
                    uint shift = (rel_j % 4) * 2;
                    code = (byte >> shift) & 0x03;
                }
                norm_angle = static_cast<half>(code) / deep_scale;
            }

            half angle = (norm_angle * 2.0h * M_PI_H) - M_PI_H;
            half k_x = r * cos(angle);
            half k_y = r * sin(angle);

            half q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
            half q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];

            private_sum += (q_x * k_x + q_y * k_y) * attention_scale;
        }

        uint local_hamming = 0;
        for (uint byte_idx = tid; byte_idx < qjl_bytes; byte_idx += 32) {
            uint offset_qjl = b * stride_qjl_b + kv_head * stride_qjl_h + s * stride_qjl_s + l * stride_qjl_l + byte_idx;
            uchar k_byte = qjl_packed_signs[offset_qjl];
            uchar q_byte = q_proj_signs[b * stride_qp_b + q_head * stride_qp_h + byte_idx];
            local_hamming += popcount(static_cast<uint>(k_byte ^ q_byte));
        }

        half total_polar_score = simd_sum(private_sum);
        uint total_hamming_dist = simd_sum(local_hamming);

        if (tid == 0) {
            half match_score = static_cast<half>(qjl_proj_dim) - 2.0h * static_cast<half>(total_hamming_dist);
            half norm_E = qjl_norms[b * stride_qn_b + kv_head * stride_qn_h + s * stride_qn_s + l * stride_qn_l];
            half sign_corr = match_score / static_cast<half>(qjl_proj_dim);
            half cos_est = sin((M_PI_H / 2.0h) * sign_corr);
            half qjl_correction = (norm_E * q_norm) * cos_est;
            uint dest_idx = b * stride_s_b + q_head * stride_s_h + (s * block_size + l) * stride_s_tok;
            scores[dest_idx] = total_polar_score + qjl_correction;
        }
    }
}

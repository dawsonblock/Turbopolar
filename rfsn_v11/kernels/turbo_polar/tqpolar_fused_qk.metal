#include <metal_stdlib>
using namespace metal;

kernel void tqpolar_fused_dequant_qk(
    device const half* q                  [[buffer(0)]],
    device const half* polar_radii        [[buffer(1)]],
    device const uchar* angle_codes_l1     [[buffer(2)]],
    device const uchar* angle_codes_deep   [[buffer(3)]],
    device half* scores                    [[buffer(4)]],
    constant uint& head_dim                [[buffer(5)]],
    constant uint& split_dim               [[buffer(6)]],
    constant uint& block_size              [[buffer(7)]],
    constant half& l1_scale                [[buffer(8)]],
    constant half& deep_scale              [[buffer(9)]],
    constant half& attention_scale         [[buffer(10)]],
    constant uint& num_queries_per_kv      [[buffer(11)]],
    device const uint* strides             [[buffer(12)]],
    uint3 tgid                             [[threadgroup_position_in_grid]],
    uint tid                               [[thread_index_in_threadgroup]])
{
    uint b = tgid.x;
    uint q_head = tgid.y;
    uint kv_head = q_head / num_queries_per_kv;
    uint s = tgid.z;
    uint half_d = head_dim / 2;
    uint split_half_d = split_dim / 2;

    uint stride_q_b = strides[0],  stride_q_h = strides[1];
    uint stride_r_b = strides[2],  stride_r_h = strides[3],  stride_r_s = strides[4],  stride_r_l = strides[5];
    uint stride_c1_b = strides[6],  stride_c1_h = strides[7],  stride_c1_s = strides[8],  stride_c1_l = strides[9];
    uint stride_cd_b = strides[10], stride_cd_h = strides[11], stride_cd_s = strides[12], stride_cd_l = strides[13];
    uint stride_s_b = strides[14], stride_s_h = strides[15], stride_s_tok = strides[16];

    for (uint l = 0; l < block_size; l++) {
        half private_sum = 0.0h;
        for (uint j = tid; j < half_d; j += 32) {
            uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;
            half r = polar_radii[offset_r];

            half norm_angle = 0.0h;
            if (j < split_half_d) {
                uint l1_byte_idx = j / 2;
                uint l1_nibble = j % 2;
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l + l1_byte_idx;
                uchar l1_byte = angle_codes_l1[offset_c1];
                uchar l1_code = (l1_nibble == 0) ? (l1_byte & 0x0F) : ((l1_byte >> 4) & 0x0F);
                norm_angle = static_cast<half>(l1_code) / l1_scale;
            } else {
                uint rel_j = j - split_half_d;
                uint deep_byte_idx = rel_j / 2;
                uint deep_nibble = rel_j % 2;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l + deep_byte_idx;
                uchar deep_byte = angle_codes_deep[offset_cd];
                uchar deep_code = (deep_nibble == 0) ? (deep_byte & 0x0F) : ((deep_byte >> 4) & 0x0F);
                norm_angle = static_cast<half>(deep_code) / deep_scale;
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
    device const half* q                  [[buffer(0)]],
    device const half* polar_radii        [[buffer(1)]],
    device const uchar* angle_codes_l1     [[buffer(2)]],
    device const uchar* angle_codes_deep   [[buffer(3)]],
    device const uchar* qjl_packed_signs   [[buffer(4)]],
    device const half* qjl_norms           [[buffer(5)]],
    device const uchar* q_proj_signs       [[buffer(6)]],
    device half* scores                    [[buffer(7)]],
    constant uint& head_dim                [[buffer(8)]],
    constant uint& split_dim               [[buffer(9)]],
    constant uint& block_size              [[buffer(10)]],
    constant uint& qjl_proj_dim            [[buffer(11)]],
    constant half& l1_scale                [[buffer(12)]],
    constant half& deep_scale              [[buffer(13)]],
    constant half& attention_scale         [[buffer(14)]],
    constant uint& num_queries_per_kv      [[buffer(15)]],
    device const uint* strides             [[buffer(16)]],
    uint3 tgid                             [[threadgroup_position_in_grid]],
    uint tid                               [[thread_index_in_threadgroup]])
{
    uint b = tgid.x;
    uint q_head = tgid.y;
    uint kv_head = q_head / num_queries_per_kv;
    uint s = tgid.z;
    uint half_d = head_dim / 2;
    uint split_half_d = split_dim / 2;
    uint qjl_bytes = qjl_proj_dim / 8;

    uint stride_q_b = strides[0];  uint stride_q_h = strides[1];
    uint stride_r_b = strides[2];  uint stride_r_h = strides[3];  uint stride_r_s = strides[4];  uint stride_r_l = strides[5];
    uint stride_c1_b = strides[6];  uint stride_c1_h = strides[7];  uint stride_c1_s = strides[8];  uint stride_c1_l = strides[9];
    uint stride_cd_b = strides[10]; uint stride_cd_h = strides[11]; uint stride_cd_s = strides[12]; uint stride_cd_l = strides[13];
    uint stride_qjl_b = strides[14]; uint stride_qjl_h = strides[15]; uint stride_qjl_s = strides[16]; uint stride_qjl_l = strides[17];
    uint stride_qn_b = strides[18]; uint stride_qn_h = strides[19]; uint stride_qn_s = strides[20]; uint stride_qn_l = strides[21];
    uint stride_qp_b = strides[22]; uint stride_qp_h = strides[23];
    uint stride_s_b = strides[24]; uint stride_s_h = strides[25]; uint stride_s_tok = strides[26];

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
            half r = polar_radii[offset_r];

            half norm_angle = 0.0h;
            if (j < split_half_d) {
                uint l1_byte_idx = j / 2;
                uint l1_nibble = j % 2;
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l + l1_byte_idx;
                uchar l1_byte = angle_codes_l1[offset_c1];
                uchar l1_code = (l1_nibble == 0) ? (l1_byte & 0x0F) : ((l1_byte >> 4) & 0x0F);
                norm_angle = static_cast<half>(l1_code) / l1_scale;
            } else {
                uint rel_j = j - split_half_d;
                uint deep_byte_idx = rel_j / 2;
                uint deep_nibble = rel_j % 2;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l + deep_byte_idx;
                uchar deep_byte = angle_codes_deep[offset_cd];
                uchar deep_code = (deep_nibble == 0) ? (deep_byte & 0x0F) : ((deep_byte >> 4) & 0x0F);
                norm_angle = static_cast<half>(deep_code) / deep_scale;
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
            half qjl_correction = (norm_E * q_norm) * (match_score / static_cast<half>(qjl_proj_dim));
            uint dest_idx = b * stride_s_b + q_head * stride_s_h + (s * block_size + l) * stride_s_tok;
            scores[dest_idx] = total_polar_score + qjl_correction;
        }
    }
}

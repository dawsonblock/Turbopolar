#include <metal_stdlib>
using namespace metal;

inline half unpack_bit_online(device const uchar* packed_signs, uint offset, uint bit_idx) {
    uchar byte_val = packed_signs[offset + (bit_idx / 8)];
    uchar bit_mask = 1 << (bit_idx % 8);
    return (byte_val & bit_mask) ? 1.0h : -1.0h;
}

kernel void tqpolar_online_attention_dense_v(
    device const half* q                  [[buffer(0)]],
    device const half* polar_radii        [[buffer(1)]],
    device const uchar* angle_codes_l1     [[buffer(2)]],
    device const uchar* angle_codes_deep   [[buffer(3)]],
    device const half* v_dense             [[buffer(4)]],
    device const uchar* qjl_packed_signs   [[buffer(5)]],
    device const half* qjl_norms           [[buffer(6)]],
    device const uchar* q_proj_signs       [[buffer(7)]],
    device half* output                    [[buffer(8)]],
    constant uint& head_dim                [[buffer(9)]],
    constant uint& split_dim               [[buffer(10)]],
    constant uint& block_size              [[buffer(11)]],
    constant uint& total_blocks            [[buffer(12)]],
    constant uint& qjl_proj_dim            [[buffer(13)]],
    constant uint& use_qjl                 [[buffer(14)]],
    constant half& l1_scale                [[buffer(15)]],
    constant half& deep_scale              [[buffer(16)]],
    constant half& attention_scale         [[buffer(17)]],
    device const uint* strides             [[buffer(18)]],
    constant uint& actual_seq_len          [[buffer(19)]],
    constant uint& num_queries_per_kv      [[buffer(20)]],
    uint3 tgid                             [[threadgroup_position_in_grid]],
    uint tid                               [[thread_index_in_threadgroup]])
{
    uint b = tgid.x;
    uint q_head = tgid.y;
    uint kv_head = q_head / num_queries_per_kv;
    uint half_d = head_dim / 2;
    uint split_half_d = split_dim / 2;
    uint qjl_bytes = qjl_proj_dim / 8;
    uint num_elements_per_thread = head_dim / 32;

    uint stride_q_b   = strides[0];  uint stride_q_h   = strides[1];
    uint stride_r_b   = strides[2];  uint stride_r_h   = strides[3];  uint stride_r_s   = strides[4];  uint stride_r_l = strides[5];
    uint stride_c1_b  = strides[6];  uint stride_c1_h  = strides[7];  uint stride_c1_s  = strides[8];  uint stride_c1_l = strides[9];
    uint stride_cd_b  = strides[10]; uint stride_cd_h  = strides[11]; uint stride_cd_s  = strides[12]; uint stride_cd_l = strides[13];
    uint stride_v_b   = strides[14]; uint stride_v_h   = strides[15]; uint stride_v_s   = strides[16]; uint stride_v_l = strides[17];
    uint stride_qjl_b = strides[18]; uint stride_qjl_h = strides[19]; uint stride_qjl_s = strides[20]; uint stride_qjl_l = strides[21];
    uint stride_qn_b  = strides[22]; uint stride_qn_h  = strides[23]; uint stride_qn_s  = strides[24]; uint stride_qn_l = strides[25];
    uint stride_qp_b  = strides[26]; uint stride_qp_h  = strides[27];
    uint stride_o_b   = strides[28]; uint stride_o_h   = strides[29];

    float m_stat = -INFINITY;
    float l_stat = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    threadgroup float shared_scores[64];
    threadgroup float shared_q_norm[1];

    if (tid == 0 && use_qjl != 0) {
        float q_sum = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float val = q[b * stride_q_b + q_head * stride_q_h + d];
            q_sum += val * val;
        }
        shared_q_norm[0] = sqrt(q_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_norm = (use_qjl != 0) ? shared_q_norm[0] : 0.0f;

    for (uint s = 0; s < total_blocks; s++) {
        for (uint l = 0; l < block_size; l++) {
            uint global_tok_idx = s * block_size + l;
            if (global_tok_idx >= actual_seq_len) {
                if (tid == 0) {
                    shared_scores[l] = -INFINITY;
                }
                continue;
            }
            float private_sum = 0.0f;
            for (uint j = tid; j < half_d; j += 32) {
                uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;
                float r = polar_radii[offset_r];
                float norm_angle = 0.0f;
                if (j < split_half_d) {
                    uint l1_byte_idx = j / 2;
                    uint l1_nibble = j % 2;
                    uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l + l1_byte_idx;
                    uchar l1_byte = angle_codes_l1[offset_c1];
                    uchar l1_code = (l1_nibble == 0) ? (l1_byte & 0x0F) : ((l1_byte >> 4) & 0x0F);
                    norm_angle = float(l1_code) / float(l1_scale);
                } else {
                    uint rel_j = j - split_half_d;
                    uint deep_byte_idx = rel_j / 4;
                    uint deep_pair = rel_j % 4;
                    uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l + deep_byte_idx;
                    uchar deep_byte = angle_codes_deep[offset_cd];
                    uchar deep_code = (deep_byte >> (deep_pair * 2)) & 0x03;
                    norm_angle = float(deep_code) / float(deep_scale);
                }
                float angle = (norm_angle * 2.0f * M_PI_F) - M_PI_F;
                float k_x = r * cos(angle);
                float k_y = r * sin(angle);
                float q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
                float q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];
                private_sum += (q_x * k_x + q_y * k_y) * float(attention_scale);
            }
            uint local_hamming = 0;
            if (use_qjl != 0) {
                for (uint byte_idx = tid; byte_idx < qjl_bytes; byte_idx += 32) {
                    uint offset_qjl = b * stride_qjl_b + kv_head * stride_qjl_h + s * stride_qjl_s + l * stride_qjl_l + byte_idx;
                    uchar k_byte = qjl_packed_signs[offset_qjl];
                    uchar q_byte = q_proj_signs[b * stride_qp_b + q_head * stride_qp_h + byte_idx];
                    local_hamming += popcount(static_cast<uint>(k_byte ^ q_byte));
                }
            }
            float total_polar_score = simd_sum(private_sum);
            uint total_hamming_dist = simd_sum(local_hamming);
            if (tid == 0) {
                if (use_qjl != 0) {
                    float match_score = float(qjl_proj_dim) - 2.0f * float(total_hamming_dist);
                    float norm_E = qjl_norms[b * stride_qn_b + kv_head * stride_qn_h + s * stride_qn_s + l * stride_qn_l];
                    float qjl_correction = (norm_E * q_norm) * (match_score / float(qjl_proj_dim));
                    shared_scores[l] = total_polar_score + qjl_correction;
                } else {
                    shared_scores[l] = total_polar_score;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_max = -INFINITY;
        for (uint l = 0; l < block_size; l++) {
            block_max = max(block_max, shared_scores[l]);
        }
        float m_new = max(m_stat, block_max);
        float alpha = exp(m_stat - m_new);
        float l_block = 0.0f;
        for (uint l = 0; l < block_size; l++) {
            uint global_tok_idx = s * block_size + l;
            if (global_tok_idx < actual_seq_len) {
                l_block += exp(shared_scores[l] - m_new);
            }
        }
        float l_new = l_stat * alpha + l_block;

        for (uint k = 0; k < num_elements_per_thread; k++) {
            uint d = tid + k * 32;
            float v_sum = 0.0f;
            for (uint l = 0; l < block_size; l++) {
                uint global_tok_idx = s * block_size + l;
                if (global_tok_idx < actual_seq_len) {
                    float p = exp(shared_scores[l] - m_new);
                    uint offset_v = b * stride_v_b + kv_head * stride_v_h + s * stride_v_s + l * stride_v_l + d;
                    v_sum += p * float(v_dense[offset_v]);
                }
            }
            acc[k] = acc[k] * alpha + v_sum;
        }
        m_stat = m_new;
        l_stat = l_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint k = 0; k < num_elements_per_thread; k++) {
        uint d = tid + k * 32;
        output[b * stride_o_b + q_head * stride_o_h + d] = half(acc[k] / l_stat);
    }
}

kernel void tqpolar_online_attention_quant_v(
    device const half* q                  [[buffer(0)]],
    device const half* polar_radii        [[buffer(1)]],
    device const uchar* angle_codes_l1     [[buffer(2)]],
    device const uchar* angle_codes_deep   [[buffer(3)]],
    device const int8_t* v_codes           [[buffer(4)]],
    device const half* v_scales           [[buffer(5)]],
    device const uchar* qjl_packed_signs   [[buffer(6)]],
    device const half* qjl_norms           [[buffer(7)]],
    device const uchar* q_proj_signs       [[buffer(8)]],
    device half* output                    [[buffer(9)]],
    constant uint& head_dim                [[buffer(10)]],
    constant uint& split_dim               [[buffer(11)]],
    constant uint& block_size              [[buffer(12)]],
    constant uint& total_blocks            [[buffer(13)]],
    constant uint& qjl_proj_dim            [[buffer(14)]],
    constant uint& group_size              [[buffer(15)]],
    constant uint& use_qjl                 [[buffer(16)]],
    constant half& l1_scale                [[buffer(17)]],
    constant half& deep_scale              [[buffer(18)]],
    constant half& attention_scale         [[buffer(19)]],
    device const uint* strides             [[buffer(20)]],
    constant uint& actual_seq_len          [[buffer(21)]],
    constant uint& num_queries_per_kv      [[buffer(22)]],
    uint3 tgid                             [[threadgroup_position_in_grid]],
    uint tid                               [[thread_index_in_threadgroup]])
{
    uint b = tgid.x;
    uint q_head = tgid.y;
    uint kv_head = q_head / num_queries_per_kv;
    uint half_d = head_dim / 2;
    uint split_half_d = split_dim / 2;
    uint qjl_bytes = qjl_proj_dim / 8;
    uint num_elements_per_thread = head_dim / 32;

    uint stride_q_b   = strides[0];  uint stride_q_h   = strides[1];
    uint stride_r_b   = strides[2];  uint stride_r_h   = strides[3];  uint stride_r_s   = strides[4];  uint stride_r_l = strides[5];
    uint stride_c1_b  = strides[6];  uint stride_c1_h  = strides[7];  uint stride_c1_s  = strides[8];  uint stride_c1_l = strides[9];
    uint stride_cd_b  = strides[10]; uint stride_cd_h  = strides[11]; uint stride_cd_s  = strides[12]; uint stride_cd_l = strides[13];
    uint stride_vc_b  = strides[14]; uint stride_vc_h  = strides[15]; uint stride_vc_s  = strides[16]; uint stride_vc_l = strides[17];
    uint stride_vs_b  = strides[18]; uint stride_vs_h  = strides[19]; uint stride_vs_s  = strides[20]; uint stride_vs_l = strides[21];
    uint stride_qjl_b = strides[22]; uint stride_qjl_h = strides[23]; uint stride_qjl_s = strides[24]; uint stride_qjl_l = strides[25];
    uint stride_qn_b  = strides[26]; uint stride_qn_h  = strides[27]; uint stride_qn_s  = strides[28]; uint stride_qn_l = strides[29];
    uint stride_qp_b  = strides[30]; uint stride_qp_h  = strides[31];
    uint stride_o_b   = strides[32]; uint stride_o_h   = strides[33];

    float m_stat = -INFINITY;
    float l_stat = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    threadgroup float shared_scores[64];
    threadgroup float shared_q_norm[1];

    if (tid == 0 && use_qjl != 0) {
        float q_sum = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float val = q[b * stride_q_b + q_head * stride_q_h + d];
            q_sum += val * val;
        }
        shared_q_norm[0] = sqrt(q_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_norm = (use_qjl != 0) ? shared_q_norm[0] : 0.0f;

    for (uint s = 0; s < total_blocks; s++) {
        for (uint l = 0; l < block_size; l++) {
            uint global_tok_idx = s * block_size + l;
            if (global_tok_idx >= actual_seq_len) {
                if (tid == 0) {
                    shared_scores[l] = -INFINITY;
                }
                continue;
            }
            float private_sum = 0.0f;
            for (uint j = tid; j < half_d; j += 32) {
                uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;
                float r = polar_radii[offset_r];
                float norm_angle = 0.0f;
                if (j < split_half_d) {
                    uint l1_byte_idx = j / 2;
                    uint l1_nibble = j % 2;
                    uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l + l1_byte_idx;
                    uchar l1_byte = angle_codes_l1[offset_c1];
                    uchar l1_code = (l1_nibble == 0) ? (l1_byte & 0x0F) : ((l1_byte >> 4) & 0x0F);
                    norm_angle = float(l1_code) / float(l1_scale);
                } else {
                    uint rel_j = j - split_half_d;
                    uint deep_byte_idx = rel_j / 4;
                    uint deep_pair = rel_j % 4;
                    uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l + deep_byte_idx;
                    uchar deep_byte = angle_codes_deep[offset_cd];
                    uchar deep_code = (deep_byte >> (deep_pair * 2)) & 0x03;
                    norm_angle = float(deep_code) / float(deep_scale);
                }
                float angle = (norm_angle * 2.0f * M_PI_F) - M_PI_F;
                float k_x = r * cos(angle);
                float k_y = r * sin(angle);
                float q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
                float q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];
                private_sum += (q_x * k_x + q_y * k_y) * float(attention_scale);
            }
            uint local_hamming = 0;
            if (use_qjl != 0) {
                for (uint byte_idx = tid; byte_idx < qjl_bytes; byte_idx += 32) {
                    uint offset_qjl = b * stride_qjl_b + kv_head * stride_qjl_h + s * stride_qjl_s + l * stride_qjl_l + byte_idx;
                    uchar k_byte = qjl_packed_signs[offset_qjl];
                    uchar q_byte = q_proj_signs[b * stride_qp_b + q_head * stride_qp_h + byte_idx];
                    local_hamming += popcount(static_cast<uint>(k_byte ^ q_byte));
                }
            }
            float total_polar_score = simd_sum(private_sum);
            uint total_hamming_dist = simd_sum(local_hamming);
            if (tid == 0) {
                if (use_qjl != 0) {
                    float match_score = float(qjl_proj_dim) - 2.0f * float(total_hamming_dist);
                    float norm_E = qjl_norms[b * stride_qn_b + kv_head * stride_qn_h + s * stride_qn_s + l * stride_qn_l];
                    float qjl_correction = (norm_E * q_norm) * (match_score / float(qjl_proj_dim));
                    shared_scores[l] = total_polar_score + qjl_correction;
                } else {
                    shared_scores[l] = total_polar_score;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_max = -INFINITY;
        for (uint l = 0; l < block_size; l++) {
            block_max = max(block_max, shared_scores[l]);
        }
        float m_new = max(m_stat, block_max);
        float alpha = exp(m_stat - m_new);
        float l_block = 0.0f;
        for (uint l = 0; l < block_size; l++) {
            uint global_tok_idx = s * block_size + l;
            if (global_tok_idx < actual_seq_len) {
                l_block += exp(shared_scores[l] - m_new);
            }
        }
        float l_new = l_stat * alpha + l_block;

        for (uint k = 0; k < num_elements_per_thread; k++) {
            uint d = tid + k * 32;
            float v_sum = 0.0f;
            uint group_idx = d / group_size;
            for (uint l = 0; l < block_size; l++) {
                uint global_tok_idx = s * block_size + l;
                if (global_tok_idx < actual_seq_len) {
                    float p = exp(shared_scores[l] - m_new);
                    uint offset_vc = b * stride_vc_b + kv_head * stride_vc_h + s * stride_vc_s + l * stride_vc_l + d;
                    uint offset_vs = b * stride_vs_b + kv_head * stride_vs_h + s * stride_vs_s + l * stride_vs_l + group_idx;
                    int8_t v_code = v_codes[offset_vc];
                    float v_scale = v_scales[offset_vs];
                    float dequantized_v = float(v_code) * v_scale;
                    v_sum += p * dequantized_v;
                }
            }
            acc[k] = acc[k] * alpha + v_sum;
        }
        m_stat = m_new;
        l_stat = l_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint k = 0; k < num_elements_per_thread; k++) {
        uint d = tid + k * 32;
        output[b * stride_o_b + q_head * stride_o_h + d] = half(acc[k] / l_stat);
    }
}

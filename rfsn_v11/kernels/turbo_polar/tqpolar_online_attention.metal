#include <metal_stdlib>
using namespace metal;

// Precomputed (cos, sin) table for 256 uniformly spaced angles in [-pi, pi].
// angle[i] = (i/255.0) * 2*pi - pi
// Used to replace per-token cos()/sin() calls for 8-bit angle codes.
constant float2 TQPOLAR_LUT_COSSIN_256[256] = {
    {-1.00000000f, -0.00000000f}, {-0.99969645f, -0.02463745f}, {-0.99878599f, -0.04925994f}, {-0.99726917f, -0.07385253f}, {-0.99514692f, -0.09840028f}, {-0.99242051f, -0.12288829f}, {-0.98909161f, -0.14730170f}, {-0.98516223f, -0.17162568f},
    {-0.98063477f, -0.19584547f}, {-0.97551197f, -0.21994636f}, {-0.96979694f, -0.24391372f}, {-0.96349314f, -0.26773300f}, {-0.95660442f, -0.29138975f}, {-0.94913494f, -0.31486959f}, {-0.94108925f, -0.33815827f}, {-0.93247223f, -0.36124167f},
    {-0.92328911f, -0.38410575f}, {-0.91354546f, -0.40673664f}, {-0.90324720f, -0.42912061f}, {-0.89240058f, -0.45124406f}, {-0.88101219f, -0.47309356f}, {-0.86908895f, -0.49465584f}, {-0.85663808f, -0.51591783f}, {-0.84366715f, -0.53686660f},
    {-0.83018403f, -0.55748944f}, {-0.81619691f, -0.57777383f}, {-0.80171428f, -0.59770746f}, {-0.78674494f, -0.61727822f}, {-0.77129796f, -0.63647424f}, {-0.75538273f, -0.65528385f}, {-0.73900892f, -0.67369564f}, {-0.72218645f, -0.69169844f},
    {-0.70492555f, -0.70928131f}, {-0.68723669f, -0.72643357f}, {-0.66913061f, -0.74314483f}, {-0.65061830f, -0.75940492f}, {-0.63171101f, -0.77520398f}, {-0.61242020f, -0.79053241f}, {-0.59275760f, -0.80538092f}, {-0.57273514f, -0.81974048f},
    {-0.55236497f, -0.83360239f}, {-0.53165947f, -0.84695821f}, {-0.51063119f, -0.85979985f}, {-0.48929292f, -0.87211951f}, {-0.46765759f, -0.88390971f}, {-0.44573836f, -0.89516329f}, {-0.42354851f, -0.90587342f}, {-0.40110153f, -0.91603360f},
    {-0.37841105f, -0.92563766f}, {-0.35549083f, -0.93467977f}, {-0.33235480f, -0.94315443f}, {-0.30901699f, -0.95105652f}, {-0.28549159f, -0.95838122f}, {-0.26179286f, -0.96512409f}, {-0.23793520f, -0.97128103f}, {-0.21393308f, -0.97684832f},
    {-0.18980109f, -0.98182256f}, {-0.16555388f, -0.98620075f}, {-0.14120615f, -0.98998021f}, {-0.11677270f, -0.99315867f}, {-0.09226836f, -0.99573418f}, {-0.06770800f, -0.99770518f}, {-0.04310654f, -0.99907048f}, {-0.01847890f, -0.99982925f},
    {0.00615995f, -0.99998103f}, {0.03079506f, -0.99952572f}, {0.05541147f, -0.99846360f}, {0.07999425f, -0.99679532f}, {0.10452846f, -0.99452190f}, {0.12899922f, -0.99164470f}, {0.15339165f, -0.98816547f}, {0.17769097f, -0.98408634f},
    {0.20188241f, -0.97940977f}, {0.22595129f, -0.97413860f}, {0.24988299f, -0.96827604f}, {0.27366299f, -0.96182564f}, {0.29727685f, -0.95479132f}, {0.32071024f, -0.94717736f}, {0.34394892f, -0.93898836f}, {0.36697879f, -0.93022931f},
    {0.38978587f, -0.92090552f}, {0.41235632f, -0.91102265f}, {0.43467642f, -0.90058670f}, {0.45673264f, -0.88960401f}, {0.47851157f, -0.87808125f}, {0.50000000f, -0.86602540f}, {0.52118488f, -0.85344380f}, {0.54205336f, -0.84034407f},
    {0.56259275f, -0.82673417f}, {0.58279060f, -0.81262237f}, {0.60263464f, -0.79801723f}, {0.62211282f, -0.78292761f}, {0.64121331f, -0.76736268f}, {0.65992453f, -0.75133189f}, {0.67823512f, -0.73484497f}, {0.69613395f, -0.71791192f},
    {0.71361015f, -0.70054304f}, {0.73065313f, -0.68274886f}, {0.74725253f, -0.66454018f}, {0.76339828f, -0.64592806f}, {0.77908057f, -0.62692381f}, {0.79428989f, -0.60753895f}, {0.80901699f, -0.58778525f}, {0.82325295f, -0.56767472f},
    {0.83698911f, -0.54721955f}, {0.85021714f, -0.52643216f}, {0.86292900f, -0.50532518f}, {0.87511698f, -0.48391142f}, {0.88677369f, -0.46220388f}, {0.89789203f, -0.44021574f}, {0.90846527f, -0.41796034f}, {0.91848699f, -0.39545121f},
    {0.92795109f, -0.37270199f}, {0.93685184f, -0.34972651f}, {0.94518383f, -0.32653871f}, {0.95294200f, -0.30315267f}, {0.96012165f, -0.27958259f}, {0.96671840f, -0.25584278f}, {0.97272827f, -0.23194764f}, {0.97814760f, -0.20791169f},
    {0.98297310f, -0.18374952f}, {0.98720184f, -0.15947579f}, {0.99083125f, -0.13510525f}, {0.99385914f, -0.11065268f}, {0.99628365f, -0.08613294f}, {0.99810333f, -0.06156091f}, {0.99931706f, -0.03695150f}, {0.99992411f, -0.01231966f},
    {0.99992411f, 0.01231966f}, {0.99931706f, 0.03695150f}, {0.99810333f, 0.06156091f}, {0.99628365f, 0.08613294f}, {0.99385914f, 0.11065268f}, {0.99083125f, 0.13510525f}, {0.98720184f, 0.15947579f}, {0.98297310f, 0.18374952f},
    {0.97814760f, 0.20791169f}, {0.97272827f, 0.23194764f}, {0.96671840f, 0.25584278f}, {0.96012165f, 0.27958259f}, {0.95294200f, 0.30315267f}, {0.94518383f, 0.32653871f}, {0.93685184f, 0.34972651f}, {0.92795109f, 0.37270199f},
    {0.91848699f, 0.39545121f}, {0.90846527f, 0.41796034f}, {0.89789203f, 0.44021574f}, {0.88677369f, 0.46220388f}, {0.87511698f, 0.48391142f}, {0.86292900f, 0.50532518f}, {0.85021714f, 0.52643216f}, {0.83698911f, 0.54721955f},
    {0.82325295f, 0.56767472f}, {0.80901699f, 0.58778525f}, {0.79428989f, 0.60753895f}, {0.77908057f, 0.62692381f}, {0.76339828f, 0.64592806f}, {0.74725253f, 0.66454018f}, {0.73065313f, 0.68274886f}, {0.71361015f, 0.70054304f},
    {0.69613395f, 0.71791192f}, {0.67823512f, 0.73484497f}, {0.65992453f, 0.75133189f}, {0.64121331f, 0.76736268f}, {0.62211282f, 0.78292761f}, {0.60263464f, 0.79801723f}, {0.58279060f, 0.81262237f}, {0.56259275f, 0.82673417f},
    {0.54205336f, 0.84034407f}, {0.52118488f, 0.85344380f}, {0.50000000f, 0.86602540f}, {0.47851157f, 0.87808125f}, {0.45673264f, 0.88960401f}, {0.43467642f, 0.90058670f}, {0.41235632f, 0.91102265f}, {0.38978587f, 0.92090552f},
    {0.36697879f, 0.93022931f}, {0.34394892f, 0.93898836f}, {0.32071024f, 0.94717736f}, {0.29727685f, 0.95479132f}, {0.27366299f, 0.96182564f}, {0.24988299f, 0.96827604f}, {0.22595129f, 0.97413860f}, {0.20188241f, 0.97940977f},
    {0.17769097f, 0.98408634f}, {0.15339165f, 0.98816547f}, {0.12899922f, 0.99164470f}, {0.10452846f, 0.99452190f}, {0.07999425f, 0.99679532f}, {0.05541147f, 0.99846360f}, {0.03079506f, 0.99952572f}, {0.00615995f, 0.99998103f},
    {-0.01847890f, 0.99982925f}, {-0.04310654f, 0.99907048f}, {-0.06770800f, 0.99770518f}, {-0.09226836f, 0.99573418f}, {-0.11677270f, 0.99315867f}, {-0.14120615f, 0.98998021f}, {-0.16555388f, 0.98620075f}, {-0.18980109f, 0.98182256f},
    {-0.21393308f, 0.97684832f}, {-0.23793520f, 0.97128103f}, {-0.26179286f, 0.96512409f}, {-0.28549159f, 0.95838122f}, {-0.30901699f, 0.95105652f}, {-0.33235480f, 0.94315443f}, {-0.35549083f, 0.93467977f}, {-0.37841105f, 0.92563766f},
    {-0.40110153f, 0.91603360f}, {-0.42354851f, 0.90587342f}, {-0.44573836f, 0.89516329f}, {-0.46765759f, 0.88390971f}, {-0.48929292f, 0.87211951f}, {-0.51063119f, 0.85979985f}, {-0.53165947f, 0.84695821f}, {-0.55236497f, 0.83360239f},
    {-0.57273514f, 0.81974048f}, {-0.59275760f, 0.80538092f}, {-0.61242020f, 0.79053241f}, {-0.63171101f, 0.77520398f}, {-0.65061830f, 0.75940492f}, {-0.66913061f, 0.74314483f}, {-0.68723669f, 0.72643357f}, {-0.70492555f, 0.70928131f},
    {-0.72218645f, 0.69169844f}, {-0.73900892f, 0.67369564f}, {-0.75538273f, 0.65528385f}, {-0.77129796f, 0.63647424f}, {-0.78674494f, 0.61727822f}, {-0.80171428f, 0.59770746f}, {-0.81619691f, 0.57777383f}, {-0.83018403f, 0.55748944f},
    {-0.84366715f, 0.53686660f}, {-0.85663808f, 0.51591783f}, {-0.86908895f, 0.49465584f}, {-0.88101219f, 0.47309356f}, {-0.89240058f, 0.45124406f}, {-0.90324720f, 0.42912061f}, {-0.91354546f, 0.40673664f}, {-0.92328911f, 0.38410575f},
    {-0.93247223f, 0.36124167f}, {-0.94108925f, 0.33815827f}, {-0.94913494f, 0.31486959f}, {-0.95660442f, 0.29138975f}, {-0.96349314f, 0.26773300f}, {-0.96979694f, 0.24391372f}, {-0.97551197f, 0.21994636f}, {-0.98063477f, 0.19584547f},
    {-0.98516223f, 0.17162568f}, {-0.98909161f, 0.14730170f}, {-0.99242051f, 0.12288829f}, {-0.99514692f, 0.09840028f}, {-0.99726917f, 0.07385253f}, {-0.99878599f, 0.04925994f}, {-0.99969645f, 0.02463745f}, {-1.00000000f, 0.00000000f}
};

// Decode (cos, sin) for an angle code using the constant-memory LUT.
// For 8-bit deep angles this eliminates two transcendental ops per pair.
// For other bit widths, falls back to the standard trig path.
inline void _tqpolar_cossin_lut(
    device const uchar* angle_codes_deep,
    uint rel_j,
    uint offset_cd,
    uint deep_bits,
    half deep_scale,
    thread float* out_cos,
    thread float* out_sin
) {
    if (deep_bits == 8) {
        uchar code = angle_codes_deep[offset_cd + rel_j];
        float2 cs = TQPOLAR_LUT_COSSIN_256[code];
        *out_cos = cs.x;
        *out_sin = cs.y;
    } else {
        // Reconstruct from normalized angle for non-8-bit configs.
        uchar code;
        if (deep_bits == 4) {
            uchar byte = angle_codes_deep[offset_cd + rel_j / 2];
            code = (rel_j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        } else {
            uchar byte = angle_codes_deep[offset_cd + rel_j / 4];
            uint shift = (rel_j % 4) * 2;
            code = (byte >> shift) & 0x03;
        }
        float norm_angle = float(static_cast<half>(code) / deep_scale);
        float angle = (norm_angle * 2.0f * M_PI_F) - M_PI_F;
        *out_cos = cos(angle);
        *out_sin = sin(angle);
    }
}

inline half unpack_bit_online(device const uchar* packed_signs, uint offset, uint bit_idx) {
    uchar byte_val = packed_signs[offset + (bit_idx / 8)];
    uchar bit_mask = 1 << (bit_idx % 8);
    return (byte_val & bit_mask) ? 1.0h : -1.0h;
}

template <typename T1, typename T2>
inline float _tqpolar_decode_radius(
    T1 polar_radii,
    T2 polar_radii_i8,
    float radii_scale,
    uint offset_r,
    uint int8_radii,
    uint log_radii
)
{
    if (int8_radii == 0) {
        return float(polar_radii[offset_r]);
    } else {
        auto code = polar_radii_i8[offset_r];
        float value = float(code) * radii_scale;
        return (log_radii != 0) ? exp(value) : value;
    }
}

template <typename T1, typename T2>
inline float _tqpolar_decode_angle(
    T1 angle_codes_l1,
    T2 angle_codes_deep,
    uint j,
    uint split_half_d,
    uint offset_c1,
    uint offset_cd,
    uint l1_bits,
    uint deep_bits,
    half l1_scale,
    half deep_scale
)
{
    if (j < split_half_d) {
        uchar code;
        if (l1_bits == 8) {
            code = angle_codes_l1[offset_c1 + j];
        } else {
            uchar byte = angle_codes_l1[offset_c1 + j / 2];
            code = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        }
        return float(static_cast<half>(code) / l1_scale);
    } else {
        uint rel_j = j - split_half_d;
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
        return float(static_cast<half>(code) / deep_scale);
    }
}

template <typename T1, typename T2, typename T3>
inline float _tqpolar_qjl_correction(
    uint b, uint q_head, uint kv_head, uint s, uint l,
    T1 qjl_packed_signs,
    T2 qjl_norms,
    T3 q_proj_signs,
    uint qjl_proj_dim, uint qjl_bytes,
    float q_norm, float attention_scale_val,
    uint use_qjl, uint tid,
    uint stride_qjl_b, uint stride_qjl_h, uint stride_qjl_s, uint stride_qjl_l,
    uint stride_qn_b, uint stride_qn_h, uint stride_qn_s, uint stride_qn_l,
    uint stride_qp_b, uint stride_qp_h
)
{
    if (use_qjl == 0) {
        return 0.0f;
    }
    uint local_hamming = 0;
    for (uint byte_idx = tid; byte_idx < qjl_bytes; byte_idx += 32) {
        uint offset_qjl = b * stride_qjl_b + kv_head * stride_qjl_h + s * stride_qjl_s + l * stride_qjl_l + byte_idx;
        uchar k_byte = qjl_packed_signs[offset_qjl];
        uchar q_byte = q_proj_signs[b * stride_qp_b + q_head * stride_qp_h + byte_idx];
        local_hamming += popcount(static_cast<uint>(k_byte ^ q_byte));
    }
    uint total_hamming_dist = simd_sum(local_hamming);
    float match_score = float(qjl_proj_dim) - 2.0f * float(total_hamming_dist);
    float norm_E = qjl_norms[b * stride_qn_b + kv_head * stride_qn_h + s * stride_qn_s + l * stride_qn_l];
    float sign_corr = match_score / float(qjl_proj_dim);
    float cos_est = sin((M_PI_F / 2.0f) * sign_corr);
    return (norm_E * q_norm) * cos_est * attention_scale_val;
}

kernel void tqpolar_online_attention_dense_v(
    device const half* q                     [[buffer(0)]],
    device const half* polar_radii           [[buffer(1)]],
    device const int8_t* polar_radii_i8      [[buffer(2)]],
    device const half* radii_scales          [[buffer(3)]],
    device const uchar* angle_codes_l1       [[buffer(4)]],
    device const uchar* angle_codes_deep     [[buffer(5)]],
    device const half* v_dense               [[buffer(6)]],
    device const uchar* qjl_packed_signs     [[buffer(7)]],
    device const half* qjl_norms             [[buffer(8)]],
    device const uchar* q_proj_signs         [[buffer(9)]],
    device half* output                      [[buffer(10)]],
    constant uint& head_dim                  [[buffer(11)]],
    constant uint& split_dim                 [[buffer(12)]],
    constant uint& block_size                [[buffer(13)]],
    constant uint& total_blocks              [[buffer(14)]],
    constant uint& qjl_proj_dim              [[buffer(15)]],
    constant uint& use_qjl                   [[buffer(16)]],
    constant half& l1_scale                  [[buffer(17)]],
    constant half& deep_scale                [[buffer(18)]],
    constant half& attention_scale           [[buffer(19)]],
    constant uint& int8_radii                [[buffer(20)]],
    constant uint& log_radii                 [[buffer(21)]],
    constant uint& l1_bits                   [[buffer(22)]],
    constant uint& deep_bits                 [[buffer(23)]],
    device const uint* strides               [[buffer(24)]],
    constant uint& actual_seq_len            [[buffer(25)]],
    constant uint& num_queries_per_kv        [[buffer(26)]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_index_in_threadgroup]])
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
    uint stride_rs_b  = strides[6];  uint stride_rs_h  = strides[7];  uint stride_rs_s  = strides[8];
    uint stride_c1_b  = strides[9];  uint stride_c1_h  = strides[10]; uint stride_c1_s  = strides[11]; uint stride_c1_l = strides[12];
    uint stride_cd_b  = strides[13]; uint stride_cd_h  = strides[14]; uint stride_cd_s  = strides[15]; uint stride_cd_l = strides[16];
    uint stride_v_b   = strides[17]; uint stride_v_h   = strides[18]; uint stride_v_s   = strides[19]; uint stride_v_l = strides[20];
    uint stride_qjl_b = strides[21]; uint stride_qjl_h = strides[22]; uint stride_qjl_s = strides[23]; uint stride_qjl_l = strides[24];
    uint stride_qn_b  = strides[25]; uint stride_qn_h  = strides[26]; uint stride_qn_s  = strides[27]; uint stride_qn_l = strides[28];
    uint stride_qp_b  = strides[29]; uint stride_qp_h  = strides[30];
    uint stride_o_b   = strides[31]; uint stride_o_h   = strides[32];

    float m_stat = -INFINITY;
    float l_stat = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // Threadgroup memory required only for q_norm (QJL path).
    // shared_scores removed: scores stay in registers via per-token online-softmax.
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
        // Hoist the per-block radii scale read outside the token loop.
        float radii_scale_val = (int8_radii == 0) ? 0.0f : float(radii_scales[b * stride_rs_b + kv_head * stride_rs_h + s * stride_rs_s]);

        for (uint l = 0; l < block_size; l++) {
            uint global_tok_idx = s * block_size + l;
            if (global_tok_idx >= actual_seq_len) {
                continue;
            }
            // Compute Q @ K dot product; each of the 32 threads covers half_d/32 dims.
            float private_sum = 0.0f;
            for (uint j = tid; j < half_d; j += 32) {
                uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l;
                float r = _tqpolar_decode_radius(polar_radii, polar_radii_i8, radii_scale_val, offset_r, int8_radii, log_radii);
                float k_x, k_y;
                if (j >= split_half_d) {
                    // Deep-bucket: use LUT for 8-bit codes, trig for narrower configs.
                    uint rel_j = j - split_half_d;
                    _tqpolar_cossin_lut(angle_codes_deep, rel_j, offset_cd,
                                        deep_bits, deep_scale, &k_x, &k_y);
                } else {
                    // L1-bucket: always use the normalized angle / trig path.
                    float norm_angle = float(static_cast<half>(
                        (l1_bits == 8)
                            ? angle_codes_l1[offset_c1 + j]
                            : ((angle_codes_l1[offset_c1 + j / 2] >> ((j % 2) * 4)) & 0x0F)
                    ) / l1_scale);
                    float angle = (norm_angle * 2.0f * M_PI_F) - M_PI_F;
                    k_x = cos(angle);
                    k_y = sin(angle);
                }
                k_x *= r;
                k_y *= r;
                float q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
                float q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];
                private_sum += (q_x * k_x + q_y * k_y) * float(attention_scale);
            }
            // simd_sum broadcasts the reduced score to all 32 threads in the SIMD group.
            float score = simd_sum(private_sum);
            score += _tqpolar_qjl_correction(
                b, q_head, kv_head, s, l,
                qjl_packed_signs, qjl_norms, q_proj_signs,
                qjl_proj_dim, qjl_bytes,
                q_norm, float(attention_scale),
                use_qjl, tid,
                stride_qjl_b, stride_qjl_h, stride_qjl_s, stride_qjl_l,
                stride_qn_b, stride_qn_h, stride_qn_s, stride_qn_l,
                stride_qp_b, stride_qp_h
            );

            // Per-token online-softmax update. All 32 threads execute identically
            // since score is broadcast by simd_sum. No threadgroup barriers needed.
            float m_new = max(m_stat, score);
            float alpha = exp(m_stat - m_new);
            float exp_score = exp(score - m_new);

            // Update V accumulator: each thread owns head_dim/32 distinct output elements.
            for (uint k = 0; k < num_elements_per_thread; k++) {
                uint d = tid + k * 32;
                uint offset_v = b * stride_v_b + kv_head * stride_v_h + s * stride_v_s + l * stride_v_l + d;
                acc[k] = acc[k] * alpha + exp_score * float(v_dense[offset_v]);
            }

            l_stat = l_stat * alpha + exp_score;
            m_stat = m_new;
        }
    }

    for (uint k = 0; k < num_elements_per_thread; k++) {
        uint d = tid + k * 32;
        output[b * stride_o_b + q_head * stride_o_h + d] = half(acc[k] / l_stat);
    }
}

kernel void tqpolar_online_attention_quant_v(
    device const half* q                     [[buffer(0)]],
    device const half* polar_radii           [[buffer(1)]],
    device const int8_t* polar_radii_i8      [[buffer(2)]],
    device const half* radii_scales          [[buffer(3)]],
    device const uchar* angle_codes_l1       [[buffer(4)]],
    device const uchar* angle_codes_deep     [[buffer(5)]],
    device const int8_t* v_codes             [[buffer(6)]],
    device const half* v_scales              [[buffer(7)]],
    device const uchar* qjl_packed_signs     [[buffer(8)]],
    device const half* qjl_norms             [[buffer(9)]],
    device const uchar* q_proj_signs         [[buffer(10)]],
    device half* output                      [[buffer(11)]],
    constant uint& head_dim                  [[buffer(12)]],
    constant uint& split_dim                 [[buffer(13)]],
    constant uint& block_size                [[buffer(14)]],
    constant uint& total_blocks              [[buffer(15)]],
    constant uint& qjl_proj_dim              [[buffer(16)]],
    constant uint& group_size                [[buffer(17)]],
    constant uint& use_qjl                   [[buffer(18)]],
    constant half& l1_scale                  [[buffer(19)]],
    constant half& deep_scale                [[buffer(20)]],
    constant half& attention_scale           [[buffer(21)]],
    constant uint& int8_radii                [[buffer(22)]],
    constant uint& log_radii                 [[buffer(23)]],
    constant uint& l1_bits                   [[buffer(24)]],
    constant uint& deep_bits                 [[buffer(25)]],
    device const uint* strides               [[buffer(26)]],
    constant uint& actual_seq_len            [[buffer(27)]],
    constant uint& num_queries_per_kv        [[buffer(28)]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_index_in_threadgroup]])
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
    uint stride_rs_b  = strides[6];  uint stride_rs_h  = strides[7];  uint stride_rs_s  = strides[8];
    uint stride_c1_b  = strides[9];  uint stride_c1_h  = strides[10]; uint stride_c1_s  = strides[11]; uint stride_c1_l = strides[12];
    uint stride_cd_b  = strides[13]; uint stride_cd_h  = strides[14]; uint stride_cd_s  = strides[15]; uint stride_cd_l = strides[16];
    uint stride_vc_b  = strides[17]; uint stride_vc_h  = strides[18]; uint stride_vc_s  = strides[19]; uint stride_vc_l = strides[20];
    uint stride_vs_b  = strides[21]; uint stride_vs_h  = strides[22]; uint stride_vs_s  = strides[23]; uint stride_vs_l = strides[24];
    uint stride_qjl_b = strides[25]; uint stride_qjl_h = strides[26]; uint stride_qjl_s = strides[27]; uint stride_qjl_l = strides[28];
    uint stride_qn_b  = strides[29]; uint stride_qn_h  = strides[30]; uint stride_qn_s  = strides[31]; uint stride_qn_l = strides[32];
    uint stride_qp_b  = strides[33]; uint stride_qp_h  = strides[34];
    uint stride_o_b   = strides[35]; uint stride_o_h   = strides[36];

    float m_stat = -INFINITY;
    float l_stat = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // Threadgroup memory required only for q_norm (QJL path).
    // shared_scores removed: scores stay in registers via per-token online-softmax.
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
        // Hoist the per-block radii scale read outside the token loop.
        float radii_scale_val = (int8_radii == 0) ? 0.0f : float(radii_scales[b * stride_rs_b + kv_head * stride_rs_h + s * stride_rs_s]);

        for (uint l = 0; l < block_size; l++) {
            uint global_tok_idx = s * block_size + l;
            if (global_tok_idx >= actual_seq_len) {
                continue;
            }
            // Compute Q @ K dot product; each of the 32 threads covers half_d/32 dims.
            float private_sum = 0.0f;
            for (uint j = tid; j < half_d; j += 32) {
                uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l;
                float r = _tqpolar_decode_radius(polar_radii, polar_radii_i8, radii_scale_val, offset_r, int8_radii, log_radii);
                float k_x, k_y;
                if (j >= split_half_d) {
                    // Deep-bucket: use LUT for 8-bit codes, trig for narrower configs.
                    uint rel_j = j - split_half_d;
                    _tqpolar_cossin_lut(angle_codes_deep, rel_j, offset_cd,
                                        deep_bits, deep_scale, &k_x, &k_y);
                } else {
                    // L1-bucket: always use the normalized angle / trig path.
                    float norm_angle = float(static_cast<half>(
                        (l1_bits == 8)
                            ? angle_codes_l1[offset_c1 + j]
                            : ((angle_codes_l1[offset_c1 + j / 2] >> ((j % 2) * 4)) & 0x0F)
                    ) / l1_scale);
                    float angle = (norm_angle * 2.0f * M_PI_F) - M_PI_F;
                    k_x = cos(angle);
                    k_y = sin(angle);
                }
                k_x *= r;
                k_y *= r;
                float q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
                float q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];
                private_sum += (q_x * k_x + q_y * k_y) * float(attention_scale);
            }
            // simd_sum broadcasts the reduced score to all 32 threads in the SIMD group.
            float score = simd_sum(private_sum);
            score += _tqpolar_qjl_correction(
                b, q_head, kv_head, s, l,
                qjl_packed_signs, qjl_norms, q_proj_signs,
                qjl_proj_dim, qjl_bytes,
                q_norm, float(attention_scale),
                use_qjl, tid,
                stride_qjl_b, stride_qjl_h, stride_qjl_s, stride_qjl_l,
                stride_qn_b, stride_qn_h, stride_qn_s, stride_qn_l,
                stride_qp_b, stride_qp_h
            );

            // Per-token online-softmax update. All 32 threads execute identically
            // since score is broadcast by simd_sum. No threadgroup barriers needed.
            float m_new = max(m_stat, score);
            float alpha = exp(m_stat - m_new);
            float exp_score = exp(score - m_new);

            // Update V accumulator: each thread owns head_dim/32 distinct output elements.
            for (uint k = 0; k < num_elements_per_thread; k++) {
                uint d = tid + k * 32;
                uint group_idx = d / group_size;
                uint offset_vc = b * stride_vc_b + kv_head * stride_vc_h + s * stride_vc_s + l * stride_vc_l + d;
                uint offset_vs = b * stride_vs_b + kv_head * stride_vs_h + s * stride_vs_s + l * stride_vs_l + group_idx;
                float dequantized_v = float(v_codes[offset_vc]) * float(v_scales[offset_vs]);
                acc[k] = acc[k] * alpha + exp_score * dequantized_v;
            }

            l_stat = l_stat * alpha + exp_score;
            m_stat = m_new;
        }
    }

    for (uint k = 0; k < num_elements_per_thread; k++) {
        uint d = tid + k * 32;
        output[b * stride_o_b + q_head * stride_o_h + d] = half(acc[k] / l_stat);
    }
}


kernel void tqpolar_online_attention_quant_v_dense_tail(
    device const half* q                     [[buffer(0)]],
    device const half* polar_radii           [[buffer(1)]],
    device const int8_t* polar_radii_i8      [[buffer(2)]],
    device const half* radii_scales          [[buffer(3)]],
    device const uchar* angle_codes_l1       [[buffer(4)]],
    device const uchar* angle_codes_deep     [[buffer(5)]],
    device const int8_t* v_codes             [[buffer(6)]],
    device const half* v_scales              [[buffer(7)]],
    device const half* tail_k                [[buffer(8)]],
    device const half* tail_v                [[buffer(9)]],
    device const uchar* qjl_packed_signs     [[buffer(10)]],
    device const half* qjl_norms             [[buffer(11)]],
    device const uchar* q_proj_signs         [[buffer(12)]],
    device half* output                      [[buffer(13)]],
    constant uint* constants                 [[buffer(14)]],
    constant half& l1_scale                  [[buffer(15)]],
    constant half& deep_scale                [[buffer(16)]],
    constant half& attention_scale           [[buffer(17)]],
    device const uint* strides               [[buffer(18)]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_index_in_threadgroup]])
{
    uint head_dim = constants[0];
    uint split_dim = constants[1];
    uint block_size = constants[2];
    uint total_blocks = constants[3];
    uint tail_length = constants[4];
    uint qjl_proj_dim = constants[5];
    uint group_size = constants[6];
    uint use_qjl = constants[7];
    uint int8_radii = constants[8];
    uint log_radii = constants[9];
    uint l1_bits = constants[10];
    uint deep_bits = constants[11];
    uint actual_seq_len = constants[12];
    uint num_queries_per_kv = constants[13];

    uint b = tgid.x;
    uint q_head = tgid.y;
    uint kv_head = q_head / num_queries_per_kv;
    uint half_d = head_dim / 2;
    uint split_half_d = split_dim / 2;
    uint qjl_bytes = qjl_proj_dim / 8;
    uint num_elements_per_thread = head_dim / 32;

    uint stride_q_b   = strides[0];  uint stride_q_h   = strides[1];
    uint stride_r_b   = strides[2];  uint stride_r_h   = strides[3];  uint stride_r_s   = strides[4];  uint stride_r_l = strides[5];
    uint stride_rs_b  = strides[6];  uint stride_rs_h  = strides[7];  uint stride_rs_s  = strides[8];
    uint stride_c1_b  = strides[9];  uint stride_c1_h  = strides[10]; uint stride_c1_s  = strides[11]; uint stride_c1_l = strides[12];
    uint stride_cd_b  = strides[13]; uint stride_cd_h  = strides[14]; uint stride_cd_s  = strides[15]; uint stride_cd_l = strides[16];
    uint stride_vc_b  = strides[17]; uint stride_vc_h  = strides[18]; uint stride_vc_s  = strides[19]; uint stride_vc_l = strides[20];
    uint stride_vs_b  = strides[21]; uint stride_vs_h  = strides[22]; uint stride_vs_s  = strides[23]; uint stride_vs_l = strides[24];
    uint stride_tk_b  = strides[25]; uint stride_tk_h  = strides[26]; uint stride_tk_l  = strides[27]; uint stride_tk_d  = strides[28];
    uint stride_tv_b  = strides[29]; uint stride_tv_h  = strides[30]; uint stride_tv_l  = strides[31]; uint stride_tv_d  = strides[32];
    uint stride_qjl_b = strides[33]; uint stride_qjl_h = strides[34]; uint stride_qjl_s = strides[35]; uint stride_qjl_l = strides[36];
    uint stride_qn_b  = strides[37]; uint stride_qn_h  = strides[38]; uint stride_qn_s  = strides[39]; uint stride_qn_l  = strides[40];
    uint stride_qp_b  = strides[41]; uint stride_qp_h  = strides[42];
    uint stride_o_b   = strides[43]; uint stride_o_h   = strides[44];

    float m_stat = -INFINITY;
    float l_stat = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // Threadgroup memory required only for q_norm (QJL path).
    // shared_scores removed: scores stay in registers via per-token online-softmax.
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

    // Phase 1: compressed completed blocks.
    for (uint s = 0; s < total_blocks; s++) {
        // Hoist the per-block radii scale read outside the token loop.
        float radii_scale_val = (int8_radii == 0) ? 0.0f : float(radii_scales[b * stride_rs_b + kv_head * stride_rs_h + s * stride_rs_s]);

        for (uint l = 0; l < block_size; l++) {
            uint global_tok_idx = s * block_size + l;
            if (global_tok_idx >= actual_seq_len) {
                continue;
            }
            // Compute Q @ K dot product; each of the 32 threads covers half_d/32 dims.
            float private_sum = 0.0f;
            for (uint j = tid; j < half_d; j += 32) {
                uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l;
                float r = _tqpolar_decode_radius(polar_radii, polar_radii_i8, radii_scale_val, offset_r, int8_radii, log_radii);
                float k_x, k_y;
                if (j >= split_half_d) {
                    // Deep-bucket: use LUT for 8-bit codes, trig for narrower configs.
                    uint rel_j = j - split_half_d;
                    _tqpolar_cossin_lut(angle_codes_deep, rel_j, offset_cd,
                                        deep_bits, deep_scale, &k_x, &k_y);
                } else {
                    // L1-bucket: always use the normalized angle / trig path.
                    float norm_angle = float(static_cast<half>(
                        (l1_bits == 8)
                            ? angle_codes_l1[offset_c1 + j]
                            : ((angle_codes_l1[offset_c1 + j / 2] >> ((j % 2) * 4)) & 0x0F)
                    ) / l1_scale);
                    float angle = (norm_angle * 2.0f * M_PI_F) - M_PI_F;
                    k_x = cos(angle);
                    k_y = sin(angle);
                }
                k_x *= r;
                k_y *= r;
                float q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
                float q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];
                private_sum += (q_x * k_x + q_y * k_y) * float(attention_scale);
            }
            // simd_sum broadcasts the reduced score to all 32 threads in the SIMD group.
            float score = simd_sum(private_sum);
            score += _tqpolar_qjl_correction(
                b, q_head, kv_head, s, l,
                qjl_packed_signs, qjl_norms, q_proj_signs,
                qjl_proj_dim, qjl_bytes,
                q_norm, float(attention_scale),
                use_qjl, tid,
                stride_qjl_b, stride_qjl_h, stride_qjl_s, stride_qjl_l,
                stride_qn_b, stride_qn_h, stride_qn_s, stride_qn_l,
                stride_qp_b, stride_qp_h
            );

            // Per-token online-softmax update. All 32 threads execute identically
            // since score is broadcast by simd_sum. No threadgroup barriers needed.
            float m_new = max(m_stat, score);
            float alpha = exp(m_stat - m_new);
            float exp_score = exp(score - m_new);

            // Update V accumulator: each thread owns head_dim/32 distinct output elements.
            for (uint k = 0; k < num_elements_per_thread; k++) {
                uint d = tid + k * 32;
                uint group_idx = d / group_size;
                uint offset_vc = b * stride_vc_b + kv_head * stride_vc_h + s * stride_vc_s + l * stride_vc_l + d;
                uint offset_vs = b * stride_vs_b + kv_head * stride_vs_h + s * stride_vs_s + l * stride_vs_l + group_idx;
                float dequantized_v = float(v_codes[offset_vc]) * float(v_scales[offset_vs]);
                acc[k] = acc[k] * alpha + exp_score * dequantized_v;
            }

            l_stat = l_stat * alpha + exp_score;
            m_stat = m_new;
        }
    }

    // Phase 2: dense partial tail.
    // simd_sum broadcasts the score to all threads; no shared memory needed.
    for (uint t = 0; t < tail_length; t++) {
        uint global_tok_idx = total_blocks * block_size + t;
        if (global_tok_idx >= actual_seq_len) {
            continue;
        }
        float private_sum = 0.0f;
        for (uint j = tid; j < head_dim; j += 32) {
            uint offset_k = b * stride_tk_b + kv_head * stride_tk_h + t * stride_tk_l + j * stride_tk_d;
            float k_val = float(tail_k[offset_k]);
            float q_val = q[b * stride_q_b + q_head * stride_q_h + j];
            private_sum += q_val * k_val * float(attention_scale);
        }
        float score = simd_sum(private_sum);

        float m_new = max(m_stat, score);
        float alpha = exp(m_stat - m_new);
        float exp_score = exp(score - m_new);

        for (uint k = 0; k < num_elements_per_thread; k++) {
            uint d = tid + k * 32;
            uint offset_v = b * stride_tv_b + kv_head * stride_tv_h + t * stride_tv_l + d * stride_tv_d;
            acc[k] = acc[k] * alpha + exp_score * float(tail_v[offset_v]);
        }
        l_stat = l_stat * alpha + exp_score;
        m_stat = m_new;
    }

    for (uint k = 0; k < num_elements_per_thread; k++) {
        uint d = tid + k * 32;
        output[b * stride_o_b + q_head * stride_o_h + d] = half(acc[k] / l_stat);
    }
}


kernel void tqpolar_dense_tail_state_raw(
    device const half* q                     [[buffer(0)]],
    device const half* tail_k                [[buffer(1)]],
    device const half* tail_v                [[buffer(2)]],
    device float* out_weighted               [[buffer(3)]],
    device float* out_max_score              [[buffer(4)]],
    device float* out_exp_sum                [[buffer(5)]],
    constant uint& head_dim                  [[buffer(6)]],
    constant uint& tail_length               [[buffer(7)]],
    constant half& attention_scale           [[buffer(8)]],
    constant uint& num_queries_per_kv        [[buffer(9)]],
    device const uint* strides               [[buffer(10)]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_index_in_threadgroup]])
{
    uint b = tgid.x;
    uint q_head = tgid.y;
    uint kv_head = q_head / num_queries_per_kv;
    uint half_d = head_dim / 2;
    uint num_elements_per_thread = head_dim / 32;

    uint stride_q_b  = strides[0];  uint stride_q_h  = strides[1];
    uint stride_tk_b = strides[2];  uint stride_tk_h = strides[3];  uint stride_tk_l = strides[4];  uint stride_tk_d = strides[5];
    uint stride_tv_b = strides[6];  uint stride_tv_h = strides[7];  uint stride_tv_l = strides[8];  uint stride_tv_d = strides[9];
    uint stride_o_b  = strides[10]; uint stride_o_h  = strides[11];

    float m_stat = -INFINITY;
    float l_stat = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint t = 0; t < tail_length; t++) {
        float private_sum = 0.0f;
        for (uint j = tid; j < half_d; j += 32) {
            float k_x = tail_k[b * stride_tk_b + kv_head * stride_tk_h + t * stride_tk_l + j * 2];
            float k_y = tail_k[b * stride_tk_b + kv_head * stride_tk_h + t * stride_tk_l + j * 2 + 1];
            float q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
            float q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];
            private_sum += (q_x * k_x + q_y * k_y) * float(attention_scale);
        }
        float score = simd_sum(private_sum);

        float m_new = max(m_stat, score);
        float alpha = exp(m_stat - m_new);
        float l_new = l_stat * alpha + exp(score - m_new);

        for (uint k = 0; k < num_elements_per_thread; k++) {
            uint d = tid + k * 32;
            float v_val = tail_v[b * stride_tv_b + kv_head * stride_tv_h + t * stride_tv_l + d * stride_tv_d];
            acc[k] = acc[k] * alpha + exp(score - m_new) * v_val;
        }
        m_stat = m_new;
        l_stat = l_new;
    }

    if (tid == 0) {
        uint num_q_heads = stride_o_b / stride_o_h;
        out_max_score[b * num_q_heads + q_head] = m_stat;
        out_exp_sum[b * num_q_heads + q_head] = l_stat;
    }
    for (uint k = 0; k < num_elements_per_thread; k++) {
        uint d = tid + k * 32;
        out_weighted[b * stride_o_b + q_head * stride_o_h + d] = acc[k];
    }
}


kernel void tqpolar_online_attention_quant_v_raw(
    device const half* q                     [[buffer(0)]],
    device const half* polar_radii           [[buffer(1)]],
    device const int8_t* polar_radii_i8      [[buffer(2)]],
    device const half* radii_scales          [[buffer(3)]],
    device const uchar* angle_codes_l1       [[buffer(4)]],
    device const uchar* angle_codes_deep     [[buffer(5)]],
    device const int8_t* v_codes             [[buffer(6)]],
    device const half* v_scales              [[buffer(7)]],
    device const uchar* qjl_packed_signs     [[buffer(8)]],
    device const half* qjl_norms             [[buffer(9)]],
    device const uchar* q_proj_signs         [[buffer(10)]],
    device float* out_weighted               [[buffer(11)]],
    device float* out_max_score              [[buffer(12)]],
    device float* out_exp_sum                [[buffer(13)]],
    constant uint& head_dim                  [[buffer(14)]],
    constant uint& split_dim                 [[buffer(15)]],
    constant uint& block_size                [[buffer(16)]],
    constant uint& total_blocks              [[buffer(17)]],
    constant uint& qjl_proj_dim              [[buffer(18)]],
    constant uint& group_size                [[buffer(19)]],
    constant uint& use_qjl                   [[buffer(20)]],
    constant half& l1_scale                  [[buffer(21)]],
    constant half& deep_scale                [[buffer(22)]],
    constant half& attention_scale           [[buffer(23)]],
    constant uint& int8_radii                [[buffer(24)]],
    constant uint& log_radii                 [[buffer(25)]],
    constant uint& l1_bits                   [[buffer(26)]],
    constant uint& deep_bits                 [[buffer(27)]],
    device const uint* strides               [[buffer(28)]],
    constant uint& actual_seq_len            [[buffer(29)]],
    constant uint& num_queries_per_kv        [[buffer(30)]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_index_in_threadgroup]])
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
    uint stride_rs_b  = strides[6];  uint stride_rs_h  = strides[7];  uint stride_rs_s  = strides[8];
    uint stride_c1_b  = strides[9];  uint stride_c1_h  = strides[10]; uint stride_c1_s  = strides[11]; uint stride_c1_l = strides[12];
    uint stride_cd_b  = strides[13]; uint stride_cd_h  = strides[14]; uint stride_cd_s  = strides[15]; uint stride_cd_l = strides[16];
    uint stride_vc_b  = strides[17]; uint stride_vc_h  = strides[18]; uint stride_vc_s  = strides[19]; uint stride_vc_l = strides[20];
    uint stride_vs_b  = strides[21]; uint stride_vs_h  = strides[22]; uint stride_vs_s  = strides[23]; uint stride_vs_l = strides[24];
    uint stride_qjl_b = strides[25]; uint stride_qjl_h = strides[26]; uint stride_qjl_s = strides[27]; uint stride_qjl_l = strides[28];
    uint stride_qn_b  = strides[29]; uint stride_qn_h  = strides[30]; uint stride_qn_s  = strides[31]; uint stride_qn_l = strides[32];
    uint stride_qp_b  = strides[33]; uint stride_qp_h  = strides[34];
    uint stride_o_b   = strides[35]; uint stride_o_h   = strides[36];

    float m_stat = -INFINITY;
    float l_stat = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // Threadgroup memory required only for q_norm (QJL path).
    // shared_scores removed: scores stay in registers via per-token online-softmax.
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
        // Hoist the per-block radii scale read outside the token loop.
        float radii_scale_val = (int8_radii == 0) ? 0.0f : float(radii_scales[b * stride_rs_b + kv_head * stride_rs_h + s * stride_rs_s]);

        for (uint l = 0; l < block_size; l++) {
            uint global_tok_idx = s * block_size + l;
            if (global_tok_idx >= actual_seq_len) {
                continue;
            }
            // Compute Q @ K dot product; each of the 32 threads covers half_d/32 dims.
            float private_sum = 0.0f;
            for (uint j = tid; j < half_d; j += 32) {
                uint offset_r = b * stride_r_b + kv_head * stride_r_h + s * stride_r_s + l * stride_r_l + j;
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l;
                float r = _tqpolar_decode_radius(polar_radii, polar_radii_i8, radii_scale_val, offset_r, int8_radii, log_radii);
                float k_x, k_y;
                if (j >= split_half_d) {
                    // Deep-bucket: use LUT for 8-bit codes, trig for narrower configs.
                    uint rel_j = j - split_half_d;
                    _tqpolar_cossin_lut(angle_codes_deep, rel_j, offset_cd,
                                        deep_bits, deep_scale, &k_x, &k_y);
                } else {
                    // L1-bucket: always use the normalized angle / trig path.
                    float norm_angle = float(static_cast<half>(
                        (l1_bits == 8)
                            ? angle_codes_l1[offset_c1 + j]
                            : ((angle_codes_l1[offset_c1 + j / 2] >> ((j % 2) * 4)) & 0x0F)
                    ) / l1_scale);
                    float angle = (norm_angle * 2.0f * M_PI_F) - M_PI_F;
                    k_x = cos(angle);
                    k_y = sin(angle);
                }
                k_x *= r;
                k_y *= r;
                float q_x = q[b * stride_q_b + q_head * stride_q_h + j * 2];
                float q_y = q[b * stride_q_b + q_head * stride_q_h + j * 2 + 1];
                private_sum += (q_x * k_x + q_y * k_y) * float(attention_scale);
            }
            // simd_sum broadcasts the reduced score to all 32 threads in the SIMD group.
            float score = simd_sum(private_sum);
            score += _tqpolar_qjl_correction(
                b, q_head, kv_head, s, l,
                qjl_packed_signs, qjl_norms, q_proj_signs,
                qjl_proj_dim, qjl_bytes,
                q_norm, float(attention_scale),
                use_qjl, tid,
                stride_qjl_b, stride_qjl_h, stride_qjl_s, stride_qjl_l,
                stride_qn_b, stride_qn_h, stride_qn_s, stride_qn_l,
                stride_qp_b, stride_qp_h
            );

            // Per-token online-softmax update (Milakov & Gimelshein streaming algorithm).
            // All 32 threads in the SIMD group execute identically since score is broadcast
            // by simd_sum above. No threadgroup memory or barriers needed in this loop.
            float m_new = max(m_stat, score);
            float alpha = exp(m_stat - m_new);
            float exp_score = exp(score - m_new);

            // Update V accumulator: each thread owns head_dim/32 distinct output elements.
            for (uint k = 0; k < num_elements_per_thread; k++) {
                uint d = tid + k * 32;
                uint group_idx = d / group_size;
                uint offset_vc = b * stride_vc_b + kv_head * stride_vc_h + s * stride_vc_s + l * stride_vc_l + d;
                uint offset_vs = b * stride_vs_b + kv_head * stride_vs_h + s * stride_vs_s + l * stride_vs_l + group_idx;
                float dequantized_v = float(v_codes[offset_vc]) * float(v_scales[offset_vs]);
                acc[k] = acc[k] * alpha + exp_score * dequantized_v;
            }

            l_stat = l_stat * alpha + exp_score;
            m_stat = m_new;
        }
    }

    if (tid == 0) {
        uint num_q_heads = stride_o_b / stride_o_h;
        out_max_score[b * num_q_heads + q_head] = m_stat;
        out_exp_sum[b * num_q_heads + q_head] = l_stat;
    }
    for (uint k = 0; k < num_elements_per_thread; k++) {
        uint d = tid + k * 32;
        out_weighted[b * stride_o_b + q_head * stride_o_h + d] = acc[k];
    }
}

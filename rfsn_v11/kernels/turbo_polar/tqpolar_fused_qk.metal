#include <metal_stdlib>
using namespace metal;

// Precomputed (cos, sin) for 256 uniformly spaced angles in [-pi, pi].
// angle[i] = (i/255.0)*2*pi - pi  (half-precision entries)
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

            half k_x, k_y;
            if (j < split_half_d) {
                // L1 bucket: trig path
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l;
                uchar code;
                if (l1_bits == 8) {
                    code = angle_codes_l1[offset_c1 + j];
                } else {
                    uchar byte = angle_codes_l1[offset_c1 + j / 2];
                    code = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                }
                half norm = static_cast<half>(code) / l1_scale;
                half angle = (norm * 2.0h * M_PI_H) - M_PI_H;
                k_x = r * cos(angle);
                k_y = r * sin(angle);
            } else {
                // Deep bucket: LUT for 8-bit, trig for narrower configs
                uint rel_j = j - split_half_d;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l;
                if (deep_bits == 8) {
                    uchar code = angle_codes_deep[offset_cd + rel_j];
                    float2 cs = TQPOLAR_LUT_COSSIN_256[code];
                    k_x = r * half(cs.x);
                    k_y = r * half(cs.y);
                } else {
                    uchar code;
                    if (deep_bits == 4) {
                        uchar byte = angle_codes_deep[offset_cd + rel_j / 2];
                        code = (rel_j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                    } else {
                        uchar byte = angle_codes_deep[offset_cd + rel_j / 4];
                        uint shift = (rel_j % 4) * 2;
                        code = (byte >> shift) & 0x03;
                    }
                    half norm = static_cast<half>(code) / deep_scale;
                    half angle = (norm * 2.0h * M_PI_H) - M_PI_H;
                    k_x = r * cos(angle);
                    k_y = r * sin(angle);
                }
            }

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

            half k_x, k_y;
            if (j < split_half_d) {
                // L1 bucket: trig path
                uint offset_c1 = b * stride_c1_b + kv_head * stride_c1_h + s * stride_c1_s + l * stride_c1_l;
                uchar code;
                if (l1_bits == 8) {
                    code = angle_codes_l1[offset_c1 + j];
                } else {
                    uchar byte = angle_codes_l1[offset_c1 + j / 2];
                    code = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                }
                half norm = static_cast<half>(code) / l1_scale;
                half angle = (norm * 2.0h * M_PI_H) - M_PI_H;
                k_x = r * cos(angle);
                k_y = r * sin(angle);
            } else {
                // Deep bucket: LUT for 8-bit, trig for narrower configs
                uint rel_j = j - split_half_d;
                uint offset_cd = b * stride_cd_b + kv_head * stride_cd_h + s * stride_cd_s + l * stride_cd_l;
                if (deep_bits == 8) {
                    uchar code = angle_codes_deep[offset_cd + rel_j];
                    float2 cs = TQPOLAR_LUT_COSSIN_256[code];
                    k_x = r * half(cs.x);
                    k_y = r * half(cs.y);
                } else {
                    uchar code;
                    if (deep_bits == 4) {
                        uchar byte = angle_codes_deep[offset_cd + rel_j / 2];
                        code = (rel_j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                    } else {
                        uchar byte = angle_codes_deep[offset_cd + rel_j / 4];
                        uint shift = (rel_j % 4) * 2;
                        code = (byte >> shift) & 0x03;
                    }
                    half norm = static_cast<half>(code) / deep_scale;
                    half angle = (norm * 2.0h * M_PI_H) - M_PI_H;
                    k_x = r * cos(angle);
                    k_y = r * sin(angle);
                }
            }

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
            scores[dest_idx] = total_polar_score + qjl_correction * attention_scale;
        }
    }
}

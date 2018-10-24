#include "neonConvolutionAlgs.h"

/* Helper functions for convolution from ARM Compute Library: https://github.com/ARM-software/ComputeLibrary */

/**
 * Use: loading first 3 elements of a 5x5 kernel row for convolver
 * @param[in] m0  Pointer to first element of row
 * @param[in] m1  Pointer to second element of row
 * @param[in] m2  Pointer to third element of row
 * @return        Struct with three 4-element float vectors, each with identical lanes equal to the corresponding input element
 */
inline float32x4x3_t load_matrix_hi(const float *const m0, const float *const m1, const float *const m2)
{
    const float32x4x3_t m00 =
    {
        {
            vld1q_dup_f32(m0),
            vld1q_dup_f32(m1),
            vld1q_dup_f32(m2)
        }
    };
    return m00;
}

/**
 * Use: loading last 2 elements of a 5x5 kernel row for convolver
 * @param[in] m3  Pointer to fourth element of row
 * @param[in] m4  Pointer to fifth element of row
 * @return        Struct with two 4-element float vectors, each with identical lanes equal to the corresponding input element
 */
inline float32x4x2_t load_matrix_lo(const float *const m3, const float *const m4)
{
    const float32x4x2_t m00 =
    {
        {
            vld1q_dup_f32(m3),
            vld1q_dup_f32(m4)
        }
    };
    return m00;
}

/**
 * Use: loading a 12-element row from input for convolver
 * @param[in] in  Pointer to input (C-style float array)
 * @return        Struct with three 4-element float vectors, lanes corresponding to consecutive elements in input row
 */
inline float32x4x3_t load_input(const float *const in)
{
    const float32x4x3_t vin =
    {
        {
            vld1q_f32(in),
            vld1q_f32(in + 4),
            vld1q_f32(in + 8)
        }
    };
    return vin;
}

/**
 * Use: convolving 5x12 input with 5x5 kernel
 * 
 * @param[in] in_0  Pointer to first row of input (C-style float array)
 * @param[in] in_1  Pointer to second row of input (C-style float array)
 * @param[in] in_2  Pointer to third row of input (C-style float array)
 * @param[in] in_3  Pointer to fourth row of input (C-style float array)
 * @param[in] in_4  Pointer to fifth row of input (C-style float array)
 * 
 * @param[in] m0    Pointer to first row of kernel (C-style float array)
 * @param[in] m1    Pointer to second row of kernel (C-style float array)
 * @param[in] m2    Pointer to third row of kernel (C-style float array)
 * @param[in] m3    Pointer to fourth row of kernel (C-style float array)
 * @param[in] m4    Pointer to fifth row of kernel (C-style float array)
 * 
 * @param[in] bias  Value of bias
 * 
 * @return          2x4 Result of convolution
 */
inline float32x4x2_t convolve_5x5(const float *in_0, const float *in_1, const float *in_2, const float *in_3, const float *in_4,
                                  const float *m0,   const float *m1,   const float *m2,   const float *m3,   const float *m4,
								  const float bias)
{
	const float32x4_t   b    = vdupq_n_f32(bias);
    const float32x4x3_t vin0 = load_input(in_0);
    const float32x4x3_t vin1 = load_input(in_1);
    const float32x4x3_t vin2 = load_input(in_2);
    const float32x4x3_t vin3 = load_input(in_3);
    const float32x4x3_t vin4 = load_input(in_4);
    const float32x4x3_t m00  = load_matrix_hi(m0, 1 + m0, 2 + m0);
    const float32x4x2_t m01  = load_matrix_lo(3 + m0, 4 + m0);
    const float32x4x3_t m10  = load_matrix_hi(m1, 1 + m1, 2 + m1);
    const float32x4x2_t m11  = load_matrix_lo(3 + m1, 4 + m1);
    const float32x4x3_t m20  = load_matrix_hi(m2, 1 + m2, 2 + m2);
    const float32x4x2_t m21  = load_matrix_lo(3 + m2, 4 + m2);
    const float32x4x3_t m30  = load_matrix_hi(m3, 1 + m3, 2 + m3);
    const float32x4x2_t m31  = load_matrix_lo(3 + m3, 4 + m3);
    const float32x4x3_t m40  = load_matrix_hi(m4, 1 + m4, 2 + m4);
    const float32x4x2_t m41  = load_matrix_lo(3 + m4, 4 + m4);

    float32x4x2_t out =
    {
        {
            vmlaq_f32(b, vin0.val[0], m00.val[0]),
            vmlaq_f32(b, vin0.val[1], m00.val[0])
        }
    };

    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin0.val[0], vin0.val[1], 1), m00.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin0.val[0], vin0.val[1], 2), m00.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin0.val[0], vin0.val[1], 3), m01.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin0.val[1], m01.val[1]);

    out.val[0] = vmlaq_f32(out.val[0], vin1.val[0], m10.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin1.val[0], vin1.val[1], 1), m10.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin1.val[0], vin1.val[1], 2), m10.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin1.val[0], vin1.val[1], 3), m11.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin1.val[1], m11.val[1]);

    out.val[0] = vmlaq_f32(out.val[0], vin2.val[0], m20.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin2.val[0], vin2.val[1], 1), m20.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin2.val[0], vin2.val[1], 2), m20.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin2.val[0], vin2.val[1], 3), m21.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin2.val[1], m21.val[1]);

    out.val[0] = vmlaq_f32(out.val[0], vin3.val[0], m30.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin3.val[0], vin3.val[1], 1), m30.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin3.val[0], vin3.val[1], 2), m30.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin3.val[0], vin3.val[1], 3), m31.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin3.val[1], m31.val[1]);

    out.val[0] = vmlaq_f32(out.val[0], vin4.val[0], m40.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin4.val[0], vin4.val[1], 1), m40.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin4.val[0], vin4.val[1], 2), m40.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin4.val[0], vin4.val[1], 3), m41.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin4.val[1], m41.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin0.val[1], vin0.val[2], 1), m00.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin0.val[1], vin0.val[2], 2), m00.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin0.val[1], vin0.val[2], 3), m01.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin0.val[2], m01.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vin1.val[1], m10.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin1.val[1], vin1.val[2], 1), m10.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin1.val[1], vin1.val[2], 2), m10.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin1.val[1], vin1.val[2], 3), m11.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin1.val[2], m11.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vin2.val[1], m20.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin2.val[1], vin2.val[2], 1), m20.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin2.val[1], vin2.val[2], 2), m20.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin2.val[1], vin2.val[2], 3), m21.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin2.val[2], m21.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vin3.val[1], m30.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin3.val[1], vin3.val[2], 1), m30.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin3.val[1], vin3.val[2], 2), m30.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin3.val[1], vin3.val[2], 3), m31.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin3.val[2], m31.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vin4.val[1], m40.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin4.val[1], vin4.val[2], 1), m40.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin4.val[1], vin4.val[2], 2), m40.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin4.val[1], vin4.val[2], 3), m41.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin4.val[2], m41.val[1]);

    return out;
}

/**
 * Use: loading first 3 elements of a 5x5 uint kernel row for convolver
 * @param[in] m0  Pointer to first element of row
 * @param[in] m1  Pointer to second element of row
 * @param[in] m2  Pointer to third element of row
 * @return        Struct with three 4-element uint32_t vectors, each with identical lanes equal to the corresponding input element
 */
inline uint32x4x3_t load_matrix_hiU(const uint32_t *const m0, const uint32_t *const m1, const uint32_t *const m2)
{
    const uint32x4x3_t m00 =
    {
        {
            vld1q_dup_u32(m0),
            vld1q_dup_u32(m1),
            vld1q_dup_u32(m2)
        }
    };
    return m00;
}

/**
 * Use: loading last 2 elements of a 5x5 uint kernel row for convolver
 * @param[in] m3  Pointer to fourth element of row
 * @param[in] m4  Pointer to fifth element of row
 * @return        Struct with two 4-element uint32_t vectors, each with identical lanes equal to the corresponding input element
 */
inline uint32x4x3_t load_matrix_loU(const uint32_t *const m3, const uint32_t *const m4)
{
    const uint32x4x3_t m00 =
    {
        {
            vld1q_dup_u32(m3),
            vld1q_dup_u32(m4)
        }
    };
    return m00;
}

/**
 * Use: loading a 12-element row from uint input for convolver
 * @param[in] in  Pointer to input (C-style uint32_t array)
 * @return        Struct with three 4-element uint32_t vectors, lanes corresponding to consecutive elements in input row
 */
inline uint32x4x3_t load_inputU(const uint32_t *const in)
{
    const uint32x4x3_t vin =
    {
        {
            vld1q_u32(in),
            vld1q_u32(in + 4),
            vld1q_u32(in + 8)
        }
    };
    return vin;
}

/**
 * Use: convolving 5x12 input with 5x5 kernel
 * 
 * @param[in] in_0  Pointer to first  row of input (C-style float array)
 * @param[in] in_1  Pointer to second row of input (C-style float array)
 * @param[in] in_2  Pointer to third  row of input (C-style float array)
 * @param[in] in_3  Pointer to fourth row of input (C-style float array)
 * @param[in] in_4  Pointer to fifth  row of input (C-style float array)
 * 
 * @param[in] n0    Pointer to first  row of XOR kernel (C-style float array)
 * @param[in] n1    Pointer to second row of XOR kernel (C-style float array)
 * @param[in] n2    Pointer to third  row of XOR kernel (C-style float array)
 * @param[in] n3    Pointer to fourth row of XOR kernel (C-style float array)
 * @param[in] n4    Pointer to fifth  row of XOR kernel (C-style float array)
 * 
 * @param[in] z0    Pointer to first  row of AND kernel (C-style float array)
 * @param[in] z1    Pointer to second row of AND kernel (C-style float array)
 * @param[in] z2    Pointer to third  row of AND kernel (C-style float array)
 * @param[in] z3    Pointer to fourth row of AND kernel (C-style float array)
 * @param[in] z4    Pointer to fifth  row of AND kernel (C-style float array)
 * 
 * @param[in] bias  Value of bias
 * 
 * @return          2x4 Result of convolution
 */
inline float32x4x2_t convolve_ternary_5x5(const uint32_t *in_0, const uint32_t *in_1, const uint32_t *in_2, const uint32_t *in_3, const uint32_t *in_4,
                                          const uint32_t *n0,   const uint32_t *n1,   const uint32_t *n2,   const uint32_t *n3,   const uint32_t *n4,
                                          const uint32_t *z0,   const uint32_t *z1,   const uint32_t *z2,   const uint32_t *z3,   const uint32_t *z4,
                                          const float bias)
{
	const float32x4_t   b    = vdupq_n_f32(bias);
    const uint32x4x3_t  vin0 = load_inputU((in_0));
    const uint32x4x3_t  vin1 = load_inputU((in_1));
    const uint32x4x3_t  vin2 = load_inputU((in_2));
    const uint32x4x3_t  vin3 = load_inputU((in_3));
    const uint32x4x3_t  vin4 = load_inputU((in_4));
    
    const uint32x4x3_t  n00  = load_matrix_hiU(n0, 1 + n0, 2 + n0);
    const uint32x4x3_t  n01  = load_matrix_loU(3 + n0, 4 + n0);
    const uint32x4x3_t  n10  = load_matrix_hiU(n1, 1 + n1, 2 + n1);
    const uint32x4x3_t  n11  = load_matrix_loU(3 + n1, 4 + n1);
    const uint32x4x3_t  n20  = load_matrix_hiU(n2, 1 + n2, 2 + n2);
    const uint32x4x3_t  n21  = load_matrix_loU(3 + n2, 4 + n2);
    const uint32x4x3_t  n30  = load_matrix_hiU(n3, 1 + n3, 2 + n3);
    const uint32x4x3_t  n31  = load_matrix_loU(3 + n3, 4 + n3);
    const uint32x4x3_t  n40  = load_matrix_hiU(n4, 1 + n4, 2 + n4);
    const uint32x4x3_t  n41  = load_matrix_loU(3 + n4, 4 + n4);
    
    const uint32x4x3_t  z00  = load_matrix_hiU(z0, 1 + z0, 2 + z0);
    const uint32x4x3_t  z01  = load_matrix_loU(3 + z0, 4 + z0);
    const uint32x4x3_t  z10  = load_matrix_hiU(z1, 1 + z1, 2 + z1);
    const uint32x4x3_t  z11  = load_matrix_loU(3 + z1, 4 + z1);
    const uint32x4x3_t  z20  = load_matrix_hiU(z2, 1 + z2, 2 + z2);
    const uint32x4x3_t  z21  = load_matrix_loU(3 + z2, 4 + z2);
    const uint32x4x3_t  z30  = load_matrix_hiU(z3, 1 + z3, 2 + z3);
    const uint32x4x3_t  z31  = load_matrix_loU(3 + z3, 4 + z3);
    const uint32x4x3_t  z40  = load_matrix_hiU(z4, 1 + z4, 2 + z4);
    const uint32x4x3_t  z41  = load_matrix_loU(3 + z4, 4 + z4);

    float32x4x2_t out =
    {
        {
            b,
            b
        }
    };

    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin0.val[0], n00.val[0]), z00.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin0.val[0], vin0.val[1], 1), n00.val[1]), z00.val[1])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin0.val[0], vin0.val[1], 2), n00.val[2]), z00.val[2])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin0.val[0], vin0.val[1], 3), n01.val[0]), z01.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin0.val[1], n01.val[1]), z01.val[1])));
    
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin1.val[0], n10.val[0]), z10.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin1.val[0], vin1.val[1], 1), n10.val[1]), z10.val[1])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin1.val[0], vin1.val[1], 2), n10.val[2]), z10.val[2])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin1.val[0], vin1.val[1], 3), n11.val[0]), z11.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin1.val[1], n11.val[1]), z11.val[1])));
    
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin2.val[0], n20.val[0]), z20.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin2.val[0], vin2.val[1], 1), n20.val[1]), z20.val[1])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin2.val[0], vin2.val[1], 2), n20.val[2]), z20.val[2])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin2.val[0], vin2.val[1], 3), n21.val[0]), z21.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin2.val[1], n21.val[1]), z21.val[1])));
    
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin3.val[0], n30.val[0]), z30.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin3.val[0], vin3.val[1], 1), n30.val[1]), z30.val[1])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin3.val[0], vin3.val[1], 2), n30.val[2]), z30.val[2])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin3.val[0], vin3.val[1], 3), n31.val[0]), z31.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin3.val[1], n31.val[1]), z31.val[1])));
    
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin4.val[0], n40.val[0]), z40.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin4.val[0], vin4.val[1], 1), n40.val[1]), z40.val[1])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin4.val[0], vin4.val[1], 2), n40.val[2]), z40.val[2])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin4.val[0], vin4.val[1], 3), n41.val[0]), z41.val[0])));
    out.val[0] = vaddq_f32(out.val[0], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin4.val[1], n41.val[1]), z41.val[1])));
    
    //
    
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin0.val[1], n00.val[0]), z00.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin0.val[1], vin0.val[2], 1), n00.val[1]), z00.val[1])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin0.val[1], vin0.val[2], 2), n00.val[2]), z00.val[2])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin0.val[1], vin0.val[2], 3), n01.val[0]), z01.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin0.val[2], n01.val[1]), z01.val[1])));
    
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin1.val[1], n10.val[0]), z10.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin1.val[1], vin1.val[2], 1), n10.val[1]), z10.val[1])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin1.val[1], vin1.val[2], 2), n10.val[2]), z10.val[2])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin1.val[1], vin1.val[2], 3), n11.val[0]), z11.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin1.val[2], n11.val[1]), z11.val[1])));
    
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin2.val[1], n20.val[0]), z20.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin2.val[1], vin2.val[2], 1), n20.val[1]), z20.val[1])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin2.val[1], vin2.val[2], 2), n20.val[2]), z20.val[2])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin2.val[1], vin2.val[2], 3), n21.val[0]), z21.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin2.val[2], n21.val[1]), z21.val[1])));
    
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin3.val[1], n30.val[0]), z30.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin3.val[1], vin3.val[2], 1), n30.val[1]), z30.val[1])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin3.val[1], vin3.val[2], 2), n30.val[2]), z30.val[2])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin3.val[1], vin3.val[2], 3), n31.val[0]), z31.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin3.val[2], n31.val[1]), z31.val[1])));
    
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin4.val[1], n40.val[0]), z40.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin4.val[1], vin4.val[2], 1), n40.val[1]), z40.val[1])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin4.val[1], vin4.val[2], 2), n40.val[2]), z40.val[2])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vextq_u32(vin4.val[1], vin4.val[2], 3), n41.val[0]), z41.val[0])));
    out.val[1] = vaddq_f32(out.val[1], vreinterpretq_f32_u32(vandq_u32(veorq_u32(vin4.val[2], n41.val[1]), z41.val[1])));
    
    return out;
}



/*
  --------------------------------------------------------------
  Exported functions, see header file for details
  --------------------------------------------------------------
 */

Matrix24f convolve28(const Matrix28f& input, const Matrix5f& kernel, const float bias)
{
	const int output_w = 24;
	const int output_h = 24;
	const int step     = 8;

	const float* k_ptr     = kernel.data();
	const float* input_ptr = input.data();

	float out[24*24];

	const float* k_r0 = k_ptr + 0;
	const float* k_r1 = k_ptr + 5;
	const float* k_r2 = k_ptr + 10;
	const float* k_r3 = k_ptr + 15;
	const float* k_r4 = k_ptr + 20;

	for (int h = 0; h < output_h; ++h)
	{
		const float* in_0  = input_ptr + (h + 0) * 28;
		const float* in_1  = input_ptr + (h + 1) * 28;
		const float* in_2  = input_ptr + (h + 2) * 28;
		const float* in_3  = input_ptr + (h + 3) * 28;
		const float* in_4  = input_ptr + (h + 4) * 28;
		float*       p_out = out + h * 24;
		for (int w = 0; w < output_w; w += step,
			in_0 += step, in_1 += step, in_2 += step, in_3 += step, in_4 += step, p_out += step)
		{
            float32x4x2_t c =  convolve_5x5(in_0, in_1, in_2, in_3, in_4, k_r0, k_r1, k_r2, k_r3, k_r4, bias);
			vst1q_f32(p_out + 0, c.val[0]);
            vst1q_f32(p_out + 4, c.val[1]);
		}
	}

	Eigen::Map<Matrix24f> res(out);
	return res;
};



Matrix8f convolve12(const Matrix12f& input, const Matrix5f& kernel, const float bias)
{
	const int output_w = 8;
	const int output_h = 8;

	const float* k_ptr     = kernel.data();
	const float* input_ptr = input.data();

	float out[64];

	const float* k_r0 = k_ptr + 0;
	const float* k_r1 = k_ptr + 5;
	const float* k_r2 = k_ptr + 10;
	const float* k_r3 = k_ptr + 15;
	const float* k_r4 = k_ptr + 20;

	for (int h = 0; h < output_h; ++h)
	{
		const float* in_0  = input_ptr + (h + 0) * 12;
		const float* in_1  = input_ptr + (h + 1) * 12;
		const float* in_2  = input_ptr + (h + 2) * 12;
		const float* in_3  = input_ptr + (h + 3) * 12;
		const float* in_4  = input_ptr + (h + 4) * 12;
		float*       p_out = out + h * 8;
        
        float32x4x2_t c =  convolve_5x5(in_0, in_1, in_2, in_3, in_4, k_r0, k_r1, k_r2, k_r3, k_r4, bias);
        vst1q_f32(p_out + 0, c.val[0]);
        vst1q_f32(p_out + 4, c.val[1]);
	}

	Eigen::Map<Matrix8f> res(out);
	return res;
};



Matrix12f maxPool24(const Matrix24f& mat)
{
	const float *data = mat.data(), *r1, *r2;
	float out[12*12];

	float32x4x2_t q11, q12, q13, q21, q22, q23;
	float32x4_t res;

	int h1, h2;

	for (int h = 0; h < 12; ++h)
	{
		h1 = 48 * h;
		h2 = h1 + 24;

		r1 = data + h1;
		r2 = data + h2;

		q11 = vld2q_f32(r1);
		q12 = vld2q_f32(r1+8);
		q13 = vld2q_f32(r1+16);
		q21 = vld2q_f32(r2);
		q22 = vld2q_f32(r2+8);
		q23 = vld2q_f32(r2+16);

		res = vmaxq_f32(q11.val[0], q11.val[1]);
		res = vmaxq_f32(res, q21.val[0]);
		res = vmaxq_f32(res, q21.val[1]);
		vst1q_f32(out + h * 12 + 0, res);

		res = vmaxq_f32(q12.val[0], q12.val[1]);
		res = vmaxq_f32(res, q22.val[0]);
		res = vmaxq_f32(res, q22.val[1]);
		vst1q_f32(out + h * 12 + 4, res);

		res = vmaxq_f32(q13.val[0], q13.val[1]);
		res = vmaxq_f32(res, q23.val[0]);
		res = vmaxq_f32(res, q23.val[1]);
		vst1q_f32(out + h * 12 + 8, res);
	}

	return Eigen::Map<Matrix12f>(out);
}



Matrix4f maxPool8(const Matrix8f& mat)
{
	const float *data = mat.data(), *r1, *r2;
	float out[4*4];

	float32x4x2_t q11, q12, q13, q21, q22, q23;
	float32x4_t res;

	int h1, h2;

	for (int h = 0; h < 4; ++h)
	{
		h1 = 16 * h;
		h2 = h1 + 8;

		r1 = data + h1;
		r2 = data + h2;

		q11 = vld2q_f32(r1);
		q21 = vld2q_f32(r2);

		res = vmaxq_f32(q11.val[0], q11.val[1]);
		res = vmaxq_f32(res, q21.val[0]);
		res = vmaxq_f32(res, q21.val[1]);
		vst1q_f32(out + h * 4, res);
	}

	return Eigen::Map<Matrix4f>(out);
}



void relu12(Matrix12f& mat)
{
	float       *data  = mat.data();
	float32x4_t zero   = vdupq_n_f32(0);

	for (int h = 0; h < 144; h+=4)
	{
		vst1q_f32(data + h, vmaxq_f32(vld1q_f32(data + h), zero));
	}
}



void relu4(Matrix4f& mat)
{
	float       *data  = mat.data();
	float32x4_t zero   = vdupq_n_f32(0);

	for (int h = 0; h < 16; h+=4)
	{
		vst1q_f32(data + h, vmaxq_f32(vld1q_f32(data + h), zero));
	}
}



void reluFC1(MatrixFCi& mat)
{
    float       *data  = mat.data();
	float32x4_t zero4  = vdupq_n_f32(0);
    float32x2_t zero2  = vdup_n_f32(0);

	for (int h = 0; h < 48; h+=4)
	{
		vst1q_f32(data + h, vmaxq_f32(vld1q_f32(data + h), zero4));
	}
    
    vst1_f32(data + 48, vmax_f32(vld1_f32(data + 48), zero2));
}



void batchNorm(Matrix12f& mat, float mean, float var, float eps, float gamma, float beta)
{
	float       *data       = mat.data();
	float       out[12*12];
	float32x4_t res,
				m           = vdupq_n_f32(mean),
				n           = vdupq_n_f32(gamma / std::sqrt(var + eps)),
				p           = vdupq_n_f32(beta);

	for (int h = 0; h < 144; h+=4)
	{
		res = vsubq_f32(vld1q_f32(data + h), m);
		res = vmlaq_f32(p, res, n);
        
		vst1q_f32(data + h, res);
	}
}



MatrixFCi fullyConnect1(const Matrix4f (&inputs)[20], const MatrixFC1& layerWeight)
{
	const float   *w       = layerWeight.data();
	float32x4_t   tRes[50] = {};
	float         res[50];
	int           m;
	float32x4x4_t mat;

	for (int32_t inputIndex = 0; inputIndex < 20; ++inputIndex)
	{
		mat = vld4q_f32(inputs[inputIndex].data());

		for (int32_t oh = 0; oh < 50; ++oh)
		{
			m = oh * 320 + inputIndex * 16;
			tRes[oh] = vmlaq_f32(tRes[oh], mat.val[0], vld1q_f32(w + m + 0));
			tRes[oh] = vmlaq_f32(tRes[oh], mat.val[1], vld1q_f32(w + m + 4));
			tRes[oh] = vmlaq_f32(tRes[oh], mat.val[2], vld1q_f32(w + m + 8));
			tRes[oh] = vmlaq_f32(tRes[oh], mat.val[3], vld1q_f32(w + m + 12));
		}

	}

	for (int oh = 0; oh < 50; ++oh)
	{
		res[oh] = tRes[oh][0] + tRes[oh][1] + tRes[oh][2] + tRes[oh][3];
	}

	return Eigen::Map<MatrixFCi>(res);
}



MatrixO fullyConnect2(const MatrixFCi& input, const MatrixFC2& layerWeight)
{
	const float *w        = layerWeight.data(), *i = input.data();
	float32x4_t tRes[10]  = {};
	float32x2_t tsRes[10] = {};
	float       res[10];
	int         m;

	for (int32_t oh = 0; oh < 10; ++oh)
	{
		m = oh * 50;
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 0),  vld1q_f32(w + m + 0));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 4),  vld1q_f32(w + m + 4));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 8),  vld1q_f32(w + m + 8));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 12), vld1q_f32(w + m + 12));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 16), vld1q_f32(w + m + 16));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 20), vld1q_f32(w + m + 20));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 24), vld1q_f32(w + m + 24));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 28), vld1q_f32(w + m + 28));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 32), vld1q_f32(w + m + 32));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 36), vld1q_f32(w + m + 36));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 40), vld1q_f32(w + m + 40));
		tRes[oh] = vmlaq_f32(tRes[oh], vld1q_f32(i + 44), vld1q_f32(w + m + 44));

		tsRes[oh] = vmla_f32(tsRes[oh], vld1_f32(i + 48),  vld1_f32(w + m + 48));
	}

	for (int oh = 0; oh < 10; ++oh)
	{
		res[oh] = tRes[oh][0] + tRes[oh][1] + tRes[oh][2] + tRes[oh][3] + tsRes[oh][0] + tsRes[oh][1];
	}

	return Eigen::Map<MatrixO>(res);
}



Matrix8f addTen8x8(const Matrix8f (&inputs)[10])
{
    float       res[8*8];
    float32x4_t tRes;
    const float *i0  = inputs[0].data(),
                *i1  = inputs[1].data(),
                *i2  = inputs[2].data(),
                *i3  = inputs[3].data(),
                *i4  = inputs[4].data(),
                *i5  = inputs[5].data(),
                *i6  = inputs[6].data(),
                *i7  = inputs[7].data(),
                *i8  = inputs[8].data(),
                *i9  = inputs[9].data();

    for ( int i = 0; i < 64; i += 4 )
    {
        tRes = vaddq_f32(vld1q_f32(i0 + i), vld1q_f32(i1 + i));
        tRes = vaddq_f32(tRes,              vld1q_f32(i2 + i));
        tRes = vaddq_f32(tRes,              vld1q_f32(i3 + i));
        tRes = vaddq_f32(tRes,              vld1q_f32(i4 + i));
        tRes = vaddq_f32(tRes,              vld1q_f32(i5 + i));
        tRes = vaddq_f32(tRes,              vld1q_f32(i6 + i));
        tRes = vaddq_f32(tRes,              vld1q_f32(i7 + i));
        tRes = vaddq_f32(tRes,              vld1q_f32(i8 + i));
        tRes = vaddq_f32(tRes,              vld1q_f32(i9 + i));

        vst1q_f32(res + i, tRes);
    }

    return Eigen::Map<Matrix8f>(res);
}



MatrixFCi addFC1Bias(MatrixFCi& mat, const float *bias)
{
    float *data = mat.data();

	for (int h = 0; h < 48; h+=4)
	{
		vst1q_f32(data + h, vaddq_f32(vld1q_f32(data + h), vld1q_f32(bias + h)));
	}
    
    vst1_f32(data + 48, vadd_f32(vld1_f32(data + 48), vld1_f32(bias + 48)));
}



Matrix24f ternary28Conv(const Matrix28f& mat, const Matrix5u& neg, const Matrix5u& zer, const float bias)
{
    const int output_w = 24;
	const int output_h = 24;
	const int step     = 8;

	const uint32_t* neg_ptr   = neg.data();
	const uint32_t* zer_ptr   = zer.data();
	const uint32_t* input_ptr = reinterpret_cast<const uint32_t*>(mat.data());

	float out[24*24];

	const uint32_t* n_r0 = neg_ptr + 0;
	const uint32_t* n_r1 = neg_ptr + 5;
	const uint32_t* n_r2 = neg_ptr + 10;
	const uint32_t* n_r3 = neg_ptr + 15;
	const uint32_t* n_r4 = neg_ptr + 20;
    
    const uint32_t* z_r0 = zer_ptr + 0;
	const uint32_t* z_r1 = zer_ptr + 5;
	const uint32_t* z_r2 = zer_ptr + 10;
	const uint32_t* z_r3 = zer_ptr + 15;
	const uint32_t* z_r4 = zer_ptr + 20;

	for (int h = 0; h < output_h; ++h)
	{
		const uint32_t* in_0  = input_ptr + (h + 0) * 28;
		const uint32_t* in_1  = input_ptr + (h + 1) * 28;
		const uint32_t* in_2  = input_ptr + (h + 2) * 28;
		const uint32_t* in_3  = input_ptr + (h + 3) * 28;
		const uint32_t* in_4  = input_ptr + (h + 4) * 28;
		float*       p_out = out + h * 24;
		for (int w = 0; w < output_w; w += step,
			in_0 += step, in_1 += step, in_2 += step, in_3 += step, in_4 += step, p_out += step)
		{
            float32x4x2_t c =  convolve_ternary_5x5(in_0, in_1, in_2, in_3, in_4, 
                                                    n_r0, n_r1, n_r2, n_r3, n_r4, 
                                                    z_r0, z_r1, z_r2, z_r3, z_r4, 
                                                    bias);
			vst1q_f32(p_out + 0, c.val[0]);
            vst1q_f32(p_out + 4, c.val[1]);
		}
	}

	Eigen::Map<Matrix24f> res(out);
	return res;
}

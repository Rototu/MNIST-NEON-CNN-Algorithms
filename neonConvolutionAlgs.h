#ifndef NEON_CONV_ALGS_H
#define NEON_CONV_ALGS_H

#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include "../Eigen/Core"



using Matrix4f  = Eigen::Matrix<float,    4,   4>;                          //!< 4x4    float  matrix
using Matrix5f  = Eigen::Matrix<float,    5,   5>;                          //!< 5x5    float  matrix
using Matrix5u  = Eigen::Matrix<uint32_t, 5,   5>;                          //!< 5x5    uint32 matrix
using Matrix8f  = Eigen::Matrix<float,    8,   8>;                          //!< 8x8    float  matrix
using Matrix12f = Eigen::Matrix<float,    12,  12>;                         //!< 12x12  float  matrix
using Matrix24f = Eigen::Matrix<float,    24,  24>;                         //!< 24x24  float  matrix
using Matrix28f = Eigen::Matrix<float,    28,  28>;                         //!< 28x28  float  matrix

using MatrixO   = Eigen::Matrix<float,    1,   10>;                         //!< 1x10   float  matrix
using MatrixFCi = Eigen::Matrix<float,    1,   50>;                         //!< 1x50   float  matrix

using MatrixXf  = Eigen::Matrix<float,    Eigen::Dynamic, Eigen::Dynamic>;  //!< Dynamic float matrix

using MatrixFC1 = Eigen::Matrix<float,    50,  320, Eigen::RowMajor>;       //!< 50x320  float matrix stored in row-major order
using MatrixFC2 = Eigen::Matrix<float,    10,  50,  Eigen::RowMajor>;       //!< 10x50   float matrix stored in row-major order



/**
 * Use: convolve 28x28 Matrix (MNIST image size) with 5x5 weight without padding, no stride
 * Adapted code from ARM Compute Library
 * NOTE: convolution is implemented as follows per block (Haskell notation)
 *       bias + sum . (coeffMul kernel) input5x5Block
 *
 * @param[in] input   28x28 input (Eigen matrix)
 * @param[in] kernel  5x5 weight (Eigen matrix)
 * @param[in] bias    Value of bias
 * @return            24x24 convolution result
 */
Matrix24f convolve28(const Matrix28f& input, const Matrix5f& kernel, const float bias);



/**
 * Use: convolution of 12x12 Matrix (MNIST image size) with 5x5 weight without padding, no stride
 * Adapted code from ARM Compute Library
 * see "convolve28" notes for algorithm details
 *
 * @param[in] input   12x12 input (Eigen matrix)
 * @param[in] kernel  5x5   weight (Eigen matrix)
 * @param[in] bias    Value of bias
 * @return            8x8 convolution result
 */
Matrix8f convolve12(const Matrix12f& input, const Matrix5f& kernel, const float bias);



/**
 * Use: 2x2 max-pooling of 12x12 matrix, stride == 2
 *
 * @param[in] mat  12x12 input (Eigen matrix)
 * @return         12x12 max-pool result
 */
Matrix12f maxPool24(const Matrix24f& mat);



/**
 * Use: 2x2 max-pooling of 8x8 matrix, stride == 2
 *
 * @param[in] mat  12x12 input (Eigen matrix)
 * @return         8x8 max-pool result
 */
Matrix4f maxPool8(const Matrix8f& mat);



/**
 * Use: In place RELU on 12x12 matrix
 *
 * @param[in,out] mat  12x12 Eigen matrix
 */
void relu12(Matrix12f& mat);



/**
 * Use: In place RELU on 4x4 matrix
 *
 * @param[in,out] mat  4x4 Eigen matrix
 */
void relu4(Matrix4f& mat);



/**
 * Use: In place RELU on 1x50 matrix
 * 
 * @param[in,out] mat  1x50 Eigen matrix
 */
void reluFC1(MatrixFCi& mat);



/**
 * Use: In place batch normalization on 12x12 matrix
 * Formula: (x - E[x]) * gamma / sqrt(Var[x] + epsilon) + beta
 *
 * @param[in,out] mat    12x12 Eigen matrix)
 * @param[in]     mean   Expectation
 * @param[in]     var    Variance
 * @param[in]     eps    Epsilon (small positive value -> 0)
 * @param[in]     gamma  Gamma
 * @param[in]     beta   Beta
 */
void batchNorm(Matrix12f& mat, float mean, float var, float eps, float gamma, float beta);



/**
 * Use: Forward propagation through a fully connected layer of 50 neurons with 320 inputs
 *
 * @param[in] inputs       C-style array containing twenty 4x4 matrices
 * @param[in] layerWeight  50x320 weight matrix for whole layer (one row per neuron)
 * @return                 1x50 Matrix as layer output (one element per neuron)
 */
MatrixFCi fullyConnect1(const Matrix4f (&inputs)[20], const MatrixFC1& layerWeight);



/**
 * Use: Forward propagation through a fully connected layer of 10 neurons with 50 inputs
 *
 * @param[in] input        1x50 input matrix
 * @param[in] layerWeight  10x50 weight matrix for whole layer (one row per neuron)
 * @return                 1x10 Matrix as layer output (one element per neuron)
 */
MatrixO fullyConnect2(const MatrixFCi& input, const MatrixFC2& layerWeight);



/**
 * Use: Add ten 12x12 input matrices
 *
 * @param[in] mats         C-style array containing ten 8x8 matrices
 * @return                 Sum of matrices
 */
Matrix8f addTen8x8(const Matrix8f (&inputs)[10]);



/**
 * Use: Add bias in place to FC1 result
 * 
 * @param[in, out] mat   1x50 Eigen matrix FC1 result
 * @param[in]      bias  50el FC1 bias as C-style array
 */
MatrixFCi addFC1Bias(MatrixFCi& mat, const float *bias);



/**
 * Use: Ternary convolution on 28x28 matrices without multiplication
 * 
 * @param[in] mat   28x28 input 
 * @param[in] neg   5x5 XOR kernel
 * @param[in] zer   5x5 AND kernel
 * @param[in] bias  convolution bias
 * @return          24x24 convolution result
 */
Matrix24f ternary28Conv(const Matrix28f& mat, const Matrix5u& neg, const Matrix5u& zer, const float bias);



#endif // NEON_CONV_ALGS

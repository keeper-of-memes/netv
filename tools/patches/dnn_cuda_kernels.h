/*
 * Copyright (c) 2024
 *
 * This file is part of FFmpeg.
 *
 * CUDA kernel declarations for DNN backend format conversion.
 */

#ifndef AVFILTER_DNN_CUDA_KERNELS_H
#define AVFILTER_DNN_CUDA_KERNELS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Convert HWC uint8 RGB24 to NCHW float32.
 *
 * @param input Input buffer on GPU (uint8, HWC format with row padding)
 * @param output Output buffer on GPU (float32, NCHW format)
 * @param height Image height
 * @param width Image width
 * @param input_linesize Input row stride in bytes (may include padding)
 * @param stream CUDA stream for async execution
 * @return 0 on success, CUDA error code on failure
 */
int cuda_hwc_uint8_to_nchw_float32(
    const void* input, void* output,
    int height, int width, int input_linesize,
    cudaStream_t stream);

/**
 * Convert NCHW float32 to HWC uint8 RGB24.
 *
 * @param input Input buffer on GPU (float32, NCHW format, values [0,1])
 * @param output Output buffer on GPU (uint8, HWC format with row padding)
 * @param height Image height
 * @param width Image width
 * @param output_linesize Output row stride in bytes (may include padding)
 * @param stream CUDA stream for async execution
 * @return 0 on success, CUDA error code on failure
 */
int cuda_nchw_float32_to_hwc_uint8(
    const void* input, void* output,
    int height, int width, int output_linesize,
    cudaStream_t stream);

/**
 * Convert 4-channel HWC uint8 to NCHW float32.
 *
 * @param input Input buffer on GPU (uint8, HWC 4-channel format)
 * @param output Output buffer on GPU (float32, NCHW 3-channel format)
 * @param height Image height
 * @param width Image width
 * @param input_linesize Input row stride in bytes
 * @param alpha_first 0 for RGB0/RGBA (alpha last), 1 for 0RGB/ARGB (alpha first)
 * @param stream CUDA stream for async execution
 * @return 0 on success, CUDA error code on failure
 */
int cuda_hwc4_uint8_to_nchw_float32(
    const void* input, void* output,
    int height, int width, int input_linesize,
    int alpha_first,
    cudaStream_t stream);

/**
 * Convert NCHW float32 to 4-channel HWC uint8.
 *
 * @param input Input buffer on GPU (float32, NCHW 3-channel format)
 * @param output Output buffer on GPU (uint8, HWC 4-channel format)
 * @param height Image height
 * @param width Image width
 * @param output_linesize Output row stride in bytes
 * @param alpha_first 0 for RGB0/RGBA (alpha last), 1 for 0RGB/ARGB (alpha first)
 * @param stream CUDA stream for async execution
 * @return 0 on success, CUDA error code on failure
 */
int cuda_nchw_float32_to_hwc4_uint8(
    const void* input, void* output,
    int height, int width, int output_linesize,
    int alpha_first,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  /* AVFILTER_DNN_CUDA_KERNELS_H */

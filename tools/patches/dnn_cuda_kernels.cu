/*
 * Copyright (c) 2024
 *
 * This file is part of FFmpeg.
 *
 * CUDA kernels for DNN backend format conversion.
 * Enables zero-copy GPU-resident processing for TensorRT inference.
 */

#include <cuda_runtime.h>
#include <cstdint>

// Kernel: HWC uint8 [0,255] -> NCHW float32 [0,1]
// Input: uint8 buffer in HWC format (height, width, 3) with possible row padding
// Output: float32 buffer in NCHW format (1, 3, height, width)
__global__ void hwc_uint8_to_nchw_float32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int height, int width, int input_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Input: HWC with potential row padding
    const uint8_t* row = input + y * input_linesize;
    uint8_t r = row[x * 3 + 0];
    uint8_t g = row[x * 3 + 1];
    uint8_t b = row[x * 3 + 2];

    // Output: NCHW (batch=1), scale to [0,1]
    int hw = height * width;
    output[0 * hw + y * width + x] = r / 255.0f;  // R channel
    output[1 * hw + y * width + x] = g / 255.0f;  // G channel
    output[2 * hw + y * width + x] = b / 255.0f;  // B channel
}

// Kernel: NCHW float32 [0,1] -> HWC uint8 [0,255]
// Input: float32 buffer in NCHW format (1, 3, height, width)
// Output: uint8 buffer in HWC format (height, width, 3) with possible row padding
__global__ void nchw_float32_to_hwc_uint8_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    int height, int width, int output_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int hw = height * width;

    // Input: NCHW (batch=1), values in [0,1]
    float r = input[0 * hw + y * width + x];
    float g = input[1 * hw + y * width + x];
    float b = input[2 * hw + y * width + x];

    // Clamp and scale to [0,255]
    r = fminf(fmaxf(r * 255.0f, 0.0f), 255.0f);
    g = fminf(fmaxf(g * 255.0f, 0.0f), 255.0f);
    b = fminf(fmaxf(b * 255.0f, 0.0f), 255.0f);

    // Output: HWC with potential row padding
    uint8_t* row = output + y * output_linesize;
    row[x * 3 + 0] = (uint8_t)r;
    row[x * 3 + 1] = (uint8_t)g;
    row[x * 3 + 2] = (uint8_t)b;
}

// Kernel: 4-channel HWC uint8 -> NCHW float32 (extract RGB, ignore alpha)
__global__ void hwc4_uint8_to_nchw_float32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int height, int width, int input_linesize,
    int r_offset, int g_offset, int b_offset)  // Channel offsets (0,1,2 or 1,2,3 depending on format)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint8_t* row = input + y * input_linesize;
    uint8_t r = row[x * 4 + r_offset];
    uint8_t g = row[x * 4 + g_offset];
    uint8_t b = row[x * 4 + b_offset];

    int hw = height * width;
    output[0 * hw + y * width + x] = r / 255.0f;
    output[1 * hw + y * width + x] = g / 255.0f;
    output[2 * hw + y * width + x] = b / 255.0f;
}

// Kernel: NCHW float32 -> 4-channel HWC uint8 (add alpha=255)
__global__ void nchw_float32_to_hwc4_uint8_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    int height, int width, int output_linesize,
    int r_offset, int g_offset, int b_offset, int a_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int hw = height * width;
    float r = input[0 * hw + y * width + x];
    float g = input[1 * hw + y * width + x];
    float b = input[2 * hw + y * width + x];

    r = fminf(fmaxf(r * 255.0f, 0.0f), 255.0f);
    g = fminf(fmaxf(g * 255.0f, 0.0f), 255.0f);
    b = fminf(fmaxf(b * 255.0f, 0.0f), 255.0f);

    uint8_t* row = output + y * output_linesize;
    row[x * 4 + r_offset] = (uint8_t)r;
    row[x * 4 + g_offset] = (uint8_t)g;
    row[x * 4 + b_offset] = (uint8_t)b;
    row[x * 4 + a_offset] = 255;  // Alpha = opaque
}

// C++ wrapper functions (extern "C" for linking with C code)
extern "C" {

// Convert HWC uint8 RGB24 to NCHW float32
// Returns 0 on success, non-zero on CUDA error
int cuda_hwc_uint8_to_nchw_float32(
    const void* input, void* output,
    int height, int width, int input_linesize,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    hwc_uint8_to_nchw_float32_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)input, (float*)output,
        height, width, input_linesize);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

// Convert NCHW float32 to HWC uint8 RGB24
int cuda_nchw_float32_to_hwc_uint8(
    const void* input, void* output,
    int height, int width, int output_linesize,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    nchw_float32_to_hwc_uint8_kernel<<<grid, block, 0, stream>>>(
        (const float*)input, (uint8_t*)output,
        height, width, output_linesize);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

// Convert 4-channel HWC uint8 to NCHW float32
// Format determines channel order: 0=RGB0/RGBA, 1=0RGB/ARGB
int cuda_hwc4_uint8_to_nchw_float32(
    const void* input, void* output,
    int height, int width, int input_linesize,
    int alpha_first,  // 0 = RGB0/RGBA, 1 = 0RGB/ARGB
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    int r_off, g_off, b_off;
    if (alpha_first) {
        // 0RGB/ARGB: A at 0, RGB at 1,2,3
        r_off = 1; g_off = 2; b_off = 3;
    } else {
        // RGB0/RGBA: RGB at 0,1,2, A at 3
        r_off = 0; g_off = 1; b_off = 2;
    }

    hwc4_uint8_to_nchw_float32_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)input, (float*)output,
        height, width, input_linesize,
        r_off, g_off, b_off);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

// Convert NCHW float32 to 4-channel HWC uint8
int cuda_nchw_float32_to_hwc4_uint8(
    const void* input, void* output,
    int height, int width, int output_linesize,
    int alpha_first,  // 0 = RGB0/RGBA, 1 = 0RGB/ARGB
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    int r_off, g_off, b_off, a_off;
    if (alpha_first) {
        a_off = 0; r_off = 1; g_off = 2; b_off = 3;
    } else {
        r_off = 0; g_off = 1; b_off = 2; a_off = 3;
    }

    nchw_float32_to_hwc4_uint8_kernel<<<grid, block, 0, stream>>>(
        (const float*)input, (uint8_t*)output,
        height, width, output_linesize,
        r_off, g_off, b_off, a_off);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

}  // extern "C"

/*
 * Copyright (c) 2024
 *
 * This file is part of FFmpeg.
 *
 * CUDA kernels for DNN backend format conversion.
 * Compiled to PTX at build time, loaded dynamically at runtime.
 * No cudart dependency - uses CUDA Driver API via FFmpeg's dynlink.
 */

extern "C" {

// Kernel: HWC uint8 [0,255] -> NCHW float32 [0,1]
// Input: uint8 buffer in HWC format (height, width, 3) with possible row padding
// Output: float32 buffer in NCHW format (1, 3, height, width)
__global__ void hwc_uint8_to_nchw_float32_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int height, int width, int input_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Input: HWC with potential row padding
    const unsigned char* row = input + y * input_linesize;
    unsigned char r = row[x * 3 + 0];
    unsigned char g = row[x * 3 + 1];
    unsigned char b = row[x * 3 + 2];

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
    unsigned char* __restrict__ output,
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
    unsigned char* row = output + y * output_linesize;
    row[x * 3 + 0] = (unsigned char)r;
    row[x * 3 + 1] = (unsigned char)g;
    row[x * 3 + 2] = (unsigned char)b;
}

// Kernel: 4-channel HWC uint8 -> NCHW float32 (extract RGB, ignore alpha)
__global__ void hwc4_uint8_to_nchw_float32_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int height, int width, int input_linesize,
    int r_offset, int g_offset, int b_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const unsigned char* row = input + y * input_linesize;
    unsigned char r = row[x * 4 + r_offset];
    unsigned char g = row[x * 4 + g_offset];
    unsigned char b = row[x * 4 + b_offset];

    int hw = height * width;
    output[0 * hw + y * width + x] = r / 255.0f;
    output[1 * hw + y * width + x] = g / 255.0f;
    output[2 * hw + y * width + x] = b / 255.0f;
}

// Kernel: NCHW float32 -> 4-channel HWC uint8 (add alpha=255)
__global__ void nchw_float32_to_hwc4_uint8_kernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
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

    unsigned char* row = output + y * output_linesize;
    row[x * 4 + r_offset] = (unsigned char)r;
    row[x * 4 + g_offset] = (unsigned char)g;
    row[x * 4 + b_offset] = (unsigned char)b;
    row[x * 4 + a_offset] = 255;  // Alpha = opaque
}

}  // extern "C"

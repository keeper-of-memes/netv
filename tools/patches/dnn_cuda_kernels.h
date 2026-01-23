/*
 * Copyright (c) 2024
 *
 * This file is part of FFmpeg.
 *
 * CUDA kernel PTX declarations for DNN backend format conversion.
 * Kernels are compiled to PTX at build time and loaded via Driver API at runtime.
 * This avoids any CUDA runtime (cudart) dependency.
 */

#ifndef AVFILTER_DNN_CUDA_KERNELS_H
#define AVFILTER_DNN_CUDA_KERNELS_H

#include <stddef.h>

/* PTX bytecode embedded at compile time via bin2c */
extern const unsigned char ff_dnn_cuda_kernels_ptx[];
extern const unsigned int ff_dnn_cuda_kernels_ptx_len;

/* Kernel names within the PTX module */
/* FP32 variants */
#define DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_FLOAT32     "hwc_uint8_to_nchw_float32_kernel"
#define DNN_CUDA_KERNEL_NCHW_FLOAT32_TO_HWC_UINT8     "nchw_float32_to_hwc_uint8_kernel"
#define DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_FLOAT32    "hwc4_uint8_to_nchw_float32_kernel"
#define DNN_CUDA_KERNEL_NCHW_FLOAT32_TO_HWC4_UINT8    "nchw_float32_to_hwc4_uint8_kernel"

/* FP16 variants */
#define DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_FLOAT16     "hwc_uint8_to_nchw_float16_kernel"
#define DNN_CUDA_KERNEL_NCHW_FLOAT16_TO_HWC_UINT8     "nchw_float16_to_hwc_uint8_kernel"
#define DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_FLOAT16    "hwc4_uint8_to_nchw_float16_kernel"
#define DNN_CUDA_KERNEL_NCHW_FLOAT16_TO_HWC4_UINT8    "nchw_float16_to_hwc4_uint8_kernel"

/* BF16 variants */
#define DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_BFLOAT16    "hwc_uint8_to_nchw_bfloat16_kernel"
#define DNN_CUDA_KERNEL_NCHW_BFLOAT16_TO_HWC_UINT8    "nchw_bfloat16_to_hwc_uint8_kernel"
#define DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_BFLOAT16   "hwc4_uint8_to_nchw_bfloat16_kernel"
#define DNN_CUDA_KERNEL_NCHW_BFLOAT16_TO_HWC4_UINT8   "nchw_bfloat16_to_hwc4_uint8_kernel"

#endif  /* AVFILTER_DNN_CUDA_KERNELS_H */

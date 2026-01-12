#!/usr/bin/env python3
"""Add dlopen for libtorch_cuda.so to fix CUDA kernel registration."""
import sys

content = open(sys.argv[1]).read()

# Find where we check for CUDA device and add dlopen before model loading
old = """    } else if (device.is_cuda()) {
        if (!at::cuda::is_available()) {
            av_log(ctx, AV_LOG_ERROR, "No CUDA device found\\n");
            goto fail;
        }
    }"""

new = """    } else if (device.is_cuda()) {
        if (!at::cuda::is_available()) {
            av_log(ctx, AV_LOG_ERROR, "No CUDA device found\\n");
            goto fail;
        }
        // Load CUDA kernels - required for libtorch CUDA ops
        static bool cuda_lib_loaded = false;
        if (!cuda_lib_loaded) {
            cuda_lib_loaded = true;
            void *cuda_handle = dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_GLOBAL);
            if (cuda_handle) {
                av_log(ctx, AV_LOG_DEBUG, "libtorch_cuda.so loaded\\n");
            } else {
                av_log(ctx, AV_LOG_WARNING, "Failed to load libtorch_cuda.so: %s\\n", dlerror());
            }
        }
    }"""

if old in content:
    content = content.replace(old, new, 1)
    print(content)
else:
    print("Pattern not found!", file=sys.stderr)
    sys.exit(1)

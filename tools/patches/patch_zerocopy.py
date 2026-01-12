#!/usr/bin/env python3
import sys

content = open(sys.argv[1]).read()

# Find the spot after tensor allocation and before the switch statement
old = """    infer_request->input_tensor = new torch::Tensor();
    infer_request->output = new torch::Tensor();

    switch (th_model->model.func_type) {"""

new = """    infer_request->input_tensor = new torch::Tensor();
    infer_request->output = new torch::Tensor();

    // Check for CUDA hardware frames (zero-copy path)
    if (task->in_frame->hw_frames_ctx && task->in_frame->format == AV_PIX_FMT_CUDA) {
        CUdeviceptr cuda_ptr = (CUdeviceptr)task->in_frame->data[0];

        if (cuda_ptr) {
            av_log(ctx, AV_LOG_INFO, "CUDA frame detected - using zero-copy path\\n");

            // Create tensor directly from CUDA memory
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            *infer_request->input_tensor = torch::from_blob(
                (void*)cuda_ptr,
                {1, input.dims[channel_idx], input.dims[height_idx], input.dims[width_idx]},
                options
            );

            return 0;
        }
    }

    switch (th_model->model.func_type) {"""

if old in content:
    content = content.replace(old, new, 1)
    print(content)
else:
    print("Pattern not found!", file=sys.stderr)
    sys.exit(1)

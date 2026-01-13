#!/usr/bin/env python3
"""Export PyTorch models to TensorRT engines for FFmpeg dnn_processing filter.

This script converts Real-ESRGAN and similar super-resolution models to TensorRT
engines (.engine files) that can be loaded by FFmpeg's TensorRT DNN backend.

Supports dynamic input shapes - a single engine handles a range of resolutions.

Usage:
    # Export with dynamic shapes (default: 256x270 to 2560x1280)
    python export-tensorrt.py --output model.engine

    # Export with custom dynamic range
    python export-tensorrt.py --min 256x270 --opt 1280x720 --max 2560x1280

    # Export from custom model
    python export-tensorrt.py --model /path/to/model.pth

Requirements:
    pip install torch onnx tensorrt

Example FFmpeg usage after export:
    ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model=model.engine" output.mp4
"""

import argparse
import os
import sys
import tempfile

def get_model(model_path=None):
    """Load or download Real-ESRGAN model."""
    import torch

    if model_path:
        # Try loading as TorchScript first
        try:
            model = torch.jit.load(model_path, map_location='cpu')
            print(f"Loaded TorchScript model from {model_path}")
            return model, True
        except:
            pass

        # Try loading as state dict with architecture
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            state_dict = torch.load(model_path, map_location='cpu')
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded RRDBNet model from {model_path}")
            return model, False
        except ImportError:
            pass

        # Try SRVGGNetCompact (Real-ESRGAN-anime/general models)
        try:
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            state_dict = torch.load(model_path, map_location='cpu')
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded SRVGGNetCompact model from {model_path}")
            return model, False
        except ImportError:
            pass

        raise RuntimeError(f"Could not load model from {model_path}. Install basicsr or realesrgan package.")

    # Download default model from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="ai-forever/Real-ESRGAN",
        filename="RealESRGAN_x4.pth"
    )
    return get_model(model_path)


def export_onnx(model, opt_shape, onnx_path, is_torchscript=False):
    """Export model to ONNX format with dynamic axes."""
    import torch

    opt_w, opt_h = opt_shape
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Optimal shape: 1x3x{opt_h}x{opt_w}")

    dummy_input = torch.randn(1, 3, opt_h, opt_w, device='cpu')

    # Dynamic axes for height and width (dimensions 2 and 3)
    dynamic_axes = {
        'input': {2: 'height', 3: 'width'},
        'output': {2: 'out_height', 3: 'out_width'}
    }

    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"  ONNX export complete (dynamic H/W)")


def build_engine(onnx_path, engine_path, min_shape, opt_shape, max_shape, fp16=False, workspace_gb=4):
    """Build TensorRT engine from ONNX model with dynamic shapes."""
    import tensorrt as trt

    min_w, min_h = min_shape
    opt_w, opt_h = opt_shape
    max_w, max_h = max_shape

    print(f"Building TensorRT engine: {engine_path}")
    print(f"  Dynamic shapes:")
    print(f"    min: {min_w}x{min_h}")
    print(f"    opt: {opt_w}x{opt_h}")
    print(f"    max: {max_w}x{max_h}")
    print(f"  FP16: {fp16}")
    print(f"  Workspace: {workspace_gb} GB")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    # Shape format: (batch, channels, height, width)
    profile.set_shape(input_name,
                      min=(1, 3, min_h, min_w),
                      opt=(1, 3, opt_h, opt_w),
                      max=(1, 3, max_h, max_w))
    config.add_optimization_profile(profile)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")
        else:
            print("  Warning: FP16 not supported on this platform")

    # Build engine
    print("  Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"  Engine saved: {engine_path} ({os.path.getsize(engine_path) / 1024 / 1024:.1f} MB)")


def height_to_shape(h, aspect=16/9):
    """Convert height to (width, height) assuming aspect ratio."""
    w = int(h * aspect)
    # Round width to multiple of 8 for GPU alignment
    w = (w + 7) // 8 * 8
    return (w, h)


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to TensorRT engines')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to PyTorch model (.pth or .pt). Downloads Real-ESRGAN if not specified.')
    parser.add_argument('--min-height', type=int, default=270,
                        help='Minimum input height (default: 270)')
    parser.add_argument('--opt-height', type=int, default=720,
                        help='Optimal input height (default: 720)')
    parser.add_argument('--max-height', type=int, default=1280,
                        help='Maximum input height (default: 1280)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output engine path. Default: realesrgan_dynamic_fp16.engine')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable FP16 precision (default: enabled)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision instead of FP16')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace size in GB (default: 4)')
    parser.add_argument('--keep-onnx', action='store_true',
                        help='Keep intermediate ONNX file')
    args = parser.parse_args()

    # Handle fp32 flag
    if args.fp32:
        args.fp16 = False

    # Convert heights to (width, height) shapes assuming 16:9
    min_shape = height_to_shape(args.min_height)
    opt_shape = height_to_shape(args.opt_height)
    max_shape = height_to_shape(args.max_height)

    # Set output path
    if args.output is None:
        suffix = '_fp16' if args.fp16 else '_fp32'
        args.output = f"realesrgan_dynamic{suffix}.engine"

    # Load model
    print("=" * 60)
    print("Real-ESRGAN to TensorRT Export (Dynamic Shapes)")
    print("=" * 60)
    model, _ = get_model(args.model)

    # Export to ONNX
    if args.keep_onnx:
        onnx_path = args.output.replace('.engine', '.onnx')
    else:
        fd, onnx_path = tempfile.mkstemp(suffix='.onnx')
        os.close(fd)

    try:
        export_onnx(model, opt_shape, onnx_path)

        # Build TensorRT engine
        build_engine(onnx_path, args.output,
                     min_shape=min_shape,
                     opt_shape=opt_shape,
                     max_shape=max_shape,
                     fp16=args.fp16,
                     workspace_gb=args.workspace)
    finally:
        if not args.keep_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)

    print()
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)
    print()
    print(f"Engine accepts input heights from {args.min_height} to {args.max_height} (16:9)")
    print()
    print("Usage with FFmpeg:")
    print(f'  ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model={args.output}" output.mp4')


if __name__ == '__main__':
    main()

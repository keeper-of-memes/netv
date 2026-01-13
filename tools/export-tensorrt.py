#!/usr/bin/env python3
"""Export PyTorch models to TensorRT engines for FFmpeg dnn_processing filter.

This script converts Real-ESRGAN models to TensorRT engines (.engine files)
that can be loaded by FFmpeg's TensorRT DNN backend.

Default model: realesr-general-x4v3 (SRVGGNetCompact) - fast, good quality
Alternative: RealESRGAN_x4plus (RRDBNet) - slow, best quality

Usage:
    # Export default compact model (recommended)
    python export-tensorrt.py --output model.engine

    # Export with custom height range
    python export-tensorrt.py --min-height 360 --max-height 1080

    # Export high-quality model (slower)
    python export-tensorrt.py --model-type rrdbnet --output hq.engine

Requirements:
    pip install torch onnx tensorrt

Example FFmpeg usage after export:
    ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model=model.engine" output.mp4
"""

import argparse
import os
import tempfile
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """Compact SR network - fast inference, good quality."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv - 2):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body[:-1]:
            out = layer(out)
        out = self.body[-1](out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDBNet."""

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet architecture for Real-ESRGAN - highest quality, slower."""

    def __init__(
        self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    ):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# Model URLs
MODELS = {
    "compact": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "filename": "realesr-general-x4v3.pth",
    },
    "rrdbnet": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
    },
}


def download_model(model_type, cache_dir):
    """Download model weights."""
    info = MODELS[model_type]
    path = os.path.join(cache_dir, info["filename"])
    if os.path.exists(path):
        print(f"Using cached model: {path}")
        return path

    print(f"Downloading {info['filename']}...")
    urllib.request.urlretrieve(info["url"], path)
    print(f"Downloaded to {path}")
    return path


def get_model(model_path=None, model_type="compact", cache_dir=None):
    """Load Real-ESRGAN model."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/realesrgan")
    os.makedirs(cache_dir, exist_ok=True)

    if model_path is None:
        model_path = download_model(model_type, cache_dir)

    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    # Auto-detect model type from state dict
    num_conv_layers = sum(1 for k, v in state_dict.items() if "weight" in k and len(v.shape) == 4)

    if num_conv_layers > 50:  # RRDBNet has many more conv layers
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        model_name = "RRDBNet"
    else:
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv_layers, upscale=4
        )
        model_name = "SRVGGNetCompact"

    model.load_state_dict(state_dict)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {model_name} ({params:.2f}M params)")
    return model


def export_onnx(model, opt_shape, onnx_path):
    """Export model to ONNX format with dynamic axes."""
    opt_w, opt_h = opt_shape
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Optimal shape: 1x3x{opt_h}x{opt_w}")

    dummy_input = torch.randn(1, 3, opt_h, opt_w, device="cpu")

    dynamic_axes = {"input": {2: "height", 3: "width"}, "output": {2: "out_height", 3: "out_width"}}

    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print("  ONNX export complete (dynamic H/W)")


def build_engine(
    onnx_path, engine_path, min_shape, opt_shape, max_shape, fp16=False, workspace_gb=4
):
    """Build TensorRT engine from ONNX model with dynamic shapes."""
    import tensorrt as trt

    min_w, min_h = min_shape
    opt_w, opt_h = opt_shape
    max_w, max_h = max_shape

    print(f"Building TensorRT engine: {engine_path}")
    print("  Dynamic shapes:")
    print(f"    min: {min_w}x{min_h}")
    print(f"    opt: {opt_w}x{opt_h}")
    print(f"    max: {max_w}x{max_h}")
    print(f"  FP16: {fp16}")
    print(f"  Workspace: {workspace_gb} GB")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name, min=(1, 3, min_h, min_w), opt=(1, 3, opt_h, opt_w), max=(1, 3, max_h, max_w)
    )
    config.add_optimization_profile(profile)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")
        else:
            print("  Warning: FP16 not supported on this platform")

    print("  Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"  Engine saved: {engine_path} ({os.path.getsize(engine_path) / 1024 / 1024:.1f} MB)")


def height_to_shape(h, aspect=16 / 9):
    """Convert height to (width, height) assuming aspect ratio."""
    w = int(h * aspect)
    w = (w + 7) // 8 * 8
    return (w, h)


def main():
    parser = argparse.ArgumentParser(description="Export Real-ESRGAN to TensorRT")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to PyTorch model (.pth). Downloads default if not specified.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="compact",
        choices=["compact", "rrdbnet"],
        help="Model type: compact (fast, default) or rrdbnet (slow, highest quality)",
    )
    parser.add_argument(
        "--min-height", type=int, default=270, help="Minimum input height (default: 270)"
    )
    parser.add_argument(
        "--opt-height", type=int, default=720, help="Optimal input height (default: 720)"
    )
    parser.add_argument(
        "--max-height", type=int, default=1080, help="Maximum input height (default: 1080)"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output engine path")
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Enable FP16 precision (default: enabled)"
    )
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision instead of FP16")
    parser.add_argument(
        "--workspace", type=int, default=8, help="TensorRT workspace size in GB (default: 8)"
    )
    parser.add_argument("--keep-onnx", action="store_true", help="Keep intermediate ONNX file")
    parser.add_argument(
        "--onnx-only", action="store_true", help="Only export ONNX, skip TensorRT engine build"
    )
    args = parser.parse_args()

    if args.fp32:
        args.fp16 = False

    min_h = min(args.min_height, args.max_height)
    max_h = args.max_height
    opt_h = min(max(args.opt_height, min_h), max_h)
    min_shape = height_to_shape(min_h)
    opt_shape = height_to_shape(opt_h)
    max_shape = height_to_shape(max_h)

    if args.output is None:
        suffix = "_fp16" if args.fp16 else "_fp32"
        args.output = f"realesrgan_{args.model_type}{suffix}.engine"

    print("=" * 60)
    print("Real-ESRGAN to TensorRT Export")
    print("=" * 60)
    model = get_model(args.model, args.model_type)

    if args.keep_onnx:
        onnx_path = args.output.replace(".engine", ".onnx")
    else:
        fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)

    try:
        export_onnx(model, opt_shape, onnx_path)

        if args.onnx_only:
            print(f"\n  ONNX saved to: {onnx_path}")
            print("  Skipping TensorRT build (--onnx-only). Build later with:")
            print(f"  trtexec --onnx={onnx_path} --saveEngine={args.output} --fp16")
            return

        build_engine(
            onnx_path,
            args.output,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            fp16=args.fp16,
            workspace_gb=args.workspace,
        )
    finally:
        if not args.keep_onnx and not args.onnx_only and os.path.exists(onnx_path):
            os.remove(onnx_path)

    print()
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)
    print()
    print(f"Engine accepts input heights from {args.min_height} to {args.max_height} (16:9)")
    print()
    print("Usage with FFmpeg:")
    print(
        f'  ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model={args.output}" output.mp4'
    )


if __name__ == "__main__":
    main()

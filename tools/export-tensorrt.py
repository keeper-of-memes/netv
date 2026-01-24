#!/usr/bin/env python3
"""Export upscaling models to TensorRT engines for FFmpeg dnn_processing filter.

This script converts AI upscaling models to TensorRT engines (.engine files)
that can be loaded by FFmpeg's TensorRT DNN backend.

Available models (use --list to see all):
  2x models (1080p → 4K):
    - 2x-liveaction-span    Best for live action TV/film

  4x models (720p → 4K, 480p → 1080p):
    - 4x-compact            Fast, good quality (SRVGGNetCompact)

Usage:
    # List available models
    python export-tensorrt.py --list

    # Export 2x model for live action
    python export-tensorrt.py --model 2x-liveaction-span -o model.engine

    # Export with custom height range
    python export-tensorrt.py --model 2x-liveaction-span --min-height 720 --max-height 1080

Requirements:
    pip install torch onnx tensorrt

Example FFmpeg usage after export:
    ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model=model.engine" output.mp4
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict


if TYPE_CHECKING:
    import tensorrt as trt

import argparse
import sys
import tempfile
import urllib.request

import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """Compact SR network - fast inference, good quality."""

    upscale: int
    body: nn.ModuleList
    upsampler: nn.PixelShuffle

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body[:-1]:
            out = layer(out)
        out = self.body[-1](out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDBNet."""

    conv1: nn.Conv2d
    conv2: nn.Conv2d
    conv3: nn.Conv2d
    conv4: nn.Conv2d
    conv5: nn.Conv2d
    lrelu: nn.LeakyReLU

    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    rdb1: ResidualDenseBlock
    rdb2: ResidualDenseBlock
    rdb3: ResidualDenseBlock

    def __init__(self, nf: int, gc: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet architecture for Real-ESRGAN - highest quality, slower."""

    scale: int
    conv_first: nn.Conv2d
    body: nn.Sequential
    conv_body: nn.Conv2d
    conv_up1: nn.Conv2d
    conv_up2: nn.Conv2d
    conv_hr: nn.Conv2d
    conv_last: nn.Conv2d
    lrelu: nn.LeakyReLU

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(
            self.conv_up1(
                F.interpolate(
                    feat,
                    scale_factor=2,
                    mode="nearest",
                )
            )
        )
        feat = self.lrelu(
            self.conv_up2(
                F.interpolate(
                    feat,
                    scale_factor=2,
                    mode="nearest",
                )
            )
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class ModelInfo(TypedDict):
    """Type definition for model registry entries."""

    description: str
    filename: str
    scale: int
    arch: str
    url: NotRequired[str]
    onnx_url: NotRequired[str]


MODELS: dict[str, ModelInfo] = {
    # 2x models - high quality, 1080p → 4K
    "2x-liveaction-span": {
        "description": "Live action TV/film - handles compression, preserves grain",
        "onnx_url": "https://github.com/jcj83429/upscaling/raw/f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.onnx",
        "filename": "2xLiveActionV1_SPAN.onnx",
        "scale": 2,
        "arch": "span",
    },
    # 4x models - 720p → 4K or 480p → 1080p
    "4x-compact": {
        "description": "Fast 4x upscale - SRVGGNetCompact",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "filename": "realesr-general-x4v3.pth",
        "scale": 4,
        "arch": "compact",
    },
    # 4x-realesrgan - not recommended (overly smooths faces)
    "4x-realesrgan": {
        "description": "RealESRGAN 4x - smooths faces (not recommended)",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "arch": "rrdbnet",
    },
    # NOTE: 4x-rrdbnet was removed because:
    # - 1080p engine build fails with OOM even on 32GB VRAM (RTX 5090)
    # - 720p engine causes "Invalid frame dimensions 0x0" errors during playback
    # - Same weights as 4x-realesrgan but different name
}


def resolve_model(model_name: str) -> tuple[str, ModelInfo]:
    """Resolve model name."""
    info = MODELS.get(model_name)
    if info is None:
        raise ValueError(f"Unknown model: {model_name}")
    return model_name, info


def download_model(model_name: str, cache_dir: Path) -> Path:
    """Download model weights (ONNX or PTH)."""
    model_name, info = resolve_model(model_name)
    # Use .name to prevent path traversal
    path = cache_dir / Path(info["filename"]).name
    if path.exists():
        print(f"Using cached model: {path}")
        return path

    url = info.get("onnx_url") or info.get("url")
    if url is None:
        raise ValueError(f"No URL for model: {model_name}")
    if not url.startswith("https://"):
        raise ValueError(f"URL must use HTTPS: {url}")
    print(f"Downloading {info['filename']}...")

    # Download to a temp file first, then rename to avoid partial downloads
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with (
            urllib.request.urlopen(url, timeout=300) as response,
            open(temp_path, "wb") as f,
        ):
            f.write(response.read())
        # Verify the download succeeded and file is not empty
        file_size = temp_path.stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {temp_path}")
        temp_path.rename(path)
        print(f"Downloaded to {path} ({file_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        # Clean up partial download
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e

    return path


def list_models() -> None:
    """Print available models."""
    print("\nAvailable models:\n")
    print("  2x models (1080p → 4K):")
    for name, info in MODELS.items():
        if name.startswith("2x-"):
            rec = " (recommended)" if name == "2x-liveaction-span" else ""
            print(f"    {name:24s} {info['description']}{rec}")
    print("\n  4x models (720p → 4K):")
    for name, info in MODELS.items():
        if name.startswith("4x-"):
            print(f"    {name:24s} {info['description']}")
    print()


def get_model_and_onnx(
    model_name: str,
    cache_dir: Path | None = None,
) -> tuple[
    nn.Module | None,
    Path | None,
    int,
]:
    """Load model and return (model_or_none, onnx_path_or_none, scale).

    For ONNX-based models, returns (None, onnx_path, scale).
    For PTH-based models, returns (model, None, scale).
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "ai_upscale"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_name, info = resolve_model(model_name)
    scale = info["scale"]
    arch = info["arch"]

    model_path = download_model(model_name, cache_dir)

    # ONNX-based models - no PyTorch loading needed
    if "onnx_url" in info:
        print(f"Using ONNX model directly: {model_path}")
        print(f"  Architecture: {arch}, Scale: {scale}x")
        return None, model_path, scale

    # PTH-based models - load PyTorch
    print(f"Loading PyTorch model from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    # Instantiate model based on explicit architecture
    if arch == "rrdbnet":
        model: nn.Module = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        arch_name = "RRDBNet"
    elif arch == "compact":
        # Count conv layers to determine num_conv for SRVGGNetCompact
        num_conv_layers = sum(
            1 for k, v in state_dict.items() if "weight" in k and len(v.shape) == 4
        )
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=num_conv_layers,
            upscale=scale,
        )
        arch_name = "SRVGGNetCompact"
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.load_state_dict(state_dict)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {arch_name} ({params:.2f}M params), Scale: {scale}x")
    return model, None, scale


def export_onnx(model: nn.Module, opt_shape: tuple[int, int], onnx_path: Path | str) -> None:
    """Export model to ONNX format with dynamic axes."""
    opt_w, opt_h = opt_shape
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Optimal shape: 1x3x{opt_h}x{opt_w}")

    dummy_input = torch.randn(1, 3, opt_h, opt_w, device="cpu")

    dynamic_axes = {
        "input": {
            2: "height",
            3: "width",
        },
        "output": {
            2: "out_height",
            3: "out_width",
        },
    }

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


def _get_trt_dtype_map() -> dict[str, trt.DataType]:
    """Get mapping from precision string to TensorRT DataType."""
    dtype_map: dict[str, trt.DataType] = {
        "fp32": trt.float32,
        "fp16": trt.float16,
    }
    if hasattr(trt, "bfloat16"):
        dtype_map["bf16"] = trt.bfloat16
    return dtype_map


def _trt_dtype_str(dtype: trt.DataType) -> str:
    """Convert TensorRT DataType to human-readable string."""
    for name, dt in _get_trt_dtype_map().items():
        if dtype == dt:
            return name.upper()
    return str(dtype)


def build_engine(
    onnx_path: Path | str,
    engine_path: Path | str,
    min_shape: tuple[int, int],
    opt_shape: tuple[int, int],
    max_shape: tuple[int, int],
    precision: str = "fp16",
    workspace_gb: int = 4,
    opt_level: int = 3,
) -> None:
    """Build TensorRT engine from ONNX model with dynamic shapes."""
    min_w, min_h = min_shape
    opt_w, opt_h = opt_shape
    max_w, max_h = max_shape

    print(f"Building TensorRT engine: {engine_path}")
    print("  Dynamic shapes:")
    print(f"    min: {min_w}x{min_h}")
    print(f"    opt: {opt_w}x{opt_h}")
    print(f"    max: {max_w}x{max_h}")
    print(f"  Precision: {precision}")
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

    # Optimization level (0-5, default is 3)
    # Higher levels enable more aggressive kernel selection/fusion but use more memory
    config.builder_optimization_level = opt_level
    print(f"  Optimization level: {opt_level}")

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        min=(1, 3, min_h, min_w),
        opt=(1, 3, opt_h, opt_w),
        max=(1, 3, max_h, max_w),
    )
    config.add_optimization_profile(profile)

    # Set compute precision
    if precision in ("fp16", "bf16"):
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("  Warning: FP16/BF16 not supported on this platform, using FP32")
            precision = "fp32"
    if precision == "bf16":
        if hasattr(trt.BuilderFlag, "BF16"):
            config.set_flag(trt.BuilderFlag.BF16)
        else:
            print("  Warning: BF16 not supported by TensorRT, using FP16")
            precision = "fp16"

    # Set I/O tensor precision (matches compute precision)
    dtype_map = _get_trt_dtype_map()
    if precision not in dtype_map:
        raise ValueError(f"Unknown precision: {precision}")
    io_dtype = dtype_map[precision]

    if io_dtype != trt.float32:
        for i in range(network.num_inputs):
            network.get_input(i).dtype = io_dtype
        for i in range(network.num_outputs):
            network.get_output(i).dtype = io_dtype

    print("  Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(
        f"  Engine saved: {engine_path} ({Path(engine_path).stat().st_size / 1024 / 1024:.1f} MB)"
    )

    # Verify the built engine has correct I/O types
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("  Verifying engine I/O:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        dtype_str = _trt_dtype_str(dtype)
        print(f"    {name}: {dtype_str} ({mode})")
        if dtype != io_dtype:
            print(f"  WARNING: {name} is {dtype_str} but {_trt_dtype_str(io_dtype)} was requested!")


def height_to_shape(h: int, aspect: float = 16 / 9) -> tuple[int, int]:
    """Convert height to (width, height) assuming aspect ratio.

    Both width and height are aligned to 8 pixels, as required by many
    neural network architectures with pooling/striding layers.
    """
    # Align height to 8 first
    h = (h + 7) // 8 * 8
    # Calculate width from aligned height
    w = int(h * aspect)
    # Align width to 8
    w = (w + 7) // 8 * 8
    return (w, h)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export AI upscaling models to TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="4x-compact",
        help="Model name (use --list to see available models)",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument(
        "--min-height",
        type=int,
        default=None,
        help="Minimum input height (default: auto)",
    )
    parser.add_argument(
        "--opt-height",
        type=int,
        default=None,
        help="Optimal input height (default: auto)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Maximum input height (default: auto)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output engine path",
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model precision for compute and I/O tensors (default: fp16)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=8,
        help="TensorRT workspace size in GB (default: 8)",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="TensorRT builder optimization level 0-5 (default: 3). Higher = more memory, potentially faster.",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only export ONNX, skip TensorRT engine build",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # Get model info for defaults
    try:
        model_name, info = resolve_model(args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        list_models()
        sys.exit(1)

    scale = info["scale"]

    # Set height defaults based on scale factor
    if scale == 2:
        # 2x: input 1080p -> output 4K
        default_min, default_opt, default_max = 720, 1080, 1080
    elif scale == 4:
        # 4x: input 720p -> output 4K, or 480p -> 1080p
        default_min, default_opt, default_max = 480, 720, 1080
    else:
        raise ValueError(f"Unsupported scale factor: {scale}")

    min_h = args.min_height or default_min
    opt_h = args.opt_height or default_opt
    max_h = args.max_height or default_max

    # Validate height constraints
    if min_h > max_h:
        raise ValueError(f"--min-height ({min_h}) cannot be greater than --max-height ({max_h})")
    if opt_h < min_h or opt_h > max_h:
        raise ValueError(
            f"--opt-height ({opt_h}) must be between --min-height ({min_h}) and --max-height ({max_h})"
        )

    min_shape = height_to_shape(min_h)
    opt_shape = height_to_shape(opt_h)
    max_shape = height_to_shape(max_h)

    if args.output is None:
        args.output = f"{model_name}_{opt_h}p_{args.precision}.engine"

    print("=" * 60)
    print("AI Upscale: TensorRT Engine Export")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"  {info['description']}")
    print()

    model, existing_onnx, _ = get_model_and_onnx(args.model)

    # Determine ONNX path
    if existing_onnx:
        # Model already has ONNX - use it directly
        onnx_path = existing_onnx
        cleanup_onnx = False
    elif args.onnx_only:
        # Save ONNX to current directory with sensible name
        onnx_path = Path(f"{model_name}_{opt_h}p.onnx")
        cleanup_onnx = False
    else:
        # Temp file for intermediate ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = Path(tmp.name)
        cleanup_onnx = True

    try:
        # Export to ONNX if needed (PTH-based models only)
        if model is not None:
            export_onnx(model, opt_shape, onnx_path)

        if args.onnx_only:
            print(f"\nONNX saved to: {onnx_path}")
            print("Skipping TensorRT build (--onnx-only). Build later with:")
            print(f"  trtexec --onnx={onnx_path} --saveEngine={args.output} --fp16")
            return

        build_engine(
            onnx_path,
            args.output,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            precision=args.precision,
            workspace_gb=args.workspace,
            opt_level=args.opt_level,
        )
    finally:
        if cleanup_onnx and (onnx_file := Path(onnx_path)).exists():
            onnx_file.unlink()

    print()
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)
    print()
    print(f"Model: {model_name} ({scale}x upscale)")
    print(f"Engine accepts input heights from {min_h} to {max_h} (16:9)")
    print()
    print("Usage with FFmpeg:")
    print(
        f'  ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model={args.output}" output.mp4'
    )


if __name__ == "__main__":
    main()

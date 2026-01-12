#!/bin/bash
# Download and convert Real-ESRGAN models to TorchScript for ffmpeg dnn_processing
set -e

MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"
VENV_DIR="${MODEL_DIR}/.venv"
# Must match LIBTORCH_VERSION in install-ffmpeg.sh
TORCH_VERSION="2.5.0"

mkdir -p "$MODEL_DIR"

# Set up Python venv with correct torch version (must match ffmpeg's libtorch)
setup_python() {
  if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required"
    exit 1
  fi

  # Check if venv exists with correct version
  if [ -f "$VENV_DIR/torch_version" ] && [ "$(cat "$VENV_DIR/torch_version")" = "$TORCH_VERSION" ]; then
    echo "Using existing venv with torch $TORCH_VERSION"
    source "$VENV_DIR/bin/activate"
    return
  fi

  echo "Setting up Python venv with torch $TORCH_VERSION (must match ffmpeg's libtorch)..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install -q --upgrade pip
  # Use CUDA 12.4 variant to match ffmpeg's libtorch (cu124)
  pip install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu124
  echo "$TORCH_VERSION" > "$VENV_DIR/torch_version"
  echo "Installed torch $TORCH_VERSION"
}

# Download Real-ESRGAN weights
download_models() {
  echo "Downloading Real-ESRGAN models..."
  cd "$MODEL_DIR"

  # RealESRGAN x2 - good for 1080p -> 4K
  if [ ! -f "RealESRGAN_x2plus.pth" ]; then
    wget -q --show-progress -O RealESRGAN_x2plus.pth \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
  fi

  # RealESRGAN x4 - for lower res sources
  if [ ! -f "RealESRGAN_x4plus.pth" ]; then
    wget -q --show-progress -O RealESRGAN_x4plus.pth \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
  fi

  # Compact models - faster, designed for real-time video
  if [ ! -f "realesr-general-x4v3.pth" ]; then
    wget -q --show-progress -O realesr-general-x4v3.pth \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
  fi

  echo "Models downloaded to $MODEL_DIR"
}

# Convert PyTorch models to TorchScript
convert_to_torchscript() {
  echo "Converting models to TorchScript..."

  python3 << 'PYTHON_SCRIPT'
import os
import sys
import torch

MODEL_DIR = os.environ.get('MODEL_DIR', os.path.expanduser('~/ffmpeg_build/models'))

# RRDBNet architecture (from Real-ESRGAN/BasicSR)
# Simplified version that matches the pretrained weights
class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(torch.nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(torch.nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        # x2plus uses pixel unshuffle: 3 channels * 2*2 = 12 input channels
        if scale == 2:
            self.pixel_unshuffle = torch.nn.PixelUnshuffle(2)
            self.conv_first = torch.nn.Conv2d(num_in_ch * 4, num_feat, 3, 1, 1)
        else:
            self.pixel_unshuffle = None
            self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling (x2plus uses unshuffle so needs 2 upsamples for 2x, x4 needs 2 for 4x)
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.pixel_unshuffle is not None:
            x = self.pixel_unshuffle(x)
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample (always 2 steps: x2plus uses unshuffle so 2+2=2x net, x4 is just 2+2=4x)
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# SRVGGNetCompact - lightweight architecture for real-time video SR
# Much faster than RRDBNet, designed for video upscaling
class SRVGGNetCompact(torch.nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale

        # Body contains interleaved Conv2d and PReLU layers
        self.body = torch.nn.ModuleList()
        # First conv
        self.body.append(torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # Body: conv + prelu pairs
        for _ in range(num_conv):
            self.body.append(torch.nn.PReLU(num_parameters=num_feat))
            self.body.append(torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        # Last activation + upsampler conv
        self.body.append(torch.nn.PReLU(num_parameters=num_feat))
        self.body.append(torch.nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))

        # Pixel shuffle for upsampling
        self.upsampler = torch.nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.body[0](x)  # First conv
        for layer in self.body[1:]:
            out = layer(out)
        out = self.upsampler(out)
        # Skip connection via bilinear upsampling
        base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return out + base


def convert_rrdb_model(pth_path, scale, num_block=23):
    """Convert RRDBNet model to TorchScript."""
    output_path = pth_path.replace('.pth', '.pt')

    if os.path.exists(output_path):
        print(f"  {os.path.basename(output_path)} already exists, skipping")
        return

    print(f"  Converting {os.path.basename(pth_path)} (RRDBNet)...")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64, num_block=num_block, num_grow_ch=32)

    state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
        traced.save(output_path)

    print(f"  Saved: {output_path}")


def convert_compact_model(pth_path, upscale=4, num_conv=16, num_feat=64):
    """Convert SRVGGNetCompact model to TorchScript (fast real-time models)."""
    output_path = pth_path.replace('.pth', '.pt')

    if os.path.exists(output_path):
        print(f"  {os.path.basename(output_path)} already exists, skipping")
        return

    print(f"  Converting {os.path.basename(pth_path)} (SRVGGNetCompact)...")

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=upscale)

    state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Use small input for tracing
    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
        traced.save(output_path)

    print(f"  Saved: {output_path}")


# Convert RRDBNet models (high quality, slow)
rrdb_models = [
    ('RealESRGAN_x2plus.pth', 2, 23),
    ('RealESRGAN_x4plus.pth', 4, 23),
]

for filename, scale, num_block in rrdb_models:
    pth_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(pth_path):
        try:
            convert_rrdb_model(pth_path, scale, num_block)
        except Exception as e:
            print(f"  Error converting {filename}: {e}")
    else:
        print(f"  {filename} not found, skipping")

# Convert compact models (fast, real-time capable)
compact_models = [
    ('realesr-general-x4v3.pth', 4, 32, 64),    # upscale, num_conv, num_feat
]

for filename, upscale, num_conv, num_feat in compact_models:
    pth_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(pth_path):
        try:
            convert_compact_model(pth_path, upscale, num_conv, num_feat)
        except Exception as e:
            print(f"  Error converting {filename}: {e}")
    else:
        print(f"  {filename} not found, skipping")

print("\nDone! TorchScript models saved to:", MODEL_DIR)
print("\nModels:")
print("  realesr-general-x4v3.pt  - Fast real-time model for general video")
print("  RealESRGAN_x4plus.pt     - High quality model (slow)")
print("\nUsage with ffmpeg:")
print('  ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3.pt" output.mp4')
PYTHON_SCRIPT
}

# Main
echo "Real-ESRGAN Model Setup for ffmpeg"
echo "==================================="
echo ""

setup_python
download_models
convert_to_torchscript


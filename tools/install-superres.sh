#!/bin/bash
# Install super-resolution model for FFmpeg TensorRT backend
# Downloads Real-ESRGAN from HuggingFace and builds TensorRT engine with dynamic shapes
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"
ENGINE_NAME="realesrgan_dynamic_fp16.engine"

echo "========================================"
echo "Super-Resolution Model Installation"
echo "========================================"
echo "Output: $MODEL_DIR/$ENGINE_NAME"
echo "Dynamic input: 270p to 1280p (16:9)"
echo ""

# Create output directory
mkdir -p "$MODEL_DIR"

# Setup Python venv
VENV_DIR="$MODEL_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -q torch huggingface_hub onnx tensorrt basicsr realesrgan 2>/dev/null || true

# Build engine with dynamic shapes
echo "Building TensorRT engine..."
python3 "$SCRIPT_DIR/export-tensorrt.py" \
    --output "$MODEL_DIR/$ENGINE_NAME"

deactivate

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo "Engine: $MODEL_DIR/$ENGINE_NAME"
echo ""
echo "Test with:"
echo "  ffmpeg -init_hw_device cuda=cu -filter_hw_device cu \\"
echo "    -f lavfi -i testsrc=duration=3:size=1280x720:rate=30 \\"
echo "    -vf \"format=rgb24,hwupload,dnn_processing=dnn_backend=8:model=$MODEL_DIR/$ENGINE_NAME\" \\"
echo "    -c:v hevc_nvenc test.mp4"

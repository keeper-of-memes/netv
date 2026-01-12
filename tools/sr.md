# Super Resolution with Real-ESRGAN

AI-powered video upscaling using Real-ESRGAN models via ffmpeg's libtorch backend.

## Quick Start

```bash
# Download and convert models
./download-sr-models.sh

# Test performance
./test-realesr.sh

# Use with ffmpeg
ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3.pt:device=cuda" output.mp4
```

## Models

### Available Models

| Model | Architecture | Size | Scale | Speed | Quality | Use Case |
|-------|-------------|------|-------|-------|---------|----------|
| realesr-general-x4v3.pt | SRVGGNetCompact | 5MB | 4x | Fast | Good | Video, real-time |
| RealESRGAN_x4plus.pt | RRDBNet | 67MB | 4x | Slow | Best | Photos, quality |
| RealESRGAN_x2plus.pt | RRDBNet | 67MB | 2x | Slow | Best | Photos, quality |

### Architecture Comparison

**SRVGGNetCompact** (realesr-general-x4v3)
- Lightweight network designed for real-time video
- ~13x fewer parameters than RRDBNet
- Best speed/quality tradeoff for video

**RRDBNet** (x4plus, x2plus)
- Full Real-ESRGAN architecture with residual-in-residual dense blocks
- Higher quality but much slower
- Better for single images or offline processing

### Model Sources

Models are downloaded from the official Real-ESRGAN releases:
- https://github.com/xinntao/Real-ESRGAN/releases

## Performance Benchmarks

### RTX 5090 (Blackwell, SM 12.0)

Test: 1280x720 input, 150 frames, CUDA

| Configuration | FPS | Realtime | Notes |
|--------------|-----|----------|-------|
| x4v3 + libtorch | 3.6 | 0.12x | Baseline |
| x4v3 + TensorRT (ffmpeg) | 4.6 | 0.15x | +28% but overhead |
| x4v3 + TensorRT (Python) | 66.1 | 2.20x | Direct inference |
| x2plus + libtorch | 3.1 | 0.10x | Heavier model = slower |
| x4plus + libtorch | ~2 | ~0.07x | Quality model, slowest |

### TITAN X (Maxwell, SM 5.2)

| Configuration | FPS | Realtime | Notes |
|--------------|-----|----------|-------|
| x4v3 + libtorch | 1.7 | 0.056x | TensorRT not supported |

### Key Findings

1. **Compact model wins for video** - x4v3 is fastest despite being 4x upscale
2. **RRDBNet models are slower even at 2x** - architecture matters more than scale
3. **TensorRT gives 18x speedup** in Python, but ffmpeg overhead reduces it to 1.3x
4. **Real-time 4K upscaling needs Python pipeline** - ffmpeg's dnn_processing has too much overhead

## TensorRT Acceleration

TensorRT provides significant speedup but requires:
- SM 7.0+ GPU (Volta, Turing, Ampere, Ada, Blackwell)
- torch-tensorrt package
- TensorRT runtime libraries

### Compile Model with TensorRT

```bash
./compile-sr-tensorrt.sh
```

This creates `realesr-general-x4v3-trt.pt` optimized for FP16 inference.

### Run TensorRT Model with ffmpeg

```bash
# Set library paths for TensorRT runtime
TRT_LIB=~/ffmpeg_build/models/.venv/lib/python3.12/site-packages/torch_tensorrt/lib
TRTCORE=~/ffmpeg_build/models/.venv/lib/python3.12/site-packages/tensorrt_libs
TORCH_LIB=~/.local/lib

LD_LIBRARY_PATH="$TRT_LIB:$TRTCORE:$TORCH_LIB:$LD_LIBRARY_PATH" \
ffmpeg -i input.mp4 \
  -vf "dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3-trt.pt:device=cuda" \
  output.mp4
```

### Why TensorRT is Slow in ffmpeg

The Python benchmark shows 66 FPS but ffmpeg only achieves 4.6 FPS. Investigation revealed
the bottleneck is **memory operations, not model inference**.

#### Benchmark Analysis

```
ffmpeg -benchmark output:
  utime=12.164s  (CPU computation)
  stime=20.526s  (kernel/memory ops)  <-- 63% of total time!
  rtime=32.375s  (wall clock)
```

Per-frame breakdown (150 frames, 32.4s total):
- Total time: 216ms/frame
- System time: **137ms/frame** (memory operations)
- User time: 81ms/frame (model + format conversion)
- TensorRT inference: ~15ms/frame (measured in Python)

#### The Real Bottleneck: Output Size

4x upscaling produces massive output frames:
```
Input:  1280 x 720  x 3 bytes =  2.7 MB/frame
Output: 5120 x 2880 x 3 bytes = 44.2 MB/frame (16x larger!)
```

For 150 frames at 4.6 fps:
- 6.6 GB of output data copied GPU→CPU
- Memory allocation/deallocation per frame
- Page faults and cache misses

#### Why Python is 14x Faster

Python benchmark keeps tensors on GPU:
1. Create input tensor on GPU (once)
2. Run forward() 10 times
3. Never copy 44MB output to CPU

ffmpeg must copy to CPU for the filter chain:
1. Decode frame to CPU
2. Convert RGB→float tensor
3. Copy to GPU
4. Run inference (~15ms)
5. Copy 44MB output back to CPU  <-- bottleneck
6. Convert float→RGB
7. Continue filter chain

#### Evidence: TRT Only Marginally Faster

- Vanilla libtorch: 3.6 fps (278ms/frame)
- TensorRT: 4.6 fps (217ms/frame)
- Difference: only 61ms saved

If inference was the bottleneck, TRT should save ~250ms (277ms - 15ms).
The 61ms savings confirms inference is only ~25% of total time.

#### Solutions Would Require ffmpeg Changes

1. **Keep tensors on GPU** - avoid CPU roundtrip entirely
2. **Direct NVENC encoding** - GPU output → GPU encoder
3. **Pinned memory** - faster GPU↔CPU transfers
4. **Async pipeline** - overlap copy with next inference
5. **Batch processing** - amortize overhead across frames

## GPU Compatibility

### libtorch Version Requirements

| GPU Generation | SM Version | Min libtorch | CUDA Variant |
|---------------|------------|--------------|--------------|
| Maxwell (TITAN X) | 5.2 | 2.5.0 | cu124 |
| Pascal (GTX 10xx) | 6.x | 2.5.0 | cu124 |
| Volta (V100) | 7.0 | 2.5.0 | cu124 |
| Turing (RTX 20xx) | 7.5 | 2.5.0 | cu124 |
| Ampere (RTX 30xx) | 8.x | 2.5.0 | cu124 |
| Ada (RTX 40xx) | 8.9 | 2.5.0 | cu124 |
| Blackwell (RTX 50xx) | 12.0 | 2.7.0+ | cu130 |

### TensorRT Requirements

- Minimum SM 7.0 (Volta or newer)
- Maxwell/Pascal GPUs cannot use TensorRT 10.x

## ffmpeg Patches Applied

The `install-ffmpeg.sh` script applies these patches to ffmpeg's torch backend:

### 1. libtorch 2.6+ Compatibility
```cpp
// initXPU() renamed to init() in libtorch 2.6+
at::detail::getXPUHooks().init();  // was initXPU()
```

### 2. CUDA Device Support
```cpp
// Upstream only supports CPU/XPU, we add CUDA
} else if (device.is_cuda()) {
    if (!at::cuda::is_available()) {
        av_log(ctx, AV_LOG_ERROR, "No CUDA device found\n");
        goto fail;
    }
}
```

### 3. TensorRT Support
```cpp
// Load TensorRT runtime for TRT-compiled models
dlopen("libtorchtrt_runtime.so", RTLD_NOW | RTLD_GLOBAL);

// Handle TRT models that return tuples
auto output = model->forward(inputs);
if (output.isTuple()) {
    result = output.toTuple()->elements()[0].toTensor();
} else {
    result = output.toTensor();
}

// TRT models may not have parameters()
if (params.begin() != params.end()) {
    device = (*params.begin()).device();
}
```

## Usage Examples

### Basic 4x Upscale
```bash
ffmpeg -i input_720p.mp4 \
  -vf "dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3.pt:device=cuda" \
  output_4k.mp4
```

### With Encoding Settings
```bash
ffmpeg -i input_720p.mp4 \
  -vf "dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3.pt:device=cuda" \
  -c:v libx265 -crf 18 -preset slow \
  -c:a copy \
  output_4k.mp4
```

### Upscale + HDR Tone Mapping
```bash
ffmpeg -i input_720p_hdr.mp4 \
  -vf "dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3.pt:device=cuda,libplacebo=tonemapping=bt.2390" \
  output_4k_sdr.mp4
```

## Troubleshooting

### "no kernel image is available for execution on the device"
GPU not supported by libtorch version. Check GPU compatibility table above.

### "Unknown type name '__torch__.torch.classes.tensorrt.Engine'"
TensorRT runtime not loaded. Set LD_LIBRARY_PATH to include torch_tensorrt/lib.

### "Expected Tensor but got None"
TRT model output format issue. Ensure ffmpeg has TensorRT patches applied.

### Terminal corrupted after ffmpeg crash
Run `reset` or the test script includes `trap 'stty sane' EXIT`.

## Future Improvements

### For Real-time Performance
1. **Python TRT pipeline** - Process with TensorRT, pipe to ffmpeg for encoding
2. **Batched inference** - Process multiple frames per forward pass
3. **Async pipeline** - Overlap inference with encoding

### Model Improvements
1. **Custom compact 2x model** - Train SRVGGNetCompact for 2x upscaling
2. **Quantized models** - INT8 inference for more speed
3. **Frame interpolation** - Combine with RIFE for frame rate upscaling

## Files

```
tools/
  download-sr-models.sh   # Download and convert models to TorchScript
  compile-sr-tensorrt.sh  # Compile models with TensorRT
  test-realesr.sh         # Benchmark performance
  install-ffmpeg.sh       # Build ffmpeg with libtorch + patches
  sr.md                   # This documentation
```

## References

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [PyTorch TensorRT](https://pytorch.org/TensorRT/)
- [FFmpeg DNN Processing](https://ffmpeg.org/ffmpeg-filters.html#dnn_005fprocessing-1)

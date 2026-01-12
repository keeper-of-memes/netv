#!/bin/bash
# Build ffmpeg from source with hardware acceleration support
# Supports: NVIDIA NVENC, AMD AMF, Intel QSV/VAAPI, LibTorch DNN
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
set -e

# =============================================================================
# Potentially Viable Pre-built FFmpeg alternatives
#
# DOCKER IMAGES
#   LinuxServer docker-ffmpeg    https://github.com/linuxserver/docker-ffmpeg
#     - Full hardware accel (NVENC, VAAPI, QSV)
#     - Comprehensive codec support, builds libva 2.23+ from source
#     - Used by Dispatcharr, basis for comparison with this script
#
# STATIC BINARIES (Linux)
#   BtbN/FFmpeg-Builds           https://github.com/BtbN/FFmpeg-Builds
#     - Daily automated builds from git master and release branches
#     - GPL/LGPL/nonfree variants, static and shared options
#     - Targets glibc 2.28+ (RHEL 8 / Ubuntu 20.04+)
#     - CUDA support: sm_52+ (Maxwell and newer)
#
#   John Van Sickle              https://johnvansickle.com/ffmpeg/
#     - Static builds for amd64, i686, armhf, arm64
#     - GPL v3 licensed, targets kernel 3.2.0+
#     - Note: static glibc = no DNS resolution (install nscd to fix)
#
# STATIC BINARIES (Windows)
#   gyan.dev                     https://www.gyan.dev/ffmpeg/builds/
#     - Essentials build: common codecs (Win 7+)
#     - Full build: all codecs including bluray, opencl (Win 10+)
#     - Official FFmpeg download page recommendation
#
# SPECIALIZED BUILDS
#   Jellyfin-ffmpeg              https://github.com/jellyfin/jellyfin-ffmpeg
#     - Modified FFmpeg with Jellyfin-specific patches
#     - Optimized for media server transcoding
#     - Ships with Jellyfin packages and Docker images
#     - Recommended only for Jellyfin; other apps should use standard builds
#
# =============================================================================

# =============================================================================
# FFmpeg library reference (checked 2026-01)
#
# Priority: high    = essential for most workflows
#           med     = useful for specific workflows
#           low     = niche use cases
#           subsumed= functionality covered by another library we use
#           legacy  = outdated, superseded by newer codecs
#
# Enable: src = built from source, apt = use apt package, - = not enabled
#
#   Library          | Build  | Pri      | Apt Ver | Latest  | Description
#   -----------------|--------|----------|---------|---------|---------------------------
#   VIDEO CODECS
#   libx264          | src    | high     | 0.164   | 0.165   | H.264/AVC encoder (8/10-bit)
#   libx265          | src    | high     | 3.5     | 4.1     | H.265/HEVC encoder (8/10/12-bit)
#   libsvtav1        | src    | high     | 1.7.0   | 3.0.2   | AV1 encoder (fast, scalable)
#   libaom           | src    | high     | 3.8.2   | 3.13.1  | AV1 reference encoder/decoder
#   libdav1d         | src    | high     | 1.4.1   | 1.5.3   | AV1 decoder (fastest)
#   libvpx           | apt    | high     | 1.14.0  | 1.14.1  | VP8/VP9 encoder/decoder
#   libvvenc         | -      | low      | -       | 1.13.1  | H.266/VVC encoder (too early)
#   librav1e         | -      | subsumed | 0.7.1   | 0.8.1   | AV1 encoder - svtav1 faster
#   libkvazaar       | -      | subsumed | 2.3.1   | 2.3.2   | HEVC encoder - x265 better
#   libopenh264      | -      | subsumed | 2.6.0   | 2.6.0   | H.264 (Cisco) - x264 better
#   libxvid          | -      | legacy   | 1.3.7   | 1.3.7   | MPEG-4 Part 2 (obsolete)
#   libtheora        | -      | legacy   | 1.2.0a1 | 1.2.0   | Theora codec (obsolete)
#
#   IMAGE CODECS
#   libwebp          | src    | high     | 1.3.2   | 1.6.0   | WebP image codec
#   libjxl           | src    | high     | 0.7.0   | 0.11.1  | JPEG XL (next-gen, HDR)
#   libopenjpeg      | -      | low      | 2.5.0   | 2.5.4   | JPEG 2000 (cinema/medical)
#   librsvg          | -      | low      | 2.58.0  | 2.61.3  | SVG rasterization
#   libsnappy        | -      | low      | 1.1.10  | 1.2.2   | Snappy compression (HAP codec)
#
#   AUDIO CODECS
#   libfdk-aac       | apt    | high     | 2.0.2   | 2.0.3   | AAC encoder (best quality)
#   libmp3lame       | apt    | high     | 3.100   | 3.100   | MP3 encoder
#   libopus          | apt    | high     | 1.5.2   | 1.6     | Opus encoder/decoder
#   libvorbis        | apt    | high     | 1.3.7   | 1.3.7   | Vorbis encoder/decoder
#   librubberband    | apt    | med      | 3.3.0   | 4.0.0   | Audio time-stretch/pitch-shift
#   liblc3           | -      | low      | 1.1.3   | 1.1.3   | LC3 Bluetooth audio codec
#   libopencore-amr  | -      | legacy   | 0.1.6   | 0.1.6   | AMR-NB/WB (old mobile audio)
#
#   SUBTITLE/TEXT
#   libass           | apt    | high     | 0.17.3  | 0.17.4  | ASS/SSA subtitle renderer
#   libfreetype      | apt    | high     | 2.13.3  | 2.14.1  | Font rendering
#   libfontconfig    | apt    | high     | 2.15.0  | 2.17.0  | Font configuration
#   libfribidi       | apt    | med      | 1.0.16  | 1.0.16  | BiDi text (RTL languages)
#   libharfbuzz      | apt    | med      | 10.2.0  | 12.3.0  | Complex text shaping
#
#   FILTERS/PROCESSING
#   libzimg          | apt    | high     | 3.0.5   | 3.0.6   | High-quality image scaling
#   libsoxr          | apt    | high     | 0.1.3   | 0.1.3   | High-quality audio resampling
#   libvmaf          | src    | med      | 2.3.1   | 3.0.0   | Video quality metrics
#   libplacebo       | src    | med      | 7.349.0 | 7.351.0 | GPU HDR tone mapping
#   libshaderc       | src*   | med      | -       | -       | GLSL->SPIRV compiler (*via Vulkan SDK)
#   libvidstab       | apt    | med      | 1.1.0   | 1.1.1   | Video stabilization
#   libmysofa        | -      | low      | 1.3.3   | 1.3.3   | HRTF spatial audio (sofalizer)
#   libtesseract     | -      | low      | 5.5.0   | 5.5.1   | OCR text extraction
#   opencl           | apt    | low      | 2.3.3   | -       | GPU compute filters
#
#   HARDWARE ACCEL
#   libva            | src    | high     | 2.20.0  | 2.23.0  | VA-API (Intel/AMD) - Xe support
#   libvpl           | src    | high     | 2023.3  | 2.16.0  | Intel QuickSync Video
#   cuda-nvcc        | src    | high     | -       | -       | NVIDIA CUDA compiler
#   nvenc            | src    | high     | -       | -       | NVIDIA hardware encoder
#   cuvid            | src    | high     | -       | -       | NVIDIA hardware decoder
#   vaapi            | src    | high     | -       | -       | VA-API hwaccel
#   nvdec            | src    | med      | -       | -       | NVIDIA hwaccel decode API
#   vulkan           | src    | med      | -       | -       | Vulkan GPU compute
#   cuda-llvm        | -      | subsumed | -       | -       | CUDA via clang - we use nvcc
#   vdpau            | -      | legacy   | 1.5     | 1.5     | NVIDIA VDPAU (use nvdec)
#
#   PROTOCOLS/NETWORK
#   openssl          | apt    | high     | 3.0.13  | 3.0.15  | TLS/HTTPS support
#   libsrt           | apt    | high     | 1.5.3   | 1.5.4   | SRT streaming protocol
#   libssh           | -      | low      | 0.10.6  | 0.11.1  | SFTP protocol
#   librist          | -      | low      | 0.2.11  | 0.2.11  | RIST broadcast protocol
#   libzmq           | -      | low      | 4.3.5   | 4.3.5   | ZeroMQ IPC messaging
#   libxml2          | -      | low      | 2.9.14  | 2.13.5  | XML/DASH manifest parsing
#
#   INPUT/OUTPUT
#   libbluray        | apt    | med      | 1.3.4   | 1.4.0   | Blu-ray disc reading
#   libv4l2          | -      | low      | 1.28.1  | 1.28.1  | V4L2 webcam/capture
#   alsa             | -      | low      | 1.2.14  | 1.2.14  | Linux ALSA audio input
#
#   META FLAGS
#   gpl              | yes    | high     | -       | -       | Enable GPL-licensed code
#   version3         | yes    | high     | -       | -       | Enable (L)GPL v3 code
#   nonfree          | yes    | high     | -       | -       | Enable non-free code (fdk-aac)
#
# =============================================================================

# Hardware acceleration (set to 1 to enable)
ENABLE_NVIDIA_CUDA=${ENABLE_NVIDIA_CUDA:-1}  # NVENC/NVDEC hardware encoding/decoding
ENABLE_AMD_AMF=${ENABLE_AMD_AMF:-1}          # AMD AMF hardware encoding (requires AMD GPU)
ENABLE_LIBTORCH=${ENABLE_LIBTORCH:-1}              # LibTorch DNN backend for AI filters

# LibTorch CUDA variant (only used if ENABLE_LIBTORCH=1)
# LIBTORCH_VARIANT options:
#   "cu124"   - (default) CUDA 12.4 - required for FFmpeg compatibility (initXPU API)
#               Note: cu124 binaries work on CUDA 12.4+ runtimes (forward compatible)
#   "auto"    - auto-detect from CUDA_VERSION, rounding minor to nearest even
#               (PyTorch only releases cu126, cu128, cu130 - even minor versions)
#               Examples: CUDA 12.9 -> cu128, CUDA 12.7 -> cu126, CUDA 13.x -> cu130
#               WARNING: auto may select LibTorch 2.6.0+ which breaks FFmpeg (initXPU->init rename)
#   "cpu"     - CPU-only (no GPU acceleration for DNN filters)
#   "cu126"   - force CUDA 12.6
#   "cu128"   - force CUDA 12.8
#   "cu130"   - force CUDA 13.0
#   "rocm6.4" - AMD ROCm 6.4 (requires ROCm installed on host)
LIBTORCH_VARIANT=${LIBTORCH_VARIANT:-cu124}

# Optional build components (set to 0 to use apt package instead)
BUILD_LIBPLACEBO=${BUILD_LIBPLACEBO:-1}  # GPU HDR tone mapping (requires Vulkan SDK)
BUILD_LIBX265=${BUILD_LIBX265:-1}        # H.265/HEVC encoder (apt: 3.5, latest: 4.1)
BUILD_LIBAOM=${BUILD_LIBAOM:-1}          # AV1 reference codec (apt: 3.8, latest: 3.13)
BUILD_LIBWEBP=${BUILD_LIBWEBP:-1}        # WebP image codec (apt: 1.3, latest: 1.6)
BUILD_LIBVPL=${BUILD_LIBVPL:-1}          # Intel QuickSync (apt: 2023.3, latest: 2.16)
BUILD_LIBDAV1D=${BUILD_LIBDAV1D:-1}      # AV1 decoder (apt: 1.4.1, latest: 1.5.0)
BUILD_LIBSVTAV1=${BUILD_LIBSVTAV1:-1}    # AV1 encoder (apt: 1.7.0, latest: 3.0.0)
BUILD_LIBVMAF=${BUILD_LIBVMAF:-1}        # Video quality metrics (apt: 2.3.1, latest: 3.0.0)
BUILD_LIBVA=${BUILD_LIBVA:-1}            # VA-API (apt: 2.20.0, latest: 2.23.0 - Xe support)
BUILD_LIBJXL=${BUILD_LIBJXL:-1}          # JPEG XL (apt: 0.7.0, latest: 0.11.1)
BUILD_LIBX264=${BUILD_LIBX264:-1}        # H.264 encoder (apt: 8-bit only, src: 8/10-bit)

# FFmpeg version: "snapshot" for latest git, or specific version like "7.1"
FFMPEG_VERSION=${FFMPEG_VERSION:-snapshot}

# Skip apt dependency installation (use if deps already installed, avoids sudo)
SKIP_DEPS=${SKIP_DEPS:-0}

# NVIDIA CUDA setup (only used if ENABLE_NVIDIA_CUDA=1)
# CUDA_VERSION options:
#   "auto"    - (default) use installed CUDA if available, else install latest
#   "12-9"    - explicit version (e.g., 12-9, 12-6, 13-0)
CUDA_VERSION=${CUDA_VERSION:-auto}
# NVCC_GENCODE options:
#   "native"  - (default) compile for build machine's GPU via nvidia-smi
#   "minimum" - lowest arch for CUDA version (sm_52 for <13, sm_75 for 13+)
#   "75"      - explicit single arch (e.g., 75, 86, 89)
NVCC_GENCODE=${NVCC_GENCODE:-native}

# Build paths
SRC_DIR="${SRC_DIR:-$HOME/ffmpeg_sources}"      # Source code cache (can be deleted after build)
BUILD_DIR="${BUILD_DIR:-$HOME/ffmpeg_build}"    # Build artifacts cache (can be deleted after build)
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"          # Final binary install location
LIB_DIR="${LIB_DIR:-$HOME/.local/lib}"          # Final shared library install location (for libva)

NPROC=$(nproc)

mkdir -p "$SRC_DIR" "$BUILD_DIR" "$BIN_DIR" "$LIB_DIR"

# Base packages (installed first, includes wget needed for CUDA repo setup)
APT_PACKAGES=(
    autoconf
    automake
    build-essential
    cmake
    doxygen
    git
    meson
    nasm
    ninja-build
    pkg-config
    texinfo
    unzip
    wget
    yasm
    libass-dev
    libbluray-dev
    libfdk-aac-dev
    libfontconfig1-dev
    libfreetype6-dev
    libfribidi-dev
    libharfbuzz-dev
    libsoxr-dev
    libsrt-openssl-dev
    libssl-dev
    libzstd-dev
    libzimg-dev
    liblzma-dev
    liblzo2-dev
    libmp3lame-dev
    libnuma-dev
    ocl-icd-opencl-dev
    libopus-dev
    librubberband-dev
    libsdl2-dev
    libtool
    python3-jinja2
    libunistring-dev
    libvdpau-dev
    libvidstab-dev
    libdrm-dev
    libx11-dev
    libvorbis-dev
    libvpx-dev
    libxcb-shm0-dev
    libxcb-xfixes0-dev
    libxcb1-dev
    zlib1g-dev
)
# Add apt packages for libraries we're not building from source
[ "$BUILD_LIBX265" != "1" ] && APT_PACKAGES+=(libx265-dev)
[ "$BUILD_LIBAOM" != "1" ] && APT_PACKAGES+=(libaom-dev)
[ "$BUILD_LIBWEBP" != "1" ] && APT_PACKAGES+=(libwebp-dev)
[ "$BUILD_LIBVPL" != "1" ] && APT_PACKAGES+=(libvpl-dev)
[ "$BUILD_LIBDAV1D" != "1" ] && APT_PACKAGES+=(libdav1d-dev)
[ "$BUILD_LIBSVTAV1" != "1" ] && APT_PACKAGES+=(libsvtav1enc-dev)
[ "$BUILD_LIBVMAF" != "1" ] && APT_PACKAGES+=(libvmaf-dev)
[ "$BUILD_LIBVA" != "1" ] && APT_PACKAGES+=(libva-dev)
[ "$BUILD_LIBJXL" != "1" ] && APT_PACKAGES+=(libjxl-dev)
[ "$BUILD_LIBX264" != "1" ] && APT_PACKAGES+=(libx264-dev)
if [ "$SKIP_DEPS" != "1" ]; then
    sudo apt-get update && sudo apt-get install -y "${APT_PACKAGES[@]}"
fi


CUDA_FLAGS=()
NVCC_ARCH=""

if [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
    # Check if CUDA is already installed
    if [ "$CUDA_VERSION" = "auto" ]; then
        if command -v nvcc &> /dev/null; then
            # Extract version from nvcc (e.g., "12.9" -> "12-9")
            NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
            CUDA_VERSION=$(echo "$NVCC_VERSION" | tr '.' '-')
            echo "Detected installed CUDA $NVCC_VERSION (using version $CUDA_VERSION)"
        else
            echo "No CUDA installed, will install latest from NVIDIA repo"
        fi
    fi

    # Add CUDA repo if not present or if we need to install
    if [ "$SKIP_DEPS" != "1" ]; then
        if [ "$CUDA_VERSION" = "auto" ] || ! command -v nvcc &> /dev/null; then
            if ! dpkg -l cuda-keyring 2>/dev/null | grep -q ^ii; then
                # Detect Ubuntu version for correct CUDA repo (24.04 -> ubuntu2404, 25.04 -> ubuntu2504)
                UBUNTU_VERSION=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2 | tr -d '.')
                CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb"
                wget -q "$CUDA_REPO_URL" -O cuda-keyring.deb
                sudo dpkg -i cuda-keyring.deb
                rm cuda-keyring.deb
                sudo apt-get update
            fi

            # Get latest CUDA version if still auto
            if [ "$CUDA_VERSION" = "auto" ]; then
                CUDA_VERSION=$(apt-cache search '^cuda-nvcc-[0-9]' | sed 's/cuda-nvcc-//' | cut -d' ' -f1 | sort -V | tail -1)
                if [ -z "$CUDA_VERSION" ]; then
                    echo "Error: No CUDA packages found. Install CUDA repo first or set CUDA_VERSION manually." >&2
                    exit 1
                fi
                echo "Will install latest CUDA version: $CUDA_VERSION"
            fi
        fi

        # Install CUDA packages
        sudo apt-get install -y libffmpeg-nvenc-dev cuda-nvcc-$CUDA_VERSION cuda-cudart-dev-$CUDA_VERSION
    fi
    echo "Using CUDA version: $CUDA_VERSION"

    # Detect CUDA installation path
    CUDA_VERSION_DOT=$(echo "$CUDA_VERSION" | tr '-' '.')
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    elif [ -d "/usr/local/cuda-${CUDA_VERSION_DOT}" ]; then
        CUDA_PATH="/usr/local/cuda-${CUDA_VERSION_DOT}"
    else
        echo "Warning: CUDA path not found, using /usr/local/cuda (headers may be missing)" >&2
        CUDA_PATH="/usr/local/cuda"
    fi
    echo "Using CUDA path: $CUDA_PATH"

    # Patch CUDA headers for glibc 2.42+ compatibility (Ubuntu 25.04+)
    # glibc 2.42 added rsqrt/rsqrtf to mathcalls.h which conflicts with CUDA's definitions
    # This causes "exception specification is incompatible" errors during nvcc compilation
    if [ "$SKIP_DEPS" != "1" ]; then
        CUDA_MATH_HEADER="$CUDA_PATH/targets/x86_64-linux/include/crt/math_functions.h"
        if [ -f "$CUDA_MATH_HEADER" ]; then
            GLIBC_VERSION=$(ldd --version | head -1 | grep -oP '\d+\.\d+$')
            GLIBC_MAJOR=$(echo "$GLIBC_VERSION" | cut -d. -f1)
            GLIBC_MINOR=$(echo "$GLIBC_VERSION" | cut -d. -f2)

            # Only patch if glibc >= 2.42 and patch not already applied
            if [ "$GLIBC_MAJOR" -gt 2 ] || ([ "$GLIBC_MAJOR" -eq 2 ] && [ "$GLIBC_MINOR" -ge 42 ]); then
                # Check for our patch OR NVIDIA's fix (they use __NV_GLIBC_PROVIDES_IEC_60559_FUNCS for similar issues)
                if grep -q "rsqrt" "$CUDA_MATH_HEADER" && \
                   ! grep -B2 "double[[:space:]]*rsqrt(double" "$CUDA_MATH_HEADER" | grep -q "GLIBC"; then
                    echo "Patching CUDA headers for glibc $GLIBC_VERSION compatibility..."
                    # Backup original if no backup exists
                    [ ! -f "${CUDA_MATH_HEADER}.bak" ] && sudo cp "$CUDA_MATH_HEADER" "${CUDA_MATH_HEADER}.bak"
                    # Add guards around rsqrt declaration (prevent conflict with glibc's rsqrt)
                    sudo sed -i '/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double[[:space:]]*rsqrt(double/c\
#if !(defined(__GLIBC__) \&\& __GLIBC_USE_IEC_60559_FUNCS_EXT_C23)\
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x);\
#endif' "$CUDA_MATH_HEADER"
                    # Add guards around rsqrtf declaration
                    sudo sed -i '/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float[[:space:]]*rsqrtf(float/c\
#if !(defined(__GLIBC__) \&\& __GLIBC_USE_IEC_60559_FUNCS_EXT_C23)\
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x);\
#endif' "$CUDA_MATH_HEADER"
                    # Verify patch was applied
                    if grep -B2 "double[[:space:]]*rsqrt(double" "$CUDA_MATH_HEADER" | grep -q "GLIBC"; then
                        echo "CUDA header patched successfully"
                    else
                        echo "Warning: CUDA header patch may have failed - rsqrt declaration not found" >&2
                        echo "CUDA version may have different header format. Check $CUDA_MATH_HEADER" >&2
                    fi
                else
                    echo "CUDA headers already patched for glibc compatibility"
                fi
            fi
        fi
    fi

    CUDA_FLAGS=(--enable-cuda-nvcc --enable-nvenc --enable-cuvid --enable-nvdec)

    CUDA_MAJOR="${CUDA_VERSION%%-*}"

    if [ "$NVCC_GENCODE" = "native" ]; then
        # Detect GPU compute capability
        if command -v nvidia-smi &> /dev/null; then
            COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
            if [ -n "$COMPUTE_CAP" ]; then
                COMPUTE_CAP_NUM=$(echo "$COMPUTE_CAP" | tr -d '.')
                NVCC_ARCH="-arch=sm_${COMPUTE_CAP_NUM}"
                echo "CUDA $CUDA_VERSION NVCC_GENCODE=native -> $NVCC_ARCH (detected via nvidia-smi)"
            else
                echo "Warning: nvidia-smi found but no GPU detected, falling back to minimum" >&2
                NVCC_GENCODE="minimum"
            fi
        else
            echo "Warning: nvidia-smi not found, falling back to minimum arch" >&2
            NVCC_GENCODE="minimum"
        fi
    fi

    if [ "$NVCC_GENCODE" = "minimum" ]; then
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            NVCC_ARCH="-arch=sm_75"
        else
            NVCC_ARCH="-arch=sm_52"
        fi
        echo "CUDA $CUDA_VERSION NVCC_GENCODE=minimum -> $NVCC_ARCH"
    elif [ "$NVCC_GENCODE" != "native" ]; then
        # Explicit arch number
        NVCC_ARCH="-arch=sm_$NVCC_GENCODE"
        echo "CUDA $CUDA_VERSION NVCC_GENCODE=$NVCC_GENCODE -> $NVCC_ARCH"
    fi

    cd "$SRC_DIR" &&
    ([ -d nv-codec-headers/.git ] && git -C nv-codec-headers pull || (rm -rf nv-codec-headers && git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git)) &&
    cd nv-codec-headers &&
    make &&
    make PREFIX="$BUILD_DIR" install
fi


# AMD AMF setup (hardware encoding for AMD GPUs)
# AMF is header-only at build time; runtime driver comes from host's AMD GPU driver
AMF_FLAGS=()
if [ "$ENABLE_AMD_AMF" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d AMF/.git ] && git -C AMF pull || (rm -rf AMF && git clone --depth 1 https://github.com/GPUOpen-LibrariesAndSDKs/AMF.git)) &&
    mkdir -p "$BUILD_DIR/include/AMF" &&
    cp -r AMF/amf/public/include/* "$BUILD_DIR/include/AMF/"
    AMF_FLAGS=(--enable-amf)
    echo "AMF headers installed for AMD GPU encoding"
fi


# libx264 (H.264/AVC encoder)
# Build with --bit-depth=all for 8-bit and 10-bit support
if [ "$BUILD_LIBX264" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d x264/.git ] && git -C x264 pull || (rm -rf x264 && git clone --depth 1 https://code.videolan.org/videolan/x264.git)) &&
    cd x264 &&
    PATH="$BIN_DIR:$PATH" ./configure --prefix="$BUILD_DIR" --enable-static --enable-pic --disable-cli --bit-depth=all &&
    PATH="$BIN_DIR:$PATH" make -j $NPROC &&
    make install
fi


# libx265 (H.265/HEVC encoder)
# Multilib build: 8-bit + 10-bit + 12-bit support (required for HDR)
# Build order: 12-bit → 10-bit → 8-bit (main links the others)
if [ "$BUILD_LIBX265" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d x265_git/.git ] && git -C x265_git pull || (rm -rf x265_git && git clone --depth 1 https://bitbucket.org/multicoreware/x265_git.git)) &&
    cd x265_git/build/linux &&

    # Clean previous builds
    rm -rf 8bit 10bit 12bit &&
    mkdir -p 8bit 10bit 12bit &&

    # Build 12-bit
    cd 12bit &&
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
        -DHIGH_BIT_DEPTH=ON \
        -DEXPORT_C_API=OFF \
        -DENABLE_SHARED=OFF \
        -DENABLE_CLI=OFF \
        -DMAIN12=ON \
        ../../../source &&
    PATH="$BIN_DIR:$PATH" make -j $NPROC &&

    # Build 10-bit
    cd ../10bit &&
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
        -DHIGH_BIT_DEPTH=ON \
        -DEXPORT_C_API=OFF \
        -DENABLE_SHARED=OFF \
        -DENABLE_CLI=OFF \
        ../../../source &&
    PATH="$BIN_DIR:$PATH" make -j $NPROC &&

    # Build 8-bit (main) and link in 10-bit and 12-bit
    cd ../8bit &&
    ln -sf ../10bit/libx265.a libx265_main10.a &&
    ln -sf ../12bit/libx265.a libx265_main12.a &&
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
        -DLIB_INSTALL_DIR="$BUILD_DIR/lib" \
        -DENABLE_SHARED=OFF \
        -DENABLE_CLI=OFF \
        -DEXTRA_LIB="x265_main10.a;x265_main12.a" \
        -DEXTRA_LINK_FLAGS="-L." \
        -DLINKED_10BIT=ON \
        -DLINKED_12BIT=ON \
        ../../../source &&
    PATH="$BIN_DIR:$PATH" make -j $NPROC &&
    # Merge 8-bit, 10-bit, and 12-bit libraries into one (cmake doesn't do this automatically)
    mv libx265.a libx265_main.a &&
    mkdir -p merged/8bit merged/10bit merged/12bit &&
    (cd merged/8bit && ar x ../../libx265_main.a) &&
    (cd merged/10bit && ar x ../../libx265_main10.a) &&
    (cd merged/12bit && ar x ../../libx265_main12.a) &&
    ar crs libx265.a merged/*/*.o &&
    rm -rf merged libx265_main.a &&
    make install &&

    # x265's cmake doesn't reliably install x265.pc, so we create it manually
    # Extract version from x265.h (format: #define X265_BUILD 215)
    X265_VERSION=$(grep '#define X265_BUILD' "$BUILD_DIR/include/x265.h" | awk '{print $3}') &&
    mkdir -p "$BUILD_DIR/lib/pkgconfig" &&
    cat > "$BUILD_DIR/lib/pkgconfig/x265.pc" << PCEOF
prefix=$BUILD_DIR
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: x265
Description: H.265/HEVC video encoder (8-bit + 10-bit + 12-bit)
Version: $X265_VERSION
Libs: -L\${libdir} -lx265
Libs.private: -lstdc++ -lm -lrt -ldl -lnuma -lpthread
Cflags: -I\${includedir}
PCEOF
fi

# libaom (AV1 reference codec)
if [ "$BUILD_LIBAOM" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d aom/.git ] && git -C aom pull || (rm -rf aom && git clone --depth 1 https://aomedia.googlesource.com/aom)) &&
    mkdir -p aom_build &&
    cd aom_build &&
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DENABLE_TESTS=OFF -DENABLE_NASM=on -DBUILD_SHARED_LIBS=OFF -DCONFIG_AV1_HIGHBITDEPTH=1 ../aom &&
    PATH="$BIN_DIR:$PATH" make -j $NPROC &&
    make install
fi

# libwebp (WebP image codec)
if [ "$BUILD_LIBWEBP" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d libwebp/.git ] && git -C libwebp pull || (rm -rf libwebp && git clone --depth 1 https://chromium.googlesource.com/webm/libwebp)) &&
    cd libwebp &&
    ./autogen.sh &&
    ./configure --prefix="$BUILD_DIR" --disable-shared --enable-static &&
    make -j $NPROC &&
    make install
fi

# libjxl (JPEG XL image codec)
# Ubuntu 24.04 ships 0.7.0 which is quite old; latest is 0.11.1 with HDR improvements
if [ "$BUILD_LIBJXL" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d libjxl/.git ] && git -C libjxl pull || (rm -rf libjxl && git clone --depth 1 --recursive https://github.com/libjxl/libjxl.git)) &&
    cd libjxl &&
    mkdir -p build &&
    cd build &&
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF \
        -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_PLUGINS=OFF -DJPEGXL_ENABLE_VIEWERS=OFF \
        -DJPEGXL_ENABLE_TOOLS=OFF -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_JPEGLI=OFF .. &&
    make -j $NPROC &&
    make install
fi

# libvpl (Intel Video Processing Library / QuickSync)
if [ "$BUILD_LIBVPL" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d libvpl/.git ] && git -C libvpl pull || (rm -rf libvpl && git clone --depth 1 https://github.com/intel/libvpl.git)) &&
    mkdir -p libvpl/build &&
    cd libvpl/build &&
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DBUILD_SHARED_LIBS=OFF .. &&
    make -j $NPROC &&
    make install
fi

# libdav1d (AV1 decoder)
if [ "$BUILD_LIBDAV1D" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d dav1d/.git ] && git -C dav1d pull || (rm -rf dav1d && git clone --depth 1 https://code.videolan.org/videolan/dav1d.git)) &&
    cd dav1d &&
    if [ -f build/build.ninja ]; then
        meson setup --reconfigure build --buildtype=release --default-library=static --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    else
        meson setup build --buildtype=release --default-library=static --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    fi &&
    ninja -C build &&
    ninja -C build install
fi

# libsvtav1 (AV1 encoder)
if [ "$BUILD_LIBSVTAV1" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d SVT-AV1/.git ] && git -C SVT-AV1 pull || (rm -rf SVT-AV1 && git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git)) &&
    mkdir -p SVT-AV1/build &&
    cd SVT-AV1/build &&
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. &&
    PATH="$BIN_DIR:$PATH" make -j $NPROC &&
    make install
fi

# libvmaf (video quality metrics)
if [ "$BUILD_LIBVMAF" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d vmaf/.git ] && git -C vmaf pull || (rm -rf vmaf && git clone --depth 1 https://github.com/Netflix/vmaf)) &&
    mkdir -p vmaf/libvmaf/build &&
    cd vmaf/libvmaf/build &&
    if [ -f build.ninja ]; then
        meson setup --reconfigure -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
    else
        meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
    fi &&
    ninja &&
    ninja install
fi

# libva (VA-API)
# Ubuntu 24.04 ships 2.20.0 which lacks Intel Xe kernel driver support (added in 2.21)
# Build from source to get Xe support for newer Intel GPUs
if [ "$BUILD_LIBVA" = "1" ]; then
    cd "$SRC_DIR" &&
    ([ -d libva/.git ] && git -C libva pull || (rm -rf libva && git clone --depth 1 https://github.com/intel/libva.git)) &&
    cd libva &&
    if [ -f build/build.ninja ]; then
        meson setup --reconfigure build --buildtype=release --default-library=shared --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    else
        meson setup build --buildtype=release --default-library=shared --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    fi &&
    ninja -C build &&
    ninja -C build install &&
    # Copy shared libs to permanent location (LIB_DIR) for runtime
    cp -a "$BUILD_DIR/lib"/libva*.so* "$LIB_DIR/"
fi

# libplacebo (for GPU tone mapping)
if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    # Download Vulkan SDK tarball (apt packages deprecated May 2025)
    VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION:-1.4.335.0}
    VULKAN_SDK_DIR="${SRC_DIR}/vulkan-sdk-${VULKAN_SDK_VERSION}"
    if [ ! -d "$VULKAN_SDK_DIR" ]; then
        echo "Downloading Vulkan SDK $VULKAN_SDK_VERSION..."
        cd "$SRC_DIR"
        rm -f vulkansdk.tar.xz    # Clean up any partial download
        wget -q -O vulkansdk.tar.xz "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
        tar xf vulkansdk.tar.xz
        mv "${VULKAN_SDK_VERSION}" "vulkan-sdk-${VULKAN_SDK_VERSION}"
        rm -f vulkansdk.tar.xz
    fi
    export VULKAN_SDK="$VULKAN_SDK_DIR/x86_64"
    export PATH="$VULKAN_SDK/bin:$PATH"
    export PKG_CONFIG_PATH="$VULKAN_SDK/lib/pkgconfig:$PKG_CONFIG_PATH"
    echo "Using Vulkan SDK: $VULKAN_SDK"

    # Use static shaderc (avoid runtime .so dependency)
    if [ ! -f "$VULKAN_SDK/lib/pkgconfig/shaderc.pc.bak" ]; then
        cp "$VULKAN_SDK/lib/pkgconfig/shaderc.pc" "$VULKAN_SDK/lib/pkgconfig/shaderc.pc.bak"
    fi
    cp "$VULKAN_SDK/lib/pkgconfig/shaderc_combined.pc" "$VULKAN_SDK/lib/pkgconfig/shaderc.pc"

    cd "$SRC_DIR" &&
    ([ -d libplacebo/.git ] && git -C libplacebo pull || (rm -rf libplacebo && git clone --depth 1 https://code.videolan.org/videolan/libplacebo.git)) &&
    cd libplacebo &&
    if [ -f build/build.ninja ]; then
        meson setup --reconfigure build --buildtype=release --default-library=static --prefer-static -Dvulkan=enabled -Dvulkan-registry="$VULKAN_SDK/share/vulkan/registry/vk.xml" -Dopengl=disabled -Dd3d11=disabled -Ddemos=false --prefix "$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    else
        meson setup build --buildtype=release --default-library=static --prefer-static -Dvulkan=enabled -Dvulkan-registry="$VULKAN_SDK/share/vulkan/registry/vk.xml" -Dopengl=disabled -Dd3d11=disabled -Ddemos=false --prefix "$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    fi &&
    ninja -C build &&
    ninja -C build install
fi

# LibTorch (PyTorch C++ library for DNN backend)
# Enables AI-based video filters like dnn_processing for upscaling, denoising, etc.
# NOTE: LibTorch 2.6.0+ renamed initXPU() to init(). We patch ffmpeg to handle both.
#       Use 2.7.0+ for RTX 50-series (Blackwell/SM 12.0) support.
LIBTORCH_FLAGS=()
if [ "$ENABLE_LIBTORCH" = "1" ]; then
    LIBTORCH_VERSION=${LIBTORCH_VERSION:-2.5.0}
    LIBTORCH_DIR="$SRC_DIR/libtorch"

    # Determine LibTorch CUDA variant
    # LIBTORCH_VARIANT: cu124 (default), auto, cpu, cu126, cu128, cu130, rocm6.4
    # PyTorch only releases for even-numbered CUDA versions
    if [ "$LIBTORCH_VARIANT" != "auto" ]; then
        TORCH_VARIANT="$LIBTORCH_VARIANT"
        echo "LibTorch: using $TORCH_VARIANT (explicit)"
    elif [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
        CUDA_MAJOR="${CUDA_VERSION%%-*}"
        CUDA_MINOR="${CUDA_VERSION#*-}"
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            TORCH_VARIANT="cu130"
        else
            # Round down to nearest even, clamp to [6, 8]
            EVEN_MINOR=$(( (CUDA_MINOR / 2) * 2 ))
            [ "$EVEN_MINOR" -gt 8 ] && EVEN_MINOR=8
            [ "$EVEN_MINOR" -lt 6 ] && EVEN_MINOR=6
            TORCH_VARIANT="cu12${EVEN_MINOR}"
        fi
        echo "LibTorch: using $TORCH_VARIANT (from CUDA $CUDA_VERSION)"
    else
        TORCH_VARIANT="cpu"
        echo "LibTorch: using CPU-only variant"
    fi

    # Download LibTorch if not present or wrong variant
    LIBTORCH_MARKER="$LIBTORCH_DIR/.variant-${TORCH_VARIANT}"
    if [ ! -f "$LIBTORCH_MARKER" ]; then
        echo "Downloading LibTorch $LIBTORCH_VERSION ($TORCH_VARIANT)..."
        cd "$SRC_DIR"
        rm -rf libtorch libtorch.zip

        # Download from pytorch.org
        # cu124 and earlier use cxx11-abi prefix, cu130+ dropped it
        if [[ "$TORCH_VARIANT" == cu13* ]] || [[ "$TORCH_VARIANT" == cu14* ]]; then
            LIBTORCH_URL="https://download.pytorch.org/libtorch/${TORCH_VARIANT}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${TORCH_VARIANT}.zip"
        else
            LIBTORCH_URL="https://download.pytorch.org/libtorch/${TORCH_VARIANT}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${TORCH_VARIANT}.zip"
        fi
        wget -q -O libtorch.zip "$LIBTORCH_URL"
        unzip -q libtorch.zip
        rm -f libtorch.zip
        touch "$LIBTORCH_MARKER"
    fi

    export LIBTORCH_PATH="$LIBTORCH_DIR"
    LIBTORCH_FLAGS=(--enable-libtorch)
    echo "Using LibTorch: $LIBTORCH_PATH"

    # Copy libtorch shared libs to permanent location (LIB_DIR)
    echo "Installing libtorch libs to $LIB_DIR..."
    cp -a "$LIBTORCH_DIR/lib"/*.so* "$LIB_DIR/" 2>/dev/null || true

    # Create pkg-config file for libtorch (FFmpeg configure uses pkg-config for detection)
    mkdir -p "$BUILD_DIR/lib/pkgconfig"
    # Include CUDA libs if using CUDA variant
    if [[ "$TORCH_VARIANT" == cu* ]]; then
        TORCH_LIBS="-ltorch -lc10 -ltorch_cpu -ltorch_cuda -lc10_cuda"
        # Needed for ffmpeg extra-libs to ensure libtorch_cuda is linked (not just dlopen'd)
        TORCH_EXTRA_LIBS="-lc10_cuda -ltorch_cuda"
    else
        TORCH_LIBS="-ltorch -lc10 -ltorch_cpu"
        TORCH_EXTRA_LIBS=""
    fi
    cat > "$BUILD_DIR/lib/pkgconfig/libtorch.pc" << PCEOF
prefix=$LIBTORCH_DIR
exec_prefix=\${prefix}
libdir=$LIB_DIR
includedir=\${prefix}/include

Name: libtorch
Description: PyTorch C++ library
Version: $LIBTORCH_VERSION
Libs: -L\${libdir} $TORCH_LIBS
Cflags: -I\${includedir} -I\${includedir}/torch/csrc/api/include -std=c++17
PCEOF
    echo "Created libtorch.pc for pkg-config detection (variant: $TORCH_VARIANT)"
fi

# ffmpeg
FFMPEG_DIR="ffmpeg-${FFMPEG_VERSION}"
cd "$SRC_DIR"
if [ ! -d "$FFMPEG_DIR" ]; then
    if [ "$FFMPEG_VERSION" = "snapshot" ]; then
        rm -f ffmpeg-snapshot.tar.bz2    # Clean up any partial download
        wget -q -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
        tar xjf ffmpeg-snapshot.tar.bz2
        mv ffmpeg "$FFMPEG_DIR"
        rm -f ffmpeg-snapshot.tar.bz2
    else
        rm -f "ffmpeg-${FFMPEG_VERSION}.tar.xz"    # Clean up any partial download
        wget -q -O "ffmpeg-${FFMPEG_VERSION}.tar.xz" "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
        tar xJf "ffmpeg-${FFMPEG_VERSION}.tar.xz"
        rm -f "ffmpeg-${FFMPEG_VERSION}.tar.xz"
    fi
fi

# Patch ffmpeg's torch backend
if [ "$ENABLE_LIBTORCH" = "1" ]; then
    TORCH_BACKEND="$FFMPEG_DIR/libavfilter/dnn/dnn_backend_torch.cpp"

    # Patch 1: Fix initXPU() -> init() for libtorch 2.6+ compatibility
    if [ -f "$TORCH_BACKEND" ] && grep -q "initXPU()" "$TORCH_BACKEND"; then
        TORCH_MAJOR=$(echo "$LIBTORCH_VERSION" | cut -d. -f1)
        TORCH_MINOR=$(echo "$LIBTORCH_VERSION" | cut -d. -f2)
        if [ "$TORCH_MAJOR" -gt 2 ] || { [ "$TORCH_MAJOR" -eq 2 ] && [ "$TORCH_MINOR" -ge 6 ]; }; then
            echo "Patching ffmpeg for libtorch 2.6+ (initXPU -> init)..."
            sed -i 's/initXPU()/init()/g' "$TORCH_BACKEND"
        fi
    fi

    # Patch 2: Add CUDA device support (upstream only supports CPU/XPU)
    if [ -f "$TORCH_BACKEND" ] && ! grep -q "device.is_cuda()" "$TORCH_BACKEND"; then
        echo "Patching ffmpeg torch backend for CUDA support..."
        # Add CUDA device support between XPU and the catch-all error
        # Also adds dlopen for libtorch_cuda.so to load CUDA kernels at runtime
        sed -i '/at::detail::getXPUHooks().init/a\
    } else if (device.is_cuda()) {\
        if (!at::cuda::is_available()) {\
            av_log(ctx, AV_LOG_ERROR, "No CUDA device found\\n");\
            goto fail;\
        }\
        // Load CUDA kernels - required for libtorch CUDA ops\
        static bool cuda_lib_loaded = false;\
        if (!cuda_lib_loaded) {\
            cuda_lib_loaded = true;\
            void *cuda_handle = dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_GLOBAL);\
            if (cuda_handle) {\
                av_log(ctx, AV_LOG_DEBUG, "libtorch_cuda.so loaded\\n");\
            } else {\
                av_log(ctx, AV_LOG_WARNING, "Failed to load libtorch_cuda.so: %s\\n", dlerror());\
            }\
        }' "$TORCH_BACKEND"
        # Add required CUDA header
        if ! grep -q "#include <ATen/cuda/CUDAContext.h>" "$TORCH_BACKEND"; then
            sed -i '/#include <torch\/torch.h>/a #include <ATen/cuda/CUDAContext.h>' "$TORCH_BACKEND"
        fi
        echo "Torch CUDA patch applied"
    fi

    # Patch 3: Add TensorRT support (load runtime + handle tuple outputs)
    if [ -f "$TORCH_BACKEND" ] && ! grep -q "isTuple" "$TORCH_BACKEND"; then
        echo "Patching ffmpeg torch backend for TensorRT support..."

        # Add dlfcn.h header for dlopen
        if ! grep -q "#include <dlfcn.h>" "$TORCH_BACKEND"; then
            sed -i '/#include <torch\/script.h>/a #include <dlfcn.h>' "$TORCH_BACKEND"
        fi

        # Add TensorRT runtime loading in model init (before torch::jit::load)
        if ! grep -q "libtorchtrt_runtime" "$TORCH_BACKEND"; then
            sed -i '/torch::jit::load(ctx->model_filename)/i\
    // Load TensorRT runtime if available (enables TRT-compiled models)\
    static bool trt_init_attempted = false;\
    if (!trt_init_attempted) {\
        trt_init_attempted = true;\
        void *trt_handle = dlopen("libtorchtrt_runtime.so", RTLD_NOW | RTLD_GLOBAL);\
        if (trt_handle) {\
            av_log(ctx, AV_LOG_INFO, "TensorRT runtime loaded\\n");\
        }\
    }' "$TORCH_BACKEND"
        fi

        # Change forward().toTensor() to handle TRT tuple outputs
        sed -i 's/\*infer_request->output = th_model->jit_model->forward(inputs)\.toTensor();/auto _fwd_out = th_model->jit_model->forward(inputs);\
    if (_fwd_out.isTuple()) {\
        *infer_request->output = _fwd_out.toTuple()->elements()[0].toTensor();\
    } else {\
        *infer_request->output = _fwd_out.toTensor();\
    }/' "$TORCH_BACKEND"

        # Fix device detection for TRT models (they may not have parameters)
        sed -i 's/c10::Device device = (\*th_model->jit_model->parameters()\.begin())\.device();/c10::Device device = torch::kCUDA;\
    auto params = th_model->jit_model->parameters();\
    if (params.begin() != params.end()) {\
        device = (*params.begin()).device();\
    }/' "$TORCH_BACKEND"

        echo "Torch TensorRT patch applied"
    fi
fi

cd "$FFMPEG_DIR" && \
# Build configure flags
# MARCH=native for CPU-specific optimizations (opt-in, not portable)
EXTRA_CFLAGS="-I$BUILD_DIR/include -O3${MARCH:+ -march=$MARCH -mtune=$MARCH}"
EXTRA_CXXFLAGS=""
# -rpath embeds library search path in binary so it finds our built libs at runtime
EXTRA_LDFLAGS="-L$BUILD_DIR/lib -s -Wl,-rpath,$LIB_DIR"
if [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
    EXTRA_CFLAGS="$EXTRA_CFLAGS -I$CUDA_PATH/include"
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$CUDA_PATH/lib64"
fi
if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    EXTRA_CFLAGS="$EXTRA_CFLAGS -I$VULKAN_SDK/include"
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$VULKAN_SDK/lib"
fi
if [ "$ENABLE_LIBTORCH" = "1" ]; then
    # LibTorch needs C++ flags (FFmpeg uses require_cxx for libtorch detection)
    # Include CUDA path for CUDA torch support
    EXTRA_CXXFLAGS="-I$LIBTORCH_PATH/include -I$LIBTORCH_PATH/include/torch/csrc/api/include"
    if [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
        EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -I$CUDA_PATH/include"
    fi
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$LIB_DIR -Wl,-rpath,$LIB_DIR"
fi
CONFIGURE_CMD=(
    ./configure
    --prefix="$BUILD_DIR"
    --pkg-config-flags="--static"
    --extra-cflags="$EXTRA_CFLAGS"
    --extra-cxxflags="$EXTRA_CXXFLAGS"
    --extra-ldflags="$EXTRA_LDFLAGS"
    --extra-libs="-lpthread -lm -ldl${TORCH_EXTRA_LIBS:+ $TORCH_EXTRA_LIBS}"
    --ld="g++"
    --bindir="$BIN_DIR"
    --disable-debug
    --enable-gpl
    --enable-version3
    --enable-openssl
    --enable-libaom
    --enable-libass
    --enable-libbluray
    --enable-libfdk-aac
    --enable-libfontconfig
    --enable-libfreetype
    --enable-libfribidi
    --enable-libharfbuzz
    --enable-libjxl
    --enable-libmp3lame
    --enable-libopus
    --enable-libsvtav1
    --enable-libdav1d
    --enable-libvmaf
    --enable-libvorbis
    --enable-libvpx
    --enable-libwebp
    --enable-libx264
    --enable-libx265
    --enable-librubberband
    --enable-libsoxr
    --enable-libsrt
    --enable-libvidstab
    --enable-libvpl
    --enable-libzimg
    --enable-opencl
    --enable-vaapi
    --enable-nonfree
    "${CUDA_FLAGS[@]}"
    "${AMF_FLAGS[@]}"
    "${LIBTORCH_FLAGS[@]}"
)

if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    CONFIGURE_CMD+=(--enable-vulkan --enable-libplacebo)
fi

if [ -n "$NVCC_ARCH" ]; then
    CONFIGURE_CMD+=(--nvccflags="$NVCC_ARCH")
fi

# Build PATH: include CUDA bin if NVIDIA enabled
BUILD_PATH="$BIN_DIR:$PATH"
[ "$ENABLE_NVIDIA_CUDA" = "1" ] && BUILD_PATH="$CUDA_PATH/bin:$BUILD_PATH"

PATH="$BUILD_PATH" PKG_CONFIG_PATH="$BUILD_DIR/lib/pkgconfig:$PKG_CONFIG_PATH" "${CONFIGURE_CMD[@]}" && \
PATH="$BUILD_PATH" make -j $NPROC && \
make install && \
hash -r

grep -q "$BUILD_DIR/share/man" "$HOME/.manpath" 2>/dev/null || echo "MANPATH_MAP $BIN_DIR $BUILD_DIR/share/man" >> "$HOME/.manpath"

# rm -rf ~/ffmpeg_build ~/.local/bin/{ffmpeg,ffprobe,ffplay,x264,x265}
# sed -i '/ffmpeg_build/d' ~/.manpath
# hash -r
# --extra-cflags="-D_GNU_SOURCE"
# cat ~/ffmpeg_sources/ffmpeg/ffbuild/config.log

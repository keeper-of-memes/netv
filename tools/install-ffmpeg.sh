#!/bin/bash
# Build ffmpeg from source with NVIDIA NVENC support
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
set -e

# =============================================================================
# Ubuntu 24.04 apt packages vs upstream (checked 2026-01)
# Sorted by staleness (most stale first)
#
#   Package      | Apt Version | Latest  | Status
#   -------------|-------------|---------|----------------------------------
#   libsvtav1    | 1.7.0       | 3.0.0   | 2 major behind - build from source
#   libx265      | 3.5         | 4.1     | 1 major behind - build from source
#   libaom       | 3.8.2       | 3.13.1  | 5 minor behind - build from source
#   libvpl       | 2023.3.0    | 2.16.0  | old API - build from source
#   libwebp      | 1.3.2       | 1.6.0   | 3 minor behind - build from source
#   libdav1d     | 1.4.1       | 1.5.0   | 1 minor behind - build from source
#   libopus      | 1.4         | 1.5.2   | 1 minor behind
#   libvpx       | 1.14.1      | 1.15.0  | 1 minor behind
#   libass       | 0.17.1      | 0.17.3  | 2 patch behind
#   nasm         | 2.16.01     | 2.16.03 | 2 patch behind
#   libfdk-aac   | 2.0.2       | 2.0.3   | 1 patch behind
#   libfreetype  | 2.13.2      | 2.13.3  | 1 patch behind
#   libx264      | 0.164       | 0.164   | current
#   libvorbis    | 1.3.7       | 1.3.7   | current
#   libmp3lame   | 3.100       | 3.100   | current
#   libfontconfig| 2.15.0      | 2.15.0  | current
#
# =============================================================================

# Optional build components (set to 0 to use apt package instead)
BUILD_NVIDIA=${BUILD_NVIDIA:-1}          # NVENC/NVDEC hardware encoding/decoding
BUILD_LIBPLACEBO=${BUILD_LIBPLACEBO:-1}  # GPU HDR tone mapping (requires Vulkan SDK)
BUILD_X265=${BUILD_X265:-1}              # H.265/HEVC encoder (apt: 3.5, latest: 4.1)
BUILD_LIBAOM=${BUILD_LIBAOM:-1}          # AV1 reference codec (apt: 3.8, latest: 3.13)
BUILD_LIBWEBP=${BUILD_LIBWEBP:-1}        # WebP image codec (apt: 1.3, latest: 1.6)
BUILD_LIBVPL=${BUILD_LIBVPL:-1}          # Intel QuickSync (apt: 2023.3, latest: 2.16)
BUILD_LIBDAV1D=${BUILD_LIBDAV1D:-1}      # AV1 decoder (apt: 1.4.1, latest: 1.5.0)

# FFmpeg version: "snapshot" for latest git, or specific version like "7.1"
FFMPEG_VERSION=${FFMPEG_VERSION:-snapshot}

# Build paths
SRC_DIR="${SRC_DIR:-$HOME/ffmpeg_sources}"
BUILD_DIR="${BUILD_DIR:-$HOME/ffmpeg_build}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

NPROC=$(nproc)

# NVIDIA CUDA setup
if [ "$BUILD_NVIDIA" = "1" ]; then
    # Add CUDA repo if not present
    if ! dpkg -l cuda-keyring 2>/dev/null | grep -q ^ii; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        rm cuda-keyring_1.1-1_all.deb
        sudo apt-get update
    fi

    # Get CUDA version
    if [ -z "${CUDA_VERSION:-}" ]; then
        CUDA_VERSION=$(apt-cache search '^cuda-nvcc-[0-9]' | sed 's/cuda-nvcc-//' | cut -d' ' -f1 | sort -V | tail -1)
        if [ -z "$CUDA_VERSION" ]; then
            echo "Error: No CUDA packages found. Install CUDA repo first or set CUDA_VERSION manually." >&2
            exit 1
        fi
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
fi

# Vulkan SDK for libplacebo (tarball - apt packages deprecated)
if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION:-1.4.328.1}
    VULKAN_SDK_DIR="${SRC_DIR}/vulkan-sdk-${VULKAN_SDK_VERSION}"
fi

APT_PACKAGES=(
    autoconf
    automake
    build-essential
    cmake
    git-core
    meson
    nasm
    ninja-build
    pkg-config
    texinfo
    wget
    yasm
    libass-dev
    libfdk-aac-dev
    libfontconfig1-dev
    libfreetype6-dev
    libsoxr-dev
    libsrt-openssl-dev
    libssl-dev
    libzimg-dev
    liblzma-dev
    liblzo2-dev
    libmp3lame-dev
    libnuma-dev
    libopus-dev
    libsdl2-dev
    libtool
    python3-jinja2
    libunistring-dev
    libva-dev
    libvdpau-dev
    libvorbis-dev
    libvpx-dev
    libx264-dev
    libxcb-shm0-dev
    libxcb-xfixes0-dev
    libxcb1-dev
    zlib1g-dev
)
# Add apt packages for libraries we're not building from source
[ "$BUILD_X265" != "1" ] && APT_PACKAGES+=(libx265-dev)
[ "$BUILD_LIBAOM" != "1" ] && APT_PACKAGES+=(libaom-dev)
[ "$BUILD_LIBWEBP" != "1" ] && APT_PACKAGES+=(libwebp-dev)
[ "$BUILD_LIBVPL" != "1" ] && APT_PACKAGES+=(libvpl-dev)
[ "$BUILD_LIBDAV1D" != "1" ] && APT_PACKAGES+=(libdav1d-dev)
if [ "$BUILD_NVIDIA" = "1" ]; then
    APT_PACKAGES+=(libffmpeg-nvenc-dev cuda-nvcc-$CUDA_VERSION cuda-cudart-dev-$CUDA_VERSION)
fi
sudo apt-get install -y "${APT_PACKAGES[@]}"

mkdir -p "$SRC_DIR"

# libx265 (H.265/HEVC encoder)
if [ "$BUILD_X265" = "1" ]; then
    cd "$SRC_DIR" &&
    git -C x265_git pull 2>/dev/null || (rm -rf x265_git && git clone --depth 1 https://bitbucket.org/multicoreware/x265_git.git) &&
    cd x265_git/build/linux &&
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DENABLE_SHARED=off ../../source &&
    PATH="$BIN_DIR:$PATH" make -j $NPROC &&
    make install
fi

# libaom (AV1 reference codec)
if [ "$BUILD_LIBAOM" = "1" ]; then
    cd "$SRC_DIR" &&
    git -C aom pull 2>/dev/null || (rm -rf aom && git clone --depth 1 https://aomedia.googlesource.com/aom) &&
    mkdir -p aom_build &&
    cd aom_build &&
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DENABLE_TESTS=OFF -DENABLE_NASM=on -DBUILD_SHARED_LIBS=OFF ../aom &&
    PATH="$BIN_DIR:$PATH" make -j $NPROC &&
    make install
fi

# libwebp (WebP image codec)
if [ "$BUILD_LIBWEBP" = "1" ]; then
    cd "$SRC_DIR" &&
    git -C libwebp pull 2>/dev/null || (rm -rf libwebp && git clone --depth 1 https://chromium.googlesource.com/webm/libwebp) &&
    cd libwebp &&
    ./autogen.sh &&
    ./configure --prefix="$BUILD_DIR" --disable-shared --enable-static &&
    make -j $NPROC &&
    make install
fi

# libvpl (Intel Video Processing Library / QuickSync)
if [ "$BUILD_LIBVPL" = "1" ]; then
    cd "$SRC_DIR" &&
    git -C libvpl pull 2>/dev/null || (rm -rf libvpl && git clone --depth 1 https://github.com/intel/libvpl.git) &&
    mkdir -p libvpl/build &&
    cd libvpl/build &&
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DBUILD_SHARED_LIBS=OFF .. &&
    make -j $NPROC &&
    make install
fi

# libdav1d (AV1 decoder)
if [ "$BUILD_LIBDAV1D" = "1" ]; then
    cd "$SRC_DIR" &&
    git -C dav1d pull 2>/dev/null || (rm -rf dav1d && git clone --depth 1 https://code.videolan.org/videolan/dav1d.git) &&
    cd dav1d &&
    if [ -f build/build.ninja ]; then
        meson setup --reconfigure build --buildtype=release --default-library=static --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    else
        meson setup build --buildtype=release --default-library=static --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    fi &&
    ninja -C build &&
    ninja -C build install
fi

# Download Vulkan SDK tarball (apt packages deprecated May 2025)
if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    if [ ! -d "$VULKAN_SDK_DIR" ]; then
        echo "Downloading Vulkan SDK $VULKAN_SDK_VERSION..."
        cd "$SRC_DIR"
        rm -f vulkansdk.tar.xz    # Clean up any partial download
        wget -O vulkansdk.tar.xz "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
        tar xf vulkansdk.tar.xz
        mv "${VULKAN_SDK_VERSION}" "vulkan-sdk-${VULKAN_SDK_VERSION}"
        rm -f vulkansdk.tar.xz
    fi
    export VULKAN_SDK="$VULKAN_SDK_DIR/x86_64"
    export PATH="$VULKAN_SDK/bin:$PATH"
    export PKG_CONFIG_PATH="$VULKAN_SDK/lib/pkgconfig:$PKG_CONFIG_PATH"
    echo "Using Vulkan SDK: $VULKAN_SDK"
fi

# libsvtav1 (AV1 encoder)
cd "$SRC_DIR" && \
git -C SVT-AV1 pull 2>/dev/null || (rm -rf SVT-AV1 && git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git) && \
mkdir -p SVT-AV1/build && \
cd SVT-AV1/build && \
PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && \
PATH="$BIN_DIR:$PATH" make -j $NPROC && \
make install

# libvmaf
cd "$SRC_DIR" &&
git -C vmaf-master pull 2>/dev/null || (rm -rf vmaf-master && git clone --depth 1 https://github.com/Netflix/vmaf vmaf-master) &&
mkdir -p 'vmaf-master/libvmaf/build' &&
cd 'vmaf-master/libvmaf/build' &&
if [ -f build.ninja ]; then
    meson setup --reconfigure -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
else
    meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
fi &&
ninja &&
ninja install


# nv-codec-headers (for NVIDIA hardware encoding)
if [ "$BUILD_NVIDIA" = "1" ]; then
    cd "$SRC_DIR" &&
    git -C nv-codec-headers pull 2>/dev/null || (rm -rf nv-codec-headers && git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git) &&
    cd nv-codec-headers &&
    make &&
    make PREFIX="$BUILD_DIR" install
fi

# libplacebo (for GPU tone mapping)
if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    # Use static shaderc (avoid runtime .so dependency)
    if [ ! -f "$VULKAN_SDK/lib/pkgconfig/shaderc.pc.bak" ]; then
        cp "$VULKAN_SDK/lib/pkgconfig/shaderc.pc" "$VULKAN_SDK/lib/pkgconfig/shaderc.pc.bak"
    fi
    cp "$VULKAN_SDK/lib/pkgconfig/shaderc_combined.pc" "$VULKAN_SDK/lib/pkgconfig/shaderc.pc"

    cd "$SRC_DIR" &&
    git -C libplacebo pull 2>/dev/null || (rm -rf libplacebo && git clone --depth 1 https://code.videolan.org/videolan/libplacebo.git) &&
    cd libplacebo &&
    if [ -f build/build.ninja ]; then
        meson setup --reconfigure build --buildtype=release --default-library=static --prefer-static -Dvulkan=enabled -Dopengl=disabled -Dd3d11=disabled -Ddemos=false --prefix "$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    else
        meson setup build --buildtype=release --default-library=static --prefer-static -Dvulkan=enabled -Dopengl=disabled -Dd3d11=disabled -Ddemos=false --prefix "$BUILD_DIR" --libdir="$BUILD_DIR/lib"
    fi &&
    ninja -C build &&
    ninja -C build install
fi

# NVIDIA CUDA flags
CUDA_FLAGS=()
NVCC_GENCODE=""
if [ "$BUILD_NVIDIA" = "1" ]; then
    CUDA_FLAGS=(--enable-cuda-nvcc --enable-nvenc --enable-cuvid)
    # Detect GPU compute capability for optimized nvcc flags
    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        if [ -n "$COMPUTE_CAP" ]; then
            COMPUTE_CAP_NUM=$(echo $COMPUTE_CAP | tr -d '.')
            NVCC_GENCODE="-gencode arch=compute_${COMPUTE_CAP_NUM},code=sm_${COMPUTE_CAP_NUM}"
            echo "Detected NVIDIA GPU with compute capability ${COMPUTE_CAP} (sm_${COMPUTE_CAP_NUM})"
        fi
    fi
fi

# ffmpeg
FFMPEG_DIR="ffmpeg-${FFMPEG_VERSION}"
cd "$SRC_DIR"
if [ ! -d "$FFMPEG_DIR" ]; then
    if [ "$FFMPEG_VERSION" = "snapshot" ]; then
        rm -f ffmpeg-snapshot.tar.bz2    # Clean up any partial download
        wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
        tar xjf ffmpeg-snapshot.tar.bz2
        mv ffmpeg "$FFMPEG_DIR"
        rm -f ffmpeg-snapshot.tar.bz2
    else
        rm -f "ffmpeg-${FFMPEG_VERSION}.tar.xz"    # Clean up any partial download
        wget -O "ffmpeg-${FFMPEG_VERSION}.tar.xz" "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
        tar xJf "ffmpeg-${FFMPEG_VERSION}.tar.xz"
        rm -f "ffmpeg-${FFMPEG_VERSION}.tar.xz"
    fi
fi
cd "$FFMPEG_DIR" && \
# Build configure flags
# MARCH=native for CPU-specific optimizations (opt-in, not portable)
EXTRA_CFLAGS="-I$BUILD_DIR/include -O3${MARCH:+ -march=$MARCH -mtune=$MARCH}"
EXTRA_LDFLAGS="-L$BUILD_DIR/lib -s"
if [ "$BUILD_NVIDIA" = "1" ]; then
    EXTRA_CFLAGS="$EXTRA_CFLAGS -I$CUDA_PATH/include"
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$CUDA_PATH/lib64"
fi
if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    EXTRA_CFLAGS="$EXTRA_CFLAGS -I$VULKAN_SDK/include"
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$VULKAN_SDK/lib"
fi

CONFIGURE_CMD=(
    ./configure
    --prefix="$BUILD_DIR"
    --pkg-config-flags="--static"
    --extra-cflags="$EXTRA_CFLAGS"
    --extra-ldflags="$EXTRA_LDFLAGS"
    --extra-libs="-lpthread -lm"
    --ld="g++"
    --bindir="$BIN_DIR"
    --enable-gpl
    --enable-version3
    --enable-openssl
    --enable-libaom
    --enable-libass
    --enable-libfdk-aac
    --enable-libfontconfig
    --enable-libfreetype
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
    --enable-libzimg
    --enable-libsoxr
    --enable-libsrt
    --enable-vaapi
    --enable-libvpl
    --enable-nonfree
    "${CUDA_FLAGS[@]}"
)

if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    CONFIGURE_CMD+=(--enable-vulkan --enable-libplacebo)
fi

if [ -n "$NVCC_GENCODE" ]; then
    CONFIGURE_CMD+=(--nvccflags="$NVCC_GENCODE")
fi

PATH="$BIN_DIR:$PATH" PKG_CONFIG_PATH="$BUILD_DIR/lib/pkgconfig:$PKG_CONFIG_PATH" "${CONFIGURE_CMD[@]}" && \
PATH="$BIN_DIR:$PATH" make -j $NPROC && \
make install && \
hash -r

grep -q "$BUILD_DIR/share/man" "$HOME/.manpath" 2>/dev/null || echo "MANPATH_MAP $BIN_DIR $BUILD_DIR/share/man" >> "$HOME/.manpath"

# rm -rf ~/ffmpeg_build ~/.local/bin/{ffmpeg,ffprobe,ffplay,x264,x265}
# sed -i '/ffmpeg_build/d' ~/.manpath
# hash -r
# --extra-cflags="-D_GNU_SOURCE"
# cat ~/ffmpeg_sources/ffmpeg/ffbuild/config.log

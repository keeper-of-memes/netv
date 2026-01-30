# netv application image
#
# Default build uses pre-built FFmpeg with full hardware support:
#   docker compose build
#
# Alternative: use apt FFmpeg (fewer codecs, no NVENC/QSV):
#   FFMPEG_IMAGE=ubuntu:24.04 docker compose build
#
# The optimized FFmpeg base image includes:
# - NVENC (NVIDIA hardware encoding)
# - VAAPI (Intel/AMD hardware encoding)
# - QSV/VPL (Intel QuickSync)
# - All major codecs (x264, x265, VP9, AV1, etc.)

ARG FFMPEG_IMAGE=ghcr.io/jvdillon/netv-ffmpeg:cuda12.4

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
# - If using apt ffmpeg (ubuntu base): install ffmpeg + python
# - If using compiled ffmpeg (netv-ffmpeg base): ffmpeg already present, just install python
# Note: The conditional must be evaluated in shell, not in Dockerfile syntax
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gosu \
        python3.11 \
        python3.11-venv \
        python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    # Conditionally install ffmpeg if not present from base image
    if [ ! -x /usr/local/bin/ffmpeg ] && [ ! -x /usr/bin/ffmpeg ]; then \
        apt-get install -y --no-install-recommends ffmpeg; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# App setup
WORKDIR /app

# Copy application files with verification
COPY pyproject.toml README.md ./
COPY *.py ./
COPY templates/ templates/
COPY static/ static/

# Verify critical files exist
RUN test -f pyproject.toml || { echo "ERROR: pyproject.toml not found"; exit 1; }

# Install Python dependencies
# --ignore-installed: avoids "Cannot uninstall X, RECORD file not found" for apt packages
# --break-system-packages: required for PEP 668 (Ubuntu 24.04+), doesn't exist in pip 22.0 (Ubuntu 22.04)
# Using try-fallback approach for maximum compatibility
RUN if python3 -m pip install --help 2>&1 | grep -q -- '--break-system-packages'; then \
        python3 -m pip install --no-cache-dir --ignore-installed --break-system-packages .; \
    else \
        python3 -m pip install --no-cache-dir --ignore-installed .; \
    fi

# Runtime config
EXPOSE 8000

# Environment variables (see README for details)
ENV NETV_PORT=8080
ENV NETV_HTTPS=""
ENV LOG_LEVEL=INFO

# Create non-root user (entrypoint handles permissions and group membership)
RUN useradd -m netv

# Copy entrypoint and set permissions with validation
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh && \
    test -x /app/entrypoint.sh || { echo "ERROR: entrypoint.sh not executable"; exit 1; }

# Healthcheck with improved error handling
# Note: start-period allows time for application startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8080/', timeout=5); exit(0 if r.status==200 else 1)" 2>/dev/null || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

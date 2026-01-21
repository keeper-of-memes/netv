#!/bin/sh
# Entrypoint: fix permissions and drop to netv user
#
# Handles two common Docker issues:
# 1. Bind-mounted ./cache owned by host user (permission denied)
# 2. /dev/dri/renderD128 GID mismatch (VAAPI unavailable)

# Fix cache directory ownership (skip if already correct to avoid slow recursive chown)
# Build/runtime note: this only applies to bind-mounted cache (e.g., NAS),
# not to image layers, so it does not affect build reproducibility.
mkdir -p /app/cache
if [ "$(stat -c '%U' /app/cache)" != "netv" ]; then
    chown -R netv:netv /app/cache 2>/dev/null || true
fi
# Ensure writable even on filesystems that ignore chown (e.g., some NAS mounts)
if ! gosu netv sh -c "touch /app/cache/.perm_test && rm /app/cache/.perm_test" 2>/dev/null; then
    chmod -R u+rwX,g+rwX /app/cache 2>/dev/null || true
    chmod g+s /app/cache 2>/dev/null || true
fi
mkdir -p /app/cache/users

# Add netv user to render device group (for VAAPI hardware encoding)
if [ -e /dev/dri/renderD128 ]; then
    RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
    groupadd --gid "$RENDER_GID" hostrender 2>/dev/null || true
    usermod -aG hostrender netv 2>/dev/null || true
fi

# Drop to netv user and run the app
exec gosu netv python3 main.py --port "${NETV_PORT:-8000}" ${NETV_HTTPS:+--https}

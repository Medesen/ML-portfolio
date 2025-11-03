#!/usr/bin/env bash
set -euo pipefail

# Who we're going to drop to
APP_USER="${NB_USER:-appuser}"
APP_UID="${NB_UID:-1000}"
APP_GID="${NB_GID:-1000}"

# Paths we care about (only writable directories, not the read-only /app/data)
DIRS=(
  /app/models
  /app/diagnostics
  /app/logs
  /app/mlruns
  /app/mlartifacts
)

# Create + fix ownership/permissions so the app user can write
for d in "${DIRS[@]}"; do
  mkdir -p "$d" 2>/dev/null || true
  chown -R "$APP_UID:$APP_GID" "$d" 2>/dev/null || true
  chmod -R ug+rwX "$d" 2>/dev/null || true
done

# Drop privileges and run the command (pass through all args)
exec gosu "$APP_UID:$APP_GID" "$@"


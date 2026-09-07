#!/usr/bin/env bash
  set -euo pipefail

  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  APP_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

  cd "$APP_DIR"

  export PYTHONPATH="$APP_DIR/src:$APP_DIR/services"

  exec python -m uvicorn processing_service.app:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8081}"
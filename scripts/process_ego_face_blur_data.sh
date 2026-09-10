#!/usr/bin/env bash
set -euo pipefail
# Resolve paths relative to this script, so activation and current directory do not matter.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PRIVACY_BLUR_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m privacy_blur.batch \
  --output-dir "$PROJECT_ROOT/outputs" \
  --face-weights "$PROJECT_ROOT/models/yolov8n-face.pt" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher for Memory/EvaluateCL.py. It discovers the task and
# checkpoint range automatically, so it also works with a partial run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CL_DATA_ROOT="${CL_DATA_ROOT:-$SCRIPT_DIR/Data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/Data/CL/runs/cl_dws_aligned/memory_checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/Memory/Eval/CL}"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

if [[ -z "$PYTHON_BIN" ]]; then
  echo "[Error] Python executable was not found. Set PYTHON=/path/to/python." >&2
  exit 1
fi
if [[ "$PYTHON_BIN" == */* && ! -x "$PYTHON_BIN" ]]; then
  echo "[Error] Python executable is not executable: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Memory${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -u -m EvaluateCL \
  --data-root "$CL_DATA_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

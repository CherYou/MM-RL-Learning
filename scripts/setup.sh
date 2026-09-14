#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=""
export UV_LINK_MODE=copy
if [[ ! -x .venv/bin/python ]]; then
    python3.12 -m venv .venv
fi
if [[ ! -x .venv/bin/uv ]]; then
    .venv/bin/python -m ensurepip
    .venv/bin/python -m pip install 'uv==0.12.11'
fi
.venv/bin/uv sync --frozen --extra dev --extra cpu
.venv/bin/arl doctor

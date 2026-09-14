#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PROFILE="${1:---cpu}"
case "$PROFILE" in
  --cpu) ENV_NAME=.venv-verl; ENV_PROJECT=environments/verl-cpu ;;
  --gpu) ENV_NAME=.venv-verl-gpu; ENV_PROJECT=environments/verl-gpu ;;
  *) echo 'Usage: bash scripts/setup_verl.sh [--cpu|--gpu]'; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=""
if [[ ! -x .venv/bin/uv ]]; then
  bash scripts/setup.sh
fi
if [[ ! -d references/verl/.git ]]; then
  git clone --depth 1 --branch v0.7.1 https://github.com/verl-project/verl.git references/verl
fi
EXPECTED_REVISION=bec9ef74768dd201881cd4e54cd0385e87caae27
ACTUAL_REVISION="$(git -C references/verl rev-parse HEAD)"
if [[ "$ACTUAL_REVISION" != "$EXPECTED_REVISION" ]]; then
  echo "verl source revision differs: $ACTUAL_REVISION; expected $EXPECTED_REVISION"
  exit 1
fi
UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/$ENV_NAME" .venv/bin/uv sync \
  --frozen --project "$ENV_PROJECT" --python "$PROJECT_DIR/.venv/bin/python"
.venv/bin/uv pip check --python "$PROJECT_DIR/$ENV_NAME/bin/python"
echo "Installed $ENV_NAME from $ENV_PROJECT/uv.lock. No training or GPU validation was run."

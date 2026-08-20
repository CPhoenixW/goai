#!/usr/bin/env bash
set -Eeuo pipefail

# One-command bootstrap and launcher for the GOAI residual policy server.
# Large base-model artifacts are intentionally not downloaded by this script.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_REPO="CPhoenixW/goai"
DEFAULT_REF="main"
DEFAULT_BANK="goai-12task-isolated-residual-bank-v1-taskmatch-v2.pt"
DEFAULT_BANK_SHA="7daf3f9f9f0542b55caaa255b3c83fc52433b0cf2c22f96a901c5d771e10dca2"
DEFAULT_COMPOSITE="goai-composite-residual-eval-v2.pt"
DEFAULT_COMPOSITE_SHA="f2b94cf1872885bcc3fb9d501b1949e03bbdfbb7c1e7feb047d99e1c4d51cfb3"
DEFAULT_PYTHON="/root/autodl-tmp/xiaomi-mibot/bin/python"
DEFAULT_BASE_MODEL="/root/autodl-tmp/goai-residual-migration-20260818/exact_base/RoboDojo-sim-arx_x5-ee-0"
DEFAULT_PROCESSOR="/root/autodl-tmp/qwen3-vl-4b"

usage() {
  cat <<'EOF'
Usage:
  bash tools/start_goai.sh [TASK]

Environment:
  GOAI_TASK                    Task name; defaults to the first argument or stack_bowls_random
  GOAI_CHECKPOINT_KIND         bank (default) or composite
  GOAI_CHECKPOINT_PATH         Override the checkpoint file path
  GOAI_PORT                    Local WebSocket port; defaults to 6000
  GOAI_GPU_ID                  CUDA device; defaults to 0
  GOAI_PYTHON                  Xiaomi inference Python executable
  GOAI_BASE_MODEL              Local Xiaomi base model directory
  GOAI_PROCESSOR               Local Qwen3-VL processor directory
  GOAI_GITHUB_REPO             Private GitHub repo; defaults to CPhoenixW/goai
  GOAI_GITHUB_REF              Download ref; defaults to main
  GOAI_REDOWNLOAD_CHECKPOINT   Set to 1 to replace a mismatched existing checkpoint

If the selected checkpoint is absent, the script downloads it from GitHub using
GH_TOKEN, GITHUB_TOKEN, or the current `gh auth token`. Existing files are always
verified before the policy server starts.
EOF
}

die() {
  echo "[goai] ERROR: $*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK="${GOAI_TASK:-${1:-stack_bowls_random}}"
KIND="${GOAI_CHECKPOINT_KIND:-bank}"
PORT="${GOAI_PORT:-6000}"
GPU="${GOAI_GPU_ID:-0}"
GITHUB_REPO="${GOAI_GITHUB_REPO:-$DEFAULT_REPO}"
GITHUB_REF="${GOAI_GITHUB_REF:-$DEFAULT_REF}"

case "${KIND}" in
  bank)
    CHECKPOINT_NAME="$DEFAULT_BANK"
    CHECKPOINT_SHA="$DEFAULT_BANK_SHA"
    ;;
  composite)
    CHECKPOINT_NAME="$DEFAULT_COMPOSITE"
    CHECKPOINT_SHA="$DEFAULT_COMPOSITE_SHA"
    ;;
  *)
    die "GOAI_CHECKPOINT_KIND must be bank or composite"
    ;;
esac

CHECKPOINT_PATH="${GOAI_CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/$CHECKPOINT_NAME}"
PYTHON="${GOAI_PYTHON:-$DEFAULT_PYTHON}"
BASE_MODEL="${GOAI_BASE_MODEL:-$DEFAULT_BASE_MODEL}"
PROCESSOR="${GOAI_PROCESSOR:-$DEFAULT_PROCESSOR}"

[[ -x "$PYTHON" ]] || die "Python executable not found or not executable: $PYTHON (set GOAI_PYTHON)"
[[ -d "$BASE_MODEL" ]] || die "Xiaomi base model directory not found: $BASE_MODEL (set GOAI_BASE_MODEL)"
[[ -d "$PROCESSOR" ]] || die "Qwen3-VL processor directory not found: $PROCESSOR (set GOAI_PROCESSOR)"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_checkpoint() {
  local path="$1"
  local actual
  [[ -f "$path" ]] || return 1
  actual="$(sha256_of "$path")"
  if [[ "$actual" != "$CHECKPOINT_SHA" ]]; then
    echo "[goai] checkpoint SHA256 mismatch" >&2
    echo "[goai] expected: $CHECKPOINT_SHA" >&2
    echo "[goai] actual:   $actual" >&2
    return 1
  fi
  echo "[goai] checkpoint verified: $path"
}

download_checkpoint() {
  local target="$1"
  local target_dir tmp token url
  command -v curl >/dev/null 2>&1 || die "curl is required to download a missing checkpoint"
  target_dir=$(dirname "$target")
  mkdir -p "$target_dir"
  tmp="$(mktemp "${TMPDIR:-/tmp}/goai-checkpoint.XXXXXX")"
  trap 'rm -f "${tmp:-}"' EXIT
  url="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_REF}/checkpoints/${CHECKPOINT_NAME}"

  token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  if [[ -z "$token" ]] && command -v gh >/dev/null 2>&1; then
    token="$(gh auth token 2>/dev/null || true)"
  fi

  echo "[goai] checkpoint missing; downloading from $GITHUB_REPO@$GITHUB_REF"
  if [[ -n "$token" ]]; then
    curl --fail --silent --show-error --location --retry 3 \
      --header "Authorization: Bearer $token" \
      --output "$tmp" "$url"
  else
    curl --fail --silent --show-error --location --retry 3 \
      --output "$tmp" "$url"
  fi
  verify_checkpoint "$tmp" || die "downloaded checkpoint failed SHA256 verification"
  mv "$tmp" "$target"
  trap - EXIT
  echo "[goai] checkpoint downloaded: $target"
}

if [[ -f "$CHECKPOINT_PATH" ]]; then
  if ! verify_checkpoint "$CHECKPOINT_PATH"; then
    [[ "${GOAI_REDOWNLOAD_CHECKPOINT:-0}" == "1" ]] || die "set GOAI_REDOWNLOAD_CHECKPOINT=1 to replace this file"
    download_checkpoint "$CHECKPOINT_PATH"
  fi
else
  download_checkpoint "$CHECKPOINT_PATH"
fi

echo "[goai] checking Python inference dependencies"
"$PYTHON" - <<'PY'
import cv2
import scipy
import torch
import transformers
import websockets
import yaml
from PIL import Image
import liger_kernel

print("[goai] dependencies OK; torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

export GOAI_PYTHON="$PYTHON"
export GOAI_BASE_MODEL="$BASE_MODEL"
export GOAI_PROCESSOR="$PROCESSOR"

echo "[goai] starting task=$TASK checkpoint=$CHECKPOINT_NAME port=$PORT gpu=$GPU"
exec bash "$REPO_ROOT/tools/start_policy_task.sh" \
  "$TASK" "$CHECKPOINT_PATH" "$CHECKPOINT_SHA" "$PORT" "$GPU"

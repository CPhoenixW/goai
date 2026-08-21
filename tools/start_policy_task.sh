#!/usr/bin/env bash
set -euo pipefail

TASK=${1:?task name required}
BANK=${2:?checkpoint path required}
BANK_SHA=${3:?checkpoint SHA256 required}
PORT=${4:-6000}
GPU=${5:-0}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${GOAI_PYTHON:-python3}
DEFAULT_BASE_MODEL="checkpoints/Xiaomi_Robotics_1/ckpt/RoboDojo/Xiaomi_Robotics_1/RoboDojo-sim-arx_x5-ee-0"
DEFAULT_PROCESSOR="models/qwen3-vl-4b"

resolve_repo_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$value"
  fi
}

BASE_MODEL="$(resolve_repo_path "${GOAI_BASE_MODEL:-$DEFAULT_BASE_MODEL}")"
PROCESSOR="$(resolve_repo_path "${GOAI_PROCESSOR:-$DEFAULT_PROCESSOR}")"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export XR1_RESIDUAL_TASK_BANK_PATH="${BANK}"
export XR1_RESIDUAL_TASK_BANK_SHA256="${BANK_SHA}"
export XR1_RESIDUAL_TAIL_EVALUATION_ONLY=${XR1_RESIDUAL_TAIL_EVALUATION_ONLY:-1}
unset XR1_ADAPTER_PATH XR1_RESIDUAL_TAIL_PATH

exec "${PYTHON}" "${REPO_ROOT}/XPolicyLab/setup_policy_server.py" \
  --config_path "${REPO_ROOT}/XPolicyLab/policy/Xiaomi_Robotics_1/deploy.yml" \
  --overrides \
    port="${PORT}" host=127.0.0.1 bench_name=RoboDojo \
    task_name="${TASK}" ckpt_name=Xiaomi_Robotics_1 \
    env_cfg_type=arx_x5 seed=0 policy_name=Xiaomi_Robotics_1 \
    action_type=ee model_dir="${BASE_MODEL}" \
    vlm_processor_path="${PROCESSOR}" eval_env=sim

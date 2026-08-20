#!/usr/bin/env bash
set -euo pipefail

TASK=${1:?task name required}
BANK=${2:?checkpoint path required}
BANK_SHA=${3:?checkpoint SHA256 required}
PORT=${4:-6000}
GPU=${5:-0}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${GOAI_PYTHON:-python}
BASE_MODEL=${GOAI_BASE_MODEL:?set GOAI_BASE_MODEL to the Xiaomi base model directory}
PROCESSOR=${GOAI_PROCESSOR:?set GOAI_PROCESSOR to the Qwen3-VL processor directory}

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

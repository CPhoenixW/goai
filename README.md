# GOAI residual inference

Inference bundle for the Xiaomi Robotics 1 RoboDojo policy with isolated residual task heads.

The two small residual checkpoints used by this runtime are included under `checkpoints/`. The large Xiaomi base model, Qwen3-VL processor, datasets, videos, logs, and credentials are intentionally excluded. The runtime validates the residual checkpoint SHA256 before loading it.

## Checkpoints

The two supported checkpoint types use the same inference implementation:

| Type | Purpose | SHA256 |
| --- | --- | --- |
| `checkpoints/isolated.pt` | Full 12-task residual bank with `*_random` task aliases and task routing metadata | `7daf3f9f9f0542b55caaa255b3c83fc52433b0cf2c22f96a901c5d771e10dca2` |
| `checkpoints/composite.pt` | Evaluation-only composite route; residuals are enabled only for the selected evidence-backed tasks and other tasks fall back to the Xiaomi base policy | `f2b94cf1872885bcc3fb9d501b1949e03bbdfbb7c1e7feb047d99e1c4d51cfb3` |

The composite checkpoint is not a separate neural inference branch. Its `task_routes` and `composite_policy` metadata are consumed by the same task-bank loader.

## Runtime flow

```text
tools/start_policy_task.sh
  -> XPolicyLab/setup_policy_server.py
  -> XPolicyLab/policy/Xiaomi_Robotics_1/model.py
  -> Xiaomi base VLA: model.generate(...)
  -> residual_tail_runtime.ResidualTailRuntime
  -> residual_tail.task_bank_runtime.load_task_model(...)
  -> task-specific residual correction
  -> RoboDojo action contract
```

The base VLA produces a 10-step action chunk. The residual runtime keeps a short state history, predicts a bounded 14-dimensional correction, applies its learned gate, composes the correction with the base end-effector action, and returns the validated action format.

## Source layout

```text
XPolicyLab/setup_policy_server.py                         WebSocket policy-server entry point
XPolicyLab/policy/Xiaomi_Robotics_1/model.py              Xiaomi adapter and inference path
XPolicyLab/policy/Xiaomi_Robotics_1/residual_tail_runtime.py  checkpoint routing and correction runtime
XPolicyLab/policy/Xiaomi_Robotics_1/action_contract.py    10x60 action contract
XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/src/ Xiaomi VLA model implementation
residual_tail/model.py                                    residual network architecture
residual_tail/task_bank_runtime.py                        isolated task-head loader
tools/start_policy_task.sh                                portable server launcher
tools/start_goai.sh                                       one-command verify/download and launcher
```

## External prerequisites

Provide the following large external artifacts locally; do not commit them:

- Xiaomi base model directory, supplied through `GOAI_BASE_MODEL`.
- Qwen3-VL processor directory, supplied through `GOAI_PROCESSOR`.
- The Xiaomi inference environment, including PyTorch/CUDA, Transformers, Pillow, SciPy, OpenCV, PyYAML, WebSockets, and the Xiaomi model dependencies such as `liger-kernel`.

The residual checkpoint files are included in `checkpoints/`; verify their exact SHA256 before deployment.

The source was extracted from the running environment under `/root/autodl-tmp/xiaomi-mibot` and the live XPolicyLab checkout. The repository does not attempt to download the large model artifacts or silently substitute a different checkpoint.

## Reproduce from a clone

### 1. Clone the source and prepare the runtime

```bash
git clone https://github.com/CPhoenixW/goai.git
cd goai
```

Use the same Xiaomi inference environment that contains the base-model dependencies. A known-good server layout used by this project was:

```text
GOAI_PYTHON=/root/autodl-tmp/xiaomi-mibot/bin/python
GOAI_BASE_MODEL=/root/autodl-tmp/goai-residual-migration-20260818/exact_base/RoboDojo-sim-arx_x5-ee-0
GOAI_PROCESSOR=/root/autodl-tmp/qwen3-vl-4b
```

The repository does not recreate that environment or download the large Xiaomi/Qwen artifacts. Before starting, verify the selected Python can import the inference dependencies:

```bash
"${GOAI_PYTHON}" - <<'PY'
import cv2
import scipy
import torch
import transformers
import websockets
import yaml
from PIL import Image
import liger_kernel

print("inference dependencies: OK")
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY
```

### 2. Verify the included residual checkpoint

The checkpoint files are already present after cloning. Verify the exact file before starting the server:

```bash
sha256sum checkpoints/isolated.pt
```

The output must match the SHA256 listed in the [Checkpoint](#checkpoints) table. A mismatch causes startup to fail closed.

### 3. One-command start

`tools/start_goai.sh` checks the local base model, processor, Python dependencies, and residual checkpoint before starting the WebSocket policy server. If the selected residual checkpoint is missing, it downloads it from this private repository using `GH_TOKEN`, `GITHUB_TOKEN`, or the current `gh auth token`, then verifies the SHA256. Existing mismatched files are not overwritten unless `GOAI_REDOWNLOAD_CHECKPOINT=1` is set.

```bash
export GOAI_BASE_MODEL=/path/to/RoboDojo-sim-arx_x5-ee-0
export GOAI_PROCESSOR=/path/to/qwen3-vl-4b
export GOAI_PYTHON=/path/to/xiaomi-mibot/bin/python

bash tools/start_goai.sh stack_bowls_random
```

Useful overrides:

```bash
GOAI_CHECKPOINT_KIND=composite GOAI_PORT=6001 GOAI_GPU_ID=1 \
  bash tools/start_goai.sh stack_bowls_random
```

The script does not download the large Xiaomi base model or Qwen3-VL processor; those must already exist locally. Run `bash tools/start_goai.sh --help` to see all options.

### 4. Start one task-specific policy server manually

```bash
export GOAI_BASE_MODEL=/path/to/RoboDojo-sim-arx_x5-ee-0
export GOAI_PROCESSOR=/path/to/qwen3-vl-4b
export GOAI_PYTHON=/path/to/xiaomi-mibot/bin/python

bash tools/start_policy_task.sh \
  stack_bowls_random \
  checkpoints/isolated.pt \
  7daf3f9f9f0542b55caaa255b3c83fc52433b0cf2c22f96a901c5d771e10dca2 \
  6000 0
```

The arguments after the script are:

```text
TASK CHECKPOINT CHECKPOINT_SHA256 PORT GPU_ID
```

The server binds to `127.0.0.1` by default, which is appropriate when RoboDojo reaches it through a local tunnel or the evaluation host. For a long-running process, use the host's process supervisor or a terminal manager such as `screen`:

```bash
screen -dmS goai-stack-bowls bash -lc '
  export GOAI_BASE_MODEL=/path/to/RoboDojo-sim-arx_x5-ee-0
  export GOAI_PROCESSOR=/path/to/qwen3-vl-4b
  export GOAI_PYTHON=/path/to/xiaomi-mibot/bin/python
  bash tools/start_policy_task.sh \
    stack_bowls_random \
    checkpoints/isolated.pt \
    7daf3f9f9f0542b55caaa255b3c83fc52433b0cf2c22f96a901c5d771e10dca2 \
    6000 0
'
```

For the composite evaluation checkpoint, replace the checkpoint path and SHA256 with:

```text
checkpoints/composite.pt
f2b94cf1872885bcc3fb9d501b1949e03bbdfbb7c1e7feb047d99e1c4d51cfb3
```

For a second task, start another server with a different port and `TASK` value. The same checkpoint file can be shared by multiple processes, but each process selects one task route at startup.

### 5. Connect the RoboDojo evaluator

This repository provides the policy server and adapter. The RoboDojo evaluator is a separate checkout. Point its Policy Server client at the running WebSocket endpoint and keep `action_type=ee`, the task name, and the checkpoint route consistent with the local server. For the official smoke test, run the evaluator's `scripts/robodojo.sh smoke` command with `--dimension generalization`; the evaluator sends observations to this server and consumes the returned action chunks.

A server process reaching `Model loaded` and printing a task-bank route is the expected readiness signal. The startup log should identify the raw and canonical task names and whether the residual route is enabled. If the route is disabled, the server intentionally returns the exact Xiaomi base action.

Use the official RoboDojo task name. Random layouts are normalized by the runtime:

```text
push_T_random      -> push_T
stack_bowls_random -> stack_bowls
stack_blocks_random -> stack_blocks
```

Unknown tasks fail closed. A disabled route returns the exact Xiaomi base action rather than guessing another task head.

## Important limitations

- `composite-eval-v2` is marked evaluation-only and is not deployment authorization by itself.
- `fold_clothes` requires the exact legacy base checkpoint SHA256 `983c5dee21cb3500ff21c093b58d83b5543d9888f81cb4dfd8dc3ff6ac13670b` and semantic adapter SHA256 `cfd7c95a6278000a3978004c85673617b6f6aa6c256aacccf99b0d72f7ff6731`.
- Restart the policy server after changing code or checkpoint files; the process does not hot-reload either.
- Keep the checkpoint path and SHA256 together in deployment configuration so a replaced file cannot be loaded accidentally.

## License and provenance

The Xiaomi adapter source retains the upstream Apache 2.0 notices present in the live files. Review the upstream Xiaomi/XPolicyLab terms before redistributing a public copy.

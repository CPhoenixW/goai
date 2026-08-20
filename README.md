# GOAI residual inference

Source-only inference bundle for the Xiaomi Robotics 1 RoboDojo policy with isolated residual task heads.

This repository intentionally contains no model weights, base checkpoints, processor files, datasets, videos, logs, or credentials. The runtime requires those artifacts to be supplied locally and validates the residual checkpoint SHA256 before loading it.

## Checkpoints

The two supported checkpoint types use the same inference implementation:

| Type | Purpose | SHA256 |
| --- | --- | --- |
| `goai-12task-isolated-residual-bank-v1-taskmatch-v2.pt` | Full 12-task residual bank with `*_random` task aliases and task routing metadata | `7daf3f9f9f0542b55caaa255b3c83fc52433b0cf2c22f96a901c5d771e10dca2` |
| `goai-composite-residual-eval-v2.pt` | Evaluation-only composite route; residuals are enabled only for the selected evidence-backed tasks and other tasks fall back to the Xiaomi base policy | `f2b94cf1872885bcc3fb9d501b1949e03bbdfbb7c1e7feb047d99e1c4d51cfb3` |

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
```

## External prerequisites

Provide the following locally; do not commit them:

- Xiaomi base model directory, supplied through `GOAI_BASE_MODEL`.
- Qwen3-VL processor directory, supplied through `GOAI_PROCESSOR`.
- One of the two residual checkpoint files and its exact SHA256.
- The Xiaomi inference environment, including PyTorch/CUDA, Transformers, Pillow, SciPy, OpenCV, PyYAML, WebSockets, and the Xiaomi model dependencies such as `liger-kernel`.

The source was extracted from the running environment under `/root/autodl-tmp/xiaomi-mibot` and the live XPolicyLab checkout. The repository does not attempt to download model artifacts or silently substitute a different checkpoint.

## Start a policy server

```bash
export GOAI_BASE_MODEL=/path/to/RoboDojo-sim-arx_x5-ee-0
export GOAI_PROCESSOR=/path/to/qwen3-vl-4b
export GOAI_PYTHON=/path/to/xiaomi-mibot/bin/python

bash tools/start_policy_task.sh \
  stack_bowls_random \
  /path/to/goai-12task-isolated-residual-bank-v1-taskmatch-v2.pt \
  7daf3f9f9f0542b55caaa255b3c83fc52433b0cf2c22f96a901c5d771e10dca2 \
  6000 0
```

For the composite evaluation checkpoint, replace the checkpoint path and SHA256 with:

```text
/path/to/goai-composite-residual-eval-v2.pt
f2b94cf1872885bcc3fb9d501b1949e03bbdfbb7c1e7feb047d99e1c4d51cfb3
```

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

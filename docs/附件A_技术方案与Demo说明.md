# 附件 A：技术方案与 Demo 说明

项目：GOAI Residual Policy for RoboDojo

模型：Xiaomi Robotics 1 VLA + task-specific residual tail

评测重点：RoboDojo Generalization、`action_type=ee`

## 1. 任务定义

面向 RoboDojo 机器人操作任务，输入多视角图像、机器人状态和任务指令，输出连续末端执行器动作。方案重点处理标准配置与 `*_random` 配置之间的分布差异。

应用价值：在不重训完整 VLA 的情况下，通过小型任务隔离残差网络补偿特定任务的动作误差、时序误差和场景差异。

## 2. 系统架构

```text
RoboDojo observation
  -> WebSocket Policy Server
  -> Xiaomi Robotics 1 VLA
  -> 10-step base action chunk
  -> canonical task / task-bank routing
  -> task-specific residual head
  -> bounded correction + learned gate + safety contract
  -> RoboDojo ee action
```

- 未知任务采用 fail-closed 策略。
- 未启用残差路由时返回精确的 Xiaomi base action。
- `fold_clothes_random` 等随机任务通过 `_random` 别名映射到 canonical task。
- 推理代码和一键启动入口位于仓库的 `XPolicyLab/`、`residual_tail/` 和 `tools/`。

## 3. 关键算法 / 模型

- 基础策略：Xiaomi Robotics 1 VLA。
- 残差策略：任务隔离的 residual head，维护短时状态历史，预测有界的 14 维动作修正。
- 动作组合：基础动作 chunk 经 residual correction、gate 和安全约束后输出。
- 权重：
  - `checkpoints/isolated.pt`：完整 12-task residual bank。
  - `checkpoints/composite.pt`：evaluation-only composite route。
- `isolated.pt` 中包含 `fold_clothes` residual head；该路由元数据标记为 `evaluation_only=true`、`deployment_enabled=false`。

## 4. 部署与复现

准备 Xiaomi 推理环境、基础模型目录和 Qwen3-VL processor：

```bash
export GOAI_PYTHON=/path/to/xiaomi-mibot/bin/python
export GOAI_BASE_MODEL=/path/to/RoboDojo-sim-arx_x5-ee-0
export GOAI_PROCESSOR=/path/to/qwen3-vl-4b
```

一键启动并校验 `isolated.pt`：

```bash
bash tools/start_goai.sh fold_clothes_random
```

脚本会依次检查：

1. Python、PyTorch/CUDA、Transformers、WebSockets 等推理依赖；
2. 基础模型和 processor 路径；
3. 本地 checkpoint 是否存在及 SHA256 是否匹配；
4. 权重缺失时从 GitHub 下载并再次校验；
5. Policy Server 是否正式启动。

基础大模型、processor、数据集和凭据不随仓库发布。

## 5. 评测与验证

叠衣服历史 A/B 结果如下。这里的残差实验使用打包前的 `fold_semantic_step800_residual_gate` source checkpoint；它随后作为 fold head 嵌入 `isolated.pt`。

| 配置 | ideal（2 runs） | random（2 runs） | 总体（4 runs） |
| --- | ---: | ---: | ---: |
| Xiaomi base | 50% 成功，平均分 60 | 0% 成功，平均分 0 | 25% 成功，平均分 30 |
| `fold_semantic_step800_residual_gate` | 50% 成功，平均分 60 | 50% 成功，平均分 50 | 50% 成功，平均分 55 |

观测到的提升：总体成功率 **+25 个百分点**，总体平均分 **+25 分**；random 成功率 **+50 个百分点**。ideal 场景没有提升。

证据视频：

- ideal / seed 0：三个相机视角全部成功，见 [`evidence/fold_clothes/ideal/seed_0/`](../evidence/fold_clothes/ideal/seed_0/)。
- random / seed 0：三个相机视角全部成功，见 [`evidence/fold_clothes/random/seed_0/`](../evidence/fold_clothes/random/seed_0/)。

## 6. 验收标准

- 启动日志出现 `Model loaded`，并显示 canonical task 与 route enabled 状态。
- checkpoint SHA256 与仓库 README 记录一致。
- RoboDojo evaluator 能连接 WebSocket Policy Server。
- 输出动作形状为 `10×60`，协议保持 `action_type=ee`。
- 正式提交前，用打包后的 `isolated.pt` 对同一批任务做 fresh A/B 复测；当前叠衣服结果属于嵌入 residual head 的证据，不应直接外推到全部 12 个任务。

## 7. 开源与复现入口

代码仓库：<https://github.com/CPhoenixW/goai>

仓库提供推理代码、一键启动脚本、checkpoint SHA256 校验、两个残差权重和成功视频证据；不包含基础大模型、数据集和凭据。

# 附件 B：项目一页纸

## 项目名称

GOAI Residual Policy for RoboDojo

## 团队名称与成员信息

团队名称：ICECREAM

团队成员：王奕涵、黄迦南、葛馨婷。

## 赛题方向与目标任务

方向一：通用双臂协作操作能力测试。

本赛题基于 X-Eval 仿真与真实一体化评测平台，面向复杂双臂机器人操作场景，要求设计并优化 VLA/WAM 等前端到端具身操作模型。目标覆盖 12 个仿真操作任务与 6 个真实机器人操作任务，并在任务成功率、泛化能力、执行效率及鲁棒性等维度进行综合优化。

本项目当前实现聚焦 RoboDojo 仿真侧 Generalization 维度，保持 `action_type=ee` 协议一致，并通过任务隔离残差网络提升标准配置与 `*_random` 配置的动作适应能力。真实机器人部署属于后续工作，当前材料中的视频和指标均为仿真证据。

## 方案概述

以 Xiaomi Robotics 1 VLA 作为基础策略，先生成 10-step base action chunk，再根据 canonical task 匹配 isolated residual head。残差网络结合短时状态历史、bounded correction、learned gate 和动作安全约束，输出修正后的末端执行器动作。

## 关键模型 / 系统架构

```text
Xiaomi base VLA
  -> WebSocket Policy Server
  -> task-bank routing
  -> isolated residual correction
  -> RoboDojo action contract
```

支持权重：`isolated.pt`、`composite.pt`。未知任务 fail-closed；没有启用残差的路由返回基础策略动作。

## Demo 运行结果与关键指标

叠衣服历史 A/B 结果：

| 配置 | 总体成功率 | 平均分 |
| --- | ---: | ---: |
| Xiaomi base | 25%（1/4） | 30 |
| residual gate | 50%（2/4） | 55 |
| 观测提升 | **+25 个百分点** | **+25 分** |

random 场景成功率由 0% 提升到 50%，ideal 场景没有提升。成功视频证据位于：

- [`evidence/fold_clothes/ideal/seed_0/`](../evidence/fold_clothes/ideal/seed_0/)
- [`evidence/fold_clothes/random/seed_0/`](../evidence/fold_clothes/random/seed_0/)

## 开放 / 开源贡献

- 推理代码与 Xiaomi adapter；
- `tools/start_goai.sh` 一键校验、下载权重并启动服务；
- `isolated.pt`、`composite.pt` 及 SHA256 校验；
- 叠衣服成功视频证据与可复现 README。

基础大模型、数据集和凭据不公开。

## 业务价值或应用前景

以较小的增量参数成本适配新任务，避免为每个任务重新训练完整 VLA。适用于仓储分拣、桌面整理、柔性物体操作及其他需要任务专门化动作修正的机器人场景。

## 后续计划与资源需求

1. 用打包后的 `isolated.pt` 对 12 个标准与随机配置做同配置 A/B 复测。
2. 补充多 seed、success rate、score、p50/p95 latency 和 GPU 利用率统计。
3. 在审核与正式评测期间保持同一 Policy Server、模型和动作类型在线。

> 说明：叠衣服历史结果直接评测的是 `fold_semantic_step800_residual_gate` source checkpoint，随后作为 fold head 嵌入 `isolated.pt`；因此该结果是对应 residual head 的证据，不是对打包文件的 fresh end-to-end A/B 结论。

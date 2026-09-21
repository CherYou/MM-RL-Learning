# 课程回顾与独立实验

你已经走完全书主线 20 章（编号见 [CHAPTERS](CHAPTERS.md)）。本页不是第 21 章算法，而是终点整理：你现在能做什么、如何按目标复习、下一步可以跑什么。

## 主线走完后的能力检查

| 主题 | 应能解释 | 若仍卡住 |
| --- | --- | --- |
| 损失与更新 | reward 与 loss 的差别；old/current；PPO 四种裁剪 | 回到 [01](../preliminary/TUTORIAL.md) 与 [L0 实验](../examples/math/README.md) |
| 策略优化 | PPO 的 critic/GAE；GRPO 组内优势；DAPO/GSPO 改的是哪一层 | [02–05](CHAPTERS.md#第二篇-策略优化与偏好学习) |
| 偏好分支 | DPO 是离线偏好，不是 GRPO 前置 | [06](../002-dpo/TUTORIAL.md) |
| 工具与环境 | 观察 mask；环境状态 ≠ 聊天记录；图是条件 | [07–10](CHAPTERS.md#第三篇-工具环境与多模态) |
| 蒸馏 | 谁生成、谁打分；step-0 / Skill 不同教师生命周期 | [11–14](CHAPTERS.md#第四篇-opd自蒸馏与应用) |
| 长程 | 恢复状态 ≠ 用旧数据更新；字段与参数分区 | [15–16](CHAPTERS.md#第五篇-长程-agent-进阶) |
| 控制与离线 | TD3/SAC 目标网络差异；HER 重标记；expectile≠分位数 | [17–20](CHAPTERS.md#第六篇-连续控制与离线强化学习) |

## 专题捷径（完整列表）

定义见 [learning_paths.json](../configs/learning_paths.json)；说明见 [LEARNING_PATH](LEARNING_PATH.md)。

| 路线 | 章节顺序 |
| --- | --- |
| 完整基础 | 01 → 02 → 03 → 04 → 05 |
| 快速 LLM | 01 → 03 → 07 → 08 → 10 |
| 离线偏好 | 01 → 06 |
| 基础蒸馏 | 01 → 11 → 12 → 13 |
| Agent 自蒸馏 | 01 → 03 → 09 → 11 → 12 → 14 |
| 长程 Agent | 01 → 02 → 03 → 07 → 08 → 09 → 15 → 16 |
| 连续控制与离线 | 01 → 17 → 18 → 19 → 20 |

## 独立实验建议（按验证层级）

1. **L0：** 重跑 [examples/math/loss_walkthrough.py](../examples/math/loss_walkthrough.py)，改优势符号或 ratio，预测导数再核对。
2. **L1：** 选一章默认后端 smoke（TRL 章用 `config.yaml`，Agent 章用 `verl.yaml`），对照 [EVIDENCE](EVIDENCE.md) 写清“只证明了管线”。
3. **L2：** 具身 FetchReach：比较 TD3 与 HER 在稀疏奖励下的到达率；**评估用原任务目标**；报告 seed 与环境步。
4. **不要**把 smoke loss 下降写成论文复现。

## 维护与贡献

- 目录与导航唯一数据源：[chapters.json](../configs/chapters.json) + [learning_paths.json](../configs/learning_paths.json)
- 生成/检查导航：`python scripts/sync_navigation.py --write` · `python scripts/check_navigation.py`
- 贡献规范：[CONTRIBUTING](../CONTRIBUTING.md)

## 重新开始

若要从头再读：[START_HERE](START_HERE.md) · [全书目录](CHAPTERS.md)。

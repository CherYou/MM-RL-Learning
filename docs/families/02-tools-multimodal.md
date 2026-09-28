# 第三篇：工具、环境与多模态

这些章节在**策略更新之外**改变了交互与条件：检索观察、真实代码执行、有状态文字环境、图像条件。本地多数组内更新仍可复用 GRPO 结构，但不能据此宣称与原论文完整方法相同。

> **ARPO / AEPO** 在全书主线中紧接 **06 DPO**（见 [CHAPTERS](../CHAPTERS.md)），因它们是智能体策略优化算法；本篇仍提供工具轨迹背景与交叉入口。

## 本篇章节

| 章号 | 教程 | 与基础 GRPO 相比 |
| ---: | --- | --- |
| 07 | [ARPO](../../07-arpo/TUTORIAL.md) | 工具后高熵分支采样 + 步级信用归因 |
| 08 | [AEPO](../../08-aepo/TUTORIAL.md) | 熵平衡 rollout / 裁剪 / 优势 |
| 09 | [Search-R1](../../09-search-r1/TUTORIAL.md) | 检索观察 + mask；轨迹字段 |
| 10 | [ReTool](../../10-retool/TUTORIAL.md) | 真实 Python 执行；观察≠奖励 |
| 11 | [ALFWorld](../../11-alfworld/TUTORIAL.md) | 环境状态账本；独立 env 实例 |
| 12 | [Vision-GRPO](../../12-vision-grpo/TUTORIAL.md) | 图文条件；L1 链路 vs L2 能力 |

默认后端均为 **verl**（各章 `verl.yaml`）。

## 智能体策略优化：ARPO 与 AEPO

| 方法 | 核心改动 | 原文 |
| --- | --- | --- |
| **ARPO** | 工具反馈后熵↑ → 自适应分支；终局优势按工具步归因 | [arXiv:2507.19849](https://arxiv.org/abs/2507.19849) |
| **AEPO** | 熵预监测预算 + 连续高熵分支惩罚；高熵 clip stop-grad；熵感知优势 | [arXiv:2510.14545](https://arxiv.org/abs/2510.14545) |

官方实现（同仓，基于 verl）：[RUC-NLPIR/ARPO](https://github.com/RUC-NLPIR/ARPO)。本地教学核心：`src/agentic_rl/{arpo,aepo}.py`。

完整非工具 DAPO 见 [06-dapo](../../04-dapo/TUTORIAL.md)（策略优化篇）。

## 实现字段提示

教学中的“轨迹字段账本”与代码中的 `Sample` / 日志键并不总同名。概念名 → 对象字段 → 日志键的映射以源码为准（见 Search-R1/ReTool 章与 [CONTRIBUTING](../../CONTRIBUTING.md)）；不要在日志里找不存在的键。

## 交叉入口

- [AgentOPSD](../../16-agentopsd/TUTORIAL.md) 主归属在[蒸馏篇](03-distillation.md)，但需要本篇 ALFWorld 与 GRPO 背景。
- [Harness-RL](../../18-harness-rl/TUTORIAL.md) / [TEMPO](../../17-tempo/TUTORIAL.md) 在[长程篇](04-long-horizon.md)。

---

[全书目录](../CHAPTERS.md) · [学习路线](../LEARNING_PATH.md)

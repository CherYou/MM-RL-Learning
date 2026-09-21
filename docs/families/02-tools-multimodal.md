# 第三篇：工具、环境与多模态

这些章节在**策略更新之外**改变了交互与条件：检索观察、真实代码执行、有状态文字环境、图像条件。本地多数组内更新仍可复用 GRPO 结构，但不能据此宣称与原论文完整方法相同。

## 本篇章节

| 章号 | 教程 | 与基础 GRPO 相比 |
| ---: | --- | --- |
| 07 | [Search-R1](../../03-search-r1/TUTORIAL.md) | 检索观察 + mask；轨迹字段 |
| 08 | [ReTool](../../05-retool/TUTORIAL.md) | 真实 Python 执行；观察≠奖励 |
| 09 | [ALFWorld](../../08-alfworld/TUTORIAL.md) | 环境状态账本；独立 env 实例 |
| 10 | [Vision-GRPO](../../09-vision-grpo/TUTORIAL.md) | 图文条件；L1 链路 vs L2 能力 |

默认后端均为 **verl**（各章 `verl.yaml`）。

## 实现字段提示

教学中的“轨迹字段账本”与代码中的 `Sample` / 日志键并不总同名。概念名 → 对象字段 → 日志键的映射以源码为准（见 Search-R1/ReTool 章与 [CONTRIBUTING](../../CONTRIBUTING.md)）；不要在日志里找不存在的键。

## 交叉入口

- [AgentOPSD](../../09-AgentOPSD/TUTORIAL.md) 主归属在[蒸馏篇](03-distillation.md)，但需要本篇 ALFWorld 与 GRPO 背景。
- [Harness-RL](../../12-harness-rl/TUTORIAL.md) / [TEMPO](../../09-tempo/TUTORIAL.md) 在[长程篇](04-long-horizon.md)。

---

[全书目录](../CHAPTERS.md) · [学习路线](../LEARNING_PATH.md)

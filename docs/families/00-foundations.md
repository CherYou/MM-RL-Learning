# 第一篇：共同基础

本篇只有一章正式主线：**01 智能体强化学习基础**（`01-foundations/TUTORIAL.md`）。目标是分清 reward、loss、梯度，建立 CE/MSE/BCE → 策略梯度 → old/ratio → PPO 裁剪的连续叙事，并衔接智能体工具 RL 所需的熵、mask 与信用分配直觉。

## 本篇页面

| 页面 | 作用 |
| --- | --- |
| [01 智能体强化学习基础](../../01-foundations/TUTORIAL.md) | 主线第 1 章 |
| [FOUNDATIONS](../../01-foundations/FOUNDATIONS.md) | MDP、价值、密度、离线等**按需**查阅 |
| [数学实验 README](../../examples/math/README.md) | L0 数值检查 |
| [全书目录](../CHAPTERS.md) | 20 章编号 |
| [开始入口](../START_HERE.md) | 唯一第一步 |

## 概念锚点（可直达）

定义见 [configs/concepts.json](../../configs/concepts.json)。常用：概率与 logprob、梯度与 detach、比率与裁剪、偏好损失、mask 与聚合。

## 读完本篇后

- 完整基础路线 → [02 PPO](../../02-ppo/TUTORIAL.md)
- 只想 LLM 组内优势 → [03 GRPO](../../03-grpo/TUTORIAL.md)（PPO 推荐非强制）
- 离线偏好 → [06 DPO](../../06-dpo/TUTORIAL.md)
- 连续控制 → [17 TD3](../../18-td3/TUTORIAL.md)（不强制读 GRPO）

---

[全书目录](../CHAPTERS.md) · [学习路线](../LEARNING_PATH.md) · [课程回顾](../NEXT_STEPS.md)

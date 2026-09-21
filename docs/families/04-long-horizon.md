# 第五篇：长程 Agent 进阶

处理“轨迹很长、交互很多、调用结构复杂”时的采样、估值与信用问题。本篇两章均标为进阶研读，默认 **verl**。

## 本篇章节

| 章号 | 教程 | 定位 |
| ---: | --- | --- |
| 15 | [TEMPO](../../09-tempo/TUTORIAL.md) | 本地实现研读：短分支、bootstrap、状态恢复 |
| 16 | [Harness-RL](../../12-harness-rl/TUTORIAL.md) | 调用账本、action/args 字段、参数分区 |

## 必须分清

- **恢复旧状态**（重放动作、核对观察）≠ **用旧数据直接更新**。
- **token mask**（数据轴）≠ **parameter mask**（参数轴）。
- TEMPO 官方页面在审阅时无法稳定读取正文；本地公式与 fallback 不得写成已核验的论文结论。
- Harness 为 **central-only**；worker 不联合训练。

## 推荐前置

- TEMPO：01 → 02 PPO（价值/bootstrap）→ 03 GRPO → 09 ALFWorld 概念
- Harness：01 → 03 GRPO → 08 ReTool

完整列表见 [LEARNING_PATH](../LEARNING_PATH.md)。

---

[全书目录](../CHAPTERS.md) · [蒸馏篇（AgentOPSD）](03-distillation.md)

# 第二篇：策略优化与偏好学习

关心“如何用反馈更新策略”的主线。PPO 先建立 actor/critic 与 GAE；GRPO 用同题组内统计替代学习式 critic；DAPO/GSPO 是组相对训练的不同扩展；DPO 是**离线偏好旁支**，不是 GRPO 的硬前置。

## 本篇章节

| 章号 | 教程 | 默认后端 |
| ---: | --- | --- |
| 02 | [PPO](../../001-ppo/TUTORIAL.md) | TRL `001-ppo/config.yaml` |
| 03 | [GRPO](../../01-grpo/TUTORIAL.md) | TRL `01-grpo/config.yaml` |
| 04 | [DAPO](../../06-dapo/TUTORIAL.md) | verl `06-dapo/verl.yaml` |
| 05 | [GSPO](../../07-gspo/TUTORIAL.md) | TRL `07-gspo/config.yaml` |
| 06 | [DPO](../../002-dpo/TUTORIAL.md) | TRL `002-dpo/config.yaml` |

## 方法联系（非继承链）

- GRPO 源自以 PPO 变体介绍的组相对优势；本仓库**推荐**先读 PPO，也允许从第 01 章直达 GRPO。
- DAPO 改的是采样供应、长度与训练流程，不能缩减成“换一条 loss”。
- GSPO 改变序列重要性权重与裁剪粒度；算术平均比率**不是** GRPO 定义。
- DPO 使用固定偏好对与 reference 校正的 logistic 目标，与在线 rollout 分支不同。

## 下一篇入口

主线第 07 章进入 [工具与多模态](02-tools-multimodal.md)。只学蒸馏可从第 01 章直达 [General OPD](03-distillation.md)。

---

[全书目录](../CHAPTERS.md) · [学习路线](../LEARNING_PATH.md)

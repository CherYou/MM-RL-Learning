# 全书目录（20 章）

**唯一显示编号来源：** 本页与 [configs/learning_paths.json](../configs/learning_paths.json) 的 `book` 列表一致。物理运行目录（`001-ppo`、`09-AgentOPSD` 等）保持不变，编号是阅读导航，不是文件夹重命名。

上一章 / 下一章固定指**全书主线**；专题捷径见 [LEARNING_PATH](LEARNING_PATH.md)。

```mermaid
flowchart TD
  F[共同基础：概率、损失、策略梯度] -->|策略梯度路线| P[PPO]
  P -->|组内统计替代价值估计| G[GRPO]
  G -->|训练流程与聚合扩展| DA[DAPO]
  G -->|序列比率与裁剪| GS[GSPO]
  F -->|偏好与 reference 背景| DP[DPO：独立偏好分支]
  G -->|本地工具应用| SR[Search-R1]
  G -->|本地工具应用| RT[ReTool]
  G -->|本地环境应用| AL[ALFWorld]
  G -->|图文条件应用| VG[Vision-GRPO]
  F -->|KL 与教师反馈基础| O[General OPD]
  O -->|本地特权信息设置| OS[OPSD]
  O -->|数据与教师调度| MED[Medical / SAR / IDT]
  OS -->|教师视角联系| AO[AgentOPSD]
  AL -->|环境前置| AO
  G -->|本地终局优势| AO
  P -->|价值与 bootstrap 背景| TE[TEMPO：本地实现研读]
  AL -->|状态恢复基础| TE
  RT -->|调用与 mask 背景| H[Harness-RL]
  F -->|连续动作、Q 与回放| C[连续控制基础]
  C -->|确定性策略路线| TD[TD3]
  C -->|随机最大熵路线| SA[SAC]
  TD -->|本仓库组合| HER[HER：数据重标记]
  C -->|固定数据与价值学习| IQ[IQL]
```

箭头表示方法联系或教学衔接，**不等于**算法继承，也不都是硬前置。硬前置见 [configs/chapters.json](../configs/chapters.json)。

## 第一篇 共同基础

| 章号 | 教程 | 实际正文 |
| ---: | --- | --- |
| 01 | **智能体强化学习基础** | [01-foundations/TUTORIAL.md](../01-foundations/TUTORIAL.md) |

按需查阅：[FOUNDATIONS](../01-foundations/FOUNDATIONS.md) · [数学实验](../examples/math/README.md) · [术语与符号](BEGINNER_GUIDE.md)

## 第二篇 策略优化与偏好学习

| 章号 | 教程 | 实际正文 |
| ---: | --- | --- |
| 02 | PPO：actor-critic、GAE 与裁剪更新 | [02-ppo/TUTORIAL.md](../02-ppo/TUTORIAL.md) |
| 03 | GRPO：同题采样与组内优势 | [03-grpo/TUTORIAL.md](../03-grpo/TUTORIAL.md) |
| 04 | DAPO：动态采样、裁剪与长度处理 | [04-dapo/TUTORIAL.md](../04-dapo/TUTORIAL.md) |
| 05 | GSPO：序列级比率与裁剪 | [05-gspo/TUTORIAL.md](../05-gspo/TUTORIAL.md) |
| 06 | DPO：从偏好对直接优化策略（离线旁支） | [06-dpo/TUTORIAL.md](../06-dpo/TUTORIAL.md) |

本篇导读：[families/01-policy-preference.md](families/01-policy-preference.md)

## 第三篇 工具、环境与多模态

| 章号 | 教程 | 实际正文 |
| ---: | --- | --- |
| 07 | ARPO：工具后高熵分支与步级信用 | [07-arpo/TUTORIAL.md](../07-arpo/TUTORIAL.md) |
| 08 | AEPO：熵平衡智能体策略优化 | [08-aepo/TUTORIAL.md](../08-aepo/TUTORIAL.md) |
| 09 | Search-R1：学习检索与观察 mask | [09-search-r1/TUTORIAL.md](../09-search-r1/TUTORIAL.md) |
| 10 | ReTool：真实执行与工具决策 | [10-retool/TUTORIAL.md](../10-retool/TUTORIAL.md) |
| 11 | ALFWorld：有状态环境中的策略学习 | [11-alfworld/TUTORIAL.md](../11-alfworld/TUTORIAL.md) |
| 12 | Vision-GRPO：图像条件下的策略学习 | [12-vision-grpo/TUTORIAL.md](../12-vision-grpo/TUTORIAL.md) |

本篇导读：[families/02-tools-multimodal.md](families/02-tools-multimodal.md)

## 第四篇 OPD、自蒸馏与应用

| 章号 | 教程 | 实际正文 |
| ---: | --- | --- |
| 13 | General OPD：学生采样与教师反馈 | [13-general-opd/TUTORIAL.md](../13-general-opd/TUTORIAL.md) |
| 14 | OPSD：特权信息与自教师 | [14-opsd/TUTORIAL.md](../14-opsd/TUTORIAL.md) |
| 16 | AgentOPSD：自教师辅助的轮次信用 | [15-agentopsd/TUTORIAL.md](../15-agentopsd/TUTORIAL.md) |

本篇导读：[families/03-distillation.md](families/03-distillation.md)

## 第五篇 长程 Agent 进阶

| 章号 | 教程 | 实际正文 |
| ---: | --- | --- |
| 17 | TEMPO：短分支、估值与状态恢复 | [16-tempo/TUTORIAL.md](../16-tempo/TUTORIAL.md) |
| 18 | Harness-RL：调用记录与参数分区 | [17-harness-rl/TUTORIAL.md](../17-harness-rl/TUTORIAL.md) |

本篇导读：[families/04-long-horizon.md](families/04-long-horizon.md)

## 第六篇 连续控制与离线强化学习

| 章号 | 教程 | 实际正文 |
| ---: | --- | --- |
| 19 | TD3：确定性策略与双 Q 稳定化 | [18-td3/TUTORIAL.md](../18-td3/TUTORIAL.md) |
| 20 | SAC：随机策略与最大熵目标 | [19-sac/TUTORIAL.md](../19-sac/TUTORIAL.md) |
| 21 | HER：目标条件经验重标记 | [20-her/TUTORIAL.md](../20-her/TUTORIAL.md) |
| 22 | IQL：固定数据、expectile 与加权模仿 | [21-iql/TUTORIAL.md](../21-iql/TUTORIAL.md) |

本篇导读：[families/05-control-offline.md](families/05-control-offline.md)

## 独立资料架（不计入 20 章主线）

- [Basic LLM 技术报告](../Basic%20LLM/README.md)
- [后端安装 VERL](VERL.md) · [数据 DATA](DATA.md) · [具身 EMBODIED](EMBODIED.md)
- [排错 TROUBLESHOOTING](TROUBLESHOOTING.md) · [证据层级 EVIDENCE](EVIDENCE.md)
- 末章之后：[课程回顾与独立实验](NEXT_STEPS.md)

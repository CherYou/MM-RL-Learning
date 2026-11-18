# 学习路线

**唯一全书编号目录：[CHAPTERS.md](CHAPTERS.md)。** 本页解释不同目标下的**专题捷径**与选择依据；上一章/下一章在教程页首/页尾固定指全书主线。

## 专题捷径

| 路线 | 章节顺序 | 适合的学习目标 |
| --- | --- | --- |
| 完整基础 | 01 基础 → 02 PPO → 03 GRPO → 04 DAPO → 05 GSPO | 用奖励更新策略 |
| 快速 LLM 实践 | 01 基础 → 03 GRPO → 07 ARPO → 08 AEPO → 09 Search-R1 → 10 ReTool → 12 Vision-GRPO | 工具调用与图文后训练 |
| 离线偏好 | 01 基础 → 06 DPO | 固定偏好对 |
| 基础蒸馏与领域案例 | 01 基础 → 13 General OPD → 14 OPSD | 学生生成与教师复评 |
| Agent 自蒸馏 | 01 基础 → 03 GRPO → 11 ALFWorld → 13 General OPD → 14 OPSD → 15 AgentOPSD | 环境中的轮次信用 |
| 长程 Agent | 01 基础 → 02 PPO → 03 GRPO → 09 Search-R1 → 10 ReTool → 11 ALFWorld → 16 TEMPO → 17 Harness-RL | 短分支与多轮调用 |
| 连续控制与离线 | 01 基础 → 18 TD3 → 19 SAC → 20 HER → 21 IQL | 连续动作与离线数据 |
| ARPO / AEPO 智能体策略优化 | 01 基础 → 03 GRPO → 07 ARPO → 08 AEPO → 10 ReTool | 工具反馈后的熵与分支 |

控制路线用到的连续动作密度、价值函数和回放概念，可以从[基础知识](../01-foundations/FOUNDATIONS.md)中按需查阅。

## 六篇导读

| 篇 | 页面 |
| --- | --- |
| 共同基础 | [families/00-foundations.md](families/00-foundations.md) |
| 策略与偏好 | [families/01-policy-preference.md](families/01-policy-preference.md) |
| 工具与多模态 | [families/02-tools-multimodal.md](families/02-tools-multimodal.md) |
| 蒸馏与应用 | [families/03-distillation.md](families/03-distillation.md) |
| 长程 Agent | [families/04-long-horizon.md](families/04-long-horizon.md) |
| 控制与离线 | [families/05-control-offline.md](families/05-control-offline.md) |

## 默认教学后端

| 章节类型 | 默认入口 |
| --- | --- |
| 01 / examples/math | PyTorch 轻量脚本 |
| 02、03、05、06 策略与偏好 | TRL + 各章 `config.yaml` |
| 04 DAPO | verl + `04-dapo/verl.yaml` |
| 07–12 工具与多模态 | verl + 各章 `verl.yaml` |
| 13–17 蒸馏与长程 | verl + 各章 `verl.yaml` |
| 18–21 控制 | PyTorch embodied + FetchReach |

跨后端命令放在章节 README 进阶区。

## 低维控制 → 还差什么（VLA 边界，集中说明）

| 已有 | 完整 VLA 还需要 |
| --- | --- |
| 低维状态/目标 + 连续动作 | 高维视觉/语言条件与对齐 |
| FetchReach 成功等环境奖励 | 更丰富任务规范与安全约束 |
| 离线/在线控制算法基础 | 大规模机器人数据与动作表示 |
| 图文条件策略（Vision-GRPO） | 动作头、控制频率、真机/高保真仿真 |

## 查阅索引

| 你想查 | 去哪里 |
| --- | --- |
| 21 章全书顺序 | [CHAPTERS.md](CHAPTERS.md) |
| 术语与符号 | [BEGINNER_GUIDE](BEGINNER_GUIDE.md) |
| 概念锚点注册 | [configs/concepts.json](../configs/concepts.json) |
| MDP / 价值 / 离线 | [FOUNDATIONS](../01-foundations/FOUNDATIONS.md) |
| 证据层级 | [EVIDENCE.md](EVIDENCE.md) |
| 末章之后 | [NEXT_STEPS.md](NEXT_STEPS.md) |

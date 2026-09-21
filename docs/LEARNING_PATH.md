# 学习路线

**唯一全书编号目录：[CHAPTERS.md](CHAPTERS.md)。** 本页解释不同目标下的**专题捷径**与选择依据；上一章/下一章在教程页首/页尾固定指全书主线。

权威路线列表：[configs/learning_paths.json](../configs/learning_paths.json)。

## 七条专题捷径

| 路线 | 完整章节 ID 顺序（显示号） | 适合谁 |
| --- | --- | --- |
| 完整基础 | 01 loss-basics → 02 ppo → 03 grpo → 04 dapo → 05 gspo | 想走通反馈→更新整条链 |
| 快速 LLM 实践 | 01 → 03 grpo → 07 search-r1 → 08 retool → 10 vision-grpo | 工具型后训练 |
| 离线偏好 | 01 → 06 dpo | 只有偏好对 |
| 基础蒸馏与领域案例 | 01 → 11 general-opd → 12 opsd → 13 medical-sar-idt | 教师反馈与调度 |
| Agent 自蒸馏 | 01 → 03 → 09 alfworld → 11 → 12 → 14 agentopsd | 轮次信用 |
| 长程 Agent | 01 → 02 → 03 → 07 → 08 → 09 → 15 tempo → 16 harness-rl | 短分支与调用分区 |
| 连续控制与离线 | 01 → 17 td3 → 18 sac → 19 her → 20 iql | 机械臂 / 离线数据 |

每条路线的硬前置都位于同一条路线之前。控制路线把连续密度、价值、replay 作为**概念补课**，不强迫读 PPO/DPO。

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
| 02 PPO / 03 GRPO / 05 GSPO / 06 DPO | TRL + 各章 `config.yaml` |
| 04 DAPO / 07–10 工具多模态 / 11–16 蒸馏与长程 | verl + 各章 `verl.yaml` |
| 17–20 控制 | PyTorch embodied + FetchReach |

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
| 20 章全书顺序 | [CHAPTERS.md](CHAPTERS.md) |
| 术语与符号 | [BEGINNER_GUIDE](BEGINNER_GUIDE.md) |
| 概念锚点注册 | [configs/concepts.json](../configs/concepts.json) |
| MDP / 价值 / 离线 | [FOUNDATIONS](../preliminary/FOUNDATIONS.md) |
| 证据层级 | [EVIDENCE.md](EVIDENCE.md) |
| 末章之后 | [NEXT_STEPS.md](NEXT_STEPS.md) |

写作者规范（勿在读者路线中夹带）见 [CONTRIBUTING](../CONTRIBUTING.md)。

# 第四篇：OPD、自蒸馏与应用

共同关注：**学生生成轨迹上的教师反馈**。不能把 OPD 一律定义成仓库这一条 sampled reverse-KL surrogate；各章改变的是教师条件、生命周期或调度。

## 本篇章节

| 章号 | 教程 | 层次 |
| ---: | --- | --- |
| 11 | [General OPD](../../02-opd/general-opd/TUTORIAL.md) | 概念入口：谁生成、谁打分 |
| 12 | [OPSD](../../04-opsd/TUTORIAL.md) | 特权 solution + 固定 step-0 教师（本地） |
| 13 | [Medical SAR/IDT](../../02-opd/TUTORIAL.md) | 领域**调度案例** |
| 14 | [AgentOPSD](../../09-AgentOPSD/TUTORIAL.md) | 环境轨迹 + 轮次信用（与 Agent 交汇） |

## 硬前置提醒

AgentOPSD 路线显式包含：01 loss-basics → 03 grpo → 09 alfworld → 11 general-opd → 12 opsd → 14 agentopsd。纯蒸馏读者可 01 → 11 直达，不必先读工具篇全文。

## 教师角色对照

| 章 | Student 可见 | Teacher 可见 | Teacher 权重 |
| --- | --- | --- | --- |
| General OPD | 题目 + 自己前缀 | 相同 token（无 solution） | 冻结教师 |
| OPSD | 题目 + 自己前缀 | 题目 + **solution** + 学生前缀 | **step-0** |
| Medical | 按调度的数据池 | Medical 或 Base Teacher | 冻结 |
| AgentOPSD | 环境历史 | **Skill** + 学生动作 | **当前批更新前** |

评估 Student 时不得保留 solution/Skill。

---

[全书目录](../CHAPTERS.md) · [工具篇交叉入口](02-tools-multimodal.md)

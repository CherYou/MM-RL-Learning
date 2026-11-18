# 第四篇：OPD、自蒸馏与应用

共同关注：**学生生成轨迹上的教师反馈**。不能把 OPD 一律定义成仓库这一条 sampled reverse-KL surrogate；各章改变的是教师条件、生命周期或调度。

## 本篇章节

| 章号 | 教程 | 层次 |
| ---: | --- | --- |
| 13 | [General OPD](../../13-general-opd/TUTORIAL.md) | 概念入口：谁生成、谁打分 |
| 14 | [OPSD](../../14-opsd/TUTORIAL.md) | 特权 solution + 固定 step-0 教师（本地） |
| 15 | [AgentOPSD](../../15-agentopsd/TUTORIAL.md) | 环境轨迹 + 轮次信用（与 Agent 交汇） |

## 硬前置提醒

AgentOPSD 会用到 GRPO 的组内优势、ALFWorld 的动作轨迹和 OPSD 的教师条件。想先学习纯蒸馏，可从第 01 章直接读第 13 章 General OPD。[完整路线](../LEARNING_PATH.md#专题捷径)

## 教师角色对照

| 章 | Student 可见 | Teacher 可见 | Teacher 权重 |
| --- | --- | --- | --- |
| General OPD | 题目 + 自己前缀 | 相同 token（无 solution） | 冻结教师 |
| OPSD | 题目 + 自己前缀 | 题目 + **solution** + 学生前缀 | **step-0** |
| AgentOPSD | 环境历史 | **Skill** + 学生动作 | **当前批更新前** |

评估 Student 时不得保留 solution/Skill。

---

[全书目录](../CHAPTERS.md) · [工具篇交叉入口](02-tools-multimodal.md)

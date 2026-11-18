# 课程回顾与独立实验

读完主线后，可以用下面的问题检查自己还需要复习哪些概念，再选一个小实验。[全书目录](CHAPTERS.md)

## 主线走完后的能力检查

| 主题 | 应能解释 | 若仍卡住 |
| --- | --- | --- |
| 损失与更新 | reward 与 loss 的差别；old/current；PPO 四种裁剪 | 回到 [01](../01-foundations/TUTORIAL.md) 与 [L0 实验](../examples/math/README.md) |
| 策略优化 | PPO 的 critic/GAE；GRPO 组内优势；DAPO/GSPO 改的是哪一层 | [02–05](CHAPTERS.md#第二篇-策略优化与偏好学习) |
| 偏好分支 | DPO 是离线偏好，不是 GRPO 前置 | [06](../06-dpo/TUTORIAL.md) |
| 工具与环境 | 观察 mask；环境状态 ≠ 聊天记录；图是条件 | [07–12](CHAPTERS.md#第三篇-工具环境与多模态) |
| 蒸馏 | 谁生成、谁打分；step-0 / Skill 不同教师生命周期 | [13–15](CHAPTERS.md#第四篇-opd自蒸馏与应用) |
| 长程 | 恢复状态 ≠ 用旧数据更新；字段与参数分区 | [16–17](CHAPTERS.md#第五篇-长程-agent-进阶) |
| 控制与离线 | TD3/SAC 目标网络差异；HER 重标记；expectile≠分位数 | [18–21](CHAPTERS.md#第六篇-连续控制与离线强化学习) |

## 选一个独立实验

1. 修改优势符号或概率比，预测 PPO 的梯度方向，再运行[数学例子](../examples/math/loss_walkthrough.py)核对。
2. 用 GRPO 的 `--smoke` 产生一批回答，找出同组奖励不同的两条样本，比较它们的优势和选中 token 的梯度方向。
3. 在 FetchReach 的同一组评估种子上比较 TD3 和 TD3 + HER。记录各自消耗的环境步数，并始终按原任务目标计算到达率。

若想继续研读一个专题，可回到[八条学习路线](LEARNING_PATH.md#专题捷径)。

## 重新开始

若要从头再读：[START_HERE](START_HERE.md) · [全书目录](CHAPTERS.md)。

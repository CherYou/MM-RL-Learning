# 第六篇：连续控制与离线强化学习

动作是连续向量；训练信号来自环境转移与固定数据。**不要求**先学完 DPO/GRPO。概念补课走 [FOUNDATIONS](../../preliminary/FOUNDATIONS.md) 锚点（价值、密度、replay/离线）。

## 本篇章节（阅读顺序）

| 章号 | 教程 | 默认入口 |
| ---: | --- | --- |
| 17 | [TD3](../../14-td3/TUTORIAL.md)（含 DDPG 短前导） | `14-td3/train.py` |
| 18 | [SAC](../../13-sac/TUTORIAL.md) | `13-sac/train.py` |
| 19 | [HER](../../15-her/TUTORIAL.md)（TD3 + 重标记） | `15-her/train.py` |
| 20 | [IQL](../../16-iql/TUTORIAL.md) | `16-iql/train.py` |

顺序 TD3→SAC→HER→IQL 是**教学阅读顺序**，不是断言四者严格逐代演化。

## 实现要点

- TD3：确定性 actor；双 Q 取小；延迟更新；目标动作平滑。
- SAC：随机最大熵；**目标动作来自当前 actor**；只慢更新目标 critic（`target_actor` 在 SAC 路径不使用）。
- HER：改的是 **goal 条件与 reward 标签**，loss 仍是 TD3；评估用原任务。
- IQL：expectile ≠ quantile；训练步 ≠ 环境交互步；评估轨迹不回流。

## 低维控制 → VLA 边界

集中说明见 [LEARNING_PATH](../LEARNING_PATH.md#低维控制--还差什么vla-边界集中说明)；各章不再重复长篇展望。

---

[全书目录](../CHAPTERS.md) · [具身安装](../EMBODIED.md) · [课程回顾](../NEXT_STEPS.md)

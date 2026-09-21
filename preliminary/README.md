# Preliminary：读算法之前的损失与更新基础

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

本章回答：模型训练到底在改变什么？有示范答案、数值目标、偏好标签和只有奖励时，反馈怎样分别进入损失？为什么需要 old、概率比率和 PPO 裁剪？

| 阅读入口 | 什么时候读 | 读完应能 |
| --- | --- | --- |
| [TUTORIAL.md](TUTORIAL.md) | 第一时间 | 解释 CE/MSE/BCE、策略梯度、old/current 与 PPO 四种裁剪方向 |
| [examples/math/README.md](../examples/math/README.md) | 读完或读到第 5–8 节时 | 运行 L0 数值与梯度检查 |
| [FOUNDATIONS.md](FOUNDATIONS.md) | 进入 PPO/GRPO/连续控制前按需查阅 | MDP、回报、V/Q、mask、离线与评估边界 |
| [学习路线](../docs/LEARNING_PATH.md) | 选分支时 | 知道下一站去哪、哪些可以跳过 |

## 唯一默认实验（L0）

```bash
# 仓库根目录；只需 PyTorch，不下载模型
python examples/math/loss_walkthrough.py
```

参考结果见 [examples/math/reference/numerical_checks.json](../examples/math/reference/numerical_checks.json)。这是构造数学检查，不是训练结果。

## 进阶：仓库 loss-demo（使用共享实现）

完整环境准备后：

```bash
.venv/bin/python preliminary/train.py
# 等价入口
.venv/bin/arl loss-demo
```

它调用 [loss_demo.py](../src/agentic_rl/loss_demo.py) 与 [losses.py](../src/agentic_rl/losses.py)，生成梯度曲线、CSV 与 TensorBoard 事件。这里不训练语言模型，不需要 `--smoke` 或 GPU。

CISPO 演示使用显式双侧裁剪教学式，与原论文完整配方的差异在 TUTORIAL 中说明。

## 前置知识按需查阅

不必先读完 [FOUNDATIONS.md](FOUNDATIONS.md) 全文再进入第一章。语言模型主线优先 TUTORIAL；连续控制或需要正式 MDP / bootstrap 时，再按 FOUNDATIONS 小节补课。完整导航见 [FOUNDATIONS 阅读地图](FOUNDATIONS.md#foundations-map)。

下一步按路线选择：[PPO](../001-ppo/TUTORIAL.md)（完整 actor-critic）或 [GRPO](../01-grpo/TUTORIAL.md)（同题组内优势）。DPO 不必成为 GRPO 的强制前置。

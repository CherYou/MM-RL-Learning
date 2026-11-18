# 从这里开始

这是一个面向 RL 初学者的中文学习与实验仓库。你不需要先读完全部章节，也不需要在第一天安装所有训练后端。

**先跑一个只需 PyTorch 的数值例子。**

```bash
# 任意已安装 PyTorch 的 Python 环境
python examples/math/loss_walkthrough.py
```

终端打印的 JSON 包含损失、梯度和一次参数更新前后的概率。结尾出现 `"status": "passed"` 时，可以回看每个数为什么增大或减小。

读完并跑通后，进入：

1. [损失函数详解：从预测一个答案，到用奖励更新模型](../01-foundations/TUTORIAL.md)
2. 按 [学习路线](LEARNING_PATH.md) 选择一条分支，不要同时打开所有算法。

## 这个仓库教什么

- 从 logits、常见损失和策略梯度，走到 PPO / GRPO 等语言模型后训练
- 工具与环境交互（检索、代码执行、ALFWorld）
- 图文条件策略学习（Vision-GRPO）
- 连续控制与离线学习基础（SAC / TD3 / HER / IQL）

想把某种算法与大模型训练中的实际用法对照起来，可以查阅[模型技术报告](../Basic%20LLM/README.md)。

## 三条阅读路线（摘要）

完整定义见 [LEARNING_PATH](LEARNING_PATH.md) 与 [CHAPTERS](CHAPTERS.md)。

| 路线 | 概要 |
| --- | --- |
| 完整基础 | 01 → 02 PPO → 03 GRPO →（04 DAPO / 05 GSPO） |
| 快速 LLM | 01 → 03 GRPO → 工具/检索 → Vision-GRPO |
| 连续控制 | 01 → 18 TD3 → 19 SAC → 20 HER / 21 IQL |

如果手头有成对的偏好数据，从[第 06 章 DPO](../06-dpo/TUTORIAL.md)开始。如果想学习如何用教师指导学生自己生成的回答，读[第 13 章 General OPD](../13-general-opd/TUTORIAL.md)。

## 完整环境安装

面向 Linux / WSL 与 Python 3.12，完整训练环境见 [README 安装节](../README.md#安装与检查)。只想读公式和跑 L0 时，不必先装 verl。

## 相关入口

- [零基础学习导航与符号表](BEGINNER_GUIDE.md)
- [基础知识查阅](../01-foundations/FOUNDATIONS.md)
- [章节目录](../README.md#章节目录与推荐顺序)
- [数据说明](DATA.md) · [后端指南](VERL.md) · [故障排查](TROUBLESHOOTING.md)

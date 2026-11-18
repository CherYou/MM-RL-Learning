# AEPO：熵平衡的智能体策略优化

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

本章用数值例子说明如何用熵预算分配分支、控制高熵 token 的梯度并调整优势。训练示例使用 verl。

- 原文：[paper](https://arxiv.org/abs/2510.14545)
- 官方代码：[RUC-NLPIR/ARPO](https://github.com/RUC-NLPIR/ARPO/tree/main/AEPO)
- 建议前置：[arpo](../07-arpo/TUTORIAL.md) · [第 01 章](../01-foundations/TUTORIAL.md)

[全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [学习路线](../docs/LEARNING_PATH.md)

## 从代码入口开始

```bash
source .venv/bin/activate
.venv/bin/arl train 08-aepo/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 08-aepo/train.py --smoke
```

`--smoke` 在 CPU 上运行微型模型和工具样例。论文中的搜索环境和多卡训练脚本见上方官方仓库。

## 本地教学模块

| 模块 | 作用 |
| --- | --- |
| [aepo.py](../src/agentic_rl/aepo.py) | 核心机制的可运行最小实现 |
| `python -m agentic_rl.aepo` | 打印 CPU 数值 demo（若以包方式安装） |

## 选择运行入口

要观察 AEPO 的熵机制，请运行 `python -m agentic_rl.aepo` 并对照[教程](TUTORIAL.md)中的手算。`train.py` 走共用的 verl 工具采样与组内优势更新路径，专章的熵计算位于独立的数值模块。

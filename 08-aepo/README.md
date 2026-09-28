# AEPO：熵平衡的智能体策略优化

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 相对基础 GRPO / 工具轨迹，本章新增：在 ARPO 的熵分支上再平衡：动态熵预算 + 分支惩罚、高熵 token 裁剪 stop-grad、熵感知优势。

- 原文：[paper](https://arxiv.org/abs/2510.14545)
- 官方代码：[RUC-NLPIR/ARPO](https://github.com/RUC-NLPIR/ARPO/tree/main/AEPO)（上游基于 verl；本仓库为最小教学适配）
- 建议前置：[arpo](../07-arpo/TUTORIAL.md) · [第 01 章](../01-foundations/TUTORIAL.md)

[全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [学习路线](../docs/LEARNING_PATH.md)

## 从代码入口开始

```bash
source .venv/bin/activate
.venv/bin/arl train 08-aepo/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 08-aepo/train.py --smoke
```

`--smoke` 使用 tiny 模型与 fixture，只验证机制链路。完整上游训练需要搜索 API、多 GPU 与真实工具环境，见原文与官方仓库。

## 本地教学模块

| 模块 | 作用 |
| --- | --- |
| [aepo.py](../src/agentic_rl/aepo.py) | 核心机制的可运行最小实现 |
| `python -m agentic_rl.aepo` | 打印 CPU 数值 demo（若以包方式安装） |

## 复现边界

本地实现覆盖算法的教学机制与数值对照，**不是**论文全量 benchmark 复现。证据层级见 [EVIDENCE](../docs/EVIDENCE.md)。

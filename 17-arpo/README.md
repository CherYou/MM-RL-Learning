# ARPO：工具后的高熵分支采样与步级信用

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 相对基础 GRPO / 工具轨迹，本章新增：在 GRPO/工具轨迹上增加：工具反馈后的熵监测、自适应分支采样、步级优势归因。

- 原文：[paper](https://arxiv.org/abs/2507.19849)
- 官方代码：[RUC-NLPIR/ARPO](https://github.com/RUC-NLPIR/ARPO)（上游基于 verl；本仓库为最小教学适配）
- 建议前置：[retool](../05-retool/TUTORIAL.md) · [第 01 章](../preliminary/TUTORIAL.md)

[全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [学习路线](../docs/LEARNING_PATH.md)

## 从代码入口开始

```bash
source .venv/bin/activate
.venv/bin/arl train 17-arpo/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 17-arpo/train.py --smoke
```

`--smoke` 使用 tiny 模型与 fixture，只验证机制链路。完整上游训练需要搜索 API、多 GPU 与真实工具环境，见原文与官方仓库。

## 本地教学模块

| 模块 | 作用 |
| --- | --- |
| [arpo.py](../src/agentic_rl/arpo.py) | 核心机制的可运行最小实现 |
| `python -m agentic_rl.arpo` | 打印 CPU 数值 demo（若以包方式安装） |

## 复现边界

本地实现覆盖算法的教学机制与数值对照，**不是**论文全量 benchmark 复现。证据层级见 [EVIDENCE](../docs/EVIDENCE.md)。

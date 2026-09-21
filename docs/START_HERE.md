# 从这里开始

这是一个面向 RL 初学者的中文学习与实验仓库。你不需要先读完全部章节，也不需要在第一天安装所有训练后端。

**唯一第一步（不下载大模型、不装 verl）：**

```bash
# 任意已安装 PyTorch 的 Python 环境
python examples/math/loss_walkthrough.py
```

终端应打印 JSON，结尾出现 `"status": "passed"`。这一步只核对损失数值、梯度方向和一次小参数更新，不能说明语言模型学会了任务。

读完并跑通后，进入：

1. [损失函数详解：从预测一个答案，到用奖励更新模型](../preliminary/TUTORIAL.md)
2. 按 [学习路线](LEARNING_PATH.md) 选择一条分支，不要同时打开所有算法。

## 这个仓库教什么

- 从 logits、常见损失和策略梯度，走到 PPO / GRPO 等语言模型后训练
- 工具与环境交互（检索、代码执行、ALFWorld）
- 图文条件策略学习（Vision-GRPO）
- 连续控制与离线学习基础（SAC / TD3 / HER / IQL）

## 这个仓库不自动等于什么

- 不是论文 benchmark 的完整复现报告
- `--smoke` / tiny 模型 / debug reward 只验证机制可运行
- Vision-GRPO 与 FetchReach 控制提供理解 VLA 的基础，但不等于完整 VLA 训练管线
- 技术报告阅读资料在 [Basic LLM](../Basic%20LLM/README.md)，不是学习策略梯度的门槛

## 三条路线怎么选

| 你是谁 | 路线 | 入口 |
| --- | --- | --- |
| 想完整理解学习循环 | 完整基础路线 | 损失基础 → PPO → GRPO |
| 主要做 LLM 后训练 | 快速 LLM 实践路线 | 损失与比率裁剪 → 最小 GRPO → ReTool / Vision-GRPO |
| 关心机器人 / 连续控制 | 连续控制路线 | 损失与 RL 基础 → SAC → TD3 → HER / IQL |

细节、前置与旁支见 [LEARNING_PATH.md](LEARNING_PATH.md)。

## 验证层级

| 层级 | 含义 | 本仓库示例 |
| --- | --- | --- |
| L0 数学演示 | 人工张量核对 loss / 梯度 | `examples/math/loss_walkthrough.py`、`arl loss-demo` |
| L1 机制 smoke | tiny 模型与真实代码链路可执行 | `python 01-grpo/train.py --smoke` |
| L2 小任务学习 | 可控任务上训练前后有变化 | 教学 bandit（规划中）/ 具身短程实验 |
| L3 方法复现 | 与论文/基准可比 | 需固定预算与设备，逐项验证 |

看到命令输出时，先确认它属于哪一层。

## 完整环境安装

面向 Linux / WSL 与 Python 3.12，完整训练环境见 [README 安装节](../README.md#安装与检查)。只想读公式和跑 L0 时，不必先装 verl。

## 相关入口

- [零基础学习导航与符号表](BEGINNER_GUIDE.md)
- [基础知识查阅](../preliminary/FOUNDATIONS.md)
- [章节目录](../README.md#章节目录与推荐顺序)
- [数据说明](DATA.md) · [后端指南](VERL.md) · [故障排查](TROUBLESHOOTING.md)

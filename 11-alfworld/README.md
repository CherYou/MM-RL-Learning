# ALFWorld：文字环境与独立 rollout

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 本章重点：环境状态账本；won/done 区分；独立 env。裁剪与组内优势见 [GRPO](../03-grpo/TUTORIAL.md) / [损失与策略梯度基础](../01-foundations/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

本章使用官方 ALFWorld + TextWorld 的文本环境。按 game.tw-pddl 找到可解游戏，实例化独立环境，reset 后获得目标和可行动作；模型每轮输出 `<action>动作</action>`，环境返回观察、done 与 won。

每道游戏采样 G 条独立轨迹，终局奖励由环境的 `won` 给出。动作 token 参与策略更新，环境观察留作下一步的上下文。[环境实现与来源](../docs/CODE_PROVENANCE.md)

文本游戏分为 `train`、`valid_seen` 和 `valid_unseen`。`--smoke` 使用小型玩具环境。评估策略时应在两种验证划分上分别运行 TextWorld。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 11-alfworld/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 11-alfworld/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 11-alfworld/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[alfworld_data.py](../src/agentic_rl/alfworld_data.py)、[alfworld_env.py](../src/agentic_rl/alfworld_env.py)、[rollout.py](../src/agentic_rl/rollout.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-11-alfworld`。

## 指标与对照实验

先检查同一游戏 reset 可复现；采样组之间状态独立；game_id 不跨 train/eval。后续评测分别汇报 seen/unseen 成功率、平均步数、无效动作数。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

本章使用 ALFWorld 的 TextWorld 文字接口。完整实验需要文本游戏数据，可以在 CPU 上运行环境。

## 一手资料

- [ALFWorld 原论文](https://arxiv.org/abs/2010.03768)

## 后端说明

CPU 和 CUDA 环境的安装、训练中断后的恢复步骤见 [verl 指南](../docs/VERL.md)。

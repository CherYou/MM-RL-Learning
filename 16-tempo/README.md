# TEMPO：状态重放与生成式 critic（进阶）

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 本章重点：恢复状态≠旧数据更新；生成式 critic。裁剪与组内优势见 [GRPO](../03-grpo/TUTORIAL.md) / [损失与策略梯度基础](../01-foundations/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

把长任务切成 H 个交互步的 macro-step。同一个起点恢复多份独立环境，分别执行分支。终局分支 future value=0；非终局边界通过同一个语言模型生成 `<value>成功概率</value>`，估计未来回报。

actor return=段内 reward+终点 value；TD target G 是同起点各分支 return 的均值。actor advantage 做组内中心化；critic 生成的 value 按 −abs(value−G) 奖励，也做组内中心化。两种样本共同更新同一套参数，不额外训练线性 value head。

StateStore 保存边界 token 前缀、动作和环境观察。下一轮恢复时重新 reset、重放动作，并逐字验证观察。旧前缀是上下文；只有新 macro-step 的模型 token 进入 loss。warmup 阶段先用完整 episode 提供初始回报，再进入截断 TD 更新。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 16-tempo/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 16-tempo/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 16-tempo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[tempo.py](../src/agentic_rl/tempo.py)、[rollout.py](../src/agentic_rl/rollout.py)。

`--smoke` 在 CPU 上用微型模型和样例数据依次运行预热、宏步更新和状态重放。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-16-tempo`。

## 指标与对照实验

专项测试终局 value 强制为零、G 取分支均值、解析失败有明确惩罚；进一步比较 H、critic_samples 和 warmup_steps。查看解析失败率、endpoint value、state store 大小和实际成功率。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

需要前缀重要性修正或跨进程恢复时，请使用 verl 后端。native 后端适合观察单进程的分支采样。若同一端点的所有估值都解析失败，使用 0.5 自举并记录失败数。

## 一手资料


## 后端说明

CPU 和 CUDA 环境的安装、训练中断后的恢复步骤见 [verl 指南](../docs/VERL.md)。

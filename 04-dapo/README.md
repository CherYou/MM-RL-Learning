# DAPO：四项机制与动态采样

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

逐项理解裁剪、有效组补采、token 权重与长度处理。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

本地主入口包含四件独立的事：

1. Clip-Higher：下界 εlow=0.2，上界 εhigh=0.28，给探索留出更宽的正向空间。
2. Dynamic Sampling：在原始任务奖励上检查组内差异，全对/全错组不进入更新，继续补采；有最大尝试次数，耗尽会显式记录 skipped，防止无限等待。
3. Token-level reduction：把所有有效生成 token 的损失相加，再除以有效 token 总数。
4. Overlong shaping：在 soft_length 到 max_new_tokens 区间线性增加负奖励；可把没有 EOS 的截断回答排除出 loss。

准备数据时先剥离原数据要求 Answer: 的模板，统一到本项目答案协议，再按题面去重、移除冲突答案。配置的 loss_type=dapo 本身并不自动完成其他三项机制；所以章节默认使用原生循环。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 04-dapo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 04-dapo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 04-dapo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[trainers.py](../src/agentic_rl/trainers.py)、[losses.py](../src/agentic_rl/losses.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-04-dapo`。

## 指标与对照实验

分别改变动态采样、Clip-Higher、token 归约和长度奖励，记录有效组数、补采次数、耗时和正确率。TRL 对照入口提供按有效 token 归一化的 DAPO loss。观察动态补采和长度处理请用本章的 verl 入口。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

小模型可能连续生成全错组。达到补采上限且没有有效组时，本次更新会跳过，可以同时查看 `sampling/attempts` 和 `update/skipped`。

## 一手资料

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。`config.yaml` 默认使用 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 04-dapo/verl.yaml --smoke --verl-workers 2
# 使用 GPU 配置前先安装 CUDA 版 verl 环境
# .venv/bin/arl train 04-dapo/verl-gpu.yaml --verl-workers 2
```

verl 用 `DataProto` 传递轨迹，由 actor 执行更新。在轨迹中可以查看 token、mask、旧概率和轮次记录。[verl 后端代码](../src/agentic_rl/verl_backend/) · [安装与恢复](../docs/VERL.md)

# PPO：Actor、Critic、GAE 与 KL

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：TRL（`config.yaml` → `backend: trl`）。** 裁剪的四种方向已在 [损失与策略梯度基础](../01-foundations/TUTORIAL.md) 讲过；本章重点是角色分工、GAE，以及一条完整 batch 的 reward→loss 数据链。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

在本章的语言模型 PPO 中，actor 生成回答，奖励函数给出任务分数，reference 提供逐 token 的 KL 约束。critic 估计剩余回报，GAE 把这些信号转成每个动作的优势。[一步 TD、完整回报和 GAE 的两步手算](TUTORIAL.md#gae)

一步 TD 将当步奖励和下一状态的折扣估值相加，再减去动作前的估值。GAE 用 λ 调节之后几步的误差传回多少。actor 用优势计算裁剪策略损失，critic 拟合回报目标。一批回答可在 TRL 中用于多轮小批量更新，其采样时的 logprob 和估值在这期间保持固定。

为了先学习 RLVR，本章用确定性数学 verifier 实现 TRL reward_model 接口，不额外依赖一个已训练的偏好 RM；可设置 reward_model 路径替换为 AutoModelForSequenceClassification。原生对照把 value head、GAE 和一次 policy/value 更新展开。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与 TRL 后端一致
source .venv/bin/activate
.venv/bin/arl train 02-ppo/config.yaml --smoke --backend trl
# 等价薄入口
python 02-ppo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 02-ppo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 02-ppo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

跨后端对照属于进阶，不要在第一次运行时同时学习算法与后端：

```bash
.venv/bin/arl train 02-ppo/config.yaml --smoke --backend native
.venv/bin/arl train 02-ppo/verl.yaml --smoke --verl-workers 2
```

从上面的命令跟读代码时，可以打开[trl_backend.py](../src/agentic_rl/trl_backend.py)、[trainers.py](../src/agentic_rl/trainers.py)、[losses.py](../src/agentic_rl/losses.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-02-ppo`。

## 指标与对照实验

同一条轨迹分别按真正终止和采样截断计算 GAE，观察末状态价值如何影响早先动作。训练时一起看任务奖励、KL、概率比、裁剪比例以及 actor 和 critic 的梯度。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

native 后端每批做一次更新。要在同一批回答上进行多轮小批量更新，使用默认的 TRL 入口。

## 一手资料

- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [TRL PPOTrainer](https://huggingface.co/docs/trl/v0.25.1/en/ppo_trainer)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。`config.yaml` 默认使用 TRL，verl 作为并列入口；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。安装与恢复见 [verl 指南](../docs/VERL.md)。

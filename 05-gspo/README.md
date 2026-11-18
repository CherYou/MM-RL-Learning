# GSPO：序列级重要性比率

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

从序列概率推到几何平均，理解裁剪粒度与梯度。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

把每个有效生成 token 的 log-ratio 先求均值，再取指数：sᵢ=exp(mean_t(log πθ−log π_old))。这相当于概率比的几何平均，不是整段概率比的乘积。

同一个 sequence ratio 参与整个回答的 clipped objective；组内 advantage 仍由回答 reward 得到。默认 TRL 配置使用 importance_sampling_level="sequence"、loss_type="grpo"、εlow=0.0003、εhigh=0.0004。原生后端显式实现同样的序列聚合，适合和 token GRPO 比较。

padding、prompt 和工具 observation 不能出现在长度均值的分母里；否则长工具返回会改变模型实际生成动作的权重。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 05-gspo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 05-gspo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 05-gspo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[losses.py](../src/agentic_rl/losses.py)、[trl_backend.py](../src/agentic_rl/trl_backend.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-05-gspo`。

## 指标与对照实验

构造两个 token 的比率 2 和 0.5，应得到 sequence ratio=1。再固定采样预算比较 GRPO 与 GSPO 的 clip fraction、长度敏感性和训练稳定性。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

使用 TRL 时，`importance_sampling_level="sequence"` 决定按回答聚合概率比，`loss_type` 取 `"grpo"`。比较 GRPO 与 GSPO 时应核对这两个字段。

## 一手资料

- [GSPO 原论文](https://arxiv.org/abs/2507.18071)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。`config.yaml` 默认使用 TRL，verl 作为并列入口；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 05-gspo/verl.yaml --smoke --verl-workers 2
# 使用 GPU 配置前先安装 CUDA 版 verl 环境
# .venv/bin/arl train 05-gspo/verl-gpu.yaml --verl-workers 2
```

verl 用 `DataProto` 传递轨迹，由 actor 执行更新。在轨迹中可以查看 token、mask、旧概率和轮次记录。[verl 后端代码](../src/agentic_rl/verl_backend/) · [安装与恢复](../docs/VERL.md)

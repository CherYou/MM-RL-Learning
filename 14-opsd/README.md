# OPSD：固定 step-0 Self-Teacher

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 概念入口是 [General OPD](../13-general-opd/TUTORIAL.md)；本章差在 **solution 信息条件 + step-0 教师**。评估时不得保留 solution。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [General OPD](../13-general-opd/TUTORIAL.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

初始化时复制 Student 得到固定 Teacher。Student 只看到题目；Teacher 的 prompt 额外包含参考 solution。随后 Teacher 对 Student 真实生成的 completion 复评，并构造 token-level reverse-KL 信号。

因此，即使最初 Student/Teacher 权重相同，两个条件分布仍不同。Teacher prompt 变长后，completion 的相对位置必须重新对齐，但 completion IDs 本身保持原样。`teacher_logprobs(..., privileged="solution")` 明确执行这一步。

训练数据取自固定版本的 Openthoughts_math_30k_opsd。可另用 AIME 2025 的 30 道题评测学生独立解题的表现。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 14-opsd/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 14-opsd/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 14-opsd/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[trainers.py](../src/agentic_rl/trainers.py)、[models.py](../src/agentic_rl/models.py)、[losses.py](../src/agentic_rl/losses.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-14-opsd`。

## 指标与对照实验

消融 Teacher 可见 solution 与不可见 solution；确认固定 Teacher 参数从 step 0 到最后都不变，Student 和评测 prompt 没有参考解答。记录采样 KL、生成长度与数学正确率。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

学习子集中的参考解有长有短。长解超过教师上下文预算时，选择模型支持的更长上下文，或筛选能放下参考解和学生回答的样本。截断后需要重新核对 token 对齐。

## 一手资料

- [OPSD 数学数据集](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。`config.yaml` 默认使用 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 14-opsd/verl.yaml --smoke --verl-workers 2
# 使用 GPU 配置前先安装 CUDA 版 verl 环境
# .venv/bin/arl train 14-opsd/verl-gpu.yaml --verl-workers 2
```

verl 用 `DataProto` 传递轨迹，由 actor 执行更新。在轨迹中可以查看 token、mask、旧概率和轮次记录。[verl 后端代码](../src/agentic_rl/verl_backend/) · [安装与恢复](../docs/VERL.md)

# General OPD：学生采样、教师评分

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**蒸馏分支概念入口；默认后端：verl（`verl.yaml`）。** 先回答“谁生成动作、谁给反馈”，再读 reverse KL 采样 surrogate。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [医学案例](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

Student 自行采样 y~π_old；Teacher 在相同问题和相同 completion token 上求 logprob。逐 token 信号 d=log π_old−log π_teacher 对参数停止梯度。使用 L=mean(exp(log πθ−log π_old) × d)，它的梯度就是 sampled reverse-KL 的 score-function 估计。

Teacher 不生成训练答案；不是拿 Teacher 自己的轨迹做普通 SFT。采样分母和 teacher 分数只在 optimizer step 前计算一次。不同大小模型可以蒸馏，但 token-level 实现要求相同 tokenizer vocabulary；代码会拒绝不一致的 vocab。

运行前按[数据说明](../docs/DATA.md)准备 DeepMath 子集。训练从本地 JSONL 读取题目。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 13-general-opd/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 13-general-opd/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 13-general-opd/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[trainers.py](../src/agentic_rl/trainers.py)、[losses.py](../src/agentic_rl/losses.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-13-general-opd`。

## 指标与对照实验

先检查 old/teacher logprob 与生成 token 数严格相同；检查 Teacher 无梯度；然后比较更强 Teacher 与同权重 Teacher。采样 reverse-KL 的有限样本估计可能为负，不能因此判断数学定义的 KL 为负。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

蒸馏信号来自学生实际采到的 token。Teacher 和 Student 必须对同一词表中的同一 token、在相应前缀下评分。默认使用较小的 Teacher 以降低显存和采样成本。

## 一手资料

- [DeepMath-103K 数据集](https://huggingface.co/datasets/zwhe99/DeepMath-103K)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。`config.yaml` 默认使用 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 13-general-opd/verl.yaml --smoke --verl-workers 2
# 使用 GPU 配置前先安装 CUDA 版 verl 环境
# .venv/bin/arl train 13-general-opd/verl-gpu.yaml --verl-workers 2
```

verl 用 `DataProto` 传递轨迹，由 actor 执行更新。在轨迹中可以查看 token、mask、旧概率和轮次记录。[verl 后端代码](../src/agentic_rl/verl_backend/) · [安装与恢复](../docs/VERL.md)

# General OPD：学生采样、教师评分

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**蒸馏分支概念入口；默认后端：verl（`verl.yaml`）。** 先回答“谁生成动作、谁给反馈”，再读 reverse KL 采样 surrogate。

[学习路线](../../docs/LEARNING_PATH.md) · [总目录](../../README.md) · [医学案例](../README.md) · [框架设计](../../docs/ARCHITECTURE.md)

## 本地实现

Student 自行采样 y~π_old；Teacher 在相同问题和相同 completion token 上求 logprob。逐 token 信号 d=log π_old−log π_teacher 对参数停止梯度。使用 L=mean(exp(log πθ−log π_old) × d)，它的梯度就是 sampled reverse-KL 的 score-function 估计。

Teacher 不生成训练答案；不是拿 Teacher 自己的轨迹做普通 SFT。采样分母和 teacher 分数只在 optimizer step 前计算一次。不同大小模型可以蒸馏，但 token-level 实现要求相同 tokenizer vocabulary；代码会拒绝不一致的 vocab。

DeepMath 使用公开原始数据的固定 revision，实验子集已下载并保存为 JSONL，不依赖 ModelScope 或远程训练账户。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 02-opd/general-opd/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 02-opd/general-opd/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 02-opd/general-opd/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `trainers.py / losses.py`，位于 `../../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-general-opd`，已存在的目录会拒绝覆盖。

## 验证思路与消融

先检查 old/teacher logprob 与生成 token 数严格相同；检查 Teacher 无梯度；然后比较更强 Teacher 与同权重 Teacher。采样 reverse-KL 的有限样本估计可能为负，不能因此判断数学定义的 KL 为负。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

只有 sampled-token reverse KL，不计算全词表 KL。默认较小 Teacher 用来降低后续实验成本，不等同于上游 Qwen3.5 的规模。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 2](https://huggingface.co/datasets/zwhe99/DeepMath-103K)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 02-opd/general-opd/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 02-opd/general-opd/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor。verl 与原生均保存 token、mask、旧概率和轮次日志。

# OPSD：固定 step-0 Self-Teacher

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 概念入口是 [General OPD](../02-opd/general-opd/TUTORIAL.md)；本章差在 **solution 信息条件 + step-0 教师**。评估时不得保留 solution。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [General OPD](../02-opd/general-opd/TUTORIAL.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

初始化时复制 Student 得到固定 Teacher。Student 只看到题目；Teacher 的 prompt 额外包含参考 solution。随后 Teacher 对 Student 真实生成的 completion 复评，并构造 token-level reverse-KL 信号。

因此，即使最初 Student/Teacher 权重相同，两个条件分布仍不同。Teacher prompt 变长后，completion 的相对位置必须重新对齐，但 completion IDs 本身保持原样。`teacher_logprobs(..., privileged="solution")` 明确执行这一步。

本章保留 Openthoughts_math_30k_opsd 的固定来源 revision，同时准备 30 道 AIME 2025 作为可选独立评测。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 04-opsd/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 04-opsd/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 04-opsd/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `trainers.py / models.py / losses.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-04-opsd`，已存在的目录会拒绝覆盖。

## 验证思路与消融

消融 Teacher 可见 solution 与不可见 solution；确认固定 Teacher 参数从 step 0 到最后都不变，Student 和评测 prompt 没有参考解答。记录采样 KL、生成长度与数学正确率。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

OPSD 数据本地是学习子集；未复现论文或上游大模型结果。长参考解可能超出上下文预算，应增加模型支持的 context 或筛选数据，不能截断后假装概率仍对齐。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 2](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 04-opsd/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 04-opsd/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor。verl 与原生均保存 token、mask、旧概率和轮次日志。

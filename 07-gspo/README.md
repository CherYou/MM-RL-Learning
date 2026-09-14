# GSPO：序列级重要性比率

**初次学习请先读 [新手算法详解](TUTORIAL.md)：直觉、公式、loss、代码对应与练习答案。**

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

把每个有效生成 token 的 log-ratio 先求均值，再取指数：sᵢ=exp(mean_t(log πθ−log π_old))。这相当于概率比的几何平均，不是整段概率比的乘积。

同一个 sequence ratio 参与整个回答的 clipped objective；组内 advantage 仍由回答 reward 得到。默认 TRL 配置使用 importance_sampling_level="sequence"、loss_type="grpo"、εlow=0.0003、εhigh=0.0004。原生后端显式实现同样的序列聚合，适合和 token GRPO 比较。

padding、prompt 和工具 observation 不能出现在长度均值的分母里；否则长工具返回会改变模型实际生成动作的权重。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 07-gspo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 07-gspo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 07-gspo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `losses.py / trl_backend.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-07-gspo`，已存在的目录会拒绝覆盖。

## 验证思路与消融

构造两个 token 的比率 2 和 0.5，应得到 sequence ratio=1。再固定采样预算比较 GRPO 与 GSPO 的 clip fraction、长度敏感性和训练稳定性。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

把 GRPO 配置文件中的 loss 改成字符串 gspo 不是 TRL API；需要改 importance_sampling_level。本项目已经分别处理两种后端的字段。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 1](https://arxiv.org/abs/2507.18071)
- [资料 2](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 保留 TRL，verl 作为并列入口；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 07-gspo/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 07-gspo/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor；CPU 逐入口证据见 [verl 审计](../reports/verl-audit-latest.json)。verl 与原生均保存 token、mask、旧概率和轮次日志。

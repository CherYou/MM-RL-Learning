# PPO：Actor、Critic、GAE 与 KL

**初次学习请先读 [新手算法详解](TUTORIAL.md)：直觉、公式、loss、代码对应与练习答案。**

这是本地新增章节。 · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

完整 PPO 包含当前 policy、冻结 reference、奖励来源与 value model。默认入口使用真实 TRL PPOTrainer，actor 负责生成，critic 估值，GAE 把末端任务奖励与逐 token KL 惩罚传回前面的动作。

delta_t=r_t+gamma V(s_{t+1})−V(s_t)，A_t=delta_t+gamma lambda A_{t+1}。优化目标由 clipped policy loss 与 clipped value regression 组成。旧 logprob、旧 value、advantage 和 return 在一轮 PPO 内保持固定，TRL 负责多个 PPO epoch 的 minibatch 更新。

为了先学习 RLVR，本章用确定性数学 verifier 实现 TRL reward_model 接口，不额外依赖一个已训练的偏好 RM；可设置 reward_model 路径替换为 AutoModelForSequenceClassification。原生对照把 value head、GAE 和一次 policy/value 更新展开。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 001-ppo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 001-ppo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 001-ppo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `trl_backend.py / trainers.py / losses.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-001-ppo`，已存在的目录会拒绝覆盖。

## 验证思路与消融

先验证 GAE 终止与截断 bootstrap，再检查 actor 和 critic 都得到梯度。训练时看 KL、policy/value loss、ratio、clip_fraction 和 task reward；不要把 PPO clipped surrogate 当成 GRPO 的完整替代说明。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

微型模型没有数学能力；奖励模型接口已实际运行但正式 RLHF 效果未验证。原生对照每轮一次更新，标准多 epoch PPO 用 TRL 默认入口。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 1](https://arxiv.org/abs/1707.06347)
- [资料 2](https://huggingface.co/docs/trl/v0.25.1/en/ppo_trainer)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 保留 TRL，verl 作为并列入口；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 001-ppo/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 001-ppo/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor；CPU 逐入口证据见 [verl 审计](../reports/verl-audit-latest.json)。verl 与原生均保存 token、mask、旧概率和轮次日志。

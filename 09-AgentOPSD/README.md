# AgentOPSD：技能 Self-Teacher 与轮级信用

**初次学习请先读 [新手算法详解](TUTORIAL.md)：直觉、公式、loss、代码对应与练习答案。**

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

Student 在没有技能的上下文里完成真实 rollout；optimizer 更新前，用同一参数快照加上 general skill 与任务类型技能，对原动作逐 token 复评。Teacher 不改变 Student 已执行动作，也不重新运行环境。

每个 turn 汇总 evidence=sum(log π_skill−log π_student)，累积 evidence 使用 gamma=0.95 衰减；通过 sigmoid belief 的变化形成轮级信用。最终 direction 仍来自终局 GRPO advantage，局部权重被限制在 [0.8,1.2]，reshape_lambda=0.5，所以最终幅度落在原始 advantage 的 [0.9,1.1] 倍，不能翻转符号。

`agentopsd.py` 直接迁移上游的确定性信用计算，并保留所有中间量。SkillBank 来源及任务映射保存在 data/skills/；teacher 前缀改变后，原生成 token IDs 直接拼接并按 offset 对齐。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 09-AgentOPSD/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 09-AgentOPSD/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 09-AgentOPSD/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `agentopsd.py / trainers.py / rollout.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-09-AgentOPSD`，已存在的目录会拒绝覆盖。

## 验证思路与消融

消融关闭技能或关闭重塑；检查 Teacher 与 Student 是同一更新前快照；检查 success 轨迹优势不会变负，退化组保持零；评测始终不注入技能。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

沿用上游的有界 advantage 机制，未逐项实现论文所有熵正则和分布式工程。检查过信用公式及同 token 复评路径，未验证上游成功率。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 1](https://arxiv.org/abs/2608.05987)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 09-AgentOPSD/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 09-AgentOPSD/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor；CPU 逐入口证据见 [verl 审计](../reports/verl-audit-latest.json)。verl 与原生均保存 token、mask、旧概率和轮次日志。

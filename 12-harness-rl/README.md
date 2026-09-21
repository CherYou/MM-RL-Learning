# Harness-RL：调用记录与参数分区

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 教程命令与此一致。本章重点：最小 action/args 调用；token mask≠参数 mask。裁剪与组内优势见 [GRPO](../01-grpo/TUTORIAL.md) / [preliminary](../preliminary/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

本章对应 Jiang 等人的 Harness-RL（arXiv:2608.29641），不是 lm-evaluation-harness，也不是只包一层工具调用循环。

第一部分是 interface-level trajectory construction。每次中央模型调用保存确切 token_in、token_out、采样 logprob、rollout/session/call ID 和 parent 关系。按 session 建 token prefix trie；共享前缀合并节点，重写上下文保留分支，但每个实际调用仍保持独立输出 span，不能重复训练 replayed context。

第二部分是 CAPO。分别标出 JSON action 与 args 的生成 token；在成功 probing 轨迹上统计 MLP 单元的正激活，按两个 token 类型各取 Top-K 单元，构造参数 mask。反向时分别求 action loss 和 args loss 的梯度，再投影到各自参数子集；交集接收两项梯度之和，未选参数冻结。

本地中央代理支持 Reason、Search、Calculate、Summary，检索/计算 worker 冻结。奖励由最终 answer F1 与有效终止 artifact 决定，额外过程奖励按同题同轮的候选组归一化后分配到当前调用 token。调用日志、prefix trees 与 CAPO partitions 都落盘。

默认 data/harness/probes.jsonl 是明确标记的人工构造、工具验证成功种子，用于在随机模型阶段启动激活检查；正式实验应先运行 collect_harness_probes.py，从目标策略成功 rollouts 收集 probing 数据。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 12-harness-rl/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 12-harness-rl/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 12-harness-rl/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `harness.py / trainers.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-12-harness-rl`，已存在的目录会拒绝覆盖。

## 验证思路与消融

专项测试不同 session 不合并、context rewrite 产生分支、action/args mask 不相交、未选择参数无梯度。比较 capo=true/false、Top-K 比例和 process_coef，观察两类梯度 cosine 与任务 F1。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

本实现选择 MLP 单元作为结构分区，冻结其他参数，暂不兼容 CAPO+LoRA；只实现 central-only、同步冻结 worker 的学习规模运行。官方 StackPlanner/slime 多机调度、joint multi-agent 训练和 7 benchmark 的分数不在本次已验证范围。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 1](https://arxiv.org/html/2608.29641v1)

## 后端说明

本章默认入口即 `verl.yaml`。完整安装、恢复与 GPU 说明见 [VERL.md](../docs/VERL.md)；不要在第一次运行时同时学习算法与后端切换。

# Harness-RL：调用记录与参数分区

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 本章重点：最小 action/args 调用；token mask≠参数 mask。裁剪与组内优势见 [GRPO](../03-grpo/TUTORIAL.md) / [损失与策略梯度基础](../01-foundations/TUTORIAL.md)。

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
.venv/bin/arl train 17-harness-rl/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 17-harness-rl/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 17-harness-rl/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[harness.py](../src/agentic_rl/harness.py)、[trainers.py](../src/agentic_rl/trainers.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-17-harness-rl`。

## 指标与对照实验

专项测试不同 session 不合并、context rewrite 产生分支、action/args mask 不相交、未选择参数无梯度。比较 capo=true/false、Top-K 比例和 process_coef，观察两类梯度 cosine 与任务 F1。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

训练只更新中央策略，worker 保持冻结。CAPO 以 MLP 单元划分参数，目前无法与 LoRA 同用。若要研究多个 agent 联合更新，需要另一套训练配置与调度实现。

## 一手资料

- [Harness-RL 原论文](https://arxiv.org/html/2608.29641v1)

## 后端说明

CPU 和 CUDA 环境的安装、训练中断后的恢复步骤见 [verl 指南](../docs/VERL.md)。

# AgentOPSD：轮次信用与 Skill 自教师

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 本章重点：evidence→belief→有界重加权；不翻转终局方向。裁剪与组内优势见 [GRPO](../03-grpo/TUTORIAL.md) / [损失与策略梯度基础](../01-foundations/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

Student 在没有技能的上下文里完成真实 rollout；optimizer 更新前，用同一参数快照加上 general skill 与任务类型技能，对原动作逐 token 复评。Teacher 不改变 Student 已执行动作，也不重新运行环境。

每个 turn 汇总 evidence=sum(log π_skill−log π_student)，累积 evidence 使用 gamma=0.95 衰减；通过 sigmoid belief 的变化形成轮级信用。最终 direction 仍来自终局 GRPO advantage，局部权重被限制在 [0.8,1.2]，reshape_lambda=0.5，所以最终幅度落在原始 advantage 的 [0.9,1.1] 倍，不能翻转符号。

`agentopsd.py` 直接迁移上游的确定性信用计算，并保留所有中间量。SkillBank 来源及任务映射保存在 data/skills/；teacher 前缀改变后，原生成 token IDs 直接拼接并按 offset 对齐。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 15-agentopsd/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 15-agentopsd/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 15-agentopsd/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[agentopsd.py](../src/agentic_rl/agentopsd.py)、[trainers.py](../src/agentic_rl/trainers.py)、[rollout.py](../src/agentic_rl/rollout.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-15-agentopsd`。

## 指标与对照实验

消融关闭技能或关闭重塑；检查 Teacher 与 Student 是同一更新前快照；检查 success 轨迹优势不会变负，退化组保持零；评测始终不注入技能。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

本章采用有界的轮次优势修正，教师在复评时读入技能提示。评估学生时移除技能提示，比较启用和关闭修正后的任务成功率。论文中的其他熵正则项未接入这条更新路径。

## 一手资料

- [AgentOPSD 原论文](https://arxiv.org/abs/2608.05987)

## 后端说明

CPU 和 CUDA 环境的安装、训练中断后的恢复步骤见 [verl 指南](../docs/VERL.md)。

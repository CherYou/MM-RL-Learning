# HER：具身强化学习基础

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

从稀疏奖励走到目标重标记，再接回 TD3 的价值更新。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [前置知识](../preliminary/FOUNDATIONS.md) · [共享环境与验证说明](../docs/EMBODIED.md)

本章重点是同一轨迹的 future goal 重标记，底层使用 TD3。使用真实 `FetchReach-v4` / MuJoCo，策略输入是 10 维机械观察与 3 维目标，输出 4 维归一化连续动作。它是理解机器人/VLA 强化学习的基础实验，没有训练视觉语言编码器。

## 安装与执行

所有命令在仓库根目录运行，默认只用 CPU：

```bash
uv sync --frozen --extra dev --extra cpu --extra embodied
.venv/bin/python 15-her/train.py --smoke --output runs/my-her-smoke
.venv/bin/python 15-her/eval.py --checkpoint runs/my-her-smoke/checkpoint-final --episodes 10 --seed 20000
```

训练实时收集环境经历；无需下载预训练模型或先生成离线数据。去掉 `--smoke` 使用 5000 个环境步，其中前 1000 步随机预热。

`--smoke` 使用真实物理环境和小网络（64 隐藏维、batch 32），运行 300 个环境步。正式默认隐藏维为 128、batch 128。每次输出必须用新目录。

可选重新采集共享数据（写到新路径，四章的准备入口等价）：

```bash
.venv/bin/python 15-her/prepare_data.py --output data/embodied/fetch-reach/my-train.npz --episodes 40
```

本章训练不读取这个离线文件；生成数据用于理解 transition 格式或供 IQL 实验使用。

## 文件与代码对应

| 文件 | 作用 |
| --- | --- |
| [config.yaml](config.yaml) | 环境、随机种子、网络、优化与评估配置 |
| [train.py](train.py) / [eval.py](eval.py) | 独立可执行入口，可从任意工作目录调用 |
| [prepare_data.py](prepare_data.py) | 采集带来源元数据的真实仿真经历 |
| [agents.py](../src/agentic_rl/embodied/agents.py) | 各算法的 loss 与更新次序 |
| [networks.py](../src/agentic_rl/embodied/networks.py) | 双 Q 与连续动作策略 |
| [replay.py](../src/agentic_rl/embodied/replay.py) | 回放、HER 和固定数据校验 |
| [runner.py](../src/agentic_rl/embodied/runner.py) | 环境循环、日志、独立评估与检查点 |

`backend: embodied` 使用本地 PyTorch 连续控制实现。共用配置里 SAC 专用 alpha、TD3 专用平滑/延迟、HER 专用 her_k、IQL 专用 expectile/weight 等只在对应分支生效。LLM 的 token mask、TRL Trainer 和 verl rollout 配置不适用于这些连续动作示例。

## 看结果

`runs/<实验>/` 保存 `metrics.jsonl`、TensorBoard、episode 记录、配置、`validation.json` 和 `checkpoint-final/agent.pt`。训练结束后会从磁盘重新加载，核对策略权重与确定性评估逐 episode 完全一致。

```bash
.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

浏览器打开 `http://127.0.0.1:6006`；远程服务器按[根 README](../README.md#本地日志)转发端口。评估同时记录“episode 内曾到达目标”与“最后一步仍在目标内”，不要混成一个成功定义。

环境、数据准备与评估方法见 [具身实验指南](../docs/EMBODIED.md)。建议先用小预算检查训练流程，再通过独立评估种子和更充分的训练观察学习效果。

# 具身强化学习实验：SAC、TD3、HER、IQL

这四章选取连续控制、稀疏目标奖励、离线数据学习中的经典基础方法，重点是它们在机器人 RL 中的问题代表性和后续方法的基础作用。原论文分别为 [SAC](https://arxiv.org/abs/1801.01290)、[TD3](https://arxiv.org/abs/1802.09477)、[HER](https://arxiv.org/abs/1707.01495)、[IQL](https://arxiv.org/abs/2110.06169)。SAC 实现采用[后续双 Q 与自动温度版本](https://arxiv.org/abs/1812.05905)。

## 先明确本实验学什么

使用真实 `FetchReach-v4` / MuJoCo 到达任务，输入为 10 维机器人观察加 3 维目标坐标，输出为 4 维归一化动作。前三维对应末端位置控制，第四维是夹爪通道（Reach 限制夹爪）。没有视觉、语言编码器，也没有动作 chunk；因而这是 VLA 所需的强化学习基础，不是已训练完成的 VLA 系统。

所有默认配置使用稀疏奖励：离目标距离大于 0.05 时为 -1，否则为 0。一个 episode 的时间上限为 50 步；到达目标通常不触发立即终止。奖励和接口以[环境文档](https://robotics.farama.org/envs/fetch/reach/)及固定依赖版本为依据。

## 一次安装，四个训练入口

固定依赖是 Gymnasium 1.3.0、Gymnasium-Robotics 1.4.2、MuJoCo 3.3.7，已经进入根 `uv.lock` 的可选 `embodied` extra。使用 Python 3.12、CPU PyTorch；不用显示窗口或 GPU。

```bash
uv sync --frozen --extra dev --extra cpu --extra embodied
.venv/bin/python 13-sac/train.py --smoke --output runs/my-sac
.venv/bin/python 14-td3/train.py --smoke --output runs/my-td3
.venv/bin/python 15-her/train.py --smoke --output runs/my-her
.venv/bin/python 16-iql/train.py --smoke --output runs/my-iql
```

`--smoke` 运行 300 步、小 MLP、真实物理与奖励。它不同于 LLM 的随机 GPT-2/debug reward smoke。去掉该选项，SAC/TD3/HER 为 5000 环境步（1000 步预热），IQL 为 5000 梯度更新。

根 CLI 也可执行 `arl train 13-sac/config.yaml --smoke`。`backend: embodied` 会分派到连续控制 runner；无需安装 verl 来运行这些章节。LLM 入口由 `scripts/smoke_all.py` 批量检查，具身章节可分别运行上面的命令。

## 数据是什么，怎样准备

在线三章边交互边建立 replay，不预先读取示范数据。HER 为每条原 transition 增加 4 条同轨迹 future goal 副本；原始状态、动作和真实未来不改变，当前/下一目标同时替换，奖励由环境重新计算。

IQL 读取 [train.npz](../data/embodied/fetch-reach/train.npz) 与 [train.json](../data/embodied/fetch-reach/train.json)。已采集 40 个 episode、2000 条真实物理 transition；按 episode 混合 75% 带噪声比例控制器与 25% 均匀随机策略。控制器用当前末端与目标之差构造位移命令，不是学习出的策略。这是程序示范，不是人类遥操作或公开 benchmark 数据。

| 字段 | 单条形状 | 含义 |
| --- | --- | --- |
| observation / next_observation | 10 | 动作前后的机械观察 |
| achieved_goal / next_achieved_goal | 3 | 实际到达位置 |
| desired_goal / next_desired_goal | 3 | 当前条件目标 |
| action | 4 | 实际执行的归一化动作 |
| reward | 标量 | 环境返回的真实奖励 |
| terminated / truncated | 布尔 | 真正终态 / 外部时间边界 |
| episode / time | 整数 | 轨迹身份与时间顺序 |
| relabeled / future_time | 布尔 / 整数 | 原始文件为 false/-1，HER 回放时才产生重标记 |

元数据记录所有字段形状、dtype、种子、依赖版本和 SHA256。采集种子为 42–81，原数据中 32/40 个 episode 曾到达目标，这只是采集器的统计，不能归为 IQL 策略成绩。

```bash
# 可选重新采集；选新路径，不覆盖随仓库提供的数据
.venv/bin/python 16-iql/prepare_data.py --output data/embodied/fetch-reach/my-train.npz --episodes 40
.venv/bin/python 16-iql/train.py --dataset data/embodied/fetch-reach/my-train.npz --output runs/my-iql-new-data
```

IQL 的训练循环没有在线环境对象，评估经历也不会写回 replay。数据 loader 检查内容 hash、字段、动作范围、reward 类型及评估种子隔离。

## 如何评估学习效果

训练前后使用同一组独立评估种子，比较平均回报、episode 内曾到达目标的比例，以及末步仍在目标内的比例。这三个指标分别反映累计表现、是否达到目标和是否稳定保持目标。

SAC、TD3 和 HER 的预算以环境交互步计，IQL 的预算以固定数据上的梯度更新计。比较算法时还需交代数据来源、训练预算与种子；小预算下的单次成功率容易波动，建议使用多个训练种子。

## 保存、评估与 TensorBoard

```bash
.venv/bin/python 16-iql/eval.py \
  --checkpoint runs/my-iql/checkpoint-final \
  --episodes 10 --seed 20000 --output reports/my-iql-evaluation.json
.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

输出包含 `config.json`、`metrics.jsonl`、`trajectories.jsonl`、TensorBoard events、`validation.json` 与 `checkpoint-final/agent.pt`。该检查点是 PyTorch 状态，包括网络、目标网络、优化器和 RNG；不是 Hugging Face 模型目录。当前入口支持重载评估，**没有提供完整训练续跑接口**，因为在线 replay 与环境状态未一起保存。

`loss/*` 按日志区间内实际执行的更新取平均，TD3/HER 不会因为延迟 actor 而漏掉 actor loss。训练开始/结束才有 `eval/*` 两组点；每 100 步的 loss 曲线不能被解读成每 100 步都完成了策略评估。

远程浏览器需在本机建立 SSH 端口转发，见[根 README](../README.md#本地日志)。如果 TensorBoard 已由项目服务启动，直接打开现有 6006 端口即可，不必启动第二个进程。

## 机制检查怎样复跑

```bash
.venv/bin/python -m pytest -q tests/test_embodied.py
```

数学测试覆盖终止 bootstrap、expectile 的手算最优值、tanh 密度与饱和梯度、TD3 更新延迟、SAC 温度、IQL 仅查询记录动作、HER 边界及重标记一致性、检查点优化状态恢复。训练生成的 `validation.json` 记录参数变化、更新次数和重载一致性；使用各章 `eval.py` 在独立种子上评估保存的模型。

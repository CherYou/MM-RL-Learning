# SAC：一边提高回报，一边保留可用的动作选择

先读 [MDP、Q 值与连续动作密度](../preliminary/FOUNDATIONS.md)，再读这一章。SAC（Soft Actor-Critic）是连续控制的经典 off-policy 方法；这里采用后续常见的双 Q、无独立 V 网络、自动温度版本。它为机器人 RL 提供基础，不是一个视觉语言动作模型。[原始论文](https://arxiv.org/abs/1801.01290)与[算法及应用论文](https://arxiv.org/abs/1812.05905)给出方法依据；以下例子、推导组织和代码讲解按本仓库实现独立撰写。

![SAC 的动作采样、双 critic、回放与熵示意](../docs/assets/algorithms/sac.png)

图中多根箭头表示同一个策略可以采出不同连续动作，双表盘表示两个 Q 估计，天平表示奖励与熵之间的权衡。表盘和天平没有数值刻度，是概念图，不是实验曲线。

## 1. 为什么机械臂不应该过早只相信一种动作

假设目标在前方，向左绕和向右绕都能到达。刚开始 critic 估得不准，某次碰巧说左边好一点；如果策略立刻把其他动作概率压到接近零，就可能再也没有足够的数据纠正这个偏好。

SAC 在“获得任务奖励”之外，给“保留动作分布的多样性”一个明确价值。它不是让机器人无目的地乱动，而是让动作质量与确定性一起接受优化。

$$J(\pi)=\mathbb E\left[\sum_t\gamma^t\{r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))\}\right],\qquad
\mathcal H(\pi)=\mathbb E_{a\sim\pi}[-\log\pi(a)].$$

$\gamma$ 是折扣；$\alpha>0$ 决定熵的相对重要性；$\pi_\theta$ 是 actor。连续动作的熵是微分熵，可能为负，不能用“负熵一定是 bug”判断。

## 2. 一次训练更新到底使用什么数据

环境返回 $(s,a,r,s',d)$，其中 $d$ 只指真正的任务终止。旧交互存进 replay buffer，之后随机抽一个 batch；所以 SAC 不要求每个更新都用刚生成的动作。**critic 拟合记录动作 $a$，actor 则在记录状态 $s$ 上重新采一个动作。** 两者不要混淆。

两个 critic $Q_{\phi_1},Q_{\phi_2}$ 各自预测动作价值；慢速目标副本记作 $\bar\phi_1,\bar\phi_2$。对下一状态采样 $a'\sim\pi_\theta(\cdot\mid s')$，构造：

$$y=r+\gamma(1-d)\left[\min_{j=1,2}Q_{\bar\phi_j}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right].$$

$$L_Q=\mathbb E_{\mathcal B}\left[(Q_{\phi_1}(s,a)-y)^2+(Q_{\phi_2}(s,a)-y)^2\right].$$

取较小 Q 是抑制高估的设计，不保证这个最小值就是真实值。本次 $y$ 全部停止梯度。我们的 `truncated` 只切断 episode，不在上式关闭 bootstrap。

## 3. Actor loss、动作变换与温度 loss

固定 critic 权重，在状态 $s$ 上用可求导采样得到 $\tilde a$：

$$L_\pi=\mathbb E_{s\sim\mathcal B,\tilde a\sim\pi_\theta}
\left[\alpha\log\pi_\theta(\tilde a\mid s)-\min_j Q_{\phi_j}(s,\tilde a)\right].$$

最小化它，一方面希望 Q 高，另一方面希望熵高。这里的动作通过重参数化 $u=\mu+\sigma\epsilon,\tilde a=\tanh u$ 产生，梯度能沿 $Q\to a\to\theta$ 回传。`rsample()` 与普通 `sample()` 的差别在这里有实际作用。

`SquashedPolicy.corrected_log_prob` 计算高斯密度并减去 tanh 的 log-Jacobian。否则熵项和梯度都会错。动作已经归一化到 $[-1,1]^4$，没有额外的缩放 Jacobian；迁移到别的执行范围时需要重新处理。

温度用 $\ell=\log\alpha$ 参数化，本地实现采用常用的 log-temperature surrogate：

$$L_\alpha=-\mathbb E\left[\ell\;\operatorname{stopgrad}(\log\pi(\tilde a\mid s)+\mathcal H_{\rm target})\right].$$

默认 $\mathcal H_{\rm target}=-4$，对应动作维数的启发式选择。它不是成功率目标，也不保证每个状态的熵精确相等。`learn_alpha: false` 可固定温度。

## 4. 手算一条 transition

设 $r=-1,\gamma=0.98,d=0$，下一动作的两个目标 Q 是 2 和 3，$\log\pi(a'\mid s')=-1,\alpha=0.2$。则：

$$y=-1+0.98(2-0.2\times(-1))=1.156.$$

假设 actor 当前采样动作的最小 Q 为 1.5、log density 为 -0.7，那么这一样本的 actor loss 是 $0.2(-0.7)-1.5=-1.64$。loss 为负完全合理；我们关心梯度推动了什么。

再看温度：如果平均 log density 是 5，目标熵为 -4，括号是 1，梯度下降会增大 $\ell$、增大熵权重；如果平均 log density 是 3，方向相反。这样可直接检查温度更新符号。

## 5. 按数据流读本仓库代码

| 位置 | 阅读重点 |
|---|---|
| [runner.py](../src/agentic_rl/embodied/runner.py) 的 `train` | reset、随机预热、交互、回放、独立评估 |
| [replay.py](../src/agentic_rl/embodied/replay.py) 的 `encode` / `Replay.sample` | 10 维观察拼 3 维目标；输出五元 batch |
| [networks.py](../src/agentic_rl/embodied/networks.py) 的 `SquashedPolicy` | 高斯参数、重参数化、tanh 与密度修正 |
| [agents.py](../src/agentic_rl/embodied/agents.py) 的 `update_online` | SAC 的软目标、双 critic、actor、alpha、Polyak |

下面是实际实现中的核心两行，省略了 optimizer 调用与分支判断：

```python
values = self.target_critic.minimum(following, next_actions) - alpha * next_log_prob
actor_loss = (alpha * log_prob - self.critic.minimum(states, proposed)).mean()
```

同一个 `Agent` 类为了让读者比较算法保留公共网络容器；SAC 不使用其中的 `value` 或 `target_actor` 做更新。不要把“对象里存在一个网络”误读成“该网络参与了 SAC 目标”。

## 6. 自己运行与看曲线

在仓库根目录安装一次具身依赖：`uv sync --frozen --extra dev --extra cpu --extra embodied`。默认训练实时从环境采集数据，不需要先下载离线示范。

```bash
.venv/bin/python 13-sac/train.py --smoke --output runs/my-sac-smoke
.venv/bin/python 13-sac/eval.py --checkpoint runs/my-sac-smoke/checkpoint-final --episodes 10 --seed 20000
.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

`--smoke` 是真实 FetchReach 的 300 步短运行。去掉它使用 [config.yaml](config.yaml) 的 5000 个环境步。输出目录需要是新目录，防止覆盖实验。TensorBoard 看 `loss/critic`、`loss/actor`、`train/alpha`、`eval/return`，以及曾经到达目标与末步到达目标两种成功率。评估只在开始、结束记录，不能把两个点画成稠密的学习过程。

## 7. 常见误区与一个自己的判断

只看 critic loss 下降，会错过“模型在大量失败轨迹上学会了预测失败”的情况。SAC 还能把熵优化得很好，但尚未发现稀疏奖励目标。检查动作分布和任务成功率，比给 actor loss 设一个“应该接近零”的指标更有用。

对 VLA 来说，SAC 值得学的是如何给连续动作赋值、如何保留探索、如何复用经验。若动作变成几十步的 action chunk，熵的维数、Q 的时间尺度与奖励归属都变了，不能只把本章 MLP 换成大模型就期待同样行为。

## 练习与答案

1. 上面手算若 $d=1$，target 是多少？**-1，未来 Q 和熵都不再进入。**
2. Actor 更新时把 critic 前向放进 `no_grad()` 行吗？**不行；这会切断 Q 对动作的梯度。应只冻结 critic 参数。**
3. SAC 的双 Q 应该取平均还是最小值？**本章目标和 actor 用最小值；平均是不同的设计。**
4. 回放数据来自旧策略，是否需要在这里直接加 PPO ratio？**不需要；这不是 PPO 的重要性采样 surrogate。**

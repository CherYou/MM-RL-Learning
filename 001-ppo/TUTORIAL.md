# 001｜PPO：一个负责行动，一个估计行动之后还有多少希望

[学习路线](../docs/BEGINNER_GUIDE.md) · [前置 loss](../preliminary/TUTORIAL.md) · [运行说明](README.md)

PPO 的 actor 决定下一步做什么，critic 估计从当前历史出发还能获得多少回报。模型得到一个奖励后，不只是问“分数高不高”，还会问“比原先预期高多少”。这种相对信号称为 advantage（优势）。

![Actor 选择动作，Critic 预测未来回报，实际结果分别调整策略与预测](../docs/assets/algorithms/ppo.png)

不要把 critic 当考试阅卷老师。奖励模型或规则校验器给出实际评分；critic 学的是未来回报的预测。Reference 则是另一种角色：一个通常冻结的策略基准。PPO 的方法来源见 [原论文](https://arxiv.org/abs/1707.06347)。

## 从环境控制映射到语言模型

在机器人控制中，状态可能是关节角，动作可能是电机力矩。在语言模型中，状态常用当前完整文字历史表示，动作是下一个 token。两者都可以写成策略 $`\pi_\theta(a_t\mid s_t)`$。长历史不一定是完全可观测的真实世界状态，实际 agent 任务常有部分可观测性。

critic 的 $`V_\phi(s_t)`$ 是预期折扣回报。若预期得 0.2，最终情况却支持 0.8，我们希望鼓励造成好结果的选择；但如何把最后的奖励分配给前面的动作，需要一个估计器。

## GAE：把多步预测误差往前传

一步 TD 误差为：

```math
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t).
```

$`\gamma`$ 控制未来奖励的折扣，$`d_t=1`$ 表示真实终局。GAE 用递推合并这些误差：

```math
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1},\qquad
\widehat G_t=A_t+V_{old}(s_t).
```

$`\lambda`$ 越靠近 0 越依赖一步估值，越靠近 1 越把后续实际误差带回来。它调节偏差与方差，不是学习率。[GAE 论文](https://arxiv.org/abs/1506.02438)解释了这一估计思想。

手算两步：奖励 `[0,1]`，旧 value `[0.2,0.4]`，第二步真实终局，取 $`\gamma=1,\lambda=0.95`$。

```math
\delta_1=1-0.4=0.6,\quad A_1=0.6;\qquad
\delta_0=0+0.4-0.2=0.2,\quad A_0=0.2+0.95\times0.6=0.77.
```

critic 的目标 return 因而为 `[0.97,1.0]`。这些是标准化之前的优势。本仓库 verl PPO 分支之后还会在有效 token 上标准化优势。

## Actor 和 critic 各自的 loss

Actor 使用 PPO clipped loss：

```math
L_\pi=-\mathbb{E}_t\left[\min\left(r_t^\pi A_t,
\mathrm{clip}(r_t^\pi,1-\epsilon,1+\epsilon)A_t\right)\right],
\quad r_t^\pi=e^{\ell_t-\ell_t^{old}}.
```

这里 $`r_t^\pi`$ 是概率比率，与环境奖励 $`r_t`$ 不同。正负优势的裁剪方向见 [preliminary loss 教程](../preliminary/TUTORIAL.md)。

本地显式 PPO 的 value loss 还限制 value 相对旧预测的变化：

```math
V^{clip}=V_{old}+\mathrm{clip}(V-V_{old},-\epsilon_V,\epsilon_V),
```

```math
L_V=\frac{1}{2}\mathbb{E}_t\left[\max((V-\widehat G)^2,(V^{clip}-\widehat G)^2)\right],\qquad
L=L_\pi+c_VL_V.
```

为什么 value 这里是 `max`？训练器保守地取未裁剪、裁剪预测中更大的误差。它和 policy 的 `min` 不矛盾，两者一个写成最小化损失，一个内部是最大化收益的 surrogate。

LLM PPO 常把偏离 reference 的惩罚放进 token reward：

```math
\widetilde r_t=-\beta(\ell_t^{old}-\ell_t^{ref})+
\mathbf{1}[t=\text{最后有效 token}]R_{task}.
```

仓库 verl 路径采用这个方式，再用 GAE 传播；actor loss 不会又加一遍同样的 KL。单个 token 的 log-ratio 可正可负，KL 非负是分布期望的性质。

## 真实代码按这个顺序读

1. [config.yaml](config.yaml)默认 `backend: trl`；[trl_backend.py](../src/agentic_rl/trl_backend.py)实例化真实 `PPOTrainer`、value model 和 reference。
2. [algorithms.py](../src/agentic_rl/verl_backend/algorithms.py)的 PPO 分支展示 token reward、最后 token 的任务奖励、GAE 和优势标准化。
3. [losses.py](../src/agentic_rl/losses.py)的 `gae` 从右向左循环；`ppo_loss` 展开 policy 与 value 两项。
4. [actor.py](../src/agentic_rl/verl_backend/actor.py)负责 CPU 显式 value 更新；GPU 接线使用官方 critic worker，但本章尚未实际 GPU 验证。

GAE 核心代码摘录：

```python
delta = rewards[:, t] + gamma * next_value - values[:, t]
carry = (delta + gamma * lam * carry) * live
adv[:, t] = carry
next_value = torch.where(live.bool(), values[:, t], next_value)
```

`carry` 带着右侧未来误差往前走；`live` 排除 padding。该 helper 可接受 `bootstrap`，但当前完整回答 PPO 调用按零尾值处理。不能因 helper 支持 bootstrap 就宣称所有截断路径已提供正确的尾值。

真实终局没有未来价值，应取 0；仅因时间上限或上下文限制停下，不一定意味着真实未来价值为 0。这个区别在具身连续控制中尤其重要，也会在 SAC、TD3 的新章节继续使用。

## 动手与诊断

```bash
.venv/bin/arl train 001-ppo/config.yaml --smoke
```

观察 policy loss、value loss、KL、奖励和梯度。critic loss 下降但奖励不涨，可能只是更准确地预测了“始终失败”；reward 上涨但 KL 急剧扩大，也不能只看分数。仓库默认 reward 是答案校验器经 reward-model 接口包装，除非明确配置 learned reward model，否则并未训练一个人类偏好奖励网络。

练习：把上面的 $`\lambda`$ 改成 0，$`A_0`$ 变为多少？答案是 0.2，因为不再带回下一步的 TD 误差；$`A_1`$ 仍为 0.6。

我的判断：PPO 的难点常在“目标值生成过程”，而不在几行 clipping。奖励放错位置、终局与截断混淆、old values 更新时机错误，都能让一个看起来正常下降的 value loss 学错对象。本章验证范围是机制与 CPU 执行，不是模型能力改善。API 对照见 [固定版本 TRL PPO 文档](https://huggingface.co/docs/trl/v0.25.1/en/ppo_trainer)。

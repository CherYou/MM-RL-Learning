# 001｜PPO：从“这次答得好”到一次稳妥的策略更新

[学习路线](../docs/BEGINNER_GUIDE.md) · [概率与梯度入门](../preliminary/TUTORIAL.md) · [运行说明](README.md)

PPO 全称 **Proximal Policy Optimization，近端策略优化**。Policy 是“当前情境下如何选择动作”的规则；optimization 是调整规则；proximal 表达更新时不要一下偏离采样策略太远。本章讲常见的 PPO-Clip 版本。它最初用于一般强化学习，后来被用于语言模型训练，方法来源是 [PPO 原论文](https://arxiv.org/abs/1707.06347)。

本章用“3 盒笔，每盒 6 支，共多少支”贯穿讲解，答对得 1，答错得 0。你只需要知道模型会给下一个 token 分配概率；token 是分词器使用的文字单位，不一定是一个汉字。接下来先分清角色，再将最终分数变成每步学习信号，最后解释怎样更新概率。读完应能自己走完一次“生成 → 评分 → 算优势 → 更新”。

## 1. 只有一个最终分数，为什么需要这么多步骤

模型偶然写出“3×6=18”，得到了 1 分。最朴素的想法是提高这段回答的概率，但立即遇到三个问题。

第一，分数属于整段回答，模型却逐 token 做选择。早先写出的“3×6”没有即时奖励，我们需要把后来的结果传回早先的动作，这叫**信用分配**。

第二，相同分数是否值得大力鼓励，取决于原先能做到什么。几乎总能答对的题再次答对，与很难的题终于答对，信息量不同。我们需要比较“结果”和“原先预期”，这个差距叫**优势 advantage**。

第三，一次答对可能有偶然性。如果反复用少量样本把某些概率放大几十倍，原有行为可能被破坏。PPO 裁剪目标用来减少过度更新的激励。

因此下面先介绍谁生成、谁评分、谁预测，再算优势，最后控制更新。Actor-critic 是角色分工，GAE 是优势估计方法，PPO clipping 是策略更新规则，三者不是同一个概念。

## 2. Actor、critic、reward、old 和 reference 各做什么

![Actor 与 Critic 的概念插图，只展示行动与估值关系](../docs/assets/algorithms/ppo.png)

上图帮助理解行动与估值。**包含 reference 的完整语言模型 PPO 流程在下方给出**，不能把上面的概念插图当作全部训练组件。

| 角色 | 输入和输出 | 何时改变 |
| --- | --- | --- |
| Actor，当前策略 | 题目与已有回答 → 下一 token 的概率 | 每次策略优化时更新 |
| Old policy，行为策略 | 生成本批回答，留下当时所选 token 的概率 | 下一批刷新；批内保存值固定 |
| Reward，奖励来源 | 完整回答 → 实际任务分数 | 本章答案校验器是固定规则 |
| Critic，价值模型 | 某一步之前的历史 → 之后回报的预测 | 用本批回报目标训练 |
| Reference，参考策略 | 对相同 token 复评 → 固定的概率基准 | 本章全程冻结 |

Old 不一定要长期保留独立网络，保存生成时的 logprob 就能提供比率分母。Reference 用于约束相对于开训起点的长期偏离。二者开训时可能相同，后面职责和数值不同。Critic 输出回报估计，reference 输出 token 概率，也不能互换。

```mermaid
flowchart TD
    X[题目与历史] --> O[本批 old policy 采样]
    O --> Y[实际回答与 token IDs]
    O --> LP[保存 old logprob]
    Y --> R[奖励规则或奖励模型]
    R --> TASK[最终任务分数]
    Y --> REF[冻结 reference 复评]
    REF --> KL[与 old 比较形成 KL 惩罚]
    LP --> KL
    TASK --> RT[逐 token 训练奖励]
    KL --> RT
    Y --> V[Critic 保存 old value]
    RT --> GAE[GAE 计算优势与回报目标]
    V --> GAE
    Y --> CUR[当前 actor 复评相同 token]
    CUR --> LOSS[PPO 裁剪损失]
    LP --> LOSS
    GAE --> LOSS
    LOSS --> UP[更新 actor]
    GAE --> VL[Value 回归损失]
    Y --> VC[当前 critic 复评历史]
    VC --> VL
    V --> VL
    VL --> UV[更新 critic]
    UP --> NEXT[下一批使用更新后的 actor 采样]
```

箭头表示数据流，不全是可导路径。采样结果、奖励、reference、old 概率与优势作为固定数据使用；当前 actor 的评分和当前 critic 的估值才反传。一般环境控制 PPO 不一定有 reference 或 KL 奖励；这两项属于本章的语言模型设置。

## 3. 把写回答拆成状态、动作、奖励与回报

角色清楚了，但 critic 在哪些位置估值尚未定义。为此先将回答拆成按时间排列的决策过程。

时刻 t 的**状态** $`s_t`$ 用“题目 + 已生成前缀”表示；**动作** $`a_t`$ 是下一个 token；**策略** $`\pi_\theta(a_t\mid s_t)`$ 是模型给这个选择的概率，$`\theta`$ 表示模型参数。接上 token 后得到下一状态。机器人可将状态换成传感器读数、动作换成电机指令，接口类似，但不是相同任务。复杂 agent 的历史也未必包含真实世界全部信息。

**奖励 reward** $`r_t`$ 是当前一步的反馈，**回报 return** $`G_t`$ 是从此刻开始未来奖励的折扣和：

```math
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.
```

$`\gamma`$ 叫折扣因子，通常在 0 到 1 之间。取 1 不打折，取 0.9 则晚一步的奖励计 0.9 倍。两步奖励 `[0,1]` 且 gamma=1 时，第一步即时奖励为 0，但回报为 1，因此早先的动作也有可学习的结果。

Critic 的 $`V_\phi(s_t)`$ 预测从该历史继续按策略行动的平均回报。它没提前看到未来答案。$`\phi`$ 表示 critic 参数，以区别 actor 的参数。单次预测会出错，训练要让预测接近经历中构造的回报目标。

## 4. 先将 reference 约束放进奖励

现在知道回报来自奖励，便能解释 reference 为什么出现在 GAE 之前。只有任务分数时，策略可能为了得分明显偏离原来的语言行为。LLM PPO 常在逐 token 奖励中加入 reference 偏离惩罚，再用合成奖励估计优势。

KL 全称 **Kullback–Leibler divergence，KL 散度**，衡量分布差异。这里没有逐项遍历整个词表，而是在 old policy 实际采到的 token 上计算 logprob 差。$`\ell_t^{old}`$、$`\ell_t^{ref}`$ 表示相同历史、相同 token 的两份对数概率：

```math
\widetilde r_t=-\beta(\ell_t^{old}-\ell_t^{ref})+
\mathbf{1}[t=\text{最后有效 token}]R_{task}.
```

$`R_{task}`$ 是任务分数，$`\beta`$ 控制偏离惩罚，指示符 $`\mathbf{1}`$ 在条件成立时取 1，否则取 0。每步都有可能获得 KL 修正，最后一步再加任务奖励。

例如 old 给实际 token 概率 0.4，reference 给 0.2，则 log-ratio 为 $`\log2\approx0.6931`$。beta=0.1 时奖励修正为 -0.06931。单个采样项也可能为负；KL 非负是分布期望的性质，并非每个 token 都成立。语言模型的角色关系可参考 [Nathan Lambert 的 RLHF 教材](https://rlhfbook.com/c/06-policy-gradients)。

后面为方便手算会直接给出奖励数组。实际 verl PPO 先完成上述合成，再做 GAE，actor loss 不会再加一份相同的 KL。不同后端的配方应分别核对。

## 5. 优势：这次结果比原先预期好多少

假设回报是 1，critic 原先预测 0.3，直接的优势估计为 0.7；若预测 0.9，优势只有 0.1。正优势推动相关动作更常出现，负优势推动它更少出现。

理想动作优势为 $`A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)`$。Q 问“先选这个动作，以后按策略行动，平均回报多少”；V 问“尚未指定动作时，平均回报多少”。本章没有额外训练 Q 网络，而是用奖励和 V 来估计优势。

为什么减 baseline（基线）不会任意改变任务？在标准策略梯度中，同一状态下对动作分布求平均，状态基线乘 logprob 梯度的期望为零。因此可以改变估计的噪声而不改变那个理想梯度的期望；有限样本仍有误差。[Spinning Up 的策略梯度教程](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)解释了这个理由。

还剩一个实际选择：等回答结束，用采样回报减 V；或者借助下一步 V，先就地估计动作的影响。前者受后面偶然结果影响较大，后者依赖 critic 准确性。下一节的 GAE 在两种信息之间建立可调的折中。

## 6. GAE：先理解 TD，再把多步误差传回来

GAE 全称 **Generalized Advantage Estimation，广义优势估计**，是一种计算优势的方法。TD 全称 **Temporal Difference，时序差分**，意思是比较相邻时间点的信息来修正价值预测。我们先讲 TD，是因为它是 GAE 加权组合的基本单元。

原先 critic 在当前状态预测 V。做完动作后，知道了即时奖励，还能用下一状态 V 预测剩余回报。“已得到的奖励 + 剩余预测”减去“原来预测”，就是一步 TD 误差：

```math
\delta_t=r_t+\gamma(1-d_t)V_{old}(s_{t+1})-V_{old}(s_t).
```

$`d_t=1`$ 表示真正终局，此后没有未来回报；否则为 0。使用的是采样阶段保存的 old value。用尚未兑现的下一步预测补全目标，称为 **bootstrapping，自举估计**。

只用 delta 相当于只看一步修正。GAE 还将同一条未终止轨迹内后续的 TD 误差带回来：

```math
\widehat A_t=\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots.
```

$`\lambda`$ 是混合参数。取 0 只剩一步 TD；取 1 且完整终局被正确处理时，各项 V 抵消，得到采样回报减基线；中间值使远处误差衰减更快。[GAE 原论文](https://arxiv.org/abs/1506.02438)研究这一偏差与方差折中：偏差是系统性偏离理想量，方差是不同采样造成的波动。Lambda 不是学习率，也不是任务的时间折扣。

代码不必每次重新求和，从最后一步向前递推即可：

```math
\widehat A_t=\delta_t+\gamma\lambda(1-d_t)\widehat A_{t+1},\qquad
\widehat G_t=\widehat A_t+V_{old}(s_t).
```

带帽子表示估计量。$`\widehat G_t`$ 给 critic 作回归目标，应在优势标准化之前构造；标准化后的优势已改变单位，不能再直接加 V 当回报。

### 两步手算：每个数从哪里来

奖励 `[0,1]`，old value `[0.2,0.4]`，第二步真实终局，gamma=1、lambda=0.95。先从最后一步算：

```math
\delta_1=1-0.4=0.6,\qquad \widehat A_1=0.6.
```

再回到第一步：

```math
\delta_0=0+0.4-0.2=0.2,\qquad
\widehat A_0=0.2+0.95\times0.6=0.77.
```

Critic 的目标为 `[0.77+0.2, 0.6+0.4]=[0.97,1.0]`。第一步没有即时奖励，但后面的好结果通过 GAE 传回来了。改 lambda=0，第一步优势为 0.2；改 lambda=1，则为 0.8，对应回报目标为 1。读者可以用这三个数检查自己是否理解递推。

## 7. 概率比与裁剪：知道方向之后，控制更新幅度

优势解决“往哪个方向学”。生成回答昂贵，PPO 往往对一批样本进行多次小批量更新；第一次之后，当前策略就不同于生成数据的 old policy。于是需要概率比衡量偏离。为避免和奖励 r 混淆，本节用 rho：

```math
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
=\exp(\ell_t-\ell_t^{old}).
```

旧概率 0.2、当前概率 0.3，rho=1.5。Old 分母与优势批内固定，只有当前 logprob 反传。Rho=1 是更新起点，不表示导数为零，因为它仍随当前参数变化。

未裁剪收益是 $`\rho_t\widehat A_t`$：正优势鼓励提高概率，负优势鼓励降低概率。一直优化可能过度相信少量样本。PPO-Clip 取一个保守收益，再加负号转成最小化 loss：

```math
L_\pi=-\mathbb{E}_t\left[\min\left(\rho_t\widehat A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right].
```

Clip 把输入夹在上下界内；epsilon 是裁剪尺度；期望在此用有效生成位置的样本平均近似。必须先乘优势再取 min，因为优势正负决定应停止鼓励哪个方向。设 epsilon=0.2：

| 情况 | 原收益 | 裁剪收益 | 实际取值与含义 |
| --- | --- | --- | --- |
| 优势 +1，比率 1.5 | 1.5 | 1.2 | 取 1.2；好动作已经增加很多，继续增加没有该项额外收益 |
| 优势 -1，比率 0.5 | -0.5 | -0.8 | 取 -0.8；坏动作已经减少很多，继续减少不再额外奖励 |
| 优势 -1，比率 1.5 | -1.5 | -1.2 | 取 -1.5；坏动作反而更多，仍应纠正 |

所以不是“越界就一律零梯度”，也不保证每个概率比最终都留在区间内。共享参数、其他样本和优化器状态仍能改变它。裁剪限制的是这个样本在特定方向的额外激励，导数对照见 [损失函数入门](../preliminary/TUTORIAL.md)。

## 8. Critic 怎样学，为什么有第二个 loss

Actor 根据优势改选择，优势又依赖 critic。策略变化后，critic 也必须学习新的回报水平。我们让它接近前面构造的回报目标，最直接的方法是平方误差。

本地显式 PPO 还使用 value clipping：先限制预测相对 old value 的变化，再取两种误差中较大的一个：

```math
V^{clip}=V_{old}+\mathrm{clip}(V_\phi-V_{old},-\epsilon_V,\epsilon_V),
```

```math
L_V=\frac12\mathbb{E}_t\left[\max\left((V_\phi-\widehat G)^2,
(V^{clip}-\widehat G)^2\right)\right],\qquad L=L_\pi+c_VL_V.
```

$`\epsilon_V`$ 是 value 裁剪尺度，$`c_V`$ 是权重。取 max 避免裁剪后的预测提供过分乐观的误差。例如 old value=0.2、当前 value=0.8、目标=1、尺度=0.2，裁剪预测为 0.4，两种平方误差为 0.04 和 0.36，value loss 为 0.18。

Value clipping 是该实现选择，并非定义 PPO 必需的唯一 critic loss。Actor、critic 使用独立参数与优化器时可以分开更新。Old value 和回报目标保持固定，避免同时移动预测与标签。

## 9. 将整轮训练接回真实代码

每个零件现在都有来由了，重新按时间拼装：

1. Actor 生成回答，保存 token IDs、old logprob、old value 与有效位置 mask。
2. 奖励来源判分，reference 复评，构造逐 token 奖励。
3. 从后向前算 GAE，保存优势和 critic 回报目标。
4. 当前 actor/critic 复评同一批 token，计算两项 loss，再反传、优化。
5. 批内更新结束后采下一批；old 基准刷新，reference 保持固定。

Mask 是“哪些位置直接算 loss”的 0/1 标记：题目与 padding（补齐长度的空位）取 0，实际生成取 1。Mask=0 不表示从模型输入删除这些文字。

| 代码入口 | 带着什么问题去读 |
| --- | --- |
| [config.yaml](config.yaml)、[trl_backend.py](../src/agentic_rl/trl_backend.py) | 默认 TRL 路径怎样构造 PPOTrainer、value、reference 和奖励接口 |
| [algorithms.py](../src/agentic_rl/verl_backend/algorithms.py) | KL 与最终任务奖励怎样合并，优势何时标准化 |
| [losses.py](../src/agentic_rl/losses.py) 的 `gae`、`ppo_loss` | 从右向左递推，哪些目标 detach，value 为什么取 max |
| [actor.py](../src/agentic_rl/verl_backend/actor.py) | 当前前向、反传及两类参数更新怎样执行 |

GAE 的实际核心摘录：

```python
delta = rewards[:, t] + gamma * next_value - values[:, t]
carry = (delta + gamma * lam * carry) * live
adv[:, t] = carry
next_value = torch.where(live.bool(), values[:, t], next_value)
```

`carry` 保存右侧已算好的优势，`live` 标记有效位置。这个 helper 面向完整回答和尾部 padding，可以接收 `bootstrap` 尾值，但当前完整回答 PPO 调用按零尾值处理。它不能不加修改就处理任意多个 episode 拼接。

真正终局应清零未来价值；达到时间或上下文预算却未必意味着未来价值为零。检查截断时，要追踪调用方有没有提供正确尾值，不能只看函数存在 bootstrap 参数。

## 10. 先预测，再运行，再解释日志

按 [README](README.md) 安装后，从仓库根目录运行下面的 Linux/WSL 命令。Windows PowerShell 按安装环境改用 `.venv\Scripts\arl.exe`。

```bash
.venv/bin/arl train 001-ppo/config.yaml --smoke
```

Smoke 是少量步骤的管线检查，使用随机微型模型，不证明数学能力改善。默认 reward 是答案校验器通过 reward-model 接口包装，接口名称不意味着训练了人类偏好奖励网络。

运行前预测：actor、critic 都应有训练路径，reference 不变。运行后一起看任务奖励、value loss、KL、概率比与梯度。Value loss 下降可能只是更准确地预测失败；actor loss 近零可能是正负项数值抵消。能力需要独立题目的自由生成正确率来判断。

后端需分清：TRL 默认入口使用其多 epoch/minibatch PPO；native 展开单次更新；verl 有独立轨迹与更新路径。当前本章尚未验证正式 GPU 能力改善，API 对照见 [固定版本 TRL PPO 文档](https://huggingface.co/docs/trl/v0.25.1/en/ppo_trainer)。

## 11. 新手自测：不用术语也能解释吗

下面是练习。先遮住答案，尝试用自己的话复述，再回到对应小节。

1. **Critic、reward、reference 都在评价，区别是什么？** Reward 给实际任务反馈；critic 预测未来合成回报；reference 提供固定概率基准。预测、判分、约束是三个问题。
2. **GAE 全称是什么，为什么先算 TD？** 广义优势估计；TD 给相邻时间点的价值修正，GAE 将未来修正加权传回当前动作。
3. **两步例子 lambda=0 时第一步优势多少？** 0.2；不再传回后面的 0.6。第二步仍为 0.6。
4. **正优势、old 概率 0.2、当前 0.3、epsilon=0.2，还继续鼓励增加吗？** 该样本落在裁剪平台，不再提供这个方向的额外激励；其他样本仍可能改变其概率。
5. **下一批谁刷新？** Old 用更新后的 actor 形成新基准，reference 固定；critic 随自己的优化更新。
6. **标准化后的优势加 old value 能当 return 吗？** 不能，它改变了单位；回报目标应先由未标准化优势构造。

## 12. 延伸阅读：每份资料解决一个疑问

- [PPO 原论文](https://arxiv.org/abs/1707.06347)：裁剪 surrogate 与更新流程。
- [GAE 原论文](https://arxiv.org/abs/1506.02438)：lambda 如何连接不同时间尺度。
- [Spinning Up：策略梯度教学](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：减 baseline 的数学理由。
- [Nathan Lambert：RLHF 中的策略梯度](https://rlhfbook.com/c/06-policy-gradients)：一般 RL 与语言模型 reward/reference 的衔接。

这些资料用于机制核对和深入阅读。本章数字例子、角色表、流程图和代码阅读顺序为本仓库重新组织，执行细节以本地配置与代码为准。

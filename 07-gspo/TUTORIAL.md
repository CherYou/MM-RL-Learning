# 07｜GSPO：先把整段回答放上秤，再决定怎样裁剪

[学习路线](../docs/BEGINNER_GUIDE.md) · [GRPO](../01-grpo/TUTORIAL.md) · [运行说明](README.md)

如果奖励只告诉你“整段解答是对的”，更新时应该逐 token 判断变化是否过大，还是让整段回答共享一个变化尺度？GSPO 选择后者。它仍可以使用组内相对奖励，但改变了重要性权重和裁剪的粒度。[GSPO 原论文](https://arxiv.org/abs/2507.18071)给出了这一设计。

![逐 token 的多个局部尺度，与整条回答共享一个尺度和裁剪判断的对比](../docs/assets/algorithms/gspo.png)

图中的单个仪表表示一个序列权重，不表示模型只在整句末尾预测一次。前向和反向仍然发生在 token 位置上。

## 先分清三种看起来相似的平均

第 i 条回答的 token ratio 是 $`r_{i,t}=\exp(\ell_{i,t}-\ell^{old}_{i,t})`$。GSPO 使用：

```math
s_i=\exp\left(\frac1{T_i}\sum_t m_{i,t}(\ell_{i,t}-\ell^{old}_{i,t})\right)
=\left(\prod_{t:m_{i,t}=1}r_{i,t}\right)^{1/T_i}.
```

它是 token ratio 的几何平均。它不等于算术平均 $`\frac1T\sum_t r_t`$，也不等于整段未经长度归一化的概率比 $`\prod_t r_t`$。

为什么从 logprob 开始？整段概率的 log 是 token logprob 的和，除以长度后再指数化，可以把序列的变化尺度标准化。这里是特定 surrogate 的设计，不应把长度归一化权重宣称为任意序列期望的精确无偏 IS 权重。

## 一个两 token 的手算例子

设两个 token 的 ratio 分别是 2 和 0.5。那么：

```math
s=\sqrt{2\times0.5}=1,\qquad \text{算术平均}=1.25.
```

GRPO 的 token 裁剪会分别看到“一个涨了很多，一个降了很多”；GSPO 的序列裁剪看到“长度归一化后的整体变化为 1”。因此两者可以对同一个回答采取不同更新。

不要误解为 GSPO 证明这两个变化都无害。局部变化可以在序列尺度上相互抵消，这是换粒度得到的性质，也是需要观察的取舍。

## loss 与梯度

令 $`A_i`$ 为该回答的组内优势，忽略可选 KL 项：

```math
L_{GSPO}=-\frac1B\sum_i\min\left(s_iA_i,\mathrm{clip}(s_i,1-\epsilon_l,1+\epsilon_h)A_i\right).
```

在未裁剪分支，对该回答某个有效 token 的 logprob：

```math
\frac{\partial L}{\partial\ell_{i,t}}=-\frac{A_is_i}{BT_i}.
```

同一回答的有效 token 共享序列权重，且长度进入导数。即使上例 $`s=1`$，只要 $`A\ne0`$ 且未处于裁剪平台，梯度依然非零；“ratio=1”不等于“没有学习”。

当前 [配置](config.yaml)设置 `clip_low=0.0003`、`clip_high=0.0004`，区间是 `[0.9997,1.0004]`。这些小数是本地实验配置，不能因为 GRPO 常见 0.2 就直接替换；粒度变了，数值尺度也变了。

## 真实代码里最关键的两行

打开 [losses.py](../src/agentic_rl/losses.py) 的 `policy_loss` 或 [verl losses.py](../src/agentic_rl/verl_backend/losses.py) 的 `objective`，找到 `kind == "gspo"`。精简摘录：

```python
delta = logp - old_logp.detach()
delta = (delta * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
ratio = delta.exp()
```

第一行原来是 `[batch, positions]`；第二行变成 `[batch, 1]`，每条回答只剩一个数；第三行得到序列 ratio。后面广播到 token 时不是重新产生一套逐 token ratio，而是把同一个数供该回答各位置共用。

分母只数有效生成 token，不能包含 prompt、padding 或工具 observation。假设一个回答有 10 个生成 token、90 个 prompt token，错误地用 100 作分母会把变化缩小十倍，悄悄改变裁剪强度。

默认配置走 TRL，入口在 [trl_backend.py](../src/agentic_rl/trl_backend.py) 的 `run_trl`，通过固定版本支持的序列重要性采样设置实现 GSPO；verl 的 [algorithms.py](../src/agentic_rl/verl_backend/algorithms.py) 明确指定 `kind="gspo"`。不要看到配置里 `loss: grpo` 就断定 ratio 仍是 token 级，必须追踪最终 trainer 参数。

## 怎么检查实现与效果

```bash
.venv/bin/arl train 07-gspo/verl.yaml --smoke --verl-workers 2
```

CPU smoke 验证管线。机制测试应至少包含“[2,0.5] 的几何平均为 1”、“padding 不改变 ratio”和“序列 loss 的梯度按有效长度分配”。这些比仅检查 loss 有限更能辨别 GSPO 是否真的实现。

练习：三个 token 的 ratio 都是 1.1，GSPO ratio 是多少？答案仍是 1.1；未经长度归一化的整段比率是 $`1.1^3=1.331`$。这正是长度归一化改变的地方。

我的阅读建议是每次遇到“平均”都写清平均对象：对 token log-ratio 平均、对 token loss 平均、对回答 loss 平均是三个不同步骤。GSPO 的核心是第一步，DAPO 主要涉及后两步和采样调度。只比较训练曲线中的 loss 数值，很容易把这些差别混在一起。

论文性能结果属于作者实验；本仓库本章目前只有机制与 CPU 执行证据，未完成 GPU 吞吐或能力对照。可进一步阅读 [GSPO 正文](https://arxiv.org/html/2507.18071v2)及 [固定版本 TRL GRPO 文档](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)。

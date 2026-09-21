# 05｜GSPO：序列级比率与裁剪

<!-- NAV:TOP:BEGIN -->
[← 上一章：04 DAPO：动态采样、裁剪与长度处理](../06-dapo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/01-policy-preference.md) · [下一章：06 DPO：从偏好对直接优化策略 →](../002-dpo/TUTORIAL.md)
<!-- NAV:TOP:END -->

[学习路线](../docs/BEGINNER_GUIDE.md) · [GRPO](../01-grpo/TUTORIAL.md) · [运行说明](README.md)

如果奖励只告诉你“整段解答是对的”，更新时应该逐 token 判断变化是否过大，还是让整段回答共享一个变化尺度？GSPO 选择后者。它仍可以使用组内相对奖励，但改变了重要性权重和裁剪的粒度。[GSPO 原论文](https://arxiv.org/abs/2507.18071)给出了这一设计。

![逐 token 的多个局部尺度，与整条回答共享一个尺度和裁剪判断的对比](../docs/assets/algorithms/gspo.png)

图中的单个仪表表示一个序列权重，不表示模型只在整句末尾预测一次。前向和反向仍然发生在 token 位置上。

## 为什么奖励按回答给，裁剪却还要选择粒度

GSPO 全称 **Group Sequence Policy Optimization，组序列策略优化**。Group 仍指同题回答的组相对奖励，sequence 指整条生成序列。它没有让语言模型跳过逐 token 生成，也没有新增一个最终评分器；变化发生在更新时怎样衡量一条回答已经偏离 old policy 多少。

GRPO 可以对每个 token 的 ratio 分别裁剪，同一回答有的位置继续学、有的位置进入裁剪平台。GSPO 则先汇总整条回答的 logprob 变化，再用同一个序列尺度决定是否裁剪。既然奖励来自整段，作者选择让更新权重也以整段为单位；这是设计动机，不意味着它为每个 token 找到了真实贡献。

先建立三个量的区别：任务分数 R 判断回答好坏；优势 A 判断相对同组好坏；ratio 判断策略相对采样时改变多少。A 与 ratio 来源不同，即使 ratio 恰好等于 1，只要 A 不为零，策略仍可以学习。

## 先分清三种看起来相似的平均

为什么还要做长度归一化？若每个 token 的 ratio 都为 1.01，100 个 token 的整段概率比约 2.70，1000 个约 20959。每步变化一样，单纯连乘却随长度剧烈变化。因此先平均 log-ratio 再指数化，让比较尺度更接近平均每步的变化。

第 i 条回答的 token ratio 是 $`r_{i,t}=\exp(\ell_{i,t}-\ell^{old}_{i,t})`$，ell 与 ell-old 分别是当前与采样时的 logprob。m 在有效生成位置取 1，其他位置取 0；T 是这些有效位置的数量。GSPO 使用：

```math
s_i=\exp\left(\frac1{T_i}\sum_t m_{i,t}(\ell_{i,t}-\ell^{old}_{i,t})\right)
=\left(\prod_{t:m_{i,t}=1}r_{i,t}\right)^{1/T_i}.
```

它是 token ratio 的几何平均。它不等于算术平均 $`\frac1T\sum_t r_t`$，也不等于整段未经长度归一化的概率比 $`\prod_t r_t`$。

为什么从 logprob 开始？整段概率的 log 是 token logprob 的和，除以长度后再指数化，可以把序列的变化尺度标准化。这里是特定 surrogate 的设计，不应把长度归一化权重宣称为任意序列期望的精确无偏 IS 权重。

## 一个两 token 的手算例子

“平均 log 再 exp”听起来抽象，下面用一升一降的两个 token 看它与算术平均为什么会给不同答案。这个区别直接影响谁进入裁剪区。

设两个 token 的 ratio 分别是 2 和 0.5。那么：

```math
s=\sqrt{2\times0.5}=1,\qquad \text{算术平均}=1.25.
```

GRPO 的 token 裁剪会分别看到“一个涨了很多，一个降了很多”；GSPO 的序列裁剪看到“长度归一化后的整体变化为 1”。因此两者可以对同一个回答采取不同更新。

不要误解为 GSPO 证明这两个变化都无害。局部变化可以在序列尺度上相互抵消，这是换粒度得到的性质，也是需要观察的取舍。

## loss 与梯度

得到序列比率后，仍要和优势相乘，才能知道这条回答该更常见还是更少见。下面将序列比率代入裁剪目标，并进一步看它对单个 token 的导数，避免把“序列级”误解成只训练最后一个位置。

令 $`A_i`$ 为该回答的组内优势，B 为本批有效回答数，epsilon-l/h 为下、上裁剪幅度。忽略可选 KL 偏离正则项：

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

数学上的区别发生在指数化之前：先聚合 logprob 差，再算 ratio。读代码时检查这两个操作的顺序与张量形状，就能区分 GSPO 和“最后把 token loss 平均一下”的普通聚合。

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

确认运算顺序之后，再做三个可预测的小实验。先验证机制会比直接比较大模型 loss 曲线更容易定位问题，因为曲线同时受到采样、长度与奖励的影响。

```bash
.venv/bin/arl train 07-gspo/verl.yaml --smoke --verl-workers 2
```

CPU smoke 验证管线。机制测试应至少包含“[2,0.5] 的几何平均为 1”、“padding 不改变 ratio”和“序列 loss 的梯度按有效长度分配”。这些比仅检查 loss 有限更能辨别 GSPO 是否真的实现。

练习：三个 token 的 ratio 都是 1.1，GSPO ratio 是多少？答案仍是 1.1；未经长度归一化的整段比率是 $`1.1^3=1.331`$。这正是长度归一化改变的地方。

我的阅读建议是每次遇到“平均”都写清平均对象：对 token log-ratio 平均、对 token loss 平均、对回答 loss 平均是三个不同步骤。GSPO 的核心是第一步，DAPO 主要涉及后两步和采样调度。只比较训练曲线中的 loss 数值，很容易把这些差别混在一起。

论文性能结果属于作者实验；本仓库本章目前只有机制与 CPU 执行证据，未完成 GPU 吞吐或能力对照。可进一步阅读 [GSPO 正文](https://arxiv.org/html/2507.18071v2)及 [固定版本 TRL GRPO 文档](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)。

## 新手自测

如果在回答末尾只增加 padding，s 应不应该变？不应该，因为 mask 排除了 padding。如果把 2 与 0.5 先取算术平均再裁剪，会不会还是 GSPO？不会，得到的是 1.25 而不是 1。最后解释：同一回答 token 共享权重，不代表它们获得相同参数梯度；每个位置的 logprob 对网络参数的导数仍不同。

<!-- NAV:BOTTOM:BEGIN -->
[← 上一章：04 DAPO：动态采样、裁剪与长度处理](../06-dapo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/01-policy-preference.md) · [下一章：06 DPO：从偏好对直接优化策略 →](../002-dpo/TUTORIAL.md)
<!-- NAV:BOTTOM:END -->

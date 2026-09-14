# 00｜loss 怎样把“这个回答更好”变成一次参数更新

[学习路线与符号表](../docs/BEGINNER_GUIDE.md) · [本章运行说明](README.md)

本章只研究一次更新的发动机：给定已经采样的 token、它的旧概率和好坏信号，怎样产生梯度？先不考虑奖励怎么来的，也先不训练完整语言模型。

![IS、PPO 和 CISPO 分别表示不限制权重、限制更新激励、限制权重但保留梯度](../docs/assets/algorithms/loss.png)

图中三个装置是比喻。真正要比较的是下文的导数；机械挡板不代表整个神经网络参数会被严格限制在某个范围内。

## 从一个 token 开始

某个 token 采样时概率为 0.2，当前概率变成 0.3，则比率 $r=0.3/0.2=1.5$。我们已经把它变得比以前更容易出现了。假设 $A=+1$，意思是它所在的回答值得鼓励；$A=-1$ 则值得降低概率。

设 $\ell=\log\pi_\theta(y_t\mid h_t)$，$\ell_{old}$ 是采样时的固定值：

$$r=e^{\ell-\ell_{old}},\qquad \frac{\partial r}{\partial\ell}=r.$$

只对当前 $\ell$ 求导。把 old 也当可训练量，会让“当时的尺子”跟着今天的参数一起变化，破坏比率的含义。

## IS：直接按相对概率调整激励

重要性采样（importance sampling）的基本恒等式是：在支持集满足要求时，$\mathbb E_{p}[f]=\mathbb E_q[(p/q)f]$。它让从分布 q 抽到的数据用于估计分布 p 下的量。

本章 token 教学目标取：

$$L_{IS}=-rA,\qquad \frac{\partial L_{IS}}{\partial\ell}=-rA.$$

当 $A=1,r=1.5$，导数为 -1.5，梯度下降继续增加这个 token 的 logprob。比率很大时激励也可能很大。LLM 多步序列中的分布修正比这条单 token 恒等式更复杂；不要据此认定任意旧轨迹只乘一个 token ratio 就完全无偏。

## PPO clipping：对已经走得够远的方向降低激励

定义 $c(r)=\operatorname{clip}(r,1-\epsilon_l,1+\epsilon_h)$，本章默认上下都是 0.2：

$$L_{clip}=-\min(rA,c(r)A).$$

为什么必须有 `min`？因为 $A$ 的正负会改变哪一侧需要停止鼓励：

| 情形 | 未裁剪项 | 裁剪项 | loss | 对 logprob 的导数 |
| --- | --- | --- | --- | --- |
| $A=+1,r=1.5$ | 1.5 | 1.2 | -1.2 | 0 |
| $A=-1,r=0.5$ | -0.5 | -0.8 | 0.8 | 0 |
| $A=-1,r=1.5$ | -1.5 | -1.2 | 1.5 | 1.5 |

第一行：好回答已经被明显提高概率，再提高没有这项 surrogate 的额外收益。第二行：坏回答已经被明显降低概率，再降低也停止额外收益。第三行：坏回答反而更常出现，仍需要把方向纠正回来。

所以“只要 ratio 超过区间，PPO 梯度就为零”是错的。裁剪也不是硬性参数约束：其它样本、KL、共享参数和优化器动量仍可能使该 token 概率变化。机制依据见 [PPO 原论文](https://arxiv.org/abs/1707.06347)。

## CISPO：限制权重，但继续传递梯度

仓库的双边裁剪教学式是：

$$L_{CISPO}=-\operatorname{sg}(c(r))A\ell,
\qquad \frac{\partial L_{CISPO}}{\partial\ell}=-c(r)A.$$

`sg` 表示这个权重在反传中当常数。$A=1,r=1.5$ 时，导数为 -1.2，而 PPO 的对应导数为 0。CISPO 仍鼓励该 token，只是给这个样本的权重封顶。

如果删掉 `.detach()`，自动求导会同时穿过 `ratio` 和 `logp`，得到另一种目标。若删掉最后的 `logp`，并只保留停止梯度的权重，则完全没有可用梯度。

[MiniMax-M1 技术报告](https://arxiv.org/abs/2506.13585)提出 CISPO，并在其具体实验中使用不同于这里对称教学设置的 IS 裁剪选择。本章复现的是核心梯度区别，不是其全部训练配方。

## 真实代码逐句读

打开 [losses.py](../src/agentic_rl/losses.py) 的 `policy_loss`。下面是对应核心的精简摘录，省略 mask 和统计：

```python
old_logp = old_logp.detach()
advantage = advantage.detach()
ratio = (logp - old_logp).exp()
objective = ratio.clamp(1 - clip_low, 1 + clip_high).detach() * advantage * logp
per_token = -objective
```

第一、二行固定本批次的采样尺子和评价信号。第三行在 log 空间求差再取指数，避免先恢复两个很小的概率。第四行是 CISPO 分支；它只允许梯度通过最后的 `logp`。第五行把要最大化的激励转成要最小化的 loss。

之后还要聚合 token。序列平均是“先对每个回答平均，再对回答平均”；token 平均是“所有有效 token 一起平均”；`dr_grpo` 分支使用“回答数 × 固定最大长度”作分母。它们的区别不是输出小数点多少位，而是长短回答各占多少权重。

例如两条回答长 2 和 8，每个 token 的损失分别都是 2 和 1。序列平均为 $(2+1)/2=1.5$；token 平均为 $(2\times2+8\times1)/10=1.2$；固定长度为 8 时，结果为 $12/(2\times8)=0.75$。本仓库的固定长度分支是对照公式，不等于新增了一个完整训练流程。

## 自己动手观察梯度

从项目根目录运行：

```bash
.venv/bin/arl loss-demo
```

执行逻辑在 [loss_demo.py](../src/agentic_rl/loss_demo.py)，会生成实际 autograd 梯度数据和图片。观察正、负优势两组曲线，重点找 $r=0.5$ 和 $r=1.5$，不要只看 loss 曲线是否更平滑。

练习：把第三行情形换成 $A=+1,r=0.5$。答案：PPO loss=-0.5，导数=-0.5；因为好回答变得更少见，需要继续提高其概率。CISPO 导数=-0.8，来自被裁剪后的权重。这是两个估计器的实际区别。

我的阅读建议是先问“梯度通过哪条线”，再问“函数叫什么”。IS、clipping、stop-gradient 这三件事看起来都只是乘法，但决定了不同的更新规则；理解这点，之后看到 GRPO、DAPO 和 GSPO 的 loss 就不会被长公式吓住。

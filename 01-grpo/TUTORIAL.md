# 01｜GRPO：让同一道题的几个回答相互当参照

[学习路线](../docs/BEGINNER_GUIDE.md) · [运行说明](README.md) · [本仓库 GPU 实测](../docs/GRPO_GPU_VALIDATION.md)

假设你让模型回答“每盒 6 支笔，买 3 盒共有多少支？”一次答错只能告诉你这条路径不好。若它对同一道题尝试四次，其中两次答对、两次答错，就可以在同样难度下比较哪些生成路径更值得重复。这是理解 GRPO 最直接的起点。

![同一问题生成多份回答，组内比较奖励后将不同方向的信号送回同一个策略](../docs/assets/algorithms/grpo.png)

图里的判分可以由程序执行，不要求另训练一个神经网络。GRPO 用组内统计提供 baseline，省去 PPO 的 value critic；它仍需要奖励来源，也可能保留 reference 模型。[DeepSeekMath](https://arxiv.org/abs/2402.03300)是这一机制的原始来源。

## 一步训练究竟发生什么

1. 保存当前采样策略的概率基准，针对每道题生成 G 个回答。
2. 用答案校验器给每个回答一个奖励。
3. 每道题内部计算优势，不跨题混算。
4. 重新计算这些已生成 token 在当前可训练模型下的 logprob。
5. 用裁剪目标更新模型；下一轮采样再使用更新后的参数。

生成答案是离散采样，本身不做普通反向传播。梯度来自第 4 步对“已选择 token 的概率”重新打分。

## 手算一组优势

设四个回答的奖励是 $R=[1,1,0,0]$：

$$\bar R=\frac1G\sum_iR_i=0.5,\quad
\sigma=\sqrt{\frac1G\sum_i(R_i-\bar R)^2}=0.5,$$

$$A_i=\frac{R_i-\bar R}{\sigma+10^{-4}}\approx[1,1,-1,-1].$$

这份仓库用总体标准差，代码中的 `correction=0` 对应分母 G，而非 G−1。不能和使用样本标准差的其它实现直接比数值。

奖励是绝对判分，优势是相对判分。若奖励变成 `[1,1,1,1]`，所有优势为 0：这些回答都成功，但本组没有告诉模型哪条比其它更好。全错组同理。不要给相同奖励随意加噪声来伪造“有学习信号”。

## 从回答分数走到 token loss

对于回答 i 的有效生成 token t，设：

$$r_{i,t}=\exp(\ell_{i,t}-\ell^{old}_{i,t}),\quad
c_{i,t}=\min\!\left(r_{i,t}A_i,\operatorname{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_i\right).$$

该回答所有生成 token 共用 $A_i$。本地 GRPO 的序列归一化目标为：

$$L=-\frac1B\sum_{i=1}^B\frac1{T_i}\sum_t m_{i,t}c_{i,t}
+\beta\frac1B\sum_{i=1}^B\frac1{T_i}\sum_t m_{i,t}k_{i,t},\qquad T_i=\sum_tm_{i,t}.$$

B 是本次更新有效回答数，题目数乘 G；空 mask 的回答排除。第一项鼓励相对更好的回答，第二项限制偏离固定 reference。仓库用的逐样本 KL 形式为：

$$d=\ell^{ref}-\ell,\qquad k=e^d-d-1.$$

它是 KL 的采样估计形式，不是“将整个词表的 KL 精确求和”。本次 GPU 检查设置 `beta=0`，未启用此正则。

重要区分：ratio 的 old 是生成该批数据的策略；KL 的 ref 通常从开训时冻结。二者不是同一件事。裁剪的正负优势行为先读 [00 章](../preliminary/TUTORIAL.md)。

## 顺着真实代码读一遍

默认 [config.yaml](config.yaml) 选择 TRL；[verl.yaml](verl.yaml) 和 [verify-gpu.yaml](verify-gpu.yaml) 选择 verl。不要只看章节名就断言运行了哪个 Trainer。

| 代码位置 | 你应该追踪的变量 |
| --- | --- |
| [agent_loop.py](../src/agentic_rl/verl_backend/agent_loop.py) 的 `collect` | G 次生成后的完整 tokens、mask、old logprob |
| [algorithms.py](../src/agentic_rl/verl_backend/algorithms.py) 的 `AlgorithmBatchBuilder.build` | `rewards` 如何分组，`advantages` 如何扩展到 token |
| [losses.py](../src/agentic_rl/losses.py) 的 `group_advantages` | 中心化、总体标准差、`.detach()` |
| [verl loss](../src/agentic_rl/verl_backend/losses.py) 的 `objective`、`reduce_tokens` | 裁剪和跨 microbatch 的全局分母 |
| [actor.py](../src/agentic_rl/verl_backend/actor.py) | 前向、反传、梯度范数与 optimizer step |

下面是教学伪代码，展示真实数据依赖，不是可直接调用的公共 API：

```python
answers = sample_each_question(G)
rewards = verify_answers(answers)
A = group_advantages(rewards, G).detach()
old = saved_generation_logprobs(answers).detach()
current = actor.score_same_token_ids(answers)
loss = clipped_sequence_mean(current, old, A, response_mask)
loss.backward()
optimizer.step()
```

`score_same_token_ids` 的“same”很重要：先解码再重新分词可能改变 token 边界，旧概率就不再对应训练位置。另一个常见错误是每个 microbatch 各自平均后相加，导致 microbatch 切法改变样本权重；verl 扩展使用全局有效序列数进行归一化。

## 如何看本仓库刚跑出的曲线

物理 1 号 A100 的 [运行报告](../docs/GRPO_GPU_VALIDATION.md)记录了真实 Qwen2.5-0.5B-Instruct、3 步、48 条 GSM8K rollout。三步梯度非零，actor 参数改变，reference 冻结。奖励均值分别是 0.3125、0.4375、0.125。

这些数字确认链路执行，不构成下降或上升趋势：每步题目不同，且样本极少。`loss/total` 接近零也不代表没更新，因为正负优势项可能在标量值上抵消，而参数梯度方向仍不同。应一起看 `update/grad_norm`、`update/skipped`、策略版本与独立评估。

## 练习与自己的判断

在项目根目录运行 CPU 链路检查：

```bash
.venv/bin/arl train 01-grpo/verl.yaml --smoke --verl-workers 2
```

手算题：G=4，奖励 `[1,0,0,0]`，优势近似是多少？答案：均值 0.25，标准差 $\sqrt{0.1875}\approx0.4330$，正确项约 1.7317，错误项各约 -0.5772，已包含 $10^{-4}$。这说明“一次成功”在罕见成功组中的标准化幅度更大。

我的判断：调 GRPO 时，先检查每道题能否产生有差异的回答，再调整 loss 超参数。没有差异时，把学习率调大只会放大其它噪声或正则；增加 G、调整题目难度和改善奖励可观测性才更直接。这个判断来自上面的梯度结构，不是本仓库已经测得的性能结论。

继续阅读 [DAPO](../06-dapo/TUTORIAL.md)，看看它怎样主动处理全同奖励组。理论与 API 对照可看 [DeepSeekMath §4](https://arxiv.org/html/2402.03300v2) 和 [固定版本 TRL GRPO 文档](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)。

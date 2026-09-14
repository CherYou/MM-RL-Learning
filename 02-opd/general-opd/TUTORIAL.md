# General OPD｜学生先尝试，教师再评价学生走过的每一步

[学习路线](../../docs/BEGINNER_GUIDE.md) · [运行说明](README.md) · [医疗调度版本](../TUTORIAL.md)

想象学骑车：只看老师骑得很好，不一定能教会你在自己快摔倒的位置怎样调整。OPD 的出发点类似：由学生生成自己的回答，再让教师对学生实际访问的历史、实际选中的 token 提供反馈。

![学生生成 token 串，冻结教师对同一串 token 评分，再把逐 token 信号送回学生](../../docs/assets/algorithms/general-opd.png)

图中 Teacher 手里的 token 和 Student 写出的 token 是同一份。这里的训练答案不是教师另写一遍；概率高低也不等于教师提供了可验证的真值。

## 先与 SFT、普通蒸馏区分

| 方法 | 训练轨迹来自谁 | 学生主要学什么 |
| --- | --- | --- |
| SFT | 固定示范数据 | 提高给定示范 token 的概率 |
| 教师生成后离线蒸馏 | 教师预先生成 | 模仿教师的轨迹 |
| 本章 OPD | 当前学生自己采样 | 在学生访问的历史上调整 token 分布 |

学生自采样与在其轨迹上接受教师监督，是 [GKD 工作](https://arxiv.org/abs/2306.13649)讨论的核心方向；reverse KL 的蒸馏动机可参考 [MiniLLM](https://arxiv.org/abs/2306.08543)。下面的数学例子解释本仓库实现，不等于这两篇论文所有细节的复现。

## 教师究竟给出什么信号

在固定历史 h 上，学生分布为 p，冻结教师分布为 q。反向 KL 定义为：

$$D_{KL}(p\|q)=\sum_v p(v\mid h)\log\frac{p(v\mid h)}{q(v\mid h)}.$$

v 遍历词表；这条式子是分布层面的目标。全词表求和很贵，本地实现只记录学生采到的 token：

$$d_t=\operatorname{sg}(\ell^{old}_t-\ell^T_t),\qquad
r_t=\exp(\ell_t-\operatorname{sg}(\ell^{old}_t)).$$

固定历史下，KL 的 score-function 梯度中可写出 $\mathbb E_p[(\log p-\log q)\nabla\log p]$；额外的常数 1 项期望梯度为零。由此得到本章采用的局部 token 采样 surrogate：

$$L_{OPD}=\frac{\sum_t m_t r_t d_t}{\sum_t m_t}.$$

这是学生轨迹上的 token 级近似训练目标。长序列中历史分布也会随策略改变，因此不要把这个简式称为任意多轮序列 KL 的完整无偏梯度，更不要与全词表 KL 数值混为一谈。

## 一个数字就能看清更新方向

学生采到了 token“18”，采样时概率为 0.2，教师在相同历史下给它 0.4：

$$d=\log(0.2)-\log(0.4)=\log(0.5)\approx-0.6931.$$

刚开始更新时当前策略等于采样策略，r=1，因而 $\partial L/\partial\ell=rd=-0.6931$，梯度下降推动学生提高该 token 的概率。

若教师只给 0.1，则 d=+0.6931，方向反转。它反映“学生相对教师是否过度偏好这个 token”，不是直接回答它是否数学正确。

一次只采到 d=-0.6931，不违反 KL 非负：KL 非负是对整个分布求期望的性质，有限样本平均可以为负，训练 surrogate 也可以为负。

## 为什么一定要停止梯度

教师参数冻结；old 概率在更新前固定；差值 d 也固定。更新后重新计算的是 current logprob。若每个 optimizer micro-step 又重算 old，ratio 总会被拉回 1；若误让 d 反传，又增加了设计之外的梯度路径。

打开 [losses.py](../../src/agentic_rl/losses.py) 的 `opd_loss`，核心代码摘录如下：

```python
gap = old_logp.detach() - teacher_logp.detach()
loss = masked_mean((logp - old_logp.detach()).exp() * gap, mask)
```

verl 路径采用等价的符号约定：在 [algorithms.py](../../src/agentic_rl/verl_backend/algorithms.py) 中设置 `advantages = teacher_log_probs - old_log_probs`，再由 `lab_opd` 最小化 `-ratio * advantages`。负号抵消后与上式相同。看到变量名 `advantages` 时，应追踪它的来源，不能认定这里也做了 GRPO 组内标准化。

## 代码阅读的重点是对齐

先读 [trainers.py](../../src/agentic_rl/trainers.py) 的 `teacher_logprobs`。它检查教师与学生 vocabulary 相同，保留学生 completion IDs，只对教师上下文进行必要构造，然后把评分移回学生 token 位置。`@torch.no_grad()` 阻止教师复评建训练图。

接着读 [models.py](../../src/agentic_rl/models.py) 的 `Policy.score` 与 [protocol.py](../../src/agentic_rl/verl_backend/protocol.py) 的 `pack`。需要核对三个事实：实际 token ID 相同，概率对应预测这个 token 的前一个位置，mask 排除 prompt 和 padding。

同一个自然语言词在不同 tokenizer 下可能拆成不同 token；单纯“输出字符串相同”无法保证逐 token 蒸馏成立。词表相等是本地保护条件，实践中还要确认 tokenizer 的规则、特殊 token 与模板兼容。

## 动手练习与诊断

```bash
.venv/bin/arl train 02-opd/general-opd/verl.yaml --smoke --verl-workers 2
```

练习：让教师和学生权重相同、上下文相同、dropout 关闭，理论上刚开始的 d 应是多少？答案是 0。若明显不是，先查采样温度、支持集、token 对齐和模型模式，再怀疑优化公式。

做能力实验时需要同时记录教师能力、学生任务准确率和采样 KL。只让学生更像一个弱教师，并不会自动提高任务得分。默认小模型用于学习成本控制，GPU 蒸馏效果尚未在本仓库验证。

我的判断：OPD 的关键资产是“教师与学生对同一条行为轨迹的两份概率账本”。账本对齐了，两行 loss 才有意义；账本错位时，loss 依然可能下降，因此优化器成功退出并不是蒸馏正确的充分证据。

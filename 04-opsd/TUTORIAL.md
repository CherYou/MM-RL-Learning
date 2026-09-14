# 04｜OPSD：同一个起点的模型，看到解题过程后能否教会只看题目的自己

[学习路线](../docs/BEGINNER_GUIDE.md) · [OPD 基础](../02-opd/general-opd/TUTORIAL.md) · [运行说明](README.md)

一个学生独立做题可能卡住，但看过参考解后能解释“这一行为什么应该这样写”。OPSD 将这种信息差用于蒸馏：Student 只看题目；Self-Teacher 看题目加训练期参考 solution，再评价 Student 自己生成的 token。

![两个同起点模型分别只看题目、看题目与参考解；教师反馈返回学生，参考解不直接进入学生输入](../docs/assets/algorithms/opsd.png)

插图里的隔板表示信息边界。教师看过答案不意味着学生在评估时也可以看。方法背景见 [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734)；下面明确描述本仓库固定 step-0 Teacher 的实现选择。

## “同样参数”不等于“同样概率”

语言模型的概率同时依赖权重与上下文：

$$p_S(y_t)=\pi_\theta(y_t\mid x,y_{1:t-1}),\qquad
p_T(y_t)=\pi_{\theta_0}(y_t\mid x,z,y_{1:t-1}),$$

z 是参考 solution，$\theta_0$ 是开训时的固定权重。即使开始时 $\theta=\theta_0$，只要上下文不同，两份分布就可能不同。

例如题目是“17×23 等于多少”。学生生成到“17×20=340，17×3=51，所以总共”时，教师已经看到参考推导，更可能支持正确的后续 token。教师评分的是学生真实写下的路径，而不是要求学生逐字复制 solution。

## loss 与 OPD 相同，改变的是教师条件

在学生采到的 token 上：

$$d_{i,t}=\operatorname{sg}\left[
\log\pi_{old}(y_{i,t}\mid x_i,y_{i,1:t-1})-
\log\pi_{\theta_0}(y_{i,t}\mid x_i,z_i,y_{i,1:t-1})\right],$$

$$L_{OPSD}=\frac{\sum_{i,t}m_{i,t}
e^{\ell_{i,t}-\ell^{old}_{i,t}}d_{i,t}}{\sum_{i,t}m_{i,t}}.$$

假设学生对某 token 给 0.25、看过解答的教师给 0.5，则 d≈−0.6931，初始 ratio=1 时推动学生提高该 token 概率。若去掉解答后两模型分布完全相同，初始 d=0；这提供了检查信息差是否真的接入的对照。

这里是 sampled-token reverse-KL surrogate，局部梯度解释与 [General OPD](../02-opd/general-opd/TUTORIAL.md)一致。没有额外的 value critic，也没有把 solution 当 Student 的 SFT 目标序列。

## 新手最容易忽略的 token 位置

假设 Student prompt 长 10，生成 4 个 token。Teacher 因加入 solution，prompt 长 30。要比较的是：

| 内容 | Student 输入位置 | Teacher 输入位置 |
| --- | --- | --- |
| 第一个生成 token | 10 | 30 |
| 它的预测 logprob 所在位置 | 9 | 29 |
| 后三个生成 token | 11–13 | 31–33 |

注意位置从 0 开始。不能直接把两张 `[positions]` 数组相减，必须从各自 prompt 尾部取同一 completion 的概率。

对应 [trainers.py](../src/agentic_rl/trainers.py) 的 `teacher_logprobs`，关键代码摘录为：

```python
suffix = s.tokens[s.prompt_length :]
t = Sample(ids + suffix, [0.0] * len(ids) + s.mask[s.prompt_length :], len(ids))
lp, _, _ = teacher.score([t])
aligned[s.prompt_length - 1 :] = lp[0, len(ids) - 1 :].to(student.device)
```

`ids` 是更长的 Teacher prompt；`suffix` 保留 Student 的 token IDs；新的 mask 排除 Teacher 多出来的上下文；最后一行把 completion 概率搬回 Student 坐标系。`privileged="solution"` 的分支负责把参考解仅放入 Teacher prompt。

这也是为什么不能偷偷截断 Teacher prompt 然后仍按旧下标对齐。上下文不够时需要明确筛选数据、调整预算或使用支持更长上下文的模型。

## 冻结与更新分别在哪里

[worker.py](../src/agentic_rl/verl_backend/worker.py)负责角色模型，OPSD Teacher 固定为 step-0 副本；[algorithms.py](../src/agentic_rl/verl_backend/algorithms.py)在 build 中选择 solution 特权复评并构造 `teacher_log_probs - old`；[verl losses.py](../src/agentic_rl/verl_backend/losses.py)通过 `lab_opd` 更新 Student。

与 AgentOPSD 区分：本章用固定 step-0 Teacher 并直接给逐 token 蒸馏信号；[AgentOPSD](../09-AgentOPSD/TUTORIAL.md)在每批使用当前策略的额外 Skill 视角来调整轮次信用，还保留终局 GRPO 优势的方向。两者不应只因为名字接近就写成同一算法。

## 自己验证什么

```bash
.venv/bin/arl train 04-opsd/verl.yaml --smoke --verl-workers 2
```

建议做三个检查：开训前后 Teacher 权重摘要相同；Student 轨迹的 prompt 没有 solution；Teacher 与 Student 的 completion IDs 相同。再做一个消融：固定同一批 Student 轨迹，只改变 Teacher 是否看到 solution，观察 gap 如何变化。

练习：Teacher prompt 从 30 token 增长为 35，Student 已生成 4 个 token，需要重新生成 Student 回答吗？答案：不需要。应保留这 4 个 token，在新 Teacher 上下文复评并重新对齐。若重生成，就换了被评价对象。

我的判断：OPSD 的有效性依赖“看到答案后更会评价”，这不是所有模型都天然具备的能力。参考解错误、过长或模型无法利用参考解时，监督也可能差。必须在不含 solution 的独立评估上检查 Student，而不能用 Teacher 看到解答时的得分代替学生效果。本地尚无本章 GPU 能力实验。

参考解数据来自 [Openthoughts_math_30k_opsd 发布页](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd)，取样与来源版本见 [DATA.md](../docs/DATA.md)。

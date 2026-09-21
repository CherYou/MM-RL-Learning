# 14｜AgentOPSD：自教师辅助的轮次信用

<!-- NAV:TOP:BEGIN -->
[← 上一章：13 Medical OPD / SAR / IDT：领域调度案例](../02-opd/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/03-distillation.md) · [下一章：15 TEMPO：短分支、估值与状态恢复 →](../09-tempo/TUTORIAL.md)
<!-- NAV:TOP:END -->

[学习路线](../docs/LEARNING_PATH.md) · [ALFWorld](../08-alfworld/TUTORIAL.md) · [General OPD / OPSD 角色](../02-opd/general-opd/TUTORIAL.md) · [运行入口](README.md)

**本章默认教学后端：verl（`verl.yaml`）。** Loss 仍是裁剪 policy gradient；真正新增的是**轮次信用重加权**。

**相对 GRPO / ALFWorld，改变了什么？**

| 组件 | GRPO 终局广播 | AgentOPSD（本章） |
| --- | --- | --- |
| 优势粒度 | 整条轨迹一个 A | 每轮 $`\widetilde A_k`$ |
| Teacher | 通常无 / 冻结 ref | **当前批次、更新前**策略 + Skill 文本 |
| 中间量 | 组均值/方差 | evidence → belief → credit → weight |
| 边界 | — | 有界乘数；**不翻转**终局方向 |
| A=0 组 | 无信号 | 仍为 0，Skill 不能凭空造监督 |

一个家庭任务用了十轮才成功，决定成败的可能只有“发现物体在另一个房间”那一轮。GRPO 把整段优势广播给所有轮次，无法表达这种差别。AgentOPSD 增加一个带 Skill 的自教师视角，给每轮提供证据，再有界地重新分配终局优势。

![带 Skill 的额外评价视角检查每轮，整段正向结果被分配成不同强度的正向信用](../docs/assets/algorithms/agentopsd.png)

图中大小不同的绿色标记表示重加权；不能据此认定某一步在真实因果意义上必然最重要。方法源自 [AgentOPSD 论文](https://arxiv.org/html/2608.05987v1)。本章按本地代码逐步解释计算，不复用论文图或原文段落。

## 一张表串起五个中间量

设同题奖励 `[1,1,0,0]`，取一条**成功**轨迹；组优势 $`A=0.5/(0.5+10^{-4})\approx0.999800`$，$`b_0=0.5`$，$`\gamma=0.95`$，$`\rho=0.2`$，$`\lambda=0.5`$。三轮证据指定为 $`[\log 3,\ 0,\ -1]`$（构造例，可复算）：

| 轮次 k | e_k | u_k | b_k | credit Δb | 标准化 z | **w**（clip 后） | **倍率 m** | 最终优势 Ã |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.098612 | 1.098612 | 0.750000 | 0.250000 | 1.247631 | **1.200000** | 1.100000 | 1.099780 |
| 2 | 0.000000 | 1.043682 | 0.739560 | −0.010440 | −0.048429 | 0.990314 | 0.995157 | 0.994958 |
| 3 | −1.000000 | −0.008502 | 0.497874 | −0.241685 | −1.199202 | **0.800000** | 0.900000 | 0.899820 |

本地公式：$`w_k=\mathrm{clip}(1+\rho z_k,\,1-\rho,\,1+\rho)`$（范围 **[0.8, 1.2]**），$`m_k=(1-\lambda)+\lambda w_k`$（范围 **[0.9, 1.1]**），$`\widetilde A_k=A\,m_k`$。**不要把 w 的上界写成 1.1。** belief 不是校准成功概率；重加权不是因果贡献证明；A=0 时倍率不能凭空造出监督。

## 为什么终局对错不足以解释每一轮

先用一个失败例子定位问题：模型先找到苹果、拿起苹果，但最后把它放到错误容器。终局奖励是 0，普通组相对更新可能让整段动作都得到负优势；然而“找到苹果”未必应该与“放错容器”承受相同力度。**Credit assignment（信用分配）**讨论的就是最终结果应怎样影响早先各步。

AgentOPSD 的 OPSD 部分沿用 On-Policy Self-Distillation（学生自采样自蒸馏）的思想，但本章用法不是直接以教师 gap 替代任务奖励。它保留环境结果决定的总体方向，再借助额外信息调整轮次力度。Skill 指训练期给教师视角的任务策略或操作知识文本，不是另一个隐藏的奖励数值，也不是模型参数里的“技能按钮”。

本章分四个层次走：教师多知道什么；把额外视角变成什么证据；怎样累计成 belief（内部信念分数）；最后如何有限度地修改优势。先不要把 belief 当成真实环境成功概率，它只是用于重加权的估计信号。

## 自教师和 OPSD 的区别

从角色开始，是因为“自己教自己”仍然可能指固定教师或当前策略两种不同设置。它们何时变化，会决定反馈的时间基准。

本章的 Teacher 视角使用**当前批次、更新前的同一策略**，额外读取训练期 Skill；复评分数停止梯度。它不是 [OPSD 章](../04-opsd/TUTORIAL.md) 从第 0 步一直冻结到最后的另一套参数，也不是一个预测 value 的 critic。

Student 正常与环境交互；Teacher 只复评这些实际动作 token。无需额外生成一套反事实轨迹。部署评估时 Student 不读取训练期 Skill。

## 第一步：把 token gap 累计为轮次证据

教师已经复评了学生真实动作，现在要把逐 token 反馈汇总到一次环境动作。这里的 turn（一轮）可能包含多个文字 token，不能把 token 与环境步混成同一计数单位。Logprob 的和对应整个动作文字概率的 log，因此本地用求和构造轮次证据。

对第 k 轮生成 token 集合 $`\mathcal T_k`$：

```math
e_k=\sum_{t\in\mathcal T_k}\left(\ell^T_t-\ell^{old}_t\right).
```

ell-T 与 ell-old 是带 Skill 的教师和正常采样策略对相同 token 的 logprob。这里使用求和，不是每轮平均。教师在 Skill 条件下更支持这轮实际生成内容时，e 通常偏正；反之偏负。它是两份条件分布的对比，不是环境给出的真实成功概率。

## 第二步：递推一个有记忆的 belief

单轮 evidence 很噪声，也缺少任务之前的背景。下一步给它一个起始水平，再累计多轮证据。b 是 0 到 1 的内部信念分数，odds 为 b/(1-b)，log-odds 是其对数；sigmoid 将任意实数变回 0 到 1。这样就能在 log-odds 空间相加，在概率尺度看变化。

如果起始 b=0.5，odds=1；加入 log3，相当于 odds 乘 3，变成 3，再换回 b=3/(1+3)=0.75。下面的公式就是将这个过程加上证据记忆衰减。它不是仅凭教师打分就获得了校准良好的贝叶斯后验。

以同题组平均成功率为初始 $`b_0`$，先夹到 $`(\varepsilon,1-\varepsilon)`$，避免取无穷 logit。设累计证据 $`u_0=0`$：

```math
u_k=\gamma u_{k-1}+e_k,\qquad
b_k=\sigma\left(\log\frac{b_0}{1-b_0}+u_k\right),\quad \Delta b_k=b_k-b_{k-1}.
```

本地默认 $`\gamma=0.95`$。它是证据记忆衰减，不是这里的环境奖励折扣。即使某轮 e=0，旧证据也可能因为衰减而改变 belief；因此“本轮没新证据”不必意味着 $`\Delta b=0`$。

例子：$`b_0=0.5,e_1=\log3\approx1.0986`$，则 $`b_1=0.75`$，变化为 0.25。若下一轮 $`e_2=0`$，则 $`u_2\approx1.0437,b_2\approx0.7396`$，变化约 -0.0104。这种递推显式依赖历史。

## 第三步：改变力度，保持终局方向

得到 belief 变化后，还不能直接把它当最终优势：辅助信息可能有误，环境成功与否仍是主要依据。因此本地先按终局优势方向调整信用，再对轮次标准化、限制幅度，最后混回原优势。Sign 取正负号，标准化使轮次差异具有可比较尺度，clip 则防止少数极端差异主导更新。

设轨迹的 GRPO 优势为 $`A`$。先按结果方向定义 $`c_k=\mathrm{sign}(A)\Delta b_k`$，对这条轨迹的轮次 c 做标准化得 $`z_k`$，再构造：

```math
w_k=\mathrm{clip}(1+\rho z_k,1-\rho,1+\rho),
```

```math
\widetilde A_k=A[(1-\lambda)+\lambda w_k].
```

本地 $`\rho=0.2,\lambda=0.5`$，所以乘数落在 `[0.9,1.1]`。若 A=2，每轮优势在 `[1.8,2.2]`；若 A=−2，则在 `[−2.2,−1.8]`。A=0 的退化组仍为零，不能被 Skill 凭空变成正负监督。

**Loss 仍是裁剪 policy loss**，只是把每个 token 的 $`A_i`$ 替换为所属轮次的 $`\widetilde A_{i,k(t)}`$。B 是轨迹数，T 是有效生成长度，m 是生成位置 mask，r 是当前/采样概率比，epsilon-l/h 是裁剪幅度，k(t) 表示 token t 所属轮次：

```math
L=-\frac1B\sum_i\frac1{T_i}\sum_tm_{i,t}
\min(r_{i,t}\widetilde A_{i,k(t)},\mathrm{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)\widetilde A_{i,k(t)}).
```

## 从代码到数组

纸上有 evidence、belief、credit、weight、advantage 五个中间量，代码也应能逐项打印。读实现时按这条链追踪，能避免把“教师更支持”直接跳读成“这个动作应被正向强化”。

[algorithms.py](../src/agentic_rl/verl_backend/algorithms.py) 先调用 `teacher_logprobs(..., "skill")`，按 `turn_spans` 汇总 gap，再调用 [agentopsd.py](../src/agentic_rl/agentopsd.py) 的 `reshape_turn_advantages`。该函数返回每轮 evidence、belief、credit、weight 和 advantage，便于逐项检查。

下面是教学伪代码：

```python
for turn in turns:
    evidence = sum(teacher_logp[turn] - behavior_logp[turn])
    memory = gamma * memory + evidence
    belief = sigmoid(initial_log_odds + memory)
    credits.append(sign(outcome_advantage) * (belief - previous_belief))
    previous_belief = belief
turn_advantages = bounded_rescale(outcome_advantage, credits)
```

实际函数还处理数值稳定、零方差与完整诊断。旧概率长度比完整 token 少一位，所以 `turn_spans` 用于概率数组时会减一；忽略这个 shift 会把证据挪到错误轮次。

## 动手与误区

最后用边界情况检查你的理解：A=0 时任何辅助证据都不应制造非零任务优势；A<0 时有界乘数应保留负号。下面的 smoke 之后，优先核对这两点与每轮索引。

```bash
.venv/bin/arl train 09-AgentOPSD/verl.yaml --smoke --verl-workers 2
```

练习：一条失败轨迹 A<0，某轮看起来很有帮助，是否应该把这轮优势改成正数？按本地有界重加权不会。它只改变负向信号的大小，保持整段的结果方向。若你设计允许翻转符号的更新，那已经是另一种信用分配规则。

检查 `turn_evidence` 和 `turn_advantages`，并确认同一轮 token 共用同一个优势、observation 不训练、Student 评估不带 Skill。论文性能数字不属于本地结果；目前本章仅有机制与 CPU 运行证据。

我的判断：把这个方法当作“额外信息对结果监督的温和修正”比当作“知道每步真贡献的裁判”更稳妥。单轮 gap 不是因果证据，递推与边界限制是在控制这种辅助信号对更新的影响。

## 新手自测

练习：第二轮 evidence=0，belief 为什么仍可能下降？因为前一轮的累计证据会按 gamma 衰减。再问：失败轨迹里的好动作能否通过当前重加权变成正优势？不能，该实现只减轻或加重原来的负向力度。这同时说明方法的保护作用与局限：它不会完全恢复每一步的真实因果贡献。

阅读 [AgentOPSD 作者论文](https://arxiv.org/html/2608.05987v1) 时可对照 recursive belief 与 advantage reshaping 两部分；本章的具体边界、零方差处理和变量由本地 `reshape_turn_advantages` 决定。

<!-- NAV:BOTTOM:BEGIN -->
[← 上一章：13 Medical OPD / SAR / IDT：领域调度案例](../02-opd/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/03-distillation.md) · [下一章：15 TEMPO：短分支、估值与状态恢复 →](../09-tempo/TUTORIAL.md)
<!-- NAV:BOTTOM:END -->

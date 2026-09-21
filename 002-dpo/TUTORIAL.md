# 06｜DPO：从偏好对直接优化策略

<!-- NAV:TOP:BEGIN -->
[← 上一章：05 GSPO：序列级比率与裁剪](../07-gspo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/01-policy-preference.md) · [下一章：07 Search-R1：学习检索与观察 mask →](../03-search-r1/TUTORIAL.md)
<!-- NAV:TOP:END -->

[学习路线](../docs/LEARNING_PATH.md) · [BCE 与偏好分差](../preliminary/TUTORIAL.md) · [运行说明](README.md)

**本章学习任务：** 用固定偏好对完成一次离线更新——先看清“分差 → 谁更好”的 logistic 目标，再接上 reference 校正的策略 logprob 差。

**建议前置：** [损失函数详解 §4](../preliminary/TUTORIAL.md)（二元交叉熵 / 偏好 margin）。不必先读完 PPO/GRPO；在线 RL 与离线偏好是两条数据条件不同的分支。

假设同一道题有两个回答，人工或规则告诉你 A 比 B 好。你没有每个 token 的分数，也未必想一边训练一边生成新答案。DPO 直接使用这种偏好对，增加模型对 preferred 回答相对 rejected 回答的偏好，并用固定 reference 作比较基准。

![固定数据中的 chosen/rejected 回答对与冻结 reference 一起决定学生偏好更新](../docs/assets/algorithms/dpo.png)

图中数据箱是固定的，不是每个更新步重新采样的 rollout。DPO 不需要单独拟合 reward model 来计算这项训练 loss，来源见 [DPO 原论文](https://arxiv.org/abs/2305.18290)。

## 为什么一对偏好能成为训练数据

DPO 全称 **Direct Preference Optimization，直接偏好优化**。Preference 是“同一问题下，更希望模型给出哪个回答”；direct 指不必先单独训练一个奖励网络，再用在线 RL 优化这个奖励。这里先研究标准离线 DPO：训练时已有回答对，模型不需要每步重新生成候选。

先想象只有 chosen：我们可以做 SFT，即 Supervised Fine-Tuning（监督微调），提高示范答案的概率。但这样没有明确利用 rejected 所说明的“哪些回答应相对不受偏好”。DPO 把两个候选放在同一个比较中学习，又用 reference 保留原来的概率尺度。它不需要知道每个 token 应该得几分，只需要一条回答相对于另一条更受偏好。

这并不意味着必须惩罚 rejected 的每个词。例如两个回答都正确写出“3 盒，每盒 6 支”，只在最后算错。整段偏好标签没有标出错误位置；DPO 的监督单位仍是完整回答，细粒度原因不会自动出现在标签里。

## 先从“分差预测谁更好”开始（BCE 桥）

在写完整 DPO 公式前，先只看偏好标签本身。设两个标量分数 s_A、s_B，分差 d=s_A−s_B，A 更好时：

```math
\sigma(d)=\frac1{1+e^{-d}},\qquad L=-\log\sigma(d).
```

d=0 时损失为 log2，对 d 的梯度为 −0.5：训练会**增大 A 相对 B 的分差**。这就是第一章的 logistic / BCE 形式，还没有出现 reference，也还没有规定“分数”必须是什么。

DPO 唯一的额外设计是：**用当前策略与 reference 的 logprob 差来构造这个分差**，而不是训练一个独立打分网络。下面各节只是在回答“分差从哪来”。

## 一行数据有什么

明确了监督单位，先检查数据能否表达它。一条样本必须让 chosen 和 rejected 对应**同一个 prompt**；如果把两道题的答案拼成一对，算法比较的就不再是同一条件下的偏好。

```json
{"prompt": "3盒，每盒6支，共多少支？", "chosen": "3×6=18，所以18支。", "rejected": "共有9支。"}
```

这是教学示例，不是对仓库数据逐字摘录。chosen/rejected 表示给定标注中的相对偏好，不保证 chosen 永远完美。仓库的 DPO 学习数据是明确标注的 GSM8K 派生偏好对，并不是大规模真实人类偏好采集结果。

## 先算回答概率，再把 logprob 差填进分差

为什么比较的是概率而不是让模型“直接读懂标签”？神经网络训练需要一个可求导的量。我们把已经给定的回答逐 token 输入，计算每个正确后续 token 的概率，这叫 teacher forcing（使用给定前缀打分），不是让模型现场自由生成。序列概率是条件概率的乘积，logprob 则是相加。例如两个 token 的条件概率 0.5、0.2，对应序列概率 0.1，logprob 为 log0.5+log0.2=log0.1。只加回答部分，排除题目和补齐空位。

用 x 表示题目，y+ 表示 chosen，y− 表示 rejected；theta 是当前模型参数，pi 表示回答的条件概率。设 $`\ell_\theta^+=\log\pi_\theta(y^+\mid x)`$，$`\ell_\theta^-=\log\pi_\theta(y^-\mid x)`$，reference 对应 $`\ell_{ref}^+,\ell_{ref}^-`$。这些是**整段生成 token 的 logprob 之和**。

现在把上一节的 d 换成“reference 校正后的策略偏好分差”：

```math
z=\beta\left[(\ell_\theta^+-\ell_\theta^-)-(\ell_{ref}^+-\ell_{ref}^-)\right],
```

```math
L_{DPO}=-\mathbb{E}_{(x,y^+,y^-)}\log\sigma(z),\qquad
\sigma(z)=\frac1{1+e^{-z}}.
```

直观上，$`\sigma(z)`$ 是偏好模型给“chosen 更好”的概率。我们希望这个概率变大，所以最小化它的负 log。这就是 BCE 桥里的 $`-\log\sigma(d)`$，只是 d 换成了 z。

Reference 校正问的是“相对于原来，你有没有更偏向 chosen”，而不是简单要求 chosen 的原始概率超过 rejected。

### 长度与合成负例：数据侧的两处混淆

| 现象 | 为什么危险 | 阅读/实验时怎么做 |
| --- | --- | --- |
| chosen 与 rejected 长度差很大 | 序列 logprob 是求和；更长回答常有更低的总 logprob，模型可能学到长度捷径，而非内容偏好 | 手算时对比“长度相近”与“长度悬殊”两对；评估不要只看 chosen 历史 logprob 是否更高 |
| 合成负例（本仓库 GSM8K 派生） | 负例往往更短、格式更简单，甚至只是答案数字 +1 | 它适合验证 **margin 与训练接口**，不能当作真实人类偏好能力的证据 |
| 把求和改平均后仍称 DPO | 统计单位变了，目标不再与标准 DPO 相同 | 若改 reduction，必须在配置与文中标明 |

仓库 DPO 学习数据是明确标注的 GSM8K 派生偏好对，并不是大规模真实人类偏好采集结果。长短回答的序列概率受长度影响，不能随手把求和改平均后仍称相同的 DPO 目标。

## 手算 margin 和梯度方向

上节的 z 叫 margin（比较间隔）：z=0 时预测偏好概率为一半，z 越大越支持 chosen。Sigmoid 将任意实数压到 0 与 1 之间，负 log 则在预测违背偏好时给较大损失。接下来用一对数字检查方向，而不只记住公式形状。

假设当前模型对两回答的 logprob 是 -2、-4，reference 是 -3、-4，取 $`\beta=0.1`$：

```math
z=0.1[( -2+4)-( -3+4)]=0.1,\qquad L=-\log\sigma(0.1)\approx0.6444.
```

若当前模型等于 reference，则 z=0，loss=log2≈0.6931。但此时梯度不为零：

```math
\frac{\partial L}{\partial z}=\sigma(z)-1;\qquad
\frac{\partial L}{\partial\ell_\theta^+}=\beta(\sigma(z)-1)\lt 0,\quad
\frac{\partial L}{\partial\ell_\theta^-}=-\beta(\sigma(z)-1)\gt 0.
```

因此梯度下降提高 chosen 的相对偏好、降低 rejected 的相对偏好。并不能保证每步 chosen 的绝对概率一定上涨，因为共享参数还受到其它样本影响。

$`\beta`$ 在理论推导里与 reference 约束相关，同时在这个 logistic loss 中缩放 margin。不能把“beta 越大”简单解释成“每一步越保守”：局部梯度大小还取决于当前 margin，跨设置比较需要实际评估。

## 为什么可以省去显式奖励模型

已经知道这项 loss 怎样工作，现在解释为什么它与奖励学习有关系。下面分三步，额外引入的 r 表示假设存在的回答奖励，并非代码真的调用了奖励模型。

第一步，假设偏好遵循 Bradley–Terry 成对比较模型：chosen 相对 rejected 的奖励差越大，被偏好的概率越高。这是一种建模假设，不是所有人类偏好必然遵守的定律。

```math
P(y^+\succ y^-\mid x)=\sigma(r(x,y^+)-r(x,y^-)).
```

第二步，KL 即 Kullback–Leibler divergence（KL 散度），用于衡量两份分布的差异。对“期望奖励减去相对 reference 的 KL 惩罚”这个理想优化问题，最优策略是 reference 按奖励指数重加权，再除以归一化常数 Z。Z 把所有候选的未归一化权重加起来，确保最终概率总和为 1；星号表示这个理想最优策略：

```math
\pi^*(y\mid x)=\frac{\pi_{ref}(y\mid x)e^{r(x,y)/\beta}}{Z(x)}.
```

第三步，两边取对数并移项，得到奖励的另一种表达：

```math
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)}+\beta\log Z(x).
```

同题比较时，两份奖励里的 Z(x) 相同，相减后消失。将可训练策略代入这个参数化，再代回第一步的偏好概率，就得到前面的 z 与负 log-sigmoid loss。因此训练可以直接调整策略，而不显式估计 Z 或先拟合 r。[原论文的推导](https://arxiv.org/abs/2305.18290)给出这一对应；这里不是声称有限数据、有限模型训练一定找到了理想最优解。

这个推导不说明“所有奖励问题都可以只靠现成偏好对解决”。DPO 依赖数据中出现了什么回答、偏好标签是否可信，以及 reference 和训练分布是否合适。

## 真实代码路径

推导中的奖励消去了，代码中就应只剩四份回答 logprob：当前 chosen/rejected 与 reference chosen/rejected。带着这四个量读实现，可以检查是否不小心引入了额外的生成或评分步骤。

默认 [config.yaml](config.yaml)调用 [trl_backend.py](../src/agentic_rl/trl_backend.py) 的真实 `DPOTrainer`，传入 prompt/chosen/rejected 三列和独立 reference 副本。本地显式对照在 [trainers.py](../src/agentic_rl/trainers.py) 的 `algo == "dpo"` 分支。

先把 chosen 与 rejected 都做 teacher-forcing 打分，排除 prompt/padding 后求 token logprob 之和，再调用 [losses.py](../src/agentic_rl/losses.py) 的 `dpo_loss`。核心摘录：

```python
margin = beta * ((chosen_logp - rejected_logp) - (ref_chosen - ref_rejected).detach())
loss = -torch.nn.functional.logsigmoid(margin).mean()
```

这里不对 chosen/rejected 的 current logprob `.detach()`，因为那是训练路径；reference 的差值固定。`logsigmoid` 避免先算很小的 sigmoid 再取 log 带来的数值问题。

## DPO 与 PPO/GRPO 怎样选择学习视角

看完执行路径，可以回到选择方法的问题：你手上是固定的可靠偏好对，还是能够让模型不断尝试并获得奖励？两种数据条件决定了系统需要哪些部分，不能只按哪条 loss 更短来判断。

| 问题 | DPO | PPO/GRPO |
| --- | --- | --- |
| 每步必须在线生成新回答吗 | 标准离线 DPO 不需要 | 训练循环包含策略 rollout |
| 学习信号来自哪里 | 同题偏好对 | 奖励与优势估计 |
| 要不要 value critic | 不需要 | PPO 需要，GRPO 不需要 |
| 容易漏掉什么 | 数据覆盖不足、偏好噪声、长度偏差 | 奖励问题、采样成本、退化组 |

## 动手练习

运行前先预测：同一对数据、相同初始策略和 reference 应给出接近 log2 的 loss，但仍可以更新；chosen 与 rejected 完全相同时则没有区分梯度。用这两种情况能区分“分数相同”和“计算图相同”。

```bash
# 默认教学入口：与本章 config.yaml 的 TRL DPOTrainer 一致
.venv/bin/arl train 002-dpo/config.yaml --smoke --backend trl
# 等价薄入口
python 002-dpo/train.py --smoke
```

练习：互换 chosen/rejected，z 怎样变化？答案是变为 -z，训练方向反转。若一对回答实际上完全相同，两个概率及其梯度相同，无法从这个对中学到区分，即便 loss 仍约为 0.6931。

看指标时，`dpo/margin` 提高只说明训练目标上的相对偏好变化。仓库的 DPO 评估使用长度敏感的 chosen/rejected logprob 比较，不能直接称为生成式数学正确率；后者需要额外进行自由生成和答案校验。

我的判断：DPO 把系统复杂度从在线采样转移到了数据设计。最值得先读的往往是偏好对本身，而不是学习率。它也是理解具身离线学习的好入口，但语言回答偏好对与机器人状态—动作数据有不同的结构，不应直接把 DPO 公式套到未定义的动作概率上。实现 API 见 [固定版本 TRL DPO 文档](https://huggingface.co/docs/trl/v0.25.1/en/dpo_trainer)。

## 新手自测

用一句话解释 reference 为什么没有被优化：它提供固定比较起点，不能和学生一起移动尺子。再解释“DPO 不需要奖励模型”是否等于“不需要偏好质量”：不等于，质量要求转移到了回答对的覆盖和标注上。如果无法说明 Z(x) 为何抵消，回到推导确认两个回答是否共享同一题目。

<!-- NAV:BOTTOM:BEGIN -->
[← 上一章：05 GSPO：序列级比率与裁剪](../07-gspo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/01-policy-preference.md) · [下一章：07 Search-R1：学习检索与观察 mask →](../03-search-r1/TUTORIAL.md)
<!-- NAV:BOTTOM:END -->

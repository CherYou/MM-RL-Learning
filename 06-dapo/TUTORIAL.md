# 06｜DAPO：除了写对 loss，还要让一个 batch 值得训练

[学习路线](../docs/BEGINNER_GUIDE.md) · [GRPO 前置知识](../01-grpo/TUTORIAL.md) · [运行说明](README.md)

设一个 batch 的每道题都生成四份答案，但所有题不是全对就是全错。GRPO 的组内优势全部为零。优化器并不缺少执行指令，它缺少可以比较的差异。DAPO 把注意力从单个公式扩展到采样、裁剪、长度和统计单位。[DAPO 论文](https://arxiv.org/abs/2503.14476)给出了这些机制的系统组合。

![DAPO 按整组过滤无差异奖励、补采样、按有效 token 聚合并对过长回答施加渐进惩罚](../docs/assets/algorithms/dapo.png)

图里的过滤对象是整组回答。混合组中的错误回答仍留下，它们是比较所必需的负向信号。

## 先把四项变化拆开

| 变化 | 针对的问题 | 本仓库位置 |
| --- | --- | --- |
| Clip-Higher | 正向概率提升过早进入裁剪区 | `clip_low`、`clip_high` |
| 动态采样 | 一个组所有奖励相同，优势无差异 | `AlgorithmBatchBuilder.build` 重采循环 |
| token 级聚合 | 不同长度回答的 token 权重不同 | `reduction="token"` |
| 长度处理 | 过长输出及不可靠截断目标 | `overlong_penalty`、`mask_truncated` |

这些开关相互独立。只设置 `loss_type="dapo"`，不会凭空得到重新采样和长度奖励。仓库完整机制走默认 verl 路径；TRL 对应入口是 loss 对照，不应称作完整 DAPO 流程。

## 关键公式：裁剪边界不必对称

仍使用组内优势 $`A_i`$ 和 token ratio $`r_{i,t}`$：

```math
L_{DAPO}=-\frac{\sum_{i,t}m_{i,t}\min\left(r_{i,t}A_i,
\mathrm{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_i\right)}{\sum_{i,t}m_{i,t}}.
```

当前 [配置](config.yaml)为 $`\epsilon_l=0.2,\epsilon_h=0.28`$。因此区间是 `[0.8,1.28]`，并不是 `[0.2,0.28]`。对正优势样本，允许比对称上界 1.2 更大的概率提升后才截住额外激励。负优势时依然要经过 `min` 判断，不能把越界率直接等同于零梯度比例。

分母是所有有效生成 token 数。两条回答分别长 2 和 8 时，每个 token 的权重都是 1/10；若用 GRPO 序列平均，短回答每 token 权重为 1/4，长回答为 1/16。前者让长回答整体贡献更多 token，后者使每个回答整体权重相同。没有一种归一化脱离任务就永远正确。

## 动态采样的真实顺序

仓库首先按原始任务奖励判断退化组：$`\max_iR_i=\min_iR_i`$ 则丢弃整组并尝试新题。保留组之后才施加长度 shaping，然后重新计算用于学习的组优势。

为什么不能反过来？若四个答案都错，只因长度不同就得到不同 shaping 分数，提前 shaping 会把“任务奖励无差异”伪装成可学习组。当前顺序明确区分任务成功差异和长度偏好。

教学伪代码：

```python
for attempt in range(max_resample_batches):
    groups = rollout(pending_questions)
    for group in groups:
        if all_task_rewards_equal(group):
            queue_another_question()
        else:
            add_length_penalty(group)
            keep(group)
    if no_pending_questions():
        break
```

注意本地上限为 8 次补采，达到上限可能只留下部分组；若一个都没有，跳过本次更新。它不会无限循环到理想 batch 大小。要一起看 `sampling/attempts`、`sampling/degenerate_groups` 和 `update/skipped`。

## 长度惩罚怎么手算

设软上限 $`L_s`$、硬上限 $`L_h`$，长度为 T：

```math
P(T)=-\mathrm{clip}\left(\frac{T-L_s}{L_h-L_s},0,1\right),\qquad R'_i=R_i+P(T_i).
```

本章配置 $`L_s=192,L_h=256`$。长度 160 的惩罚是 0；224 是 -0.5；256 是 -1。若一个正确回答长 224，其 shaped reward 为 0.5，而不是原来的 1。

`mask_truncated: true` 还会把被标记为截断的回答的训练 mask 清零。这是另外一项操作：改变 reward 与决定是否让该回答参与 loss 不能混为一谈。原始奖励非退化，也不保证清 mask 后还有足够的有效梯度。

## 代码阅读路线与常见错误

先看 [algorithms.py](../src/agentic_rl/verl_backend/algorithms.py) 的 `while active` 循环，再读 [losses.py](../src/agentic_rl/losses.py) 的 `overlong_penalty`，最后读 [verl losses.py](../src/agentic_rl/verl_backend/losses.py) 的 `reduce_tokens`。分布式时分母必须是全局 token 数，而不能每个 GPU 先各自平均再把结果当全局平均。

精简代码摘录：

```python
penalty = -((lengths - soft_limit) / (hard_limit - soft_limit)).clamp(0, 1)
numerator = (values * mask).sum()
denominator = config["global_tokens"]
```

第一行只依赖长度；第二、三行决定有效 token 的权重。若改了裁剪边界，奖励分布不一定直接改变；若改了补采策略，题目分布会改变。比较实验时要记录这些差别，不能把所有变化都归功于 loss。

## 动手实验与思考

```bash
.venv/bin/arl train 06-dapo/verl.yaml --smoke --verl-workers 2
```

这个 CPU 检查使用明确标注的 debug 奖励，不代表数学表现。真实实验可固定同一批题、种子和生成预算，先只比较是否动态补采，再比较 token/sequence reduction。记录“每秒完成多少有效 token 更新”和“用了多少生成 token”，否则反复补采的成本会被隐藏。

练习：原始奖励 `[0,0,0,0]`，长度 `[100,200,224,256]`，会不会因为长度不同而被保留？答案：不会。当前代码先判原始奖励退化，整组进入重采路径。

我的判断是把 DAPO 看成“训练数据供应与更新规则的联合设计”更有帮助：没有有效对比的 batch、被截断的回答、不同长度的权重都发生在 loss 周围。实际阅读应核对 [论文](https://arxiv.org/html/2503.14476v1)和本地三个代码位置，不能只搜一个函数名。本章尚无 GPU 能力验证。

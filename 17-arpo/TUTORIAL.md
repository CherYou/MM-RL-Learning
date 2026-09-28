# 07｜ARPO：工具后的高熵分支采样与步级信用

<!-- NAV:TOP:BEGIN -->
[← 上一章：06 DPO：从偏好对直接优化策略](../002-dpo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [下一章：08 AEPO：熵平衡的智能体策略优化 →](../18-aepo/TUTORIAL.md)
<!-- NAV:TOP:END -->

[← 上一章：06 DPO](../002-dpo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [运行说明](README.md)

**本章默认教学后端：verl（`verl.yaml`）。** 建议先掌握 [GRPO](../01-grpo/TUTORIAL.md) 的组内优势，以及 [ReTool](../05-retool/TUTORIAL.md) 中「工具观察 mask=0」的约定。

**本章学习任务：** 说清 ARPO 相对「多轮工具 + GRPO」新增的两件事——（1）在**工具反馈后的高不确定步**附近自适应多采一点；（2）把终局优势**按工具步归因**，而不是整条轨迹一个数。

**相对 GRPO / 普通工具轨迹，改变了什么？**

| 组件 | 基础工具 GRPO | ARPO（本章） |
| --- | --- | --- |
| Rollout 预算 | 每题固定 G 条完整轨迹 | 全局采样 + **高熵工具步分支**（entropy 预监测） |
| 优势 | 终局组统计，整轨广播 | **步级归因**（可加熵权重），再进 clipped PG |
| 工具观察 | mask=0 | **仍是 mask=0** |
| 主要风险 | 工具步探索不足 | 分支过多浪费预算；归因≠因果真值 |

来源：[ARPO 原文](https://arxiv.org/abs/2507.19849)（ICLR 2026）· [官方代码 RUC-NLPIR/ARPO](https://github.com/RUC-NLPIR/ARPO)（基于 verl + 工具搜索栈）。以下机制为**本地教学最小实现**，超参与完整工程以原文为准。

---

## 1. 一个具体问题：工具回来以后，模型在“想什么”

固定历史下，用温度 1 的分类分布模拟“下一 token 分布”。假设模型要决定下一条检索查询怎么写。

**工具尚未返回时**，分布较集中（熵较低）：

| 候选 | 概率 |
| --- | ---: |
| `search 谁写了红楼梦` | 0.70 |
| `search 红楼梦 作者` | 0.20 |
| `search 曹雪芹 生卒` | 0.10 |

熵约 $`H\approx0.90`$ nat（手算：$`-\sum p\log p`$）。

**工具返回「作者：曹雪芹」之后**，下一轮要问出生地，但表述很多，分布变散（熵升高）：

| 候选 | 概率 |
| --- | ---: |
| `search 曹雪芹 出生地` | 0.35 |
| `search 曹雪芹 出生于` | 0.25 |
| `search Cao Xueqin birthplace` | 0.25 |
| 其它 | 0.15 |

熵升到约 $`H\approx1.34`$。原文的观察是：**每轮工具反馈后的首批 token 往往熵更高**——外部结果给推理带来了新的不确定性，也恰恰是“下一条查询写什么”的关键决策点。

若整条轨迹只做 G 条全局 rollout，这些步的探索预算被摊薄；ARPO 的回答是：**哪里更不确定，就哪里多采一点**，并把信用算清楚。

---

## 2. 分支采样：预算怎样跟着熵走

### 2.1 单步是否分支

工具返回后算 token 熵 $`H`$。若 $`H\ge\tau`$（阈值），以概率 $`p_{branch}`$ 在该步附加分支，且受 $`max\_branches\_per\_turn`$ 限制：

```math
\mathrm{branch}(H)=0\ \mathrm{if}\ H\lt\tau,\qquad
\mathrm{branch}(H)\sim\mathrm{Bernoulli}(p_{branch})\ \mathrm{if}\ H\ge\tau.
```

教学实现中还用 $`beam\_size`$ 限制单步并行候选数。**连续高熵步不应无限开叉**——这正是下一章 AEPO 要收紧的地方。

### 2.2 多步之间的预算分配

设总 rollout 预算 $`B`$，各工具步熵 $`H_i`$，额外权重：

```math
w_i=\max(0,H_i-\tau)+\varepsilon,\qquad
n_i=1+\Big\lfloor (B-T)\frac{w_i}{\sum_j w_j}\Big\rfloor,
```

$`T`$ 为步数（每步至少 1 条），$`n_i`$ 再截断到 $`beam\_size\times max\_branches`$。

**手算例：** $`\tau=1`$，熵 $`[0.4,1.8,2.2]`$，$`B=10`$，$`T=3`$。  
$`w=[\varepsilon,0.8+\varepsilon,1.2+\varepsilon]`$（取 $`\varepsilon=10^{-6}`$ 时前两步几乎 0）。  
额外预算 7 几乎全给后两步 → demo 中 $`[1,3,5]`$ 一类的分配（和为 9，因取整）。请核对：低熵步只保底 1 条，高熵步显著更多。

---

## 3. 步级优势归因：终局优势怎样落到“哪一次工具调用”

### 3.1 从终局到步

设轨迹 $`i`$ 的终局组优势为 $`A_i=R_i-b`$（GRPO 亦可）。若有步级信号 $`s_{i,t}`$（如工具成功指示）或熵 $`H_{i,t}`$，构造权重再归一：

```math
\alpha_{i,t}=\frac{\tilde w_{i,t}}{\sum_{t'}\tilde w_{i,t'}},\qquad
\tilde w_{i,t}=|s_{i,t}|+\varepsilon+\lambda_H\,\widehat H_{i,t},
```

$`\widehat H`$ 为行内 min-max 归一化熵。步优势：

```math
\widetilde A_{i,t}=A_i\,\alpha_{i,t},\qquad
\sum_t\alpha_{i,t}=1.
```

无步级信号时 $`\alpha`$ 均匀，退化为整轨广播。

### 3.2 手算例（与本地 demo 一致）

两条轨迹，终局奖励 $`R=[1,0]`$，组均值 $`b=0.5`$，故 $`A=[+0.5,-0.5]`$。  
步信号（示意“哪一步与答案更相关”）与熵：

| 轨迹 | t=0 信号 / 熵 | t=1 | t=2 |
| --- | --- | --- | --- |
| i=0（成功） | 0 / 0.2 | 1 / **2.0** | 0 / 0.1 |
| i=1（失败） | 0 / **1.5** | 0 / 0.3 | 1 / **1.8** |

则 $`\widetilde A`$ 的形状是：

- i=0：几乎全部正质量落在 **t=1**（高熵 + 有信号）→ 约 $`[0.005,\,0.495,\,0.000]`$
- i=1：质量落在 t=0 与 t=2 → 约 $`[-0.06,\,0,\,-0.44]`$

**读表规则：** 同一轨迹内 $`\sum_t\widetilde A_{i,t}=A_i`$；符号不因归因而翻转。失败轨迹里“看起来有用”的步仍然只能减轻或分担负号，**不能**自动变成正监督。

### 3.3 接到 token loss

步优势广播到该步的有效生成 token（观察仍 mask=0），再用 clipped PG（与 GRPO/DAPO 同型）：

```math
L=-\frac1B\sum_i\frac1{T_i}\sum_t m_{i,t}
\min\bigl(r_{i,t}\widetilde A_{i,k(t)},\,
\mathrm{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)\widetilde A_{i,k(t)}\bigr).
```

$`k(t)`$ 为 token $`t`$ 所属工具步。比率 $`r=\exp(\ell-\ell^{old})`$ 定义见 [第 01 章](../preliminary/TUTORIAL.md#behavior-ratio)。

---

## 4. 为什么不能把归因当成因果证明

| 常见误读 | 正确理解 |
| --- | --- |
| 高熵步权重最大 ⇒ 该步“最重要” | 只说明**更不确定 / 更值得探索** |
| 成功轨迹所有步都是正的 | 是；但幅度由 $`\alpha`$ 调节 |
| 失败轨迹可以靠 Skill/熵翻正 | 本地实现**不翻转**终局方向 |
| 工具文本进 loss | 观察 mask=0，只作条件 |

---

## 5. 数值实验与代码

```bash
# 独立 CPU demo（无需训练环境）
python examples/math/agentic_rl_demo.py --algorithm arpo

# 仓库 verl smoke（tiny 模型 + fixture）
.venv/bin/arl train 17-arpo/verl.yaml --smoke --verl-workers 2
python 17-arpo/train.py --smoke
```

| 位置 | 看什么 |
| --- | --- |
| [src/agentic_rl/arpo.py](../src/agentic_rl/arpo.py) | `token_entropy`、`should_branch_after_tool`、`allocate_branch_budget`、`attribute_step_advantages`、`arpo_surrogate_loss` |
| [verl.yaml](verl.yaml) | 默认 verl 配置 |
| [官方仓库](https://github.com/RUC-NLPIR/ARPO) | 同步带工具的 rollout、搜索 API、`initial_rollouts` / `beam_size` / `branch_probability` / `entropy_weight` 等完整栈 |

上游训练默认名（对照用，非本地 smoke）：`initial_rollouts=8`、`beam_size=2`、`branch_probability=0.5`、`entropy_weight=0.2`、组内 $`G=16`$ 一类设置；见官方脚本。

---

## 6. 练习

1. **预测再运行：** 熵全为 $`0.5\lt\tau`$ 时，`allocate_branch_budget` 与「每步 1 条、其余随机摊」有何差别？  
2. 把 $`\lambda_H`$（demo 中 `entropy_weight`）改为 0，步优势变成什么？何时这等于 GRPO 整轨广播？  
3. 一条失败轨迹某步熵极高且工具“看起来成功”，$`\widetilde A`$ 会变成正吗？为什么？  
4. 若把观察 token 的 mask 误设为 1，ARPO 的归因会怎样污染学习目标？

---

## 7. 边界与下一步

- 本地只验证机制与数值方向，**不等于** GAIA / HLE / 13 个 benchmark 的论文复现。  
- 工具预算、缓存、多线程执行等工程优化不在本章展开。  
- **下一篇：** [08 AEPO：熵平衡的智能体策略优化](../18-aepo/TUTORIAL.md)——在熵分支之上防止过分支与高熵梯度被裁掉。  
- 需要先补工具轨迹字段时：[Search-R1](../03-search-r1/TUTORIAL.md) / [ReTool](../05-retool/TUTORIAL.md)。

<!-- NAV:BOTTOM:BEGIN -->
[← 上一章：06 DPO：从偏好对直接优化策略](../002-dpo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [下一章：08 AEPO：熵平衡的智能体策略优化 →](../18-aepo/TUTORIAL.md)
<!-- NAV:BOTTOM:END -->

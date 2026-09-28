# 08｜AEPO：熵平衡的智能体策略优化

<!-- NAV:TOP:BEGIN -->
[← 上一章：07 ARPO：工具后的高熵分支采样与步级信用](../17-arpo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [下一章：09 Search-R1：学习检索与观察 mask →](../03-search-r1/TUTORIAL.md)
<!-- NAV:TOP:END -->

[← 上一章：07 ARPO](../17-arpo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [运行说明](README.md)

**本章默认教学后端：verl（`verl.yaml`）。** 硬前置：[ARPO](../17-arpo/TUTORIAL.md)（熵分支与步级信用）；工具轨迹约定见 [ReTool](../05-retool/TUTORIAL.md)。

**本章学习任务：** 解释为什么「只靠熵做分支」会翻车，并掌握 AEPO 在 **rollout** 与 **策略更新** 两侧的熵平衡——动态预算 + 分支惩罚、高熵裁剪 stop-grad、熵感知优势。

**相对 ARPO，改变了什么？**

| 组件 | ARPO | AEPO（本章） |
| --- | --- | --- |
| Rollout | 熵高就可能分支 | 熵**预监测**分配预算；**连续高熵**加 branch penalty |
| 裁剪 | 标准 clipped PG | 高熵 token：clip 权重 **stop-grad**，保留 $`\log\pi`$ 路径 |
| 优势 | 步级归因（可含熵权重） | **熵感知缩放** $`A(1+w\tanh(H-\bar H))`$ |
| 目标 | 探索工具步 | 探索与稳定的**折中**，抑制过分支与训练崩溃 |

来源：[AEPO 原文](https://arxiv.org/abs/2510.14545)（WWW 2026 Oral）· [官方代码 AEPO/](https://github.com/RUC-NLPIR/ARPO/tree/main/AEPO)。本地为教学最小实现。

---

## 1. ARPO 的隐患：熵是好信号，但是个坏老板

ARPO 在工具后高熵处多采样，能提升探索。但若**每一步高熵都无限开叉**：

1. **预算被少数长轨迹烧光**（连续工具步一直高熵）；  
2. **训练不稳定**：高熵 token 上 PPO 式裁剪容易把探索关键梯度**直接置零**，策略熵塌缩或抖动；  
3. **过分支**：同一状态反复分叉，样本相关、信息重复。

因此 AEPO 把熵同时用在**采样分配**与**梯度整形**上，并显式惩罚“连环开叉”。

---

## 2. 动态熵平衡 Rollout

### 2.1 预监测与预算

对轨迹上各工具步预估熵 $`H_t`$，高熵步（$`H_t\gt\tau_H`$）获得额外分支权重，再按比例分预算：

```math
\mathrm{extra}_t \propto \mathbf 1[H_t\gt\tau_H],\qquad
n_t = 1 + \mathrm{round}\bigl((B-T)\,\mathrm{extra}_t\bigr).
```

### 2.2 连续高熵分支惩罚

若 $`t-1`$ 与 $`t`$ 均为高熵，对 $`t`$ 的额外分支施加惩罚（教学式）：

```math
\mathrm{extra}_t \leftarrow
\max\bigl(0,\ \mathrm{extra}_t - \lambda_{pen}\cdot t\bigr),
```

再重新归一。直觉：**连环高熵说明模型在同一不确定上打转**，应限制分叉，而不是加倍分叉。

### 2.3 手算例

$`H=[0.3,1.4,2.1,2.3]`$，$`\tau_H=1`$，$`B=10`$，$`T=4`$，$`max\_branch=4`$，$`\lambda_{pen}=0.1`$：

| t | H | 高熵？ | 惩罚后 extra（示意） | 预算 $`n_t`$（demo） |
| ---: | ---: | --- | --- | ---: |
| 0 | 0.3 | 否 | 0 | 1 |
| 1 | 1.4 | 是 | 1 | 3 |
| 2 | 2.1 | 是（连续） | $`1-0.2`$ | 3 |
| 3 | 2.3 | 是（连续） | $`1-0.3`$ | 3 |

与 ARPO「熵越高越多」相比，**后两个连续高熵步不会无限膨胀**。请运行 `agentic_rl_demo.py --algorithm aepo` 核对数字。

---

## 3. 熵平衡裁剪：高熵 token 不要被 clip 剪没

标准 PPO/GRPO 裁剪在平台区把**该样本**的额外激励去掉；对高熵工具决策 token，这可能恰好剪掉探索梯度。

AEPO 教学式（单 token）：令 $`H`$ 为该位置熵，$`\rho`$ 为比率，$`A`$ 为（步）优势，$`\ell=\log\pi`$：

**低熵（正常）：**

```math
L=-\min(\rho A,\ \mathrm{clip}(\rho,1-\epsilon_l,1+\epsilon_h)A).
```

**高熵（clip 权重 stop-grad，保留 $`\ell`$ 路径）：**

```math
L=-\underbrace{\mathrm{sg}\bigl(\mathrm{clip}(\rho,1-\epsilon_l,1+\epsilon_h)\bigr)}_{\text{仅作权重}}\cdot A\cdot \ell.
```

比较：

| | 低熵 token | 高熵 token |
| --- | --- | --- |
| 平台区 | 可能近零梯度 | **仍有** $`\partial L/\partial\ell=-\mathrm{sg}(\mathrm{clip}(\rho))A`$ |
| 数值 | 标准 clipped PG | 权重被夹在 $`[1-\epsilon_l,1+\epsilon_h]`$，但不切断 $`\ell`$ |

**手算：** $`\rho=1.5,A=1,\epsilon=0.2`$。  
低熵：裁剪收益 1.2，平台区导数 0。  
高熵：权重 $`\mathrm{sg}(1.2)=1.2`$，$`L=-1.2\ell`$，$`\partial L/\partial\ell=-1.2`$——**鼓励继续提高该 token 概率**，与「好行为已推高就停」相反，专为高不确定探索位设计。

Demo 中对比 `high_entropy_token_loss` 与 `low_entropy_token_loss` 的结构差异。

---

## 4. 熵感知优势

在已有优势（含 ARPO 步级归因）上再按相对熵缩放：

```math
\widetilde A_t=A_t\cdot\bigl(1+w\tanh(H_t-\bar H)\bigr).
```

$`\bar H`$ 为行内平均熵。高于平均则略放大，低于平均则略缩小；$`\tanh`$ 限制幅度，避免单点熵爆炸主导 batch。

**手算：** $`A=[1,-1]`$，$`H=[2.0,0.1]`$，$`\bar H=1.05`$，$`w=0.2`$：  
$`\widetilde A\approx[1.15,-0.85]`$（demo 同量级）。注意**负优势幅度也略减小**——不是翻正，而是降低低熵位的相对压力。

---

## 5. 三个开关（与官方脚本对应）

| 开关 | 作用 | 关闭后 |
| --- | --- | --- |
| `enable_dynamic_rollouts` | 熵预算 + 分支惩罚 | 近似均匀预算 |
| `enable_entropy_balanced_clipping` | 高熵 stop-grad 权重 | 标准 clipped PG |
| `enable_entropy_balanced_advantage` | 熵感知优势缩放 | 原优势 |

官方 AEPO 脚本中的 `ENABLE_DYNAMIC_ROLLOUTS` / `ENABLE_ENTROPY_BALANCED_CLIPPING` / `ENABLE_ENTROPY_BALANCED_ADVANTAGE` 与上述一一对应，便于做消融。

---

## 6. 数值实验与代码

```bash
python examples/math/agentic_rl_demo.py --algorithm aepo

.venv/bin/arl train 18-aepo/verl.yaml --smoke --verl-workers 2
python 18-aepo/train.py --smoke
```

| 位置 | 看什么 |
| --- | --- |
| [src/agentic_rl/aepo.py](../src/agentic_rl/aepo.py) | `dynamic_entropy_budget`、`entropy_balanced_clip`、`entropy_aware_advantage` |
| [verl.yaml](verl.yaml) | 默认配置 |
| [官方 AEPO](https://github.com/RUC-NLPIR/ARPO/tree/main/AEPO) | 完整 verl 分支、工具缓存、多卡脚本 |

---

## 7. 练习

1. 关闭 clipping 开关后，$`\rho=1.5,A=1`$ 的高熵 token 梯度变成多少？与第 01 章四象限哪一格相同？  
2. 为何连续高熵要**惩罚**而不是每步 $`\times 2`$ beam？用预算守恒解释。  
3. 熵感知优势能否单独制造监督？当 $`A=0`$ 时 $`\widetilde A`$ 是多少？  
4. 对比 ARPO 步级归因 + AEPO 熵缩放：两者是替换还是叠加？写出你的一次 batch 伪代码。

---

## 8. 边界与下一步

- 不是 GAIA / HLE / AIME 的论文分数复现；开源模型与数据见 [HF AEPO Collection](https://huggingface.co/collections/dongguanting/aepo-68ef6832c99697ee03d5e1c7)。  
- 熵是**探索指示**，不是正确率；需独立评估。  
- **下一篇（工具环境）：** [09 Search-R1](../03-search-r1/TUTORIAL.md) 或 [ALFWorld](../08-alfworld/TUTORIAL.md)；蒸馏分支见 [General OPD](../02-opd/general-opd/TUTORIAL.md)。

<!-- NAV:BOTTOM:BEGIN -->
[← 上一章：07 ARPO：工具后的高熵分支采样与步级信用](../17-arpo/TUTORIAL.md) · [全书目录](../docs/CHAPTERS.md) · [本篇目录](../docs/families/02-tools-multimodal.md) · [下一章：09 Search-R1：学习检索与观察 mask →](../03-search-r1/TUTORIAL.md)
<!-- NAV:BOTTOM:END -->

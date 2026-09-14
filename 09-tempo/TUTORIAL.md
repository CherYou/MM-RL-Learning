# TEMPO｜不必每次走到结局，先学会评价下一段路

[学习路线](../docs/BEGINNER_GUIDE.md) · [PPO/GAE](../001-ppo/TUTORIAL.md) · [运行入口](README.md)

长任务可能要交互几十轮才成功。每个候选都从头走到结尾，既昂贵，也很难给早期决策信号。TEMPO 把若干轮交互合成一个 macro-step，在段末估计未来价值，并把部分非终局边界保存为以后继续采样的起点。

![从保存的边界展开短分支，在非终局估值，并回到保存状态继续采样](../docs/assets/algorithms/tempo.png)

图中 REPLAY 表示恢复先前保存的非终局状态；并不表示可以从成功终局继续行动。方法出处是 [Dots 团队 TEMPO 博客](https://studio.dots.ai/dots/tempo-blog.html)。该站当前页面抓取可能返回站点脚本；以下公式和数值明确以仓库学习实现为依据，不冒充已核验的最新论文全部细节。

## 一段路的回报由两部分组成

从相同起点分出 G 条短轨迹。第 i 条段内得到奖励 $`r_i`$，若还没结束，估计段末未来价值 $`v_i`$；真实终局未来价值为零：

```math
q_i=r_i+(1-d_i)v_i,\qquad G=\frac1n\sum_iq_i,\qquad A_i=q_i-G.
```

这里 G 表示共享起点的 TD target，n 是分支数，避免与其他章节的 group size 符号混淆。本地采用不折扣的段内累计/任务成功设定，不额外乘 $`\gamma^H`$；不要把公式当成所有奖励定义下的通用宏步 return。

手算：三个分支的段内奖励 `[0,0,1]`，终点 value `[0.2,0.6,0.9]`，第三条已成功终局。有效 return 是 `[0.2,0.6,1.0]`，而不是 `[0.2,0.6,1.9]`；目标 G=0.6，actor 优势为 `[-0.4,0,0.4]`。

## 生成式 critic 怎样训练

本章不是在模型顶上加一个线性 value head。**同一语言模型**以 critic prompt 生成 `<value>0到1之间的数</value>`，将它解析为成功概率估计。当前起点的多个 critic 回答，对照上面得到的 G 打分：

```math
r_j^V=-|v_j-G|,\qquad A_j^V=r_j^V-\overline{r^V}.
```

例如 G=0.6，两个生成估值为 0.5 与 0.1，奖励为 -0.1 与 -0.5，中心化优势为 +0.2 与 -0.2。随后对生成这些估值的 **token 概率**做裁剪 policy gradient；并不是通过正则表达式解析出的浮点数直接反传 MSE。

最终 batch 合并 actor 轨迹与 critic 生成，沿用序列平均 clipped loss：

```math
L=-\frac1B\sum_i\frac1{T_i}\sum_tm_{i,t}
\min(r_{i,t}A_{i,t},\mathrm{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_{i,t}).
```

actor 与 critic 是两种 prompt 角色，共享参数。`parse_value` 要求恰好一个合法标签；解析失败明确计数，critic 训练奖励取 -1。段末所有估值都解析失败时，本地以 0.5 fallback bootstrap，属于教学实现选择，需要监测而不是当作可靠估值。

## warmup、保存、重放各解决什么

warmup 先用完整 episode 得到回报，为 critic 提供起始训练信号；随后才改用短分支与 bootstrap。没有 warmup 的随机估值可能主要是在用噪声指导另一段噪声。

`ReplayStore` 保存完整 token 前缀、动作、环境观察和行为概率。恢复时 reset 环境、重放动作，并逐条比较观察是否相同。仅把旧文本塞回模型，并不能恢复真实环境状态。

历史前缀只作为条件，不重复进入本次 loss。只有新段的模型 token 训练；观察继续 mask=0。context/turn limit 耗尽的轨迹不再放回可继续采样池。

## verl 路径的前缀分布修正

旧前缀由过去策略生成，而今天的策略已变化。verl 扩展保存历史 assistant token 的行为概率并复评：

```math
w_{prefix}=\exp\left(\sum_{t\in\text{历史模型 token}}
[\ell_t^{current}-\ell_t^{behavior}]\right).
```

该权重停止梯度后乘到新段 surrogate。用户与环境 token 排除；新段自身的 ratio 仍由对应 token 的 current/old 概率计算。若前缀 log-ratio 总和为 log2，则权重为 2。

`prefix_is_clip` 可显式限制权重，降低极端数值，但会改变估计量。原生对照路径没有这项修正；不能把 native 和 verl 的重放视为完全等价。

## 代码阅读与练习

读 [tempo.py](../src/agentic_rl/tempo.py) 的 `signals`、`parse_value`、`critic_rollouts`，再读 [verl tempo.py](../src/agentic_rl/verl_backend/tempo.py) 的 `ReplayStore`、`prefix_correction`、`tempo_batch`，最后回看 [rollout.py](../src/agentic_rl/rollout.py) 的恢复分支。这条路线把数学信号、模型调用和环境状态分开。

```bash
.venv/bin/arl train 09-tempo/verl.yaml --smoke --verl-workers 2
```

本章 smoke 跑三步，覆盖 warmup、macro-step 与重放。看 `tempo/replayed`、`tempo/state_store_size`、`tempo/critic_parse_failures`、`tempo/prefix_is_weight`，不能只看程序退出码。

练习：终局分支的 critic 恰好输出 0.9，是否加到已得到的成功奖励 1 上？答案：不加。未来已不存在，必须用 0。多算这项会人为偏好已结束的分支。

我的判断：TEMPO 的核心风险是“用不准的未来估值放大采样偏好”。短 macro-step 节省交互，但更依赖 critic；长 macro-step 更接近实际终局，又更昂贵。应在相同环境交互预算下比较，而不是只统一 optimizer 步数。本地尚未复现长任务正式效果或 GPU 训练。

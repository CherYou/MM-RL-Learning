# 从零读懂这个仓库：先认清数据，再认清梯度

你不需要先学会分布式训练。读完这组教程，首先应该能回答三个问题：模型这一次看见了什么？哪个信号在告诉它好坏？这个信号最后改变了哪些参数？

每章 README 开头都有 **“直接阅读本章新手教程：TUTORIAL.md”** 链接，根目录表格也可直接进入各章教程。`TUTORIAL.md` 负责概念、动机、推导、手算和自测，README 保留安装与运行说明。

建议一章读两遍。第一遍顺着“遇到什么困难 → 为什么引入下一部分 → 一次更新发生什么”理解，不急着记全部符号。第二遍自己重算数字例子，再沿代码链接查对应变量。节末练习先遮住答案；如果只能复述名词而说不清它解决的问题，就返回该概念出现前的动机段。

图中的机器人表示模型角色，不表示一定是独立网络。原概念插图用于辅助直觉，PPO 另有包含 old、reference、reward、critic 和更新路径的完整流程图；以图旁说明的范围为准。图不是实验测量，也不能替代具体公式和执行代码。

## 遇到这些词，先用普通话读一遍

| 术语 | 本书中的读法 |
| --- | --- |
| Policy / actor | 根据当前信息做选择的模型 |
| Reward / return | 一步获得的反馈 / 从当前起未来反馈的总账 |
| Critic / value | 预测未来回报的模型 / 它预测的数值 |
| Advantage / baseline | 相对基准好多少 / 拿来比较的预期水平 |
| Rollout / trajectory / episode | 一次采样过程 / 记录下的行动与反馈 / 从初始化到停止的一回合 |
| Group / batch | 同题的多个候选 / 一次更新使用的一批样本 |
| On-policy / off-policy / offline | 当前策略采样 / 可以用旧行为数据 / 训练只用固定数据 |
| Bootstrap / target network | 用预测补全尚未知的未来 / 缓慢更新的目标副本 |
| Mask / detach | 哪些位置直接计损失 / 哪条计算分支停止梯度 |
| Reference / teacher | 固定行为参照 / 提供学习反馈的教师；两者用途不同 |

这些是帮助阅读的简写，各章会给出具体条件。例如 critic 在 PPO 中预测 V，在 SAC/TD3 中预测 Q，在 TEMPO 中则通过生成数字字符串扮演估值角色，不能仅凭同一个名字认为实现相同。

## 一条适合第一次学习的路线

| 阶段 | 详解 | 读完应能解释 |
| --- | --- | --- |
| 先补基础 | [基础知识](../preliminary/FOUNDATIONS.md)、[损失函数](../preliminary/TUTORIAL.md) | MDP、Bellman、概率、梯度、mask 与统计单位 |
| 理解评价与更新 | [001 PPO](../001-ppo/TUTORIAL.md) | actor、critic、奖励、优势分别做什么 |
| 理解离线偏好 | [002 DPO](../002-dpo/TUTORIAL.md) | 为什么可以只用好坏回答对训练 |
| 不使用 value critic | [01 GRPO](../01-grpo/TUTORIAL.md) | 同一道题的回答如何互相比较 |
| 改采样与统计单位 | [06 DAPO](../06-dapo/TUTORIAL.md)、[07 GSPO](../07-gspo/TUTORIAL.md) | 采样、归一化和裁剪粒度各自改变什么 |
| 学生向教师学习 | [General OPD](../02-opd/general-opd/TUTORIAL.md)、[Medical/SAR/IDT](../02-opd/TUTORIAL.md) | 谁生成答案、谁给概率、教师怎样切换 |
| 使用训练期额外信息 | [04 OPSD](../04-opsd/TUTORIAL.md) | 同样权重为何能产生不同监督 |
| 开始使用工具 | [03 Search-R1](../03-search-r1/TUTORIAL.md)、[05 ReTool](../05-retool/TUTORIAL.md) | 为什么检索结果和执行结果不直接进入 policy loss |
| 连续与环境交互 | [08 ALFWorld](../08-alfworld/TUTORIAL.md) | 观察、行动、终局、截断如何衔接 |
| 处理多轮信用 | [09 AgentOPSD](../09-AgentOPSD/TUTORIAL.md)、[09 TEMPO](../09-tempo/TUTORIAL.md) | 轮次重加权和分段 bootstrap 有何区别 |
| 增加图像条件 | [09 Vision-GRPO](../09-vision-grpo/TUTORIAL.md) | 图像如何进入模型，文字奖励如何反传 |
| 训练调用与编排 | [12 Harness-RL](../12-harness-rl/TUTORIAL.md) | 调用树、字段 mask 与参数分区是什么 |
| 连续动作与最大熵 | [13 SAC](../13-sac/TUTORIAL.md) | 连续策略密度、双 Q 与温度怎样配合 |
| 稳定连续控制更新 | [14 TD3](../14-td3/TUTORIAL.md) | 延迟 actor 和目标动作平滑防止什么问题 |
| 利用稀疏奖励经历 | [15 HER](../15-her/TUTORIAL.md) | 只改目标、不改动作历史怎样产生训练信息 |
| 从固定机器人数据学习 | [16 IQL](../16-iql/TUTORIAL.md) | expectile 价值学习与优势加权模仿 |

Medical 章节包含 SFT、Medical OPD、SAR-OPD、IDT-OPD；preliminary 损失教程包含 IS、PPO clipping、CISPO，并补充固定长度归一化的区别。HER 则修改经验数据，底层使用 TD3 的 loss；目录与独立 loss 不一一对应。

## 先认识模型的一次“选择”

语言模型不是一次写出整篇文章。给定已经出现的文字，它为下一个 token 分配概率，再选择一个 token，接到历史后面，重复这一过程。Token 是分词器的单位：可能是字、词的一部分、标点，也可能是特殊标记，不保证一个 token 等于一个汉字。

例如历史是“2+3=”，模型给“5”分配 0.6，给“6”分配 0.1，其余 token 合计 0.3。采样到“5”只说明这次选择了它，不代表其它选项的概率为零。

对一个回答 $`y=(y_1,\ldots,y_T)`$：

```math
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{1:t-1}).
```

```math
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{1:t-1}).
```

乘很多小数容易下溢，所以代码常保存 `logprob`。`exp(logp - old_logp)` 就是“现在的概率 / 采样时的概率”。整段的概率与单个 token 的概率不能混用。

## 全书符号表

| 符号 | 读法与用途 | 常见代码名 |
| --- | --- | --- |
| $`x`$、$`h_t`$ | 问题；到第 t 步为止的全部历史 | `prompt`、`tokens` |
| $`y_i`$、$`y_{i,t}`$ | 第 i 个回答；它的第 t 个 token | `Sample.tokens` |
| $`\pi_\theta`$ | 正在训练、参数可变化的策略 | `policy`、`actor` |
| $`\pi_{old}`$ | 生成这一批数据时的行为策略 | `old_log_probs` |
| $`\pi_{ref}`$ | 通常固定的参考策略，控制偏离 | `ref_log_prob` |
| $`\pi_T`$ | 给学生 token 打分的教师 | `teacher_log_probs` |
| $`R_i`$、$`A_i`$ | 回答得分；相对基准的好坏信号 | `reward`、`advantages` |
| $`m_{i,t}\in\{0,1\}`$ | 该位置是否直接参与训练 | `response_mask` |
| $`r_{i,t}`$ | 当前/行为策略的 token 概率比 | `ratio` |
| $`\mathrm{sg}`$ | 当常数使用，停止沿该分支求导 | `.detach()` |
| $`\epsilon`$、$`\beta`$ | 裁剪幅度；正则强度，含义依章节而定 | `clip_low/high`、`beta` |

“old”和“reference”可能初始化相同，但职责不同。Old 跟随采样批次更新；reference 常从训练开始固定。Teacher 也不等于 critic：teacher 给 token 概率，critic 估计未来回报。

## loss 为什么常带负号

优化器通常执行 $`\theta\leftarrow\theta-\eta\nabla_\theta L`$，也就是减小 loss。若我们想增加一个好回答的概率，可以取 $`L=-A\log\pi_\theta(y)`$，其中 $`A\gt 0`$。这时对 logprob 的导数是负数，沿梯度下降会推动它增大。$`A\lt 0`$ 时方向相反。

这只是理解局部方向的办法。模型参数是共享的，增加一个 token 概率会通过 softmax 和网络牵动其它概率；不能把每个 token 想成互不相干的独立旋钮。

奖励回答“任务做得怎么样”，loss 回答“本批数据要怎样改变参数”。优势经过中心化后，有些 loss 一开始接近零，但不同样本的梯度并不相互抵消，因此仍可能有效更新。判断训练不要只看 loss 正负。

## mask 是理解 Agent 训练的钥匙

假设一条轨迹是：题目 → 模型写查询 → 工具返回文档 → 模型写答案。题目和工具文档的训练 mask 为 0，模型生成部分为 1。mask 为 0 不表示模型看不见那些文字：它们仍影响后续生成，也会通过上下文计算间接影响参数梯度。它只表示没有要求模型把这些位置当成自己的预测目标。

在自回归模型中，第 t 个输入位置预测第 t+1 个 token。因此完整 `tokens` 和完整 `mask` 通常比 `old_logp` 多一个位置；打分使用 `mask[1:]`。读 [models.py](../src/agentic_rl/models.py) 的 `Sample` 和 [protocol.py](../src/agentic_rl/verl_backend/protocol.py) 的 `pack` 时，可以先在纸上写出 5 个 token 的索引。

## 语言模型算法为什么有三条代码路径

| 路径 | 最适合读什么 | 需要留意 |
| --- | --- | --- |
| native | 公式、mask、显式 Python 训练循环 | 主要是单进程教学对照 |
| TRL | 标准 Trainer 怎样接收模型、数据和配置 | 配置只控制该固定版本支持的功能 |
| verl | 轨迹分发、角色打分、全局归一化、actor 更新与生成同步 | 本仓库扩展了控制器和 loss，并非直接运行原版 RayPPOTrainer |

共享阅读路线是 `章节 config.yaml → CLI → 后端 trainer → 轨迹 → 信号 → loss → optimizer`。verl 路径重点读 [trainer.py](../src/agentic_rl/verl_backend/trainer.py)、[algorithms.py](../src/agentic_rl/verl_backend/algorithms.py)、[actor.py](../src/agentic_rl/verl_backend/actor.py)。先理解一个 batch，再理解多卡，会容易很多。

新增 SAC/TD3/HER/IQL 使用 `embodied` 路径：`config → embodied.runner → Replay → Agent.update → 连续控制评估`。它使用真实 MuJoCo 环境、13 维状态-目标输入和 4 维动作，没有 token 或语言模型。与 VLA 的关系、数据来源和验证结果见 [具身实验说明](EMBODIED.md)。

## 怎样判断自己真的读懂了

对每章问自己：随机采样来自谁？学习目标由谁决定？如果所有回答一样好，会发生什么？哪些张量要 `.detach()`？哪些 token 不训练？一次 optimizer step 后谁改变、谁冻结？如果指标不好，首先检查的是数据、生成、奖励还是梯度？

每篇末尾有可在 CPU 上完成的练习。LLM 的 `--smoke` 使用随机微型模型与显式 debug 奖励；具身的 `--smoke` 使用真实物理环境、环境奖励和小 MLP。两者都用于检查执行流程。研究学习效果时，需要充分训练、独立评估与多个训练种子；GPU 启动方法见 [verl 指南](VERL.md#gpu-启动方式与边界)。

## 资料和原创范围

教程采用“先解释来源解决什么疑问，再给链接”的引用方式。论文用于核对机制；作者教材、官方教程用于补充教学视角；本地数值与代码说明以本仓库实现为准。研究主张、教学例子和本地运行结果分别标明，未把其他项目成绩写成本仓库成绩。

算法名称及机制依据链接到作者论文、官方文档或作者博客；本地行为依据实际代码。例子、阅读顺序、中文解释和诊断建议为本教程重新设计。上游参考副本继续保留原作者归属，不当作本教程原创内容。

理论背景可查 [PPO 原论文](https://arxiv.org/abs/1707.06347)、[DeepSeekMath](https://arxiv.org/abs/2402.03300) 与 [DPO 原论文](https://arxiv.org/abs/2305.18290)。每章给出更具体的来源，不要求初学者先读完论文才能开始。

需要另一种详细解释时，可补读 [Spinning Up 的策略梯度课](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)、[SAC 教程](https://spinningup.openai.com/en/latest/algorithms/sac.html)、[TD3 教程](https://spinningup.openai.com/en/latest/algorithms/td3.html) 和 [Nathan Lambert 的 RLHF 教材](https://rlhfbook.com/c/06-policy-gradients)。这里引用概念依据并重新设计例子，没有整段搬运博客正文。

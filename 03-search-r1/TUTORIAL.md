# Search-R1｜把“去查一下”变成模型学会的行动

[学习路线](../docs/BEGINNER_GUIDE.md) · [运行入口](README.md) · [GRPO 基础](../01-grpo/TUTORIAL.md)

普通问答模型看到问题后直接作答。Search-R1 允许它在回答中途提出检索请求，读到材料后继续推理。训练要学习的包括：什么时候需要查、查什么、拿到材料后如何使用。搜索引擎返回的文章并不是模型自己说的话。

![查询进入检索器，文档成为观察，模型再生成答案；下方只有模型生成段落带训练标记](../docs/assets/algorithms/search-r1.png)

图中几台机器人表示同一模型在不同时间点的工作；图书馆代表固定检索器。方法背景与 retrieved-token masking 见 [Search-R1 原论文](https://arxiv.org/abs/2503.09516)。本地实现用 BM25 检索已准备的语料，不调用实时互联网搜索。

## 跟着一条轨迹走

教学问题：“某部小说作者出生在哪个城市？”合理轨迹可能是：

```text
模型：<search>小说名 作者</search>
工具：作者是某人。
模型：<search>该作者 出生地</search>
工具：出生于某城市。
模型：<answer>该城市</answer>
```

这些内容只是原创教学示意，不是事实问答或实际运行输出。环境只负责收到 `<search>` 后检索，并把结果接回上下文；它不会替模型决定下一条查询。

若模型生成 `<answer>`，该问答 rollout 结束。格式不合法时，工具错误信息也成为 observation，模型随后可以纠正。上下文长度与轮数上限则防止无限交互。

## 奖励与 loss：新的是交互，不一定是目标公式

本章默认按最终答案匹配给任务奖励，同题生成 G 条完整检索轨迹后计算：

$$A_i=\frac{R_i-\bar R}{\sigma_R+10^{-4}}.$$

生成 token 的历史 $h_{i,t}$ 现在包含已经返回的文档。概率比仍为：

$$r_{i,t}=\frac{\pi_\theta(y_{i,t}\mid h_{i,t})}{\pi_{old}(y_{i,t}\mid h_{i,t})}.$$

本地使用 GRPO 型序列平均裁剪 loss：

$$L=-\frac1B\sum_i\frac1{T_i}\sum_t m_{i,t}
\min\{r_{i,t}A_i,\operatorname{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_i\}.$$

可选 reference KL 见 GRPO 章。这里的 $T_i$ 只统计模型生成 token。奖励没有逐条证明每个查询有用，整条轨迹共享终局优势；这也是多轮信用分配仍然困难的原因。

## 最重要的数组是 mask

假设题目 3 个 token、查询 2 个、检索结果 4 个、答案 2 个：

```text
段落：题目       查询    文档          答案
mask：0 0 0      1 1     0 0 0 0       1 1
```

有效 token 总数是 4，不是 11。若 loss numerator 是 8，token 平均应除以 4；但本章还会先对每条轨迹平均，再平均轨迹。

文档 mask=0 不表示文档被屏蔽掉。模型仍读取文档，后续生成概率依赖它，梯度可以通过“文档影响答案”的网络计算路径传播；只是没有直接要求模型预测文档文字。

## 对照代码逐步阅读

先打开 [rollout.py](../src/agentic_rl/rollout.py) 的 `agent_rollout`。按 `search → observation → ids.extend → mask.extend` 追踪。下面是关键摘录：

```python
ids.extend(out)
mask.extend([1.0] * len(out))
# 检索与 observation 编码发生在两段代码之间
ids.extend(obs_ids)
mask.extend([0.0] * len(obs_ids))
```

旧 token IDs 原样保留，仅追加新内容。若每次交互都将整段对话解码再编码，旧 logprob 可能与新的 token 边界错位。

[environments.py](../src/agentic_rl/environments.py) 的 `LocalSearch.search` 按词频、逆文档频率和文档长度计算 BM25 风格得分，再取 top-k；它没有可训练参数。多任务并发由 [agent_loop.py](../src/agentic_rl/verl_backend/agent_loop.py) 协调；[algorithms.py](../src/agentic_rl/verl_backend/algorithms.py) 把完整轨迹转成 GRPO 学习信号。

## 自己实验时先检查什么

```bash
.venv/bin/arl train 03-search-r1/verl.yaml --smoke --verl-workers 2
```

CPU smoke 检查工具调用路径，随机模型未必生成合法查询。真正评估时，先检查语料中是否存在回答所需材料，再统计合法查询率、空结果率、查询次数与最终正确率。单看工具调用次数增加，不说明检索能力提高。

练习：文档从 4 token 变成 40 token，模型生成位置不变，训练 mask 应增加几个 1？答案是 0。上下文计算量会上升，但直接参与 policy loss 的 token 数不会因此增加。

我的判断：检索训练的第一道质量关是“观察是否值得相信，训练归属是否正确”。没有证据的语料和错误的 mask，都会让最终奖励难以解释。本章只验证本地语料检索和 CPU 管线，未复现论文多数据集得分，也没有验证实时联网检索。

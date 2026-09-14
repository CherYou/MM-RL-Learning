# ReTool｜学会何时计算，而不只是学会写出像代码的文字

[学习路线](../docs/BEGINNER_GUIDE.md) · [运行入口](README.md) · [Search-R1 的 mask](../03-search-r1/TUTORIAL.md)

面对需要多步算术的问题，模型可以自己推导，也可以写一小段程序，请工具执行后继续思考。ReTool 研究这种自然语言与实际代码执行交替的学习过程。工具必须真的运行；把模型预测的“执行结果”直接当事实，会失去这个机制。

![模型思考、写代码、工具执行返回结果，再继续生成最终答案](../docs/assets/algorithms/retool.png)

图是调用流程示意，不是某次实验测量。[ReTool 原论文](https://arxiv.org/abs/2504.11536)讨论战略性工具使用；本地学习实现选用 GRPO 信号与受限数值 Python，不覆盖论文训练规模和全部工程配置。

## 一次具体的工具交互

假设需要求 $1+2+\cdots+100$。模型可以输出：

```text
<python>print(sum(range(1, 101)))</python>
```

工具返回 `5050`，随后模型输出 `<answer>5050</answer>`。第一次生成的是代码，第二次生成的是答案。中间的 `5050` 是外部执行结果，两处同样的字符串在训练上有不同归属。

也可以不用工具，直接利用等差数列求和。当前奖励并不会因为“调用过 Python”就自动加分；最终答案是否正确仍是主要任务信号。因此合理目标是学会有用调用，而不是最大化工具次数。

## 关键公式与 loss

一条轨迹写成 $\tau=(y_1,o_1,y_2,o_2,\ldots,y_K)$，$y_k$ 是模型生成段，$o_k$ 是执行结果。环境转换可以是离散、不可微的程序。Policy gradient 不要求对 Python 执行器求导。

本地对完整轨迹判分，组内计算 $A_i=(R_i-\bar R)/(\sigma+10^{-4})$，再优化：

$$L=-\frac1B\sum_i\frac1{\sum_tm_{i,t}}\sum_tm_{i,t}
\min\{r_{i,t}A_i,\operatorname{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_i\}.$$

其中 $m=1$ 覆盖模型的代码、推理和回答；$m=0$ 覆盖题目及工具 observation。$r$ 比较当前与行为策略对相同已生成 token 的概率。Reward 并不直接进入代码执行器的梯度，而是重新加权生成这些 token 的概率。

例子：一组两条轨迹，一条代码计算正确最终答对，另一条虽然代码运行成功但抄错最终答案，奖励为 `[1,0]`。优势近似 `[1,-1]`，第二条不会仅因工具没有报错就得到正向终局信号。

## 代码中三处责任边界

| 文件/函数 | 负责什么 |
| --- | --- |
| [rollout.py](../src/agentic_rl/rollout.py) / `agent_rollout` | 解析 `<python>`、发起执行、追加观察、判断结束 |
| [environments.py](../src/agentic_rl/environments.py) / `python_tool` | 检查允许的语法，在独立进程中实际计算 |
| [algorithms.py](../src/agentic_rl/verl_backend/algorithms.py) | 组奖励、优势、mask 与 actor 更新数据 |

`python_tool` 的执行范围是有意限制的：允许常用数值操作和选定 `math` 函数，有时间、内存与输出限制；不等于完整 Python 开发环境。语法错误、超时、未支持操作以 `ToolError` 返回，Agent 可以把错误当下一轮输入。具体限制见 [TOOLS.md](../docs/TOOLS.md)。

教学伪代码如下，显示控制流而不是可导计算图：

```python
model_tokens = generate(history)
code = extract_python(model_tokens)
observation = python_tool(code)
history += model_tokens + encode(observation)
mask += [1] * len(model_tokens) + [0] * len(encode(observation))
```

实现中还要保留精确 token、旧概率、上下文预算和结束条件，不应把上面几行直接当完整训练器。

## 一个无需大模型的动手检查

```bash
.venv/bin/python - <<'PY'
from agentic_rl.environments import python_tool
print(python_tool('print(sum(range(1, 101)))'))
PY
.venv/bin/arl train 05-retool/verl.yaml --smoke --verl-workers 2
```

第一条应得到 `5050`，它单独验证环境；第二条验证 CPU 训练路径。二者都不证明随机模型已经学会何时调用工具。

练习：把工具结果的位置也设为 mask=1，会发生什么？答案：模型会被要求模仿外部执行器输出，并把它们当自己采样的行为来更新；若这些位置还没有真实行为 logprob，重要性比率也失去依据。

我的判断：工具型 RL 的一次失败应拆成四段检查——模型是否正确描述问题、代码是否正确、执行器是否支持、模型是否正确使用返回值。只看最终零奖励会把这些完全不同的错误混在一起。正式学习效果还需固定题目与工具预算比较，本章尚无 GPU 能力验证。

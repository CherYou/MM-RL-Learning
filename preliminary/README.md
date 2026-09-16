# Preliminary：读算法之前的基础课

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

先理解概率、梯度和停止梯度，再比较 IS、PPO、CISPO。术语陌生时，可先读下方的基础知识详解。

[基础知识详解](FOUNDATIONS.md) · [损失函数详解](TUTORIAL.md) · [总目录](../README.md)

建议先学“数据是怎么来的、模型在预测什么、梯度改变了谁”，再比较各章 loss。

| 前置内容 | 阅读入口 | 练习目标 |
| --- | --- | --- |
| MDP、观察与状态、部分可观察性 | [FOUNDATIONS](FOUNDATIONS.md) 第 1 节 | 区分环境状态与模型能看到的观察 |
| 回报、V/Q、优势与 Bellman 方程 | 同文第 2–3 节 | 手算一步 bootstrap |
| 真实终止与时间截断 | 同文第 4 节 | 理解同样结束 episode 为何 target 不同 |
| 概率、连续动作密度、tanh 变换 | 同文第 5 节 | 分清概率和密度，理解 Jacobian |
| autograd、detach、重参数化 | 同文第 6 节 | 运行小例子检查梯度 |
| Mask、统计单位、回放与离线学习 | 同文第 7–8 节 | 知道哪些位置和数据参与训练 |
| VLA 的表示、动作约定与数据覆盖 | 同文第 9 节 | 明白低维机械臂与 VLA 的距离 |
| 基线、独立评估、种子和数据来源 | 同文第 10 节 | 避免将执行成功当能力结论 |
| IS、PPO clipping、CISPO | [TUTORIAL](TUTORIAL.md) | 比较同一 ratio 下的梯度 |

损失实验可以直接运行：

```bash
.venv/bin/python preliminary/train.py
# 等价入口
.venv/bin/arl loss-demo
```

它计算 PyTorch autograd 梯度并保存真实数值图、CSV 和 TensorBoard 事件。这里不训练语言模型，不需要 `--smoke`、模型下载或 GPU。输出路径由 [loss_demo.py](../src/agentic_rl/loss_demo.py) 创建。

核心实现是 [losses.py](../src/agentic_rl/losses.py)。CISPO 演示使用显式双侧裁剪教学式，原论文的具体裁剪设置与它的区别在详解中说明。接着读 [001 PPO](../001-ppo/TUTORIAL.md) 和 [002 DPO](../002-dpo/TUTORIAL.md)。

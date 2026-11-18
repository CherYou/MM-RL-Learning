# 数学与梯度轻量实验

本目录提供不依赖仓库训练包、预训练模型、网络 API 或物理仿真的 CPU 数值例子，用于核对：

- 交叉熵 / MSE / BCE 与训练对象
- 一次真实 logits 更新
- old 快照与 `detach`
- PPO 正负优势四种裁剪方向
- mask 与 reduction 统计单位
- 单样本 log-ratio 与完整 KL 的区别

## 运行

```bash
# 需要当前环境已安装 PyTorch；过程不使用网络
python examples/math/loss_walkthrough.py
```

结尾应出现 `"status": "passed"`。保存一份新的结果：

```bash
python examples/math/loss_walkthrough.py --output examples/math/results/numerical_checks.json
```

脚本拒绝覆盖已存在的输出文件。

## 怎样读输出

在 `cross_entropy` 中找到 `logit_gradient`。目标类别的梯度为负，按梯度下降做一步更新会提高该类的 logit。在 `value_regression` 中，预测值 0.3 离目标 1 还差 0.7，所以平方误差是 0.49，梯度是 −1.4。其余输入与输出可以逐项对照[参考数值](reference/numerical_checks.json)。

## 与仓库其他入口的关系

| 入口 | 层级 | 说明 |
| --- | --- | --- |
| `python examples/math/loss_walkthrough.py` | L0 | 纯数学，适合第一章 |
| `.venv/bin/arl loss-demo` / `01-foundations/train.py` | L0/L1 | 使用仓库 `policy_loss`，生成曲线与 CSV |
| `python 03-grpo/train.py --smoke` | L1 | tiny 模型与真实 Trainer 链路 |

阅读顺序建议：先跑通本目录脚本，再读 [01-foundations/TUTORIAL.md](../../01-foundations/TUTORIAL.md)，然后按需运行仓库 smoke。

## 源码

- [loss_walkthrough.py](loss_walkthrough.py)：独立断言脚本
- 仓库损失实现：[src/agentic_rl/losses.py](../../src/agentic_rl/losses.py)

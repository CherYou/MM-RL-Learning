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

## 证据边界

| 可以说明 | 不能说明 |
| --- | --- |
| 构造例子中的 loss 数值与梯度方向 | 语言模型任务能力提升 |
| PPO 四象限符号与导数结构 | 论文 benchmark 复现 |
| mask/reduction 的算术含义 | 分布式训练或 GPU 性能 |

仓库内 `reference/numerical_checks.json` 是审阅时的一次参考运行结果，不是能力实验报告。

## 与仓库其他入口的关系

| 入口 | 层级 | 说明 |
| --- | --- | --- |
| `python examples/math/loss_walkthrough.py` | L0 | 纯数学，适合第一章 |
| `.venv/bin/arl loss-demo` / `preliminary/train.py` | L0/L1 | 使用仓库 `policy_loss`，生成曲线与 CSV |
| `python 01-grpo/train.py --smoke` | L1 | tiny 模型与真实 Trainer 链路 |

阅读顺序建议：先跑通本目录脚本，再读 [preliminary/TUTORIAL.md](../../preliminary/TUTORIAL.md)，然后按需运行仓库 smoke。

## 源码

- [loss_walkthrough.py](loss_walkthrough.py)：独立断言脚本
- 仓库损失实现：[src/agentic_rl/losses.py](../../src/agentic_rl/losses.py)

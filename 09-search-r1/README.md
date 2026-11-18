# Search-R1：多轮检索与观察屏蔽

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 本章重点：检索观察 mask 与轨迹字段。裁剪与组内优势见 [GRPO](../03-grpo/TUTORIAL.md) / [损失与策略梯度基础](../01-foundations/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

模型生成 `<search>query</search>`，本地 BM25 返回真实公开文档片段，模型继续生成，最后用 `<answer>回答</answer>` 结束。训练时仍按同题多轨迹计算组内优势。

最关键的工程约束是 token 前缀：只追加模型实际生成的 IDs，工具输出编码后追加为 observation；不重渲染旧历史。模型自己的 token mask=1，检索返回内容和模板 mask=0。否则模型会被训练去模仿检索系统的文档。

数据包含 Search-R1 的 NQ/Hotpot 混合问答子集。离线示例用 HotpotQA 的训练和验证问题，从本地文档检索证据。索引只包含文档文本，问答标签和 gold supporting facts 不进入索引。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 09-search-r1/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 09-search-r1/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 09-search-r1/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[rollout.py](../src/agentic_rl/rollout.py)、[environments.py](../src/agentic_rl/environments.py)、[losses.py](../src/agentic_rl/losses.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-09-search-r1`。

## 指标与对照实验

比较不调用检索、一次检索、最多多轮检索；看 EM/F1、工具调用数和失败次数。先用已知文档标题检查检索能返回证据，再检查 observation token 的梯度确实为零。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

检索范围限于本地的五千余个文档片段。如果一直答错，先看返回的片段是否覆盖答案所需的证据，再检查模型有没有使用它。

## 一手资料

- [HotpotQA 数据集](https://huggingface.co/datasets/hotpotqa/hotpot_qa)

## 后端说明

CPU 和 CUDA 环境的安装、训练中断后的恢复步骤见 [verl 指南](../docs/VERL.md)。

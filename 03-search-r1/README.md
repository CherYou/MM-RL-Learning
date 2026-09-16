# Search-R1：多轮检索与观察屏蔽

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

从检索轨迹开始，理解动作、观察、奖励与 mask。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

模型生成 `<search>query</search>`，本地 BM25 返回真实公开文档片段，模型继续生成，最后用 `<answer>回答</answer>` 结束。训练时仍按同题多轨迹计算组内优势。

最关键的工程约束是 token 前缀：只追加模型实际生成的 IDs，工具输出编码后追加为 observation；不重渲染旧历史。模型自己的 token mask=1，检索返回内容和模板 mask=0。否则模型会被训练去模仿检索系统的文档。

已准备原 Search-R1 的 NQ/Hotpot 混合问答学习子集；快速离线配置使用 HotpotQA train/validation 子集及其公开 distractor 文档构成的本地检索库，不需要 API key。检索库不存 question/answer 标签或 gold supporting-facts 注释。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 03-search-r1/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 03-search-r1/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 03-search-r1/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `rollout.py / environments.py / losses.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-03-search-r1`，已存在的目录会拒绝覆盖。

## 验证思路与消融

比较不调用检索、一次检索、最多多轮检索；看 EM/F1、工具调用数和失败次数。先用已知文档标题检查检索能返回证据，再检查 observation token 的梯度确实为零。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

5 千余篇本地片段构成的是小型离线检索任务，检索覆盖率与全 Wikipedia 环境不同，不可把这个结果写成原论文 Search-R1 benchmark。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 2](https://huggingface.co/datasets/hotpotqa/hotpot_qa)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 03-search-r1/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 03-search-r1/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor。verl 与原生均保存 token、mask、旧概率和轮次日志。

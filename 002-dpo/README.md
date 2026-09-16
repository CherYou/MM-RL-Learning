# DPO：离线偏好优化

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

从偏好对开始，推到 reference 校正、DPO loss 与具体更新。建议先读教程完成手算和自测，再回到本页运行代码。

这是本地新增章节。 · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

每条数据包含 prompt、chosen、rejected。DPO 用冻结 reference 对两种回答的 logprob 差进行校正：z=beta[(log πθ(y+)−log πθ(y−))−(log πref(y+)−log πref(y−))]，L=−log sigmoid(z)。

completion logprob 先在有效回答 token 上求和，prompt/pad 不参与比较。与 PPO 不同，DPO 不在训练时在线采样、不构造 GAE，也不训练 value model。默认入口直接使用 TRL DPOTrainer，同时保留原生 loss 方便观察 chosen 上升、rejected 下降的梯度。

本地偏好数据由 GSM8K 正确推理解答作为 chosen，数值答案加 1 的错误回答作为 rejected；分别从独立 train/test 派生，明确标注 synthetic negatives。它适合验证流程，不能代表真实人类偏好数据。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 002-dpo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 002-dpo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 002-dpo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `trl_backend.py / trainers.py / losses.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-002-dpo`，已存在的目录会拒绝覆盖。

## 验证思路与消融

检查初始 policy=reference 时 loss≈log(2)，一步反传应鼓励 chosen、压低 rejected。比较 beta、回答长度和 synthetic-negative 策略，记录 reward margin。评测附带 chosen/rejected 原始 logprob 偏好率，注意长度偏差。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

这些错误回答比正确长解更短、更简单，存在明显风格捷径；正式偏好实验应换成经审查的难负例。准备和验证本学习数据不等于完成 RLHF 效果复现。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 1](https://arxiv.org/abs/2305.18290)
- [资料 2](https://huggingface.co/docs/trl/v0.25.1/en/dpo_trainer)

# DAPO：四项机制与动态采样

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

逐项理解裁剪、有效组补采、token 权重与长度处理。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

本地主入口包含四件独立的事：

1. Clip-Higher：下界 εlow=0.2，上界 εhigh=0.28，给探索留出更宽的正向空间。
2. Dynamic Sampling：在原始任务奖励上检查组内差异，全对/全错组不进入更新，继续补采；有最大尝试次数，耗尽会显式记录 skipped，防止无限等待。
3. Token-level reduction：把所有有效生成 token 的损失相加，再除以有效 token 总数。
4. Overlong shaping：在 soft_length 到 max_new_tokens 区间线性增加负奖励；可把没有 EOS 的截断回答排除出 loss。

准备数据时先剥离原数据要求 Answer: 的模板，统一到本项目答案协议，再按题面去重、移除冲突答案。配置的 loss_type=dapo 本身并不自动完成其他三项机制；所以章节默认使用原生循环。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 06-dapo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 06-dapo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 06-dapo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `trainers.py / losses.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-06-dapo`，已存在的目录会拒绝覆盖。

## 验证思路与消融

逐项关闭 dynamic sampling、Clip-Higher、token reduction、长度奖励，比较有效组数、补采次数、采样耗时和正确率。TRL 对照入口只验证 DAPO loss，不宣称它包含完整四项。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

小模型可能连续遇到全错组，动态采样上限触发后没有参数更新是正确行为。CPU smoke 用调试奖励验证补采逻辑，不证明数学训练有效。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 2](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 06-dapo/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 06-dapo/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor。verl 与原生均保存 token、mask、旧概率和轮次日志。

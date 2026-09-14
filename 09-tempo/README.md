# TEMPO：macro-step 与生成式 critic

**初次学习请先读 [新手算法详解](TUTORIAL.md)：直觉、公式、loss、代码对应与练习答案。**

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

把长任务切成 H 个交互步的 macro-step。同一个起点恢复多份独立环境，分别执行分支。终局分支 future value=0；非终局边界通过同一个语言模型生成 <value>成功概率</value>，估计未来回报。

actor return=段内 reward+终点 value；TD target G 是同起点各分支 return 的均值。actor advantage 做组内中心化；critic 生成的 value 按 −abs(value−G) 奖励，也做组内中心化。两种样本共同更新同一套参数，不额外训练线性 value head。

StateStore 保存边界 token 前缀、动作和环境观察。下一轮恢复时重新 reset、重放动作，并逐字验证观察。旧前缀是上下文；只有新 macro-step 的模型 token 进入 loss。warmup 阶段先用完整 episode 提供初始回报，再进入截断 TD 更新。

## 从代码入口开始

```bash
cd /data/xionglei-extract/agentic-rl-lab
source .venv/bin/activate
python 09-tempo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 09-tempo/train.py
# 每次运行自动生成唯一 runs/ 子目录，替换下方实际路径
python 09-tempo/eval.py --checkpoint runs/<本次实验>/checkpoint-final --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `tempo.py / rollout.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、三步（warmup、macro TD、状态重放），并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-09-tempo`，已存在的目录会拒绝覆盖。

## 验证思路与消融

专项测试终局 value 强制为零、G 取分支均值、解析失败有明确惩罚；进一步比较 H、critic_samples 和 warmup_steps。查看解析失败率、endpoint value、state store 大小和实际成功率。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

这是参考仓库的算法级 TEMPO 教学实现。原生路径未做 prefix importance correction 与跨进程恢复；verl 扩展已实现并验证这两项机制。全解析失败用 0.5 bootstrap 并单独计数。长上下文训练和原文正式实验尚未验证。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料


## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 09-tempo/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 09-tempo/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor；CPU 逐入口证据见 [verl 审计](../reports/verl-audit-latest.json)。verl 与原生均保存 token、mask、旧概率和轮次日志。

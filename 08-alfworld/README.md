# ALFWorld：真实家务文本环境

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

先认识回合、状态、观察和环境动作，再理解训练。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

本章使用官方 ALFWorld + TextWorld 的文本环境。按 game.tw-pddl 找到可解游戏，实例化独立环境，reset 后获得目标和可行动作；模型每轮输出 `<action>动作</action>`，环境返回观察、done 与 won。

每道游戏采样 G 条独立轨迹，终局 reward 是环境 won 的 0/1。动作 token 学习、观察 token 只作上下文。环境和数据加载工具由原仓库迁移，文件头保留来源与修改说明，模型采样与更新已改为本地 Transformers/PyTorch。

已准备全部支持的可解文本游戏：train、valid_seen、valid_unseen；seen 与 unseen 必须分别汇报。`--smoke` 切换到明确命名的 toy fixture；真实 TextWorld reset/step 在专项测试中单独执行。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 08-alfworld/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 08-alfworld/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 08-alfworld/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `alfworld_data.py / alfworld_env.py / rollout.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-08-alfworld`，已存在的目录会拒绝覆盖。

## 验证思路与消融

先检查同一游戏 reset 可复现；采样组之间状态独立；game_id 不跨 train/eval。后续评测分别汇报 seen/unseen 成功率、平均步数、无效动作数。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

这里只使用文本环境，不下载或启动 THOR 视觉模拟器，也不需要 GPU detector。真实环境已通过 CPU 交互检查，但大模型家务成功率没有训练验证。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 2](https://arxiv.org/abs/2010.03768)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 08-alfworld/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 08-alfworld/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor。verl 与原生均保存 token、mask、旧概率和轮次日志。

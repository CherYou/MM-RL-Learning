# ALFWorld：文字环境与独立 rollout

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 教程命令与此一致。本章重点：环境状态账本；won/done 区分；独立 env。裁剪与组内优势见 [GRPO](../01-grpo/TUTORIAL.md) / [preliminary](../preliminary/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

本章使用官方 ALFWorld + TextWorld 的文本环境。按 game.tw-pddl 找到可解游戏，实例化独立环境，reset 后获得目标和可行动作；模型每轮输出 `<action>动作</action>`，环境返回观察、done 与 won。

每道游戏采样 G 条独立轨迹，终局 reward 是环境 won 的 0/1。动作 token 学习、观察 token 只作上下文。环境和数据加载工具由原仓库迁移，文件头保留来源与修改说明，模型采样与更新已改为本地 Transformers/PyTorch。

已准备全部支持的可解文本游戏：train、valid_seen、valid_unseen；seen 与 unseen 必须分别汇报。`--smoke` 切换到明确命名的 toy fixture；真实 TextWorld reset/step 在专项测试中单独执行。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 08-alfworld/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 08-alfworld/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
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

## 后端说明

本章默认入口即 `verl.yaml`。完整安装、恢复与 GPU 说明见 [VERL.md](../docs/VERL.md)；不要在第一次运行时同时学习算法与后端切换。

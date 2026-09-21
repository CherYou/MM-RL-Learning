# ReTool：工具执行与代码/观察交织

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 教程命令与此一致。本章重点：真实 Python 执行；观察不直接训练。裁剪与组内优势见 [GRPO](../01-grpo/TUTORIAL.md) / [preliminary](../preliminary/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

模型可以交替输出推理、`<python>代码</python>`、读取 `<observation>结果</observation>`，直到给出最终答案。工具运行与参数更新分离：工具侧只执行，优化侧只学习模型生成 token。

本地 Python 工具运行在独立子进程和临时目录，数值代码经过 AST 白名单检查，具有 CPU 时间、地址空间、输出大小和墙钟超时限制。支持常用算术、循环、列表、sum/range 及部分 math 函数。工具报错作为 observation 返回，允许后续轮次纠正。

训练题来自 DAPO-Math；验证器不会把执行成功直接等价为答案正确。真正 task reward 仍来自最终答案，工具成功只用于诊断。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 05-retool/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 05-retool/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 05-retool/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `rollout.py / environments.py / rewards.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-05-retool`，已存在的目录会拒绝覆盖。

## 验证思路与消融

固定任务比较禁用工具、单次工具、多轮修正；记录代码执行失败率、最终正确率和每条轨迹有效 token 比例。专项测试用真实 Python 计算 2+3，并确认返回的 5 没有进入策略 loss。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

工具是受限数值 Python，不是支持任意包和系统操作的生产容器沙箱。详见 docs/TOOLS.md；论文级 ReTool 的采样规模与训练结果未验证。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料


## 后端说明

本章默认入口即 `verl.yaml`。完整安装、恢复与 GPU 说明见 [VERL.md](../docs/VERL.md)；不要在第一次运行时同时学习算法与后端切换。

# ReTool：工具执行与代码/观察交织

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 本章重点：真实 Python 执行；观察不直接训练。裁剪与组内优势见 [GRPO](../03-grpo/TUTORIAL.md) / [损失与策略梯度基础](../01-foundations/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

模型可以交替输出推理、`<python>代码</python>`、读取 `<observation>结果</observation>`，直到给出最终答案。工具运行与参数更新分离：工具侧只执行，优化侧只学习模型生成 token。

本地 Python 工具运行在独立子进程和临时目录，数值代码经过 AST 白名单检查，具有 CPU 时间、地址空间、输出大小和墙钟超时限制。支持常用算术、循环、列表、sum/range 及部分 math 函数。工具报错作为 observation 返回，允许后续轮次纠正。

训练题来自 DAPO-Math；验证器不会把执行成功直接等价为答案正确。真正 task reward 仍来自最终答案，工具成功只用于诊断。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 10-retool/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 10-retool/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 10-retool/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[rollout.py](../src/agentic_rl/rollout.py)、[environments.py](../src/agentic_rl/environments.py)、[rewards.py](../src/agentic_rl/rewards.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-10-retool`。

## 指标与对照实验

固定任务比较禁用工具、单次工具、多轮修正；记录代码执行失败率、最终正确率和每条轨迹有效 token 比例。专项测试用真实 Python 计算 2+3，并确认返回的 5 没有进入策略 loss。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

执行器支持受限的数值 Python，限制可用语法、模块和运行时间。运行前可先查[工具说明](../docs/TOOLS.md)中的支持范围。

## 一手资料


## 后端说明

CPU 和 CUDA 环境的安装、训练中断后的恢复步骤见 [verl 指南](../docs/VERL.md)。

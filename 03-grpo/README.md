# GRPO：组内相对奖励

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

默认使用 TRL 的 `GRPOTrainer`。先用同一题的多条回答算出组内优势，再看裁剪如何限制策略更新。[损失与策略梯度基础](../01-foundations/TUTORIAL.md)

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

同一道题采样 G 个回答，先算可验证奖励，再得到 Aᵢ=(Rᵢ−mean(R))/(std(R)+1e−4)。每个回答内部的生成 token 共用 Aᵢ，prompt 与 padding 不参与 loss。

默认入口实际实例化 TRL GRPOTrainer。原生对照入口把采样、token logprob、组内归一化和损失完全展开，适合逐行阅读。训练没有另建 critic；reference policy 仅在 beta>0 时提供 KL 约束。整组奖励相同就保持零优势，不给它人为制造排序。

GSM8K 使用官方 main train/test。答案同时支持 `<answer>`、嵌套 boxed、`####`、数字等价；数学等价通过 math-verify 处理。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与 TRL 后端一致
source .venv/bin/activate
.venv/bin/arl train 03-grpo/config.yaml --smoke --backend trl
# 等价薄入口
python 03-grpo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 03-grpo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 03-grpo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

跨后端对照属于进阶：

```bash
.venv/bin/arl train 03-grpo/config.yaml --smoke --backend native
.venv/bin/arl train 03-grpo/verl.yaml --smoke --verl-workers 2
```

从上面的命令跟读代码时，可以打开[losses.py](../src/agentic_rl/losses.py)、[trl_backend.py](../src/agentic_rl/trl_backend.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-03-grpo`。

## 指标与对照实验

固定 prompt、模型和种子，比较 group_size=4/8、beta=0/0.02；观察 reward、reward_std、KL、梯度范数和回答长度。比较 native 与 TRL 时统一 reduction 和采样设置。教程中的 `[1,1,0,0]` 完整数据链可用来预测日志方向，而不是只看 loss 是否下降。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

CPU 示例使用 `debug_token` 奖励，以便观察组内奖励差异。评估数学能力时，使用独立 GSM8K 题目并按最终答案计算正确率。

## 一手资料

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)
- [DeepSeekMath 与 GRPO](https://arxiv.org/abs/2402.03300)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。`config.yaml` 默认使用 TRL。安装与恢复见 [verl 指南](../docs/VERL.md)。

```bash
.venv/bin/arl train 03-grpo/verl.yaml --smoke --verl-workers 2
# GPU 检查（需本机 GPU）
.venv/bin/python scripts/run_grpo_gpu_check.py --gpu-index 0 --output runs/verl-grpo-gpu-repeat
.venv/bin/python scripts/verify_grpo_gpu.py runs/verl-grpo-gpu-repeat \
  --report reports/verl-grpo-gpu-repeat-verification.json
```

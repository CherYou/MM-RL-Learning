# GRPO：组内相对奖励

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

理解同题比较、组优势、token 更新和退化组。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

同一道题采样 G 个回答，先算可验证奖励，再得到 Aᵢ=(Rᵢ−mean(R))/(std(R)+1e−4)。每个回答内部的生成 token 共用 Aᵢ，prompt 与 padding 不参与 loss。

默认入口实际实例化 TRL GRPOTrainer。原生对照入口把采样、token logprob、组内归一化和损失完全展开，适合逐行阅读。训练没有另建 critic；reference policy 仅在 beta>0 时提供 KL 约束。整组奖励相同就保持零优势，不给它人为制造排序。

GSM8K 使用官方 main train/test。答案同时支持 `<answer>`、嵌套 boxed、`####`、数字等价；数学等价通过 math-verify 处理。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 01-grpo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 01-grpo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 01-grpo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `losses.py / trl_backend.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-01-grpo`，已存在的目录会拒绝覆盖。

## 验证思路与消融

固定 prompt、模型和种子，比较 group_size=4/8、beta=0/0.02；观察 reward、reward_std、KL、梯度范数和回答长度。比较 native 与 TRL 时统一 reduction 和采样设置。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

CPU smoke 使用随机 tiny 模型和明确标记的 debug_token 奖励，只检查优化链路；它不能证明 GSM8K 正确率提高。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告。本地已完成 CPU 执行检查，并在物理 1 号 A100 上完成 verl GRPO 三步真实训练和 checkpoint 重载；尚未证明与上游相同的训练效果。

## 一手资料

- [资料 1](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)
- [资料 2](https://arxiv.org/abs/2402.03300)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 保留 TRL，verl 作为并列入口；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 01-grpo/verl.yaml --smoke --verl-workers 2
# 选择本机可用的 GPU；示例使用物理 index 0
.venv/bin/python scripts/run_grpo_gpu_check.py --gpu-index 0 --output runs/verl-grpo-gpu-repeat
.venv/bin/python scripts/verify_grpo_gpu.py runs/verl-grpo-gpu-repeat \
  --report reports/verl-grpo-gpu-repeat-verification.json
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor；CPU 逐入口证据见 [verl 审计](../reports/verl-audit-latest.json)。verl 与原生均保存 token、mask、旧概率和轮次日志。

GPU 验证使用 [verify-gpu.yaml](verify-gpu.yaml)。启动器通过 `--gpu-index` 选择设备，并核对 worker 与 `nvidia-smi` 监测到的 UUID 是否一致；配置文件不绑定某台机器。历史实测完成 48 条 rollout、三步非零梯度，actor 参数变化且 reference 冻结；保存后重载，独立 eval 子集答对 2/4。完整指标和单卡验证边界见 [GPU 验证记录](../docs/GRPO_GPU_VALIDATION.md)。

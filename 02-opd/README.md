# Medical OPD、SAR-OPD、IDT-OPD

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

分清 SFT、学生与两位教师，再比较先后训练和交替训练。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

这章保留参考仓库的完整实验关系：医疗 SFT 先得到固定 Medical Teacher；fresh Base Student 自行采样，Teacher 仅复评 Student 已生成 token。

SAR-OPD 是参考作者命名的 Staged Anchor-Restoration OPD：前一阶段在医疗 prompt 上蒸馏 Medical Teacher；后一阶段改用通用 C-Eval prompt 和冻结 Base Teacher，恢复通用能力。`medical_steps` 控制阶段边界，默认总步数的一半。

IDT-OPD 是 Interleaved Dual-Teacher OPD：偶数步医疗 Teacher + 医疗题，奇数步 Base Teacher + 通用题。两位 Teacher 都冻结，Student 参数连续更新。这里没有把分阶段 SAR 偷换成混合 batch。

`sft.yaml`、`medical-opd.yaml`、`idt-opd.yaml` 给出独立入口。最顺畅的方式是 `python scripts/medical_pipeline.py --variant sar-opd`，它先训练医疗 SFT Teacher，再自动把 checkpoint 传给 Student 阶段。可加 `--smoke` 验证完整链路。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 02-opd/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 02-opd/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 02-opd/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `trainers.py / scripts/medical_pipeline.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-02-opd`，已存在的目录会拒绝覆盖。

## 验证思路与消融

同时固定 MedQA 与 C-Eval 的评测题，比较 Base、Medical SFT Teacher、医疗 OPD 中间点、最终恢复点。画医疗/通用双指标，而非只看其中一项；SAR/IDT 是实验调度名称，不是论文标准算法名。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

本地尚未进行模型能力验证。普通章节配置设为 `teacher_model: medical_sft`，会先自动训练医疗 SFT Teacher，再把其 checkpoint 传给 OPD；也可以显式传入已有医疗 Teacher。`--smoke` 的微型 Teacher 只检查 OPD 链路，完整三阶段可运行 `medical_pipeline.py --smoke`。准备的是学习规模医疗训练子集与去重后的 MedQA 测试集，未复现原文百分比。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 2](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 02-opd/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 02-opd/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor。verl 与原生均保存 token、mask、旧概率和轮次日志。

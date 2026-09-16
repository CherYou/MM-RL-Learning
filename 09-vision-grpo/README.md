# Vision GRPO：图片真正参与策略

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

追踪图片怎样成为模型条件，再理解 GRPO 与视觉梯度。建议先读教程完成手算和自测，再回到本页运行代码。

[总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

算法目标仍是 GRPO，但 prompt 包含真实图片。生产入口使用 Transformers AutoProcessor 与 AutoModelForImageTextToText，配置为 Qwen2.5-VL。图片处理后的 pixel_values 和网格信息与 token sample 绑定，采样和打分都使用同一份视觉输入。

离线 CPU 检查构造标准 Transformers 微型 LLaVA（CLIP vision encoder + Llama text decoder），并使用真实 GeoQA 图片。它没有预训练能力，但图片特征确实进入模型：替换图片会改变 logprob，视觉投影层会收到非零梯度。不是把文件名或标准答案转成文字来替代图片。

GeoQA 原始字段 subject/choices/label 被转换为题目、A-D 选项和标签，原始 answer 推理解答只存为 solution。保留官方 original_split，train=3503，test=759；dev 图片保留在下载资产中，未混入训练或测试。

## 从代码入口开始

```bash
# 从仓库根目录运行
source .venv/bin/activate
python 09-vision-grpo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 09-vision-grpo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 09-vision-grpo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

本章 `config.yaml` 保存实验配置，`train.py`、`eval.py` 是直接可运行入口。共用实现见 `models.py / rollout.py / losses.py`，位于 `../src/agentic_rl/`；薄入口让修复 token 对齐、设备或日志问题时可以统一维护各章训练逻辑。

`--smoke` 强制 CPU、随机微型模型、两次更新，并使用标明的学习 fixture。普通配置读取 `data/` 的公开实验数据；没有远程训练服务调用。修改输出路径可用 `--output runs/my-09-vision-grpo`，已存在的目录会拒绝覆盖。

## 验证思路与消融

做图片置零/换图消融；确认 processor 和模型的视觉 token 数一致；检查训练 token 的图像条件与 rollout 一致；数学奖励必须看正确选项或正确表达式。

建议读 `tests/test_mechanisms.py` 中相应检查，再打开 `runs/<实验>/metrics.jsonl` 与 TensorBoard。Native 运行还保留 `trajectories.jsonl`，其中有 token IDs、mask、旧 logprob、turn spans 和 reward，可逐段检查信用分配。

## 复现边界

完整 Qwen2.5-VL 模型权重不属于本次 CPU 验证；默认不会加载大型 VLM 做训练。微型 LLaVA 只验证多模态数据与梯度链路，不能证明 GeoQA 能力。

上游原文中的 GPU、耗时、付费服务和准确率是参考作者报告；本地复现完成的是代码、数据、依赖和 CPU 执行检查，尚未声称得到相同训练效果。

## 一手资料

- [资料 1](https://huggingface.co/datasets/hz2475/geoQA)
- [资料 2](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/llava)

## verl 训练路径

本章提供 `verl.yaml`（CPU）和 `verl-gpu.yaml`（显式 CUDA）。默认 config.yaml 已切换为 verl；`--backend native` 可读取原生参考实现。根环境 CLI 会自动切换到独立 verl 环境。

```bash
# 在项目根目录运行；两个实际 CPU worker
.venv/bin/arl train 09-vision-grpo/verl.yaml --smoke --verl-workers 2
# GPU 配置供后续实验使用，本次未运行 GPU 验证
# .venv/bin/arl train 09-vision-grpo/verl-gpu.yaml --verl-workers 2
```

真实更新代码在 `src/agentic_rl/verl_backend/`，本章机制对应关系、安装和恢复方式见 [verl 指南](../docs/VERL.md)。轨迹通过 DataProto 传递，参数更新调用官方 verl actor。verl 与原生均保存 token、mask、旧概率和轮次日志。

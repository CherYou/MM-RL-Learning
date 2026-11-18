# Vision-GRPO：图像条件与 VLM logprob

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：verl（`verl.yaml`）。** 本章重点：两层检查：输入链路 vs 能力。裁剪与组内优势见 [GRPO](../03-grpo/TUTORIAL.md) / [损失与策略梯度基础](../01-foundations/TUTORIAL.md)。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

算法目标仍是 GRPO，但 prompt 包含真实图片。生产入口使用 Transformers AutoProcessor 与 AutoModelForImageTextToText，配置为 Qwen2.5-VL。图片处理后的 pixel_values 和网格信息与 token sample 绑定，采样和打分都使用同一份视觉输入。

CPU 示例使用随机初始化的微型 LLaVA（CLIP 视觉编码器和 Llama 文本解码器）读取 GeoQA 图片。固定题目、替换图片，再比较生成 token 的 logprob，可以观察图像是否影响预测。

GeoQA 原始字段 subject/choices/label 被转换为题目、A-D 选项和标签，原始 answer 推理解答只存为 solution。保留官方 original_split，train=3503，test=759；dev 图片保留在下载资产中，未混入训练或测试。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与本章 verl 后端一致
source .venv/bin/activate
.venv/bin/arl train 12-vision-grpo/verl.yaml --smoke --verl-workers 2
# 等价薄入口
python 12-vision-grpo/train.py --smoke
# 每次运行自动生成唯一 runs/ 子目录
RUN_DIR="runs/RUN_NAME"
python 12-vision-grpo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[models.py](../src/agentic_rl/models.py)、[rollout.py](../src/agentic_rl/rollout.py)、[losses.py](../src/agentic_rl/losses.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-12-vision-grpo`。

## 指标与对照实验

做图片置零/换图消融；确认 processor 和模型的视觉 token 数一致；检查训练 token 的图像条件与 rollout 一致；数学奖励必须看正确选项或正确表达式。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 在 native 后端，`trajectories.jsonl` 还保存 token、mask、旧 logprob 和轮次范围，可以定位某条轨迹上参与更新的位置。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

常规配置使用 Qwen2.5-VL，需要相应的预训练权重和容纳模型的设备。评估时在相同的 GeoQA 题目上对照原图、换图和置零图。

## 一手资料

- [GeoQA 数据集](https://huggingface.co/datasets/hz2475/geoQA)
- [Transformers LLaVA 接口](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/llava)

## 后端说明

CPU 和 CUDA 环境的安装、训练中断后的恢复步骤见 [verl 指南](../docs/VERL.md)。

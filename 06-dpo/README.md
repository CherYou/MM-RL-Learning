# DPO：离线偏好优化

**[直接阅读本章新手教程：TUTORIAL.md](TUTORIAL.md)**

**本章默认教学后端：TRL（`config.yaml` → `backend: trl`）。** 建议先读 [第 01 章的 BCE 与偏好分差](../01-foundations/TUTORIAL.md)；PPO/GRPO 不是强制前置。

[学习路线](../docs/LEARNING_PATH.md) · [总目录](../README.md) · [框架设计](../docs/ARCHITECTURE.md)

## 本地实现

每条数据包含 prompt、chosen、rejected。DPO 用冻结 reference 对两种回答的 logprob 差进行校正：z=beta[(log πθ(y+)−log πθ(y−))−(log πref(y+)−log πref(y−))]，L=−log sigmoid(z)。

completion logprob 先在有效回答 token 上求和，prompt/pad 不参与比较。与 PPO 不同，DPO 不在训练时在线采样、不构造 GAE，也不训练 value model。默认入口直接使用 TRL DPOTrainer，同时保留原生 loss 方便观察 chosen 上升、rejected 下降的梯度。

偏好对分别从 GSM8K 的训练集和测试集生成。chosen 保留正确解答，rejected 使用数值答案加 1 的合成错误回答。

## 从代码入口开始

```bash
# 从仓库根目录运行；默认与 TRL 后端一致
source .venv/bin/activate
.venv/bin/arl train 06-dpo/config.yaml --smoke --backend trl
# 等价薄入口
python 06-dpo/train.py --smoke
# 正式学习配置（CPU 默认；较大模型可能较慢）
python 06-dpo/train.py
# 每次运行自动生成唯一 runs/ 子目录；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
python 06-dpo/eval.py --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

从上面的命令跟读代码时，可以打开[trl_backend.py](../src/agentic_rl/trl_backend.py)、[trainers.py](../src/agentic_rl/trainers.py)、[losses.py](../src/agentic_rl/losses.py)。

`--smoke` 在 CPU 上用微型模型和样例数据运行一个短训练示例。常规配置从 `data/` 读取数据。若要指定输出位置，请用新目录，例如 `--output runs/my-06-dpo`。

## 指标与对照实验

检查初始 policy=reference 时 loss≈log(2)，一步反传应鼓励 chosen、压低 rejected。比较 beta、回答长度和 synthetic-negative 策略，记录 reward margin。评测附带 chosen/rejected 原始 logprob 偏好率，注意长度偏差。

在 `runs/<实验>/metrics.jsonl` 或 TensorBoard 查看指标。 [机制测试](../tests/test_mechanisms.py)

## 使用范围

错误回答通常比正确解答短，模型可能仅凭长度判断哪条更受偏好。比较偏好学习效果时，可以加入长度相近、推理中有错误的回答，并检查模型能否区分它们。

## 一手资料

- [DPO 原论文](https://arxiv.org/abs/2305.18290)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/v0.25.1/en/dpo_trainer)

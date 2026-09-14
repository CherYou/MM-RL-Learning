# verl GRPO：物理 1 号 GPU 实际验证

2026-09-10，使用第二张卡（物理 index 1，NVIDIA A100-SXM4-40GB）完成本仓库 **verl GRPO 的真实采样、反向传播、三步参数更新、checkpoint 保存及重载评估**。运行进程退出码为 0。训练采用预训练 Qwen2.5-0.5B-Instruct、真实 GSM8K 数据和数学答案奖励。

本次是小规模功能验证。没有测量训练前后的能力提升，也没有复现论文准确率、进行 TRL/verl 吞吐对比或验证多卡训练。

## 配置与设备证据

| 项目 | 实际值 |
| --- | --- |
| 物理 GPU | 历史运行使用 index 1；公开报告将设备 UUID 归一化为 `${GPU_UUID}` |
| 进程设备映射 | 历史运行选择物理 index 1，进程内为 `cuda:0` |
| 运行环境 | `.venv-verl-gpu`；verl 0.7.1、Torch 2.9.0+cu129、vLLM 0.12.0、Transformers 4.57.6、Ray 2.49.2 |
| 模型 | `models/Qwen--Qwen2.5-0.5B-Instruct` |
| 数据与奖励 | `data/gsm8k/train.jsonl`；真实答案匹配，`reward: math` |
| 训练量 | 3 步，每步 2 道题，每题 8 个回答，共 6 道题、48 条 rollout |
| 长度与采样 | 最多生成 512 token，上下文上限 2048；temperature=1、top_p=1、top_k=-1 |
| 优化器设置 | learning_rate=5e-6，beta=0，ppo_epochs=1，micro_batch_size=1，seed=42 |
| worker 与推理 | 单个真实 verl Ray GPU worker，官方 DataParallelPPOActor 接口与异步 vLLM rollout |
| 分片边界 | 使用 FSDP1 路径；world_size=1 时 PyTorch 自动使用 NO_SHARD，未验证 GPU 多卡分片 |
| 执行方式 | `TORCHDYNAMO_DISABLE=1`，实际 CUDA eager 运算；本次不评价编译优化后的吞吐 |

[专用配置](../01-grpo/verify-gpu.yaml)使用仓库相对模型和数据路径，不绑定设备编号或 UUID。启动器记录所选物理设备，核对脚本比较 worker 与 `nvidia-smi` 采样的 UUID。实际身份、初始/最终参数摘要已汇总进[机器可读报告](../reports/verl-grpo-gpu1-verification.json)。

历史报告中的 `01-grpo/verify-gpu1.yaml` 是运行当时的文件名；当前可移植配置已更名为 `verify-gpu.yaml`。历史命令保留原名，以免把旧证据误写成使用了新配置。

## 实际指标与检查

| 更新步（日志 step） | 奖励均值 | 裁剪前梯度范数 | loss | 有效训练 token | 更新后策略版本 |
| --- | --- | --- | --- | --- | --- |
| 1（0） | 0.3125 | 4.602261 | -0.0000883676 | 3419 | 1 |
| 2（1） | 0.4375 | 12.193846 | -0.0017877854 | 3670 | 2 |
| 3（2） | 0.1250 | 3.409316 | 0.0000819825 | 4035 | 3 |

三步 `update/skipped` 均为 0，梯度非零且指标有限。Actor 参数摘要发生变化，reference 参数摘要保持一致。beta=0，因此这次未验证非零 KL 正则项的 GPU 更新。采样奖励合计 14/48；不同训练步使用不同题目，奖励序列不构成学习趋势。

[自动核对脚本](../scripts/verify_grpo_gpu.py)检查真实 CUDA/verl 身份、监测进程与 worker 的设备 UUID 一致性、更新步、参数变化、reference 冻结、策略版本、组大小、token/mask/行为概率长度对齐、奖励独立重算、非退化组、模型/优化器/控制器保存及 TensorBoard 文件。加入 checkpoint 重载评估与 tokenizer 核对后，共 **22 项检查通过**，见 [机器可读报告](../reports/verl-grpo-gpu1-verification.json)。

完整成功训练进程（含启动和保存）耗时 **185.51 秒**。约每 2 秒采样的 GPU 显存峰值为 **19,604 MiB，约 19.14 GiB**；这是采样观测值，不是精确瞬时峰值。记录见 [GPU 监测](../reports/verl-grpo-gpu1-20260910-a6-monitor.json)和 [训练日志](../reports/verl-grpo-gpu1-20260910-a6.log)。

原运行的 `checkpoint-final/` 包含可供 Transformers 加载的 HF 模型、FSDP 模型/优化器状态、worker RNG/策略版本、冻结 reference 和控制器状态。本次实际验证了模型重载推理；未执行该 GPU checkpoint 的优化器续训。`runs/` 目录未随公开仓库发布。

在同一张卡重新加载 checkpoint，使用独立 GSM8K eval split 的 4 道题进行贪心生成，答对 **2/4**。评估入口检查 train/eval 题目无交集，评估奖励再次独立重算通过；结果保存在[机器可读报告](../reports/verl-grpo-gpu1-verification.json)的 `evaluation` 字段。样本很小且未做同题训练前对照，这个分数只用于确认保存模型可评估。

Transformers 4.57.6 重载时对保存的 Qwen tokenizer 发出 Mistral regex 警告。检查本地版本源码发现，其 config 版本判断会把此版本保存的非 Mistral 模型也带入警告分支。核对原始与保存 tokenizer 的词表、pre-tokenizer、特殊 token、chat template，并逐一比较本地 GSM8K 全部题目及本次生成文本的 token IDs；结果一致。保留原 Qwen 分词规则，未应用 Mistral 正则替换。核对结论已收录在[机器可读报告](../reports/verl-grpo-gpu1-verification.json)的 `tokenizer_audit` 字段。

## 本次修复

- `gpu_worker.py`：训练/rollout 切换采用异步方法，避免 Ray 已运行的事件循环内再次调用 `run_until_complete`；按 verl 0.7.1 接口释放 rollout 缓存，并记录、校验实际 GPU UUID。
- `gpu_rollout.py`：官方 vLLM replicas 创建后先进入 sleep，再通过 worker 权重同步唤醒；修复首次重复唤醒造成的 CuMem 错误。退出时清理本任务创建的服务。
- `gpu_config.py`：采样 seed 通过 `engine_kwargs.vllm.seed` 传入；增加实际 RolloutConfig dataclass 转换检查。
- GPU 验证启动器：历史运行选择物理 GPU 1 并记录显存；当前入口以 `--gpu-index` 参数选择设备。它关闭 TorchDynamo，避免缺少主机编译工具链时触发 vLLM logprob 辅助函数编译失败。`enforce_eager=True` 本身不会关闭该辅助函数的 `torch.compile`。

相关机制回归检查 **23 项通过**，包括异步切换、replica 初始休眠和真实 GPU 配置 schema 转换；见 [回归日志](../reports/verl-gpu-regression-tests.log)。这些回归测试使用 CPU，CUDA 端到端证据来自上述实际运行。

## 复现命令

从项目根目录执行；用 `--gpu-index` 选择当前机器上的物理设备。启动器与核对脚本会比较监测进程和 worker 的 UUID，输出目录必须是新目录。

```bash
# 从仓库根目录运行
.venv/bin/python scripts/run_grpo_gpu_check.py --gpu-index 0 --output runs/verl-grpo-gpu-repeat
.venv/bin/python scripts/verify_grpo_gpu.py runs/verl-grpo-gpu-repeat \
  --report reports/verl-grpo-gpu-repeat-verification.json
```

仅重载本次已经保存的模型进行评估：

```bash
GPU_INDEX=0
RUN_DIR="runs/verl-grpo-gpu-repeat"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
ACCELERATE_USE_CPU=false HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1 \
.venv-verl-gpu/bin/arl eval 01-grpo/verify-gpu.yaml \
  --checkpoint "$RUN_DIR/checkpoint-final" \
  --limit 4 --output "$RUN_DIR/evaluation"
```

训练完成时核对脚本有 19 项训练检查；若运行目录内另有 `evaluation/summary.json` 和 `evaluation/tokenizer-audit.json`，会纳入相应检查及结果。历史训练与评估进程退出后，所选卡显存回到 0 MiB。其他 GPU 上的已有任务未被该次清理流程停止。

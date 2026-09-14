# 验证范围与实验记录

初始原生/TRL/verl 批量检查使用 CPU；对应两个环境安装 `torch==2.9.1+cpu`，`torch.version.cuda` 为 `None`，smoke 设置 `CUDA_VISIBLE_DEVICES=''`、`ACCELERATE_USE_CPU=true`、`HF_HUB_OFFLINE=1`。2026-09-10 按用户指定新增物理 1 号 GPU 的 verl GRPO 验证，使用独立 CUDA 环境。两阶段证据分别记录，不请求远程训练服务。

## 哪些证据能证明什么

| 证据 | 覆盖范围 | 不代表什么 |
| --- | --- | --- |
| `reports/pytest.log` | loss 数学不变量、梯度屏蔽、Teacher freeze、DPO 方向、GAE、CAPO、图像作用、checkpoint 等 | 正式任务学习效果 |
| `reports/cpu-audit-latest.json` | 原有 LLM 章节训练入口及 native/TRL 对照能实际生成、执行更新路径并保存状态 | 每个随机模型都能获得有效 task reward |
| `reports/embodied-validation.json` | SAC/TD3/HER/IQL 的真实物理训练、参数更新、重载、独立评估及数据边界 | 多种子 benchmark 或 VLA 能力 |
| `reports/medical-pipeline-smoke.log` | SFT Teacher checkpoint 传递到 staged OPD，阶段切换 | 医疗正确率改善 |
| `reports/medical-auto-teacher-smoke.log` | 医学章节默认配置自动准备 SFT Teacher，再执行两个 SAR 阶段 | 预训练医疗模型的实际效果 |
| `reports/checkpoint-eval.log` | TRL checkpoint 能重新加载并跑独立问题评测 | GSM8K benchmark 分数 |
| `reports/vision-checkpoint-eval.log` | 视觉 checkpoint 重载后能读取独立图像并完成评测 | GeoQA benchmark 分数 |
| `reports/data-verification.json` | 文件 hash、字段、题目去重、split 隔离、真实图像和游戏存在 | 全量论文训练数据都已下载 |
| `reports/idempotent-data-preparation.log` | 已有数据的准备流程可在离线模式下重复运行 | 未下载数据可离线获取 |
| `reports/local-services.json` | 本地服务的进程归属与 HTTP 200 | 远程浏览器已经建立 SSH 转发 |

第一阶段原生/TRL 交付核对报告另存为 [completion-audit-initial.json](../reports/completion-audit-initial.json)，保留当时 15 个顶层章节与 22 个 LLM 训练入口的历史范围。当前 [completion-audit.json](../reports/completion-audit.json) 按扩展后的 19 个顶层目录、20 篇详解与 57 个训练章节薄入口核对，并引用新的具身证据。

2026-09-11 教程开发回归日志为 [tutorial-tests.log](../reports/tutorial-tests.log) 和 [tutorial-verl-tests.log](../reports/tutorial-verl-tests.log)；改名后的 PPO/DPO 实际训练入口见 [tutorial-renamed-entrypoints.log](../reports/tutorial-renamed-entrypoints.log)。

## 专项检查

- PPO 正、负优势对应不同裁剪方向；CISPO 裁剪权重仍保留梯度。
- 全同奖励组保持零优势；没有为了 smoke 伪造组内正负任务结果。
- GSPO 使用几何平均序列比率；DAPO 按有效 token 聚合。
- OPD Teacher 不接收梯度；DPO 提高 chosen、降低 rejected。
- GAE 区分 terminal 与提供 bootstrap 的截断状态。
- TEMPO 终局 value=0，target 是同起点分支 return 均值。
- TEMPO smoke 额外跑第三步，覆盖 warmup → macro TD → StateStore 重放，context_limit 不存为可继续训练的边界。
- AgentOPSD 轮优势范围有界且不翻转终局方向。
- 真实 Python 工具算出结果后，observation 不进入 loss。
- 真实 TextWorld 游戏 reset/step 可用；本地 BM25 有真实文档可检索。
- Harness 的 session 保持隔离，rewrite 前缀形成分支，CAPO mask 外参数不接收梯度。
- 不同图片能改变视觉模型 logprob；投影层反传非零；保存重载后概率一致。
- 视觉专用占位 token 在生成与打分两侧使用同一支持集，避免随机模型输出占位符后破坏图像特征对齐。

## 如何扩展成能力复现实验

先选一个小型预训练模型，固定数据版本、学习/验证 split、种子和采样预算。记录训练前 baseline，再训练并对同样的 held-out id 评测。多种子重复后才比较准确率、成功率和采样效率。每个章节 README 给出应对照的机制和未实现的论文级工程范围。

对退化采样尤其要看 `update/skipped`、有效 token 数和 gradient norm：DAPO 全错组、Harness 不合法 JSON、TEMPO critic 全解析失败都可能没有某部分有效梯度。日志如实显示这一点，不能仅以程序退出 0 就宣布算法有效。

## verl 扩展的实际验证

- `reports/verl-audit-latest.json`：16 个入口（15 个算法名加 Medical OPD 变体），每项使用两个真实 verl Ray CPU worker，检查进程退出、状态、模型 checkpoint。
- `reports/verl-mechanism-tests.log`：46 项通过，包括原有 25 项以及精确轨迹往返、跨 microbatch/rank 的 loss 分母、冻结 Teacher、prefix correction、重放状态和 GPU Hydra schema 等 21 项。
- `reports/verl-fsdp-capo.json`：两个 CPU rank 的真实 FULL_SHARD FSDP，调用官方 actor 前向与优化器，检查 CAPO 选择区内确有参数更新、选择区外零更新、两 rank 完整权重一致。另验证 FSDP 推理后重新分片，以及官方 FSDPCheckpointManager 保存/恢复后的模型参数逐项一致。
- `reports/verl-resume.json`：通过根 CLI 自动切换环境，TEMPO 三步 checkpoint 恢复后继续第四步；检查恢复权重、策略版本与实际步数，并重载 checkpoint 评测。另执行医疗 SFT → verl SAR 链路。
- `reports/verl-completion-audit.json`：重新检查默认配置、所有算法真实运行目录、worker 身份、指标、checkpoint、锁文件和报告，区别于第一阶段核对。

初始 CUDA 依赖检查未执行 GPU 模型。后续单卡 GRPO 已补充真实 CUDA vLLM 服务、权重同步、反传和 checkpoint 重载证据，详见下一节；GPU VLM/critic、多卡/多节点、其他 GPU 算法及正式训练效果仍未验证。Harness 随机 rollout 可以没有合法 action/args，不能把其程序退出成功当作有非零 CAPO 学习；非零分区更新由独立 FSDP 机制实验明确验证。

## 物理 1 号 GPU 的 GRPO 验证（2026-09-10）

- [详细运行记录](GRPO_GPU_VALIDATION.md)：A100-SXM4-40GB、Qwen2.5-0.5B-Instruct、真实 GSM8K，48 条 rollout、三步非零梯度更新。Actor 发生变化、reference 保持冻结；单 rank FSDP1 自动使用 NO_SHARD。
- [核对报告](../reports/verl-grpo-gpu1-verification.json)：22 项通过，包括 19 项训练检查及 checkpoint CUDA 重载、独立 eval 奖励重算、tokenizer 保存前后一致性。
- [GPU 监测](../reports/verl-grpo-gpu1-20260910-a6-monitor.json)：实际进程退出码 0、耗时 185.51 秒、采样显存峰值 19,604 MiB；worker UUID 与物理 GPU 1 相符。
- [checkpoint 评估](../runs/verl-grpo-gpu1-20260910-a6/evaluation/summary.json)：同一卡重载，独立 eval 子集答对 2/4。未运行训练前对照，不能据此推断能力提升。
- [相关机制回归](../reports/verl-gpu-regression-tests.log)：23 项通过，覆盖异步事件循环、replica 初始 sleep、真实 RolloutConfig schema 转换及既有 verl 机制。

`reports/verl-completion-audit.json` 保留初始 CPU 接入阶段的历史范围；新 GPU 结论以本节报告为准。GPU 优化器状态已经保存，但本次未执行其续训恢复验证。

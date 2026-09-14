# 调研与框架选择

调研日期：2026-09-09–10。优先依据原仓库源码、论文、官方框架文档；实现 API 以实际安装版本检查为准。

## 原仓库固定范围

读取了 [原仓库](https://github.com/KMnO4-zx/agentic-rl-lab) 并固定提交 `d745e6e26f96485f7689da962630026723546cd5`。这个提交包含 00–08、09-vision-grpo、09-tempo、09-AgentOPSD，共 12 个顶层算法目录。General OPD 是 02-opd 下的子目录。README 提及 Slime，但没有对应独立目录；Harness-RL 当时仍是计划章节。

完整来源存于 `references/upstream/`。`alfworld_data.py`、`alfworld_env.py`、`agentopsd.py` 复用了不依赖训练服务的确定性工具，修改了包引用/数据路径并保留文件头归属；训练程序重新使用本地框架实现。

## 框架取舍

| 方案 | 一手依据 | 本项目决定 |
| --- | --- | --- |
| TRL | [0.25.1 GRPO](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)、[DPO](https://huggingface.co/docs/trl/v0.25.1/en/dpo_trainer)、[PPO](https://huggingface.co/docs/trl/v0.25.1/en/ppo_trainer) | 标准算法直接实例化真实 Trainer |
| PyTorch/Transformers/Accelerate | [PyTorch CPU 安装](https://docs.pytorch.org/get-started/locally/)、[Accelerate](https://huggingface.co/docs/accelerate/index) | 特殊算法的显式采样、概率、loss 与更新 |
| verl | [官方仓库](https://github.com/verl-project/verl) | 已固定 v0.7.1；复杂在线算法默认接入真实 WorkerGroup、AgentLoop 和 actor，CPU/GPU 独立环境 |
| 官方 Harness-RL / slime | [官方实现](https://github.com/jiangxinke/Harness-RL) | 参考接口记录、轨迹树与 CAPO；本地重建机制，不安装其 GPU 分布式栈 |
| ALFWorld/TextWorld | [官方实现](https://github.com/alfworld/alfworld) | 安装真实文本环境与实际游戏资产 |

TRL 最新版本已进入 1.x，接口相较 v0 有迁移，见 [官方发布记录](https://github.com/huggingface/trl/releases) 与 [迁移说明](https://github.com/huggingface/trl/blob/main/MIGRATION.md)。本项目明确固定 **TRL 0.25.1 + Transformers 4.57.6**，以同一套已测试 API 同时支撑 PPO、DPO、GRPO/GSPO。不是把旧版参数复制到最新版后仅做 import 检查。CPU 小训练实际执行了这些 Trainer。

## 关键算法判断

- **DAPO**：`loss_type="dapo"` 只代表损失归一化，不等于完整 Dynamic Sampling、Clip-Higher 和 overlong 处理。因此完整章节走显式循环，额外 TRL 入口只作为 loss 对照。[官方 DAPO](https://github.com/BytedTsinghua-SIA/DAPO)
- **GSPO**：序列 ratio 是有效 token log-ratio 的均值取指数，配置对应 `importance_sampling_level="sequence"`。[GSPO 论文](https://arxiv.org/abs/2507.18071)
- **OPD/OPSD**：Teacher 复评 Student 实际采样 token，不能退化成拿 Teacher 生成答案做 SFT；OPSD 还需要固定 step-0 参数和不同上下文。[上游 OPSD 代码](https://github.com/KMnO4-zx/agentic-rl-lab/tree/d745e6e26f96485f7689da962630026723546cd5/04-opsd)
- **SAR/IDT**：参考作者的实验方案命名；SAR 是分阶段恢复，IDT 是交替双 Teacher，不能当成同一个混合数据训练。[上游实验讲解](https://github.com/KMnO4-zx/agentic-rl-lab/blob/d745e6e26f96485f7689da962630026723546cd5/02-opd/readme.md)
- **AgentOPSD**：当前同一参数快照加 Skill 进行轮级信用分配，评测必须去掉 Skill。[论文](https://arxiv.org/abs/2608.05987)
- **TEMPO**：用生成式 critic 与 macro-step 拼接 TD 信号；上游本身标注算法级复现，并说明 prefix correction 未完成。原生路径保留该参考边界；verl 扩展已增加行为前缀 logprob、精确 prefix importance correction、状态保存与恢复。[上游源码](https://github.com/KMnO4-zx/agentic-rl-lab/tree/d745e6e26f96485f7689da962630026723546cd5/09-tempo)
- **Harness-RL**：明确对应 [2608.29641](https://arxiv.org/html/2608.29641v1)。它同时需要 Interface Call Records、session-wise prefix trees、分段奖励和基于激活的 CAPO 参数分区；不是普通“多工具 GRPO”。本项目读了论文方法部分及官方仓库，选择 central-only MLP 单元分区的轻量实现。

## 新增章节依据

PPO 根据 [原始论文](https://arxiv.org/abs/1707.06347) 和 TRL 的 actor/ref/reward/value 四部分接口实现；DPO 根据 [原始论文](https://arxiv.org/abs/2305.18290) 和 TRL 成对偏好数据契约实现。数学答案验证使用 [Hugging Face math-verify](https://github.com/huggingface/Math-Verify)，优先精确/数值比较，必要时在有界超时内进行符号等价比较。

文档中的架构取舍属于本地工程选择，不能据此推断哪个框架在本机 GPU 上更快，也没有推断任何尚未运行的模型准确率。

## verl 接入后的取舍

最初选 TRL 是为了尽快验证标准 Trainer 与 CPU 教学循环。对多轮、多模型、状态重放及分区梯度而言，把分布式基础设施推迟到以后会增加重复建设，因此现在保留 TRL 标准入口，复杂在线算法以 verl 为默认。TRL 并非只能单卡或不能做 Agent；最新文档的能力也不能直接归于固定的 0.25.1。实际版本、API 和算法扩展对应关系见 [VERL.md](VERL.md)。

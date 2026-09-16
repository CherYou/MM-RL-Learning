# MM-RL-Learning

面向大模型、多模态与强化学习的中文学习仓库，包含技术报告阅读资料、算法新手教程、可运行的训练实现与实验验证记录。

## 参考与感谢

感谢 [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 提供章节组织、算法资料与早期工具实现的参考，固定版本为 `d745e6e26f96485f7689da962630026723546cd5`。本项目重新编写面向初学者的中文教程、手算例子、代码阅读说明和练习，使用本地 PyTorch / TRL / verl 实现训练，并增加连续控制课程。SAR/IDT 的实验调度名称也来自该参考资料。

感谢算法论文作者，以及 [TRL](https://github.com/huggingface/trl)、[verl](https://github.com/verl-project/verl)、[Search-R1](https://github.com/PeterGriffinJin/Search-R1)、[ReTool](https://github.com/ReTool-RL/ReTool)、[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)、[ALFWorld](https://github.com/alfworld/alfworld)、[AgentOPSD](https://github.com/ZethWang/AgentOPSD)、[Harness-RL](https://github.com/jiangxinke/Harness-RL)、[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) 等项目。论文及官方接口依据在各教程中就近列出，参考仓库链接集中在这里。

来源按实际情况区分：训练循环、后端适配、新增连续控制实现和教程为本地编写；`agentopsd.py`、`alfworld_data.py`、`alfworld_env.py` 有已注明的 Apache-2.0 移植来源，技能文本和第三方数据也各有归属。未将这些内容宣称为原创。详情见 [代码来源核对](docs/CODE_PROVENANCE.md)、[NOTICE](NOTICE) 和 [LICENSE](LICENSE)。`references/upstream/` 保留未修改的参考快照及原许可证，本地入口不导入其远程训练程序。

## 这套课程包含什么

强化学习部分目前共 **19 个顶层学习目录、20 篇算法详解**（包括 General OPD 子章），每篇配有内置图像生成工具制作的原创概念图。另有 `preliminary/FOUNDATIONS.md` 补足 MDP、Bellman 方程、概率、自动求导、回放、离线学习和实验评估等前置知识。开始阅读：[零基础学习导航](docs/BEGINNER_GUIDE.md)。大模型技术报告资料收录在 [Basic LLM](Basic%20LLM/README.md)。

PPO 与 DPO 位于 `001-ppo`、`002-dpo`；原损失函数目录扩展为 `preliminary`。新增 SAC、TD3、HER、IQL 用真实 FetchReach / MuJoCo 演示连续动作、稀疏奖励和固定数据学习，是理解具身/VLA 强化学习的基础课；输入为低维机械状态与目标，尚未训练视觉语言动作模型。

项目可以克隆到任意目录。文档、配置和命令中的相对路径都以仓库根目录为基准；完整约定见[路径与可移植性说明](docs/PORTABILITY.md)。

已完成代码、实验数据、依赖、本地日志与 CPU 执行验证。2026-09-10 新增 **物理 1 号 A100 上的 verl GRPO 实际验证**：Qwen2.5-0.5B-Instruct、真实 GSM8K 奖励、48 条 rollout、3 次非零梯度更新及 checkpoint 重载评估。详见 [GPU 验证记录](docs/GRPO_GPU_VALIDATION.md)。这属于功能验证，尚未证明模型能力提升或复现论文准确率。

## 大模型技术报告

[Basic LLM](Basic%20LLM/README.md) 按“模型系列 → 模型版本”组织技术报告、中文翻译与解读、原文资料和配图。阅读报告可帮助理解模型架构、训练和推理系统，再与本仓库的强化学习实现对照。

| 模型 | 内容 | 阅读与下载 |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/README.md) | CED、CSA2、KV 缓存压缩、训练与推理基础设施、后训练和评估 | [中文合订稿](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/report.zh.md) · [PDF](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/documents/translation.zh.pdf) · [Word](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/documents/translation.zh.docx) |

DeepSeek 资料整理自提供的中文阅读文档，包含四份分章稿、12 张报告配图、英文提取文本、页面预览和文档导出工具。正式来源见 [DeepSeek 官方发布说明](https://api-docs.deepseek.com/news/news260910/) 与 [作者技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)。中文资料属于学习用翻译与解读，报告中的实验结果属于原作者，不是本仓库的复现实验；文件对应关系见该目录的 `manifest.json`。

## 五分钟开始

项目面向 Linux / WSL 和 Python 3.12。初始化脚本会在当前仓库内创建 `.venv/`；verl CPU 与 GPU 环境分别使用 `.venv-verl/` 和 `.venv-verl-gpu/`。复杂算法默认走 verl，CLI 会按设备切换环境。

```bash
git clone https://github.com/CherYou/MM-RL-Learning.git
cd MM-RL-Learning
bash scripts/setup.sh
source .venv/bin/activate

# 查看版本、设备和本地数据
arl doctor

# 阅读并计算 loss 梯度，生成曲线图
arl loss-demo

# 使用真正的 TRL Trainer 跑两步离线 CPU smoke
python 01-grpo/train.py --smoke
python 001-ppo/train.py --smoke
python 002-dpo/train.py --smoke

# 多轮 agent / 自蒸馏 / 特殊机制
python 05-retool/train.py --smoke
python 09-vision-grpo/train.py --smoke
python 12-harness-rl/train.py --smoke

# 完整医疗 SFT → Medical Teacher → SAR-OPD 链路
python scripts/medical_pipeline.py --smoke

# 启动或查看本地面板
make logs
make status
```

`--smoke` 使用本地随机微型 GPT-2 / LLaVA、独立 fixture 和显式 debug reward，强制 CPU。它验证采样、反传、环境接口、保存与日志，不代表任务成功率。实验配置文件使用真实公开数据；去掉 `--smoke` 后会加载配置中的预训练模型，执行实际训练，但仍默认 CPU。

## 章节目录与推荐顺序

语言模型路线：**preliminary → 001 PPO → 002 DPO → 01 GRPO → DAPO/GSPO → OPD/OPSD → 工具 Agent → ALFWorld/AgentOPSD/TEMPO → Harness-RL**。连续控制路线：**preliminary → SAC → TD3 → HER → IQL**。每章 README 的第一条学习入口指向详细教程。

| 目录与运行说明 | 算法教程 | 本地重点 | 默认后端 |
| --- | --- | --- | --- |
| [preliminary](preliminary/README.md) | [TUTORIAL.md](preliminary/TUTORIAL.md) | 前置知识、IS、PPO surrogate、CISPO 梯度图 | PyTorch autograd |
| [001-ppo](001-ppo/README.md) | [TUTORIAL.md](001-ppo/TUTORIAL.md) | Actor、value model、GAE、KL、PPO epochs | TRL PPOTrainer |
| [002-dpo](002-dpo/README.md) | [TUTORIAL.md](002-dpo/TUTORIAL.md) | chosen/rejected 与 reference 校正 | TRL DPOTrainer |
| [01-grpo](01-grpo/README.md) | [TUTORIAL.md](01-grpo/TUTORIAL.md) | GSM8K、组内 advantage、KL | TRL GRPOTrainer |
| [02-opd](02-opd/README.md) | [TUTORIAL.md](02-opd/TUTORIAL.md) | 医疗 SFT、SAR 分阶段恢复、IDT 交替 Teacher | TRL SFT + verl OPD |
| [02-opd/general-opd](02-opd/general-opd/README.md) | [TUTORIAL.md](02-opd/general-opd/TUTORIAL.md) | 学生采样、教师复评、sampled reverse KL | verl WorkerGroup |
| [03-search-r1](03-search-r1/README.md) | [TUTORIAL.md](03-search-r1/TUTORIAL.md) | 本地真实文档检索、多轮 observation mask | verl WorkerGroup |
| [04-opsd](04-opsd/README.md) | [TUTORIAL.md](04-opsd/TUTORIAL.md) | 固定 step-0 Teacher，特权 solution 条件 | verl WorkerGroup |
| [05-retool](05-retool/README.md) | [TUTORIAL.md](05-retool/TUTORIAL.md) | 实际数值 Python 执行、代码/观察交织 | verl WorkerGroup |
| [06-dapo](06-dapo/README.md) | [TUTORIAL.md](06-dapo/TUTORIAL.md) | Clip-Higher、补采、token reduction、长度处理 | verl WorkerGroup |
| [07-gspo](07-gspo/README.md) | [TUTORIAL.md](07-gspo/TUTORIAL.md) | 序列级几何平均 ratio 与 clipping | TRL GRPOTrainer |
| [08-alfworld](08-alfworld/README.md) | [TUTORIAL.md](08-alfworld/TUTORIAL.md) | 真实 TextWorld 游戏，独立 rollout 组 | verl WorkerGroup |
| [09-AgentOPSD](09-AgentOPSD/README.md) | [TUTORIAL.md](09-AgentOPSD/TUTORIAL.md) | 当前快照 + Skill，turn-level credit | verl WorkerGroup |
| [09-tempo](09-tempo/README.md) | [TUTORIAL.md](09-tempo/TUTORIAL.md) | 状态重放、macro-step、生成式 critic | verl WorkerGroup |
| [09-vision-grpo](09-vision-grpo/README.md) | [TUTORIAL.md](09-vision-grpo/TUTORIAL.md) | 实际图片输入、VLM logprob 与反传 | verl + Transformers VLM |
| [12-harness-rl](12-harness-rl/README.md) | [TUTORIAL.md](12-harness-rl/TUTORIAL.md) | 接口记录、session prefix trees、CAPO | verl WorkerGroup |
| [13-sac](13-sac/README.md) | [TUTORIAL.md](13-sac/TUTORIAL.md) | 连续策略、最大熵、双 Q 与自动温度 | PyTorch embodied |
| [14-td3](14-td3/README.md) | [TUTORIAL.md](14-td3/TUTORIAL.md) | 双 Q、延迟 actor、目标动作平滑 | PyTorch embodied |
| [15-her](15-her/README.md) | [TUTORIAL.md](15-her/TUTORIAL.md) | 同轨迹 future goal 重标记 + TD3 | PyTorch embodied |
| [16-iql](16-iql/README.md) | [TUTORIAL.md](16-iql/TUTORIAL.md) | 固定数据、expectile V、优势加权模仿 | PyTorch embodied |

上游提到 Slime，但固定提交没有独立 Slime 算法目录；这里按实际目录完整映射。框架选择与版本依据见 [调研记录](docs/RESEARCH.md)。

## 训练、对照和评测

每章有中文 README、TUTORIAL、配置、训练入口、数据准备入口；训练章节另有评测入口。实际优化代码集中在 [src/agentic_rl](src/agentic_rl)，模型与日志逻辑共享。

```bash
# 对同一 GRPO 配置选择不同后端
arl train 01-grpo/config.yaml --smoke --backend native
arl train 01-grpo/config.yaml --smoke --backend trl
arl train 01-grpo/verl.yaml --smoke --verl-workers 2

# 用保存的 checkpoint 重新加载并评测；把 RUN_NAME 改为终端输出的目录名
RUN_DIR="runs/RUN_NAME"
arl eval 01-grpo/config.yaml --smoke \
  --checkpoint "$RUN_DIR/checkpoint-final" --limit 32

# 使用已下载的小型预训练模型做后续实验（不会自动改用 GPU）
arl train 01-grpo/config.yaml \
  --model models/Qwen--Qwen2.5-0.5B-Instruct --steps 20

# 可选下载更强 Teacher / 大型 VLM，只下载不推理
python scripts/prepare_models.py --model Qwen/Qwen2.5-1.5B-Instruct
python scripts/prepare_models.py --model Qwen/Qwen2.5-VL-3B-Instruct
```

每次训练生成唯一 `runs/<algorithm>-<时间>/`。指定 `--output` 时若目录已存在会拒绝覆盖。Native 运行保存 `config.json`、`metrics.jsonl`、`trajectories.jsonl`、TensorBoard event、模型与训练状态；TRL 运行保存真实 Trainer 指标、Trainer state 和模型。verl 额外保存每个 worker 的优化器/RNG/策略版本、冻结模型和控制器/TEMPO 状态，通过 `--resume` 恢复到新的输出目录；原生后端和 TRL 的现有 CLI 未提供该恢复接口。

## 实验数据已落地

具身课程另提供本地真实 MuJoCo 采集的 **40 个 episode、2000 条 transition** 及种子、版本、SHA256 元数据，来源是带噪声程序控制器与随机策略。详见 [具身实验说明](docs/EMBODIED.md)。它不属于下面的公开 LLM 数据清单。

完整清单、revision、hash、字段与规模见 [数据说明](docs/DATA.md) 和 [实际数据验证报告](reports/data-verification.json)。包括完整去重后的 GSM8K、完整 GeoQA train/test 与图像、AIME25、ALFWorld 支持的真实游戏、医疗/通用/蒸馏学习子集、MedQA、检索问答与本地文档库。DPO 是明确标注的 GSM8K 派生学习偏好对。

```bash
# 已准备数据会复用本地文件；需要时补齐
make data

# 检查文件 hash、字段、图片、游戏文件和 train/eval 隔离
python scripts/verify_data.py

# 扩大某个学习子集，0 表示读完该来源 split
python scripts/prepare_data.py --dataset opsd --limit 0
```

## 本地日志

- **Streamlit**：<http://127.0.0.1:8501>，选择实验、看曲线、生成样本、token mask、配置和数据检查。
- **TensorBoard**：<http://127.0.0.1:6006>，叠加多个运行的 reward/loss/KL 等曲线。
- `make status` 查看真实进程与 HTTP 状态；`make stop` 只停止本项目服务。

如果代码在远程服务器，通过本机终端建立转发后访问上述地址：

```bash
SSH_USER="your-user"
SERVER="server.example.com"
ssh -L 8501:127.0.0.1:8501 -L 6006:127.0.0.1:6006 "${SSH_USER}@${SERVER}"
```

只记录本地文件，不要求 SwanLab/W&B 登录。端口已有服务时不会替换它；可用 `scripts/services.py start --dashboard-port 8502 --tensorboard-port 6007` 指定其他端口。

## 安装与检查

项目锁定 Linux x86_64、Python 3.12，版本已固定在 `pyproject.toml` 和 `uv.lock`。相同平台的新机器可以：

```bash
bash scripts/setup.sh
bash scripts/setup_verl.sh --cpu
make data
make check
make smoke
make check-verl
make smoke-verl
```

检查包括数学/梯度不变量、真实 Python 工具、真实 ALFWorld reset/step、session 分支隔离、CAPO 梯度路由、视觉输入影响与 checkpoint 重载。批量 smoke 以真实进程退出码和运行状态文件为证据，报告在 [reports](reports)。具体已验证范围见 [验证说明](docs/VALIDATION.md)。

## 复现范围

这是一套可换模型、可查看数据与梯度的学习规模实现。原生后端按单进程设计，TRL 提供标准训练器，verl 提供复杂在线算法的 WorkerGroup、分布式 actor 和异步 rollout 接口。TEMPO 的 prefix importance correction 与状态恢复已在 verl 路径实现。GPU 依赖单独位于 `.venv-verl-gpu/`；单卡 GRPO 的 vLLM 采样、权重同步、actor 更新及保存重载已实际通过，单 rank 的 FSDP1 路径自动使用 NO_SHARD。其他 GPU 算法、多卡/多节点、吞吐和正式 benchmark 尚未运行验证。Harness-RL 保留 central-only MLP 分区，ReTool 保留受限数值 Python。

详见 [TRL 与 verl 的选型、算法映射、恢复和 GPU 命令](docs/VERL.md)，初始 CPU 接入验收见 [verl 验收报告](reports/verl-completion-audit.json)，后续 GPU GRPO 结果见 [GPU 核对报告](reports/verl-grpo-gpu1-verification.json)。

参考代码遵循 Apache-2.0，归属与修改说明见 [NOTICE](NOTICE)、[LICENSE](LICENSE)；公开数据与论文分别遵循其来源条款。

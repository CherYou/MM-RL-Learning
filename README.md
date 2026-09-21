# MM-RL-Learning

这是一个面向 RL 初学者的中文学习与实验仓库。我们从模型概率、常见损失和策略梯度开始，逐步学习语言模型后训练、工具与环境交互、图文条件策略学习和连续控制。每个主题区分原始方法、本地教学实现与已完成验证；数值演示、管线 smoke 和能力实验分别报告。

你可以沿 [学习路线](docs/LEARNING_PATH.md) 顺序阅读，也可以通过术语、公式和代码索引查找问题。

## 适合谁

- 会读基础 Python，希望从概念、公式走到可运行实验
- 关心 LLM 后训练（PPO / GRPO / 工具 Agent）或具身控制基础
- 需要中文教程 + 可核对数值例子 + 本地训练入口

**不自动等于：** 论文完整复现、临床/生产可靠性结论，或“安装后所有 benchmark 立刻可比”。

## 从这里开始

**唯一第一步（不需要下载大模型，也不需要 verl）：**

```bash
git clone https://github.com/CherYou/MM-RL-Learning.git
cd MM-RL-Learning
python examples/math/loss_walkthrough.py
```

需要已安装 PyTorch。终端 JSON 应出现 `"status": "passed"`。这一步只做 L0 数学与梯度核对。

接着阅读：

1. [开始入口](docs/START_HERE.md) — 范围、验证层级、路线选择
2. [损失函数详解](preliminary/TUTORIAL.md) — 从预测答案到用奖励更新模型
3. [学习路线](docs/LEARNING_PATH.md) — 完整基础 / LLM 实践 / 连续控制

需要完整训练环境时再进入下方安装节。完整安装下载较多，耗时取决于网络与机器，不保证固定分钟数完成。

## 仓库教什么

| 分支 | 内容 | 目录 |
| --- | --- | --- |
| 损失与策略梯度基础 | CE / MSE / BCE、PG、old/current、PPO 裁剪 | `preliminary`、`examples/math` |
| 语言模型策略学习 | PPO、DPO、GRPO、DAPO、GSPO | `001-ppo` … `07-gspo` |
| 蒸馏与领域配方 | General OPD、OPSD、医学 SAR/IDT | `02-opd`、`04-opsd` |
| 工具与环境 Agent | Search-R1、ReTool、ALFWorld、AgentOPSD、TEMPO、Harness-RL | `03`–`12` 系列目录 |
| 图文条件 | Vision-GRPO | `09-vision-grpo` |
| 连续控制与离线 | SAC、TD3、HER、IQL | `13-sac`–`16-iql` |
| 技术报告阅读 | 模型系列报告与中文资料 | [Basic LLM](Basic%20LLM/README.md) |

Vision-GRPO 是图文问答策略学习；ALFWorld 主要使用文字接口；SAC/TD3/HER/IQL 是低维 FetchReach 控制。它们共同提供理解 VLA 的基础，但不等于已覆盖完整视觉—语言—动作训练管线。路径可移植性见 [PORTABILITY](docs/PORTABILITY.md)。

## 三条阅读路线

| 路线 | 顺序概要 | 详见 |
| --- | --- | --- |
| 完整基础 | 损失 → PPO → GRPO →（DAPO/GSPO） | [LEARNING_PATH](docs/LEARNING_PATH.md#路线一完整基础) |
| 快速 LLM 实践 | 比率与裁剪 → 最小 GRPO → ReTool / Vision-GRPO | [同上](docs/LEARNING_PATH.md#路线二快速-llm-实践) |
| 连续控制 | 基础 → SAC/TD3 → HER / IQL | [同上](docs/LEARNING_PATH.md#路线三连续控制与离线) |

偏好与蒸馏是旁支：General OPD 入口在概念上先于医学配方，不必嵌在应用案例之后才能学习。

## 章节目录与推荐顺序

每章 README 第一条链接指向详细教程。下表的“默认后端”是该章教学命令应对齐的后端；跨后端对照放在进阶说明。

| 目录与运行说明 | 算法教程 | 本地重点 | 默认后端 |
| --- | --- | --- | --- |
| [preliminary](preliminary/README.md) | [TUTORIAL.md](preliminary/TUTORIAL.md) | 常见损失→PG→ratio/PPO；L0 数值实验 | PyTorch math |
| [001-ppo](001-ppo/README.md) | [TUTORIAL.md](001-ppo/TUTORIAL.md) | Actor、value model、GAE、KL、PPO epochs | TRL PPOTrainer |
| [002-dpo](002-dpo/README.md) | [TUTORIAL.md](002-dpo/TUTORIAL.md) | chosen/rejected 与 reference 校正 | TRL DPOTrainer |
| [01-grpo](01-grpo/README.md) | [TUTORIAL.md](01-grpo/TUTORIAL.md) | GSM8K、组内 advantage、KL | TRL GRPOTrainer |
| [02-opd/general-opd](02-opd/general-opd/README.md) | [TUTORIAL.md](02-opd/general-opd/TUTORIAL.md) | 学生采样、教师复评、sampled reverse KL | verl WorkerGroup |
| [02-opd](02-opd/README.md) | [TUTORIAL.md](02-opd/TUTORIAL.md) | 医疗 SFT、SAR 分阶段恢复、IDT 交替 Teacher（领域案例） | TRL SFT + verl OPD |
| [03-search-r1](03-search-r1/README.md) | [TUTORIAL.md](03-search-r1/TUTORIAL.md) | 本地真实文档检索、多轮 observation mask | verl WorkerGroup |
| [04-opsd](04-opsd/README.md) | [TUTORIAL.md](04-opsd/TUTORIAL.md) | 固定 step-0 Teacher，特权 solution 条件 | verl WorkerGroup |
| [05-retool](05-retool/README.md) | [TUTORIAL.md](05-retool/TUTORIAL.md) | 实际数值 Python 执行、代码/观察交织 | verl WorkerGroup |
| [06-dapo](06-dapo/README.md) | [TUTORIAL.md](06-dapo/TUTORIAL.md) | Clip-Higher、补采、token reduction、长度处理 | verl WorkerGroup |
| [07-gspo](07-gspo/README.md) | [TUTORIAL.md](07-gspo/TUTORIAL.md) | 序列级几何平均 ratio 与 clipping | TRL GRPOTrainer |
| [08-alfworld](08-alfworld/README.md) | [TUTORIAL.md](08-alfworld/TUTORIAL.md) | 真实 TextWorld 游戏，独立 rollout 组 | verl WorkerGroup |
| [09-AgentOPSD](09-AgentOPSD/README.md) | [TUTORIAL.md](09-AgentOPSD/TUTORIAL.md) | 当前快照 + Skill，turn-level credit | verl WorkerGroup |
| [09-tempo](09-tempo/README.md) | [TUTORIAL.md](09-tempo/TUTORIAL.md) | 状态重放、macro-step、生成式 critic（进阶研读） | verl WorkerGroup |
| [09-vision-grpo](09-vision-grpo/README.md) | [TUTORIAL.md](09-vision-grpo/TUTORIAL.md) | 实际图片输入、VLM logprob 与反传 | verl + Transformers VLM |
| [12-harness-rl](12-harness-rl/README.md) | [TUTORIAL.md](12-harness-rl/TUTORIAL.md) | 接口记录、session prefix trees、CAPO | verl WorkerGroup |
| [13-sac](13-sac/README.md) | [TUTORIAL.md](13-sac/TUTORIAL.md) | 连续策略、最大熵、双 Q 与自动温度 | PyTorch embodied |
| [14-td3](14-td3/README.md) | [TUTORIAL.md](14-td3/TUTORIAL.md) | 双 Q、延迟 actor、目标动作平滑 | PyTorch embodied |
| [15-her](15-her/README.md) | [TUTORIAL.md](15-her/TUTORIAL.md) | 同轨迹 future goal 重标记 + TD3 | PyTorch embodied |
| [16-iql](16-iql/README.md) | [TUTORIAL.md](16-iql/TUTORIAL.md) | 固定数据、expectile V、优势加权模仿 | PyTorch embodied |

逻辑前置、路线次序与稳定 ID 见 [configs/chapters.json](configs/chapters.json) 与 [学习路线](docs/LEARNING_PATH.md)。

## 验证层级

| 层级 | 含义 | 典型入口 |
| --- | --- | --- |
| L0 | 数学与梯度演示 | `examples/math/loss_walkthrough.py`、`arl loss-demo` |
| L1 | 机制 / plumbing smoke | `python 01-grpo/train.py --smoke` |
| L2 | 可控小任务学习 | 具身短程实验；需独立评估 |
| L3 | 论文/基准可比复现 | 需明确模型、数据、预算与设备 |

`--smoke` 使用本地随机微型模型、fixture 和显式 debug reward，强制 CPU。它验证采样、反传、环境接口、保存与日志，**不代表任务成功率**。具身 smoke 使用真实 FetchReach 物理与奖励，与 LLM fixture 不是同一种实验。

## 完整环境安装

项目锁定 Linux / WSL、Python 3.12，版本固定在 `pyproject.toml` 和 `uv.lock`。verl CPU / GPU 环境分别为 `.venv-verl/`、`.venv-verl-gpu/`。

```bash
bash scripts/setup.sh
source .venv/bin/activate
arl doctor
arl loss-demo
python 01-grpo/train.py --smoke
```

更多命令（数据、检查、多后端）见下方进阶节或 [docs/VERL.md](docs/VERL.md)。

## 训练、对照和评测

每章有中文 README、TUTORIAL、配置、训练入口；训练章节另有评测入口。优化代码在 [src/agentic_rl](src/agentic_rl)。

```bash
# 同一 GRPO 配置的后端对照（进阶，非首次运行必做）
arl train 01-grpo/config.yaml --smoke --backend native
arl train 01-grpo/config.yaml --smoke --backend trl
arl train 01-grpo/verl.yaml --smoke --verl-workers 2

RUN_DIR="runs/RUN_NAME"
arl eval 01-grpo/config.yaml --smoke \
  --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

每次训练生成唯一 `runs/<algorithm>-<时间>/`。指定已存在的 `--output` 会拒绝覆盖。Native / TRL / verl 的保存与恢复范围不同，详见章节 README 与 [VERL.md](docs/VERL.md)。

## 实验数据

数据来源、revision、hash、字段与规模见 [数据说明](docs/DATA.md)。具身课程另提供本地 MuJoCo 采集数据，见 [EMBODIED](docs/EMBODIED.md)。

```bash
make data
python scripts/verify_data.py
```

学习子集、合成偏好对、程序示范与官方划分的用途不同，不能互相冒充 benchmark 证据。

## 本地日志

- Streamlit：`http://127.0.0.1:8501`
- TensorBoard：`http://127.0.0.1:6006`
- `make status` / `make stop`

远程开发时用 SSH 端口转发访问上述地址。只记录本地文件，不要求 SwanLab/W&B 登录。

## 安装与检查

```bash
bash scripts/setup.sh
bash scripts/setup_verl.sh --cpu
make data
make docs          # Markdown 链接/数学 + 教程结构审计
make check-math    # L0 数值实验与单测（需 PyTorch）
make check         # 完整检查（.venv、数据、pytest、ruff）
make smoke
```

检查报告写入 `reports/`，训练输出写入 `runs/`；两者默认不纳入版本控制。

分层 CI 见 [.github/workflows](.github/workflows)：文档与 L0 数学适合普通 PR；机制 / 环境 / GPU 需自备环境，不作为文档改动的默认阻塞。贡献前请读 [CONTRIBUTING](CONTRIBUTING.md) 与 [证据层级](docs/EVIDENCE.md)。

## 复现范围

这是一套学习规模的实现：原生 PyTorch 用于公式对照，TRL 提供标准训练器，verl 提供复杂在线算法的分布式接口。Harness-RL 使用 central-only MLP 分区，ReTool 使用受限数值 Python 工具。模型效果需针对具体数据、预算和设备独立评估。

**L0–L3 验证层级见 [docs/EVIDENCE.md](docs/EVIDENCE.md)。** 后端安装与恢复见 [docs/VERL.md](docs/VERL.md)。

## 许可证与来源

参考代码遵循 Apache-2.0。归属与修改见 [NOTICE](NOTICE)、[LICENSE](LICENSE)、[来源说明](docs/CODE_PROVENANCE.md)。公开数据与论文分别遵循其来源条款。贡献规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 参考与感谢

感谢 [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 提供章节组织、算法资料与工具实现的参考。感谢算法论文作者，以及 [TRL](https://github.com/huggingface/trl)、[verl](https://github.com/verl-project/verl)、[Search-R1](https://github.com/PeterGriffinJin/Search-R1)、[ReTool](https://github.com/ReTool-RL/ReTool)、[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)、[ALFWorld](https://github.com/alfworld/alfworld)、[AgentOPSD](https://github.com/ZethWang/AgentOPSD)、[Harness-RL](https://github.com/jiangxinke/Harness-RL)、[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) 等项目。论文及官方接口依据在各教程中就近列出。

### 大模型技术报告

[Basic LLM](Basic%20LLM/README.md) 按“模型系列 → 模型版本”组织技术报告、中文翻译与解读。它是扩展阅读，不是开始策略梯度的前置。

| 模型 | 内容 | 阅读与下载 |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/README.md) | 架构、训练与推理基础设施、后训练和评估 | [原始技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) · [中文合订稿](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/report.zh.md) · [PDF](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/documents/translation.zh.pdf) · [Word](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/documents/translation.zh.docx) |

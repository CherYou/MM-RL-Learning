# MM-RL-Learning

这是面向强化学习初学者的中文课程。从一次模型更新出发，你会学到语言模型如何利用奖励、怎样与工具和环境交互，以及怎样在连续控制任务中训练策略。每章都有算法教程和可运行的例子。

你可以沿 [学习路线](docs/LEARNING_PATH.md) 顺序阅读，也可以通过术语、公式和代码索引查找问题。

## 从这里开始

**先用 PyTorch 看一次梯度更新。** 这个例子只在 CPU 上运算。

```bash
git clone https://github.com/CherYou/MM-RL-Learning.git
cd MM-RL-Learning
python examples/math/loss_walkthrough.py
```

终端输出包含损失值、梯度方向和更新前后的概率。结尾的 `"status": "passed"` 表示这些数值与脚本中的手算结果一致。

接着阅读：

1. **主线 01** [损失函数与策略梯度基础](01-foundations/TUTORIAL.md)
2. [全书目录](docs/CHAPTERS.md)按六篇组织 21 章，每章都可以直接打开教程。
3. 专题捷径：[LEARNING_PATH](docs/LEARNING_PATH.md) · 末章后：[NEXT_STEPS](docs/NEXT_STEPS.md)

## 全书六篇

| 篇 | 章号 | 内容 |
| --- | ---: | --- |
| [共同基础](docs/families/00-foundations.md) | 01 | 损失与策略梯度 |
| [策略与偏好](docs/families/01-policy-preference.md) | 02–06 | PPO、GRPO、DAPO、GSPO、DPO |
| [工具与多模态](docs/families/02-tools-multimodal.md) | 07–12 | ARPO、AEPO、Search-R1、ReTool、ALFWorld、Vision-GRPO |
| [蒸馏与应用](docs/families/03-distillation.md) | 13–15 | General OPD、OPSD、AgentOPSD |
| [长程 Agent](docs/families/04-long-horizon.md) | 16–17 | TEMPO、Harness-RL |
| [控制与离线](docs/families/05-control-offline.md) | 18–21 | TD3、SAC、HER、IQL |

## 仓库内容

算法教程配有手算例子、源码入口和运行命令。工具篇里，Search-R1 读取检索结果，ALFWorld 接收文字观察，Vision-GRPO 根据图像回答问题。连续控制篇使用低维状态和目标坐标驱动 FetchReach 机械臂。[路径与环境配置](docs/PORTABILITY.md)

模型技术报告单独收在 [Basic LLM](Basic%20LLM/README.md)，可以对照算法章节阅读 DeepSeek 和 MiMo 的后训练方法。

## 三条阅读路线

| 路线 | 顺序概要 | 详见 |
| --- | --- | --- |
| 完整基础 | 损失 → PPO → GRPO →（DAPO/GSPO） | [LEARNING_PATH](docs/LEARNING_PATH.md#专题捷径) |
| 快速 LLM 实践 | 比率与裁剪 → 最小 GRPO → ReTool / Vision-GRPO | [学习路线](docs/LEARNING_PATH.md#专题捷径) |
| 连续控制 | 基础 → SAC/TD3 → HER / IQL | [学习路线](docs/LEARNING_PATH.md#专题捷径) |

偏好与蒸馏是旁支：从 [General OPD](13-general-opd/TUTORIAL.md) 进入即可。

## 章节目录与推荐顺序

每章 README 汇集运行命令，开头的链接直达算法教程。下表列出各章示例默认使用的训练后端。

| 目录与运行说明 | 算法教程 | 本地重点 | 默认后端 |
| --- | --- | --- | --- |
| [01-foundations](01-foundations/README.md) | [TUTORIAL.md](01-foundations/TUTORIAL.md) | 常见损失→PG→ratio/PPO；L0 数值实验 | PyTorch math |
| [02-ppo](02-ppo/README.md) | [TUTORIAL.md](02-ppo/TUTORIAL.md) | Actor、value model、GAE、KL、PPO epochs | TRL PPOTrainer |
| [03-grpo](03-grpo/README.md) | [TUTORIAL.md](03-grpo/TUTORIAL.md) | GSM8K、组内 advantage、KL | TRL GRPOTrainer |
| [04-dapo](04-dapo/README.md) | [TUTORIAL.md](04-dapo/TUTORIAL.md) | Clip-Higher、补采、token reduction、长度处理 | verl WorkerGroup |
| [05-gspo](05-gspo/README.md) | [TUTORIAL.md](05-gspo/TUTORIAL.md) | 序列级几何平均 ratio 与 clipping | TRL GRPOTrainer |
| [06-dpo](06-dpo/README.md) | [TUTORIAL.md](06-dpo/TUTORIAL.md) | chosen/rejected 与 reference 校正 | TRL DPOTrainer |
| [07-arpo](07-arpo/README.md) | [TUTORIAL.md](07-arpo/TUTORIAL.md) | 工具后高熵分支、步级优势归因 | verl WorkerGroup |
| [08-aepo](08-aepo/README.md) | [TUTORIAL.md](08-aepo/TUTORIAL.md) | 熵平衡 rollout/裁剪/优势 | verl WorkerGroup |
| [09-search-r1](09-search-r1/README.md) | [TUTORIAL.md](09-search-r1/TUTORIAL.md) | 本地真实文档检索、多轮 observation mask | verl WorkerGroup |
| [10-retool](10-retool/README.md) | [TUTORIAL.md](10-retool/TUTORIAL.md) | 实际数值 Python 执行、代码/观察交织 | verl WorkerGroup |
| [11-alfworld](11-alfworld/README.md) | [TUTORIAL.md](11-alfworld/TUTORIAL.md) | 真实 TextWorld 游戏，独立 rollout 组 | verl WorkerGroup |
| [12-vision-grpo](12-vision-grpo/README.md) | [TUTORIAL.md](12-vision-grpo/TUTORIAL.md) | 实际图片输入、VLM logprob 与反传 | verl + Transformers VLM |
| [13-general-opd](13-general-opd/README.md) | [TUTORIAL.md](13-general-opd/TUTORIAL.md) | 学生采样、教师复评、sampled reverse KL | verl WorkerGroup |
| [14-opsd](14-opsd/README.md) | [TUTORIAL.md](14-opsd/TUTORIAL.md) | 固定 step-0 Teacher，特权 solution 条件 | verl WorkerGroup |
| [15-agentopsd](15-agentopsd/README.md) | [TUTORIAL.md](15-agentopsd/TUTORIAL.md) | 当前快照 + Skill，turn-level credit | verl WorkerGroup |
| [16-tempo](16-tempo/README.md) | [TUTORIAL.md](16-tempo/TUTORIAL.md) | 状态重放、macro-step、生成式 critic（进阶研读） | verl WorkerGroup |
| [17-harness-rl](17-harness-rl/README.md) | [TUTORIAL.md](17-harness-rl/TUTORIAL.md) | 接口记录、session prefix trees、CAPO | verl WorkerGroup |
| [18-td3](18-td3/README.md) | [TUTORIAL.md](18-td3/TUTORIAL.md) | 双 Q、延迟 actor、目标动作平滑 | PyTorch embodied |
| [19-sac](19-sac/README.md) | [TUTORIAL.md](19-sac/TUTORIAL.md) | 连续策略、最大熵、双 Q 与自动温度 | PyTorch embodied |
| [20-her](20-her/README.md) | [TUTORIAL.md](20-her/TUTORIAL.md) | 同轨迹 future goal 重标记 + TD3 | PyTorch embodied |
| [21-iql](21-iql/README.md) | [TUTORIAL.md](21-iql/TUTORIAL.md) | 固定数据、expectile V、优势加权模仿 | PyTorch embodied |

## 怎样读实验结果

想弄清公式，先比较数值例子中的梯度和更新前后的概率。LLM 章节的 `--smoke` 用微型模型和样例数据，适合跟踪一批回答怎样进入 loss。评估答题能力时，另用独立题目，让模型自由生成，再按答案计分。

具身章节的 `--smoke` 运行 FetchReach 短实验。比较控制策略时，固定评估种子，记录机械臂到达原任务目标的比例。

## 完整环境安装

项目锁定 Linux / WSL、Python 3.12，版本固定在 `pyproject.toml` 和 `uv.lock`。verl CPU / GPU 环境分别为 `.venv-verl/`、`.venv-verl-gpu/`。

```bash
bash scripts/setup.sh
source .venv/bin/activate
arl doctor
arl loss-demo
python 03-grpo/train.py --smoke
```

更多命令（数据、检查、多后端）见下方进阶节或 [docs/VERL.md](docs/VERL.md)。

## 训练、对照和评测

每章有中文 README、TUTORIAL、配置、训练入口；训练章节另有评测入口。优化代码在 [src/agentic_rl](src/agentic_rl)。

```bash
# 同一 GRPO 配置的后端对照（进阶，非首次运行必做）
arl train 03-grpo/config.yaml --smoke --backend native
arl train 03-grpo/config.yaml --smoke --backend trl
arl train 03-grpo/verl.yaml --smoke --verl-workers 2

RUN_DIR="runs/RUN_NAME"
arl eval 03-grpo/config.yaml --smoke \
  --checkpoint "$RUN_DIR/checkpoint-final" --limit 32
```

每次训练生成唯一 `runs/<algorithm>-<时间>/`。指定已存在的 `--output` 会拒绝覆盖。Native / TRL / verl 的保存与恢复范围不同，详见章节 README 与 [VERL.md](docs/VERL.md)。

## 实验数据

数据来源、revision、hash、字段与规模见 [数据说明](docs/DATA.md)。具身课程另提供本地 MuJoCo 采集数据，见 [EMBODIED](docs/EMBODIED.md)。

```bash
make data
python scripts/verify_data.py
```

对照实验要使用相同的数据划分。DPO 章的负例由规则生成，IQL 章的示范来自仿真控制器。评估时使用各章单独的测试数据或环境种子。

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

## 训练后端

想逐行观察损失，可以从原生 PyTorch 实现入手。TRL 提供 PPO、GRPO 和 DPO 等训练器，verl 用于多轮工具交互和分布式训练。不同后端的检查点和恢复条件见 [verl 指南](docs/VERL.md)。

## 许可证与来源

参考代码遵循 Apache-2.0。归属与修改见 [NOTICE](NOTICE)、[LICENSE](LICENSE)、[来源说明](docs/CODE_PROVENANCE.md)。公开数据与论文分别遵循其来源条款。贡献规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 参考与感谢

感谢 [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 提供章节组织、算法资料与工具实现的参考。感谢算法论文作者，以及 [TRL](https://github.com/huggingface/trl)、[verl](https://github.com/verl-project/verl)、[Search-R1](https://github.com/PeterGriffinJin/Search-R1)、[ReTool](https://github.com/ReTool-RL/ReTool)、[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)、[ALFWorld](https://github.com/alfworld/alfworld)、[AgentOPSD](https://github.com/ZethWang/AgentOPSD)、[Harness-RL](https://github.com/jiangxinke/Harness-RL)、[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) 等项目。论文及官方接口依据在各教程中就近列出。

### 大模型技术报告

技术报告目录按模型系列和版本整理，提供原文与中文阅读材料。[Basic LLM](Basic%20LLM/README.md)

| 模型 | 内容 | 阅读与下载 |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/README.md) | 架构、训练与推理基础设施、后训练和评估 | [原始技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) · [中文合订稿](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/report.zh.md) · [PDF](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/documents/translation.zh.pdf) · [Word](Basic%20LLM/DeepSeek/DeepSeek-V4.1-Flash/documents/translation.zh.docx) |
| [MiMo-V2.6](Basic%20LLM/MiMo/MiMo%20V2.6/README.md) | 全模态 MoE、智能体训练与评估 | [原始技术报告](Basic%20LLM/MiMo/MiMo%20V2.6/MiMo_V2_6_technical_report.pdf) · [中文译稿](Basic%20LLM/MiMo/MiMo%20V2.6/report.zh.md) |

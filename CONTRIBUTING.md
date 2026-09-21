# 贡献指南

感谢你改进 **MM-RL-Learning**。这是一份面向初学者的中文学习与实验仓库：贡献应优先保证**读者能理解、能定位、能运行**，而不是先追求目录编号整齐或篇幅统一。

## 先读什么

1. [开始入口](docs/START_HERE.md) · [学习路线](docs/LEARNING_PATH.md)
2. [证据与验证层级](docs/EVIDENCE.md)（L0–L3 不可混称）
3. 相关章节的 `TUTORIAL.md` 与 `README.md`（默认后端以章节为准）

## 改动类型（请在 PR 中勾选）

| 类型 | 要求 |
| --- | --- |
| 教学表达 | 保持公式语义与代码一致；练习先给预测空间，避免只改文风 |
| 实现逻辑 | 说明改的是 objective / reduction / 数据流哪一层；附数值或机制测试 |
| 数据 | 更新 [docs/DATA.md](docs/DATA.md) 与 hash/来源；区分学习子集与 benchmark |
| 依赖 | 不把完整训练栈变成阅读 L0 数学的必要条件 |
| 结果 / 证据 | 按 [EVIDENCE](docs/EVIDENCE.md) 标注层级；未跑通不得写成“已复现” |
| 配图 | 概念图注明来源；数值图附生成脚本或参数；caption 写清“看什么” |
| 导航 | 同步 `configs/chapters.json`、`docs/LEARNING_PATH.md`、前后章链接 |

**不要：** 为了通过检查而把感谢挪回首页、每章强行一张图、或在教程旁禁止链接上游论文/源码。审计只校验有效性与一致性。

## 目录与导航（唯一数据源）

| 文件 | 作用 |
| --- | --- |
| [configs/chapters.json](configs/chapters.json) | 章节身份、家族、教程路径、前置、入口、验证层级 |
| [configs/learning_paths.json](configs/learning_paths.json) | 全书 `book` 顺序 + 专题捷径完整 ID 列表 |
| [configs/concepts.json](configs/concepts.json) | 概念锚点（path + anchor） |
| [docs/CHAPTERS.md](docs/CHAPTERS.md) | 读者可见的 20 章总目录 |
| `scripts/build_chapter_registry.py` | 由权威顺序生成 chapters.json |
| `scripts/sync_navigation.py` | 读写各 TUTORIAL 的 NAV:TOP/BOTTOM 块 |
| `scripts/check_navigation.py` | 校验 ID、路线闭合、NAV 双向一致 |

```bash
make nav          # 重建注册表 + 写入页首页尾 + 检查
python scripts/sync_navigation.py --check
```

**不要**在每章手工维护互相冲突的“下一章”列表；显示编号由 `learning_paths.book` 派生。物理运行目录本轮**不搬迁**。

## 默认教学后端

| 章节 | 默认入口 |
| --- | --- |
| preliminary / examples/math | PyTorch 轻量脚本（无需 verl） |
| PPO / DPO / GRPO / GSPO | TRL + 各章 `config.yaml` |
| Agent / 多模态 / 蒸馏在线机制 | verl + 各章 `verl.yaml` |
| SAC / TD3 / HER / IQL | PyTorch embodied + FetchReach |

跨后端命令放在 README「进阶」区；教程正文的**第一条运行命令**应与默认后端一致。

## 本地检查

```bash
# 文档：Markdown 链接/数学 + 教程结构审计
make docs

# 轻量数学（需 PyTorch；不下载模型）
python examples/math/loss_walkthrough.py
python -m pytest -q tests/test_loss_math_walkthrough.py

# 完整仓库检查（需已 setup 的 .venv；含数据校验）
make check
```

CI 分层见 [.github/workflows](.github/workflows)：文档 job 不依赖 GPU；数学 job 只跑 L0；机制/环境/GPU 不作为普通文档 PR 的默认阻塞。

## 教程写作约定

- 每章有一个可观察的学习任务；章首可用「相对 GRPO / PPO 改变了什么」表，避免重复展开已学公式。
- 重要公式交代：预测对象、数据来源、可训练量、统计单位、至少一个数值例子、边界。
- 行内数学用 `$`...`$`，块级用 ` ```math `；初学材料避免 GitHub 不安全宏（如裸 `\operatorname`、`\begin{cases}`）与原始 `<` `>`。
- 旧路径若迁移，旧文件只保留简短“已移动”链接，不复制整篇正文。

## 来源与许可

代码 Apache-2.0，归属见 [NOTICE](NOTICE)、[CODE_PROVENANCE](docs/CODE_PROVENANCE.md)。第三方数据/论文遵循各自条款；不要删除既有来源声明来“让文字更顺”。

## 提交信息

建议说明**为什么改**，并区分教学、实现、数据、文档或检查。例如：

- `Clarify PPO batch walkthrough and link clipping to preliminary`
- `Relax tutorial audit style gates; keep link and asset checks`

不强制特定英文模板；清晰即可。

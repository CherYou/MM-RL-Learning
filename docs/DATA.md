# 本地实验数据

本地数据实际通过 `scripts/verify_data.py` 检查。以下规模来自落盘文件，不是远端仓库宣称的规模。

| 数据目录 | 训练 | 评测 | 说明 |
| --- | ---: | ---: | --- |
| `gsm8k` | 7,473 | 1,319 | 官方 main，去重后；原始 train 7473，不把同题重复计入训练 |
| `dpo` | 7,473 | 1,319 | GSM8K 正确推理 vs 数值 +1 错误答案，synthetic negatives |
| `deepmath` | 960 | 64 | DeepMath-103K 前 1024 行学习子集，固定 seed 切分 |
| `opsd` | 960 | 64 | Openthoughts_math_30k_opsd 前 1024 行学习子集 |
| `dapo` | 953 | 64 | 前 1024 原始行按题面去重/冲突过滤后，再分学习集 |
| `aime25` | 0 | 30 | 全部 30 题，只评测 |
| `medical` | 952 | 64 | medical-o1 reasoning SFT zh，前 1024 行去重后 |
| `medqa` | 0 | 3,420 | 中文 4-option test，原始 3426 行去重后 3420 |
| `ceval` | 204 | 45 | 8 个通用科目，带标签 dev+val，分科目 80/20 |
| `search` | 1,024 | 1,024 | Search-R1 混合问答，各 source split 前 1024 行 |
| `hotpot` | 256 | 256 | HotpotQA distractor，真实检索学习查询 |
| `geoqa` | 3,503 | 759 | 保留 official original_split；所有图片文件在本地 |
| `alfworld-manifests` | 3,553 | 274 | 真实可解游戏；seen=140，unseen=134 |

## 检索、图像、技能与 fixture

`data/search/corpus.jsonl` 有 5,038 篇真实公开文档片段，来自 HotpotQA 查询的 distractor context。它包含 title/text，不含 query、answer 或 gold supporting facts。默认 Search-R1/Harness-RL 配置使用与本地 corpus 配套的 `data/hotpot/` 查询；`data/search/train.jsonl` 保留原始 NQ/Hotpot 混合来源以便后续扩展检索库。

`data/geoqa/images/` 保存真实 GeoQA 图片。官方 dev 图像也保留，但训练/评测 JSONL 只包含 train/test。`data/skills/` 保存 AgentOPSD skill 资产和来源说明。`data/harness/probes.jsonl` 是两个明确标记的种子 probing 调用，不是宣称由随机策略成功生成的训练轨迹。

`data/fixtures/` 仅用于离线 CPU plumbing 检查。其数学、偏好、视觉标签和调试奖励不参与正式实验结论；评测与训练 fixture 问题分离。

## 契约

- 通用行：`id, prompt, answer`，可选 `answers, solution, source`。
- DPO：`id, prompt, chosen, rejected, source`。
- 视觉：在通用行上增加项目相对路径 `image`；图片由 processor 读入。
- 游戏：`id, split, game_file, task_type`；`prompt` 保存唯一游戏标识，实际观察由 reset 产生。
- 每个数据目录的 `manifest.json` 记录 source revision、license/source URL、行数、字节数和 SHA-256。

## 可重复准备

```bash
make data
python scripts/verify_data.py
# 单独扩大某种数据；0 表示完整 source split
python scripts/prepare_data.py --dataset deepmath --limit 0
# 专门重建八科 C-Eval
python scripts/prepare_ceval.py
# 重建真实 ALFWorld 文本资产
python scripts/prepare_environments.py --only alfworld --download-alfworld
```

原始大数据可留在 Hub cache；直接训练只需要清洗后的 JSONL、图像或游戏文件。学习子集不等于全量论文 benchmark。不要拿第一个 1024 行的结果估计完整数据分布，也不要把调试 reward 当成任务正确率。

## 数据范围

为支持快速本地学习，DeepMath、OPSD、DAPO、医疗和 Search-R1 使用学习子集；GSM8K、GeoQA train/test、AIME25 和 ALFWorld 文本游戏采用表中所列范围。C-Eval 使用八科有标签 dev/val 重新划分训练与评测，MedQA 评测集按题目去重。

## 数据归属

数据许可证独立于代码的 Apache-2.0。各 manifest 保留来源和 license 元数据；不把上游数据重新宣称为自己生成。C-Eval 使用 CC-BY-NC-SA-4.0，HotpotQA 使用 CC-BY-SA-4.0，其他集合以相应固定 revision 的 dataset card 为准。

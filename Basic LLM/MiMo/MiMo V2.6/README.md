# MiMo-V2.6 技术报告

**原文：** MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement  
**团队：** LLM-Core Xiaomi  
**关联模型：** [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) · [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)

## 阅读与下载

| 材料 | 本地路径 | 说明 |
| --- | --- | --- |
| **英文技术报告 PDF** | [MiMo_V2_6_technical_report.pdf](MiMo_V2_6_technical_report.pdf) | 官方原文（自 HF 模型仓库下载） |
| **中文翻译** | [report.zh.md](report.zh.md) | 学习用途整理译稿 |
| 英文抽取文本 | [source/report.en.txt](source/report.en.txt) | 便于全文检索 |

## 内容摘要

- **模型：** MiMo-V2.6-Pro（1.02T 总参 / 42B 激活）、MiMo-V2.6-Flash（310B / 15B），全模态 MoE
- **架构：** Hybrid SWA + Global Attention 稀疏 MoE；MiMo-ViT；Audio Tokenizer + patch 编码器；MTP/DFlash 投机解码
- **训练：** 两阶段预训练 + 智能体中心中期训练（含 Muown、MXFP4 QAT）
- **RL 扩展：** 大 batch 异步训练（每步 1,568 prompts × G=16）、多域环境与 harness、组级智能体评分（GRS/GAR）
- **稳定性：** 冻结 MoE 路由器；多层反奖励作弊
- **开源：** MiMo-V2.6-Distill-Qwen-9B、RL 环境、训练框架、mini-harness

## 来源说明

- 原文 PDF 与模型版权：Xiaomi LLM-Core / XiaomiMiMo，许可证以模型卡为准（MIT License on HF）。
- 中文译稿由本仓库整理，仅供学习对照，**不替代官方报告**；数字若有出入以 PDF 为准。
- 下载地址（报告文件在模型仓库内）：  
  `https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf`

[← MiMo 系列](../README.md) · [← Basic LLM](../../README.md)

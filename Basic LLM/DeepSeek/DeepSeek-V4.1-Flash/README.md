# DeepSeek-V4.1-Flash · 技术报告与中文解读

[DeepSeek 目录](../README.md) · [Basic LLM](../../README.md) · [仓库首页](../../../README.md)

原报告：**[DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)**，作者 **DeepSeek-AI**。本目录整理提供的中文翻译、深度解读与配套文件，主题包括 CED 因果编码器–解码器、CSA2 压缩稀疏注意力、FP4 KV 缓存、有界重放、DSpark，以及预训练、后训练和评估。

## 从这里开始读

- [中文合订稿](report.zh.md)包含导读、正文翻译与配图。
- 离线阅读可用 [58 页 PDF](documents/translation.zh.pdf)，编辑或批注可用 [Word 文档](documents/translation.zh.docx)。
- 按主题查阅时，可以从下面四份分章稿中选择。

| 顺序 | 阅读稿 | 覆盖范围 |
| --- | --- | --- |
| 01 | [概览与深度解读](chapters/01-overview.zh.md) | 核心贡献、技术动机、结果和局限 |
| 02 | [摘要、引言与模型架构](chapters/02-architecture.zh.md) | CED、CSA2、mHC、Engram、DSpark、FP4 与优化 |
| 03 | [基础设施与预训练](chapters/03-infrastructure-pretraining.zh.md) | 训练、推理、缓存管理、数据与基座评估 |
| 04 | [后训练、评估与附录](chapters/04-post-training-appendices.zh.md) | 智能体 RL、异步训练、OPD、推理努力与术语表 |

## 文件怎么组织

```text
DeepSeek-V4.1-Flash/
├── README.md                 # 阅读入口与来源
├── report.zh.md              # 中文合订稿
├── chapters/                 # 01–04 分章阅读稿
├── documents/                # Word、PDF 成品
├── source/                   # 英文提取文本、原整理大纲
├── assets/
│   ├── figures/              # 图 01–12，按报告图号排序
│   └── pages/
│       ├── en/               # 英文原报告页面截图
│       └── zh/               # 中文文档制作阶段的抽页预览
├── visualizations/           # KV 缓存机制 HTML 示意
├── tools/                    # Markdown→Word、Word→PDF 工具
└── manifest.json             # 原文件名、目标路径、SHA256 与整理说明
```

查术语时可搜索[英文提取文本](source/report.en.txt)。查图时可按论文图号和页码打开[配图与页面索引](assets/README.md)。

想看压缩前后缓存结构的示意，可下载 [KV 缓存图解](visualizations/kv-cache.html)，用浏览器打开。

## 作者与来源

- [DeepSeek 官方发布说明](https://api-docs.deepseek.com/news/news260910/)
- [作者技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
- [作者模型与代码发布页](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

报告和配图来自 DeepSeek-AI。中文材料为非官方学习译稿，引用方法或实验数字时请对照上方作者报告。原文件与目录的对应关系见[文件清单](manifest.json)。

## 文档导出工具

重新导出 Word 或 PDF 时，工具默认将文件写入 `build/`。[文档导出工具](tools/README.md)

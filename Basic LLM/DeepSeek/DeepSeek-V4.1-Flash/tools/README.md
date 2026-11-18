# 文档导出工具

[返回资料目录](../README.md)

脚本默认从本版本目录读取 `report.zh.md` 和 `assets/figures/`，生成的 Word 和 PDF 写入 `build/`。

## Markdown 转 Word

在仓库根目录执行，路径中的空格需要保留引号：

```bash
python -m pip install -r "Basic LLM/DeepSeek/DeepSeek-V4.1-Flash/tools/requirements.txt"
python "Basic LLM/DeepSeek/DeepSeek-V4.1-Flash/tools/md2docx.py"
```

生成 `Basic LLM/DeepSeek/DeepSeek-V4.1-Flash/build/translation.zh.docx`。可用 `--source`、`--output`、`--figures` 指定输入、输出及配图目录；显式相对参数以调用时的工作目录为准。

`mathml2omml.py` 将公式转换为 Word 的 OMML。导出器识别本报告的标题、表格和图注，按图号插图。系统字体或 Word 版本改变时，重新导出后请检查分页。

## Word 转 PDF

此脚本需要 **Windows 和已安装的 Microsoft Word**。先生成上面的 Word，再从仓库根目录运行：

```powershell
pwsh -File "Basic LLM/DeepSeek/DeepSeek-V4.1-Flash/tools/docx2pdf.ps1"
```

也可在 Windows PowerShell 中直接调用脚本。使用 `-InputPath` 和 `-OutputPath` 可指定其他文档；默认 PDF 与输入 Word 同名、同目录，已有目标文件会拒绝覆盖。脚本以只读方式打开输入，并在结束时关闭 Word。

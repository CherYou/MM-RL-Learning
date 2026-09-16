# 路径与可移植性

仓库可以克隆到任意目录。命令、配置、数据与输出路径以仓库根目录为基准；运行章节命令前先进入仓库根目录。

## 路径约定

- 代码和数据使用 `src/`、`data/`、`models/`、`runs/` 等仓库相对路径。
- Python 代码通过 `Path(__file__).resolve()` 推导仓库位置，通过 `tempfile` 选择系统临时目录，通过 `os.defpath` 获取平台默认命令搜索路径。
- GPU 验证使用 `scripts/run_grpo_gpu_check.py --gpu-index INDEX` 选择设备；配置文件不保存某台机器的 GPU UUID。
- GPU 监测所需的原始 UUID 只保存在已忽略的 `runs/RUN_NAME/` 中，包括 `gpu-monitor.json` 和 `verl-runtime.json`。各验证脚本的原始 stdout 写入 `runs/.logs/`；结构化检查结果只记录相对路径，并自动移除 GPU UUID。
- `reports/` 保存本地运行生成的检查结果，与 `runs/` 一起由 Git 忽略。
- `references/upstream/` 是固定提交的第三方原始快照。为保持来源审计可复核，其中的上游示例和路径不做改写，也不会被本项目代码导入或执行。

## 自动检查

完成环境安装后，从仓库根目录运行：

```bash
make portability
make docs
```

第一条检查遍历 Git 跟踪的文本文件，忽略冻结的第三方快照，并拒绝机器专属路径、主机或 GPU 标识再次进入主项目内容。第二条检查 Markdown 数学公式的分隔符、花括号和容易触发渲染错误的历史下标。新增文档、配置或报告后应同时运行。

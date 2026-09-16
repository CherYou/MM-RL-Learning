# 常见问题

## 运行入口与相对路径

先 `source .venv/bin/activate`。`arl` 使用项目根目录解析配置/数据路径；各章 Python 入口也能从其他工作目录调用。输出目录已存在时拒绝覆盖，换一个 `--output` 或直接使用默认带时间戳目录。

## 模型或 Teacher 找不到

`--smoke` 不需要下载预训练模型。普通配置使用 Hugging Face ID，可先运行 `scripts/prepare_models.py --model ...`，之后把 `--model` / `--teacher-model` 指向本地权重目录。医疗场景推荐 `scripts/medical_pipeline.py`，它会先产生 SFT Teacher，再自动传入 OPD。

Teacher 和 Student 必须使用同一 vocab；token-level 蒸馏不会把不同 tokenizer 的位置硬对齐。较长 teacher solution/skill 可能超出模型上下文；调整 max_new_tokens/最大轮数，或者使用支持更长上下文的模型。

## 没有梯度、奖励全零

小型随机模型全错是正常情况。普通 GRPO 的全同组 advantage 为零，DAPO 会补采并在上限处跳过。Harness 输出没有合法 action/args 时没有结构化 token loss。用 `--smoke` 检查执行链路，再使用预训练模型或 SFT 初始化做实际任务实验。

## 依赖和 CPU

先执行 `bash scripts/setup.sh`，再 `arl doctor`。本环境安装 CPU torch，不能通过修改 device 字符串获得 CUDA 训练能力。verl CPU 环境用 `bash scripts/setup_verl.sh --cpu`，独立 CUDA 环境用 `bash scripts/setup_verl.sh --gpu` 安装。GPU 训练仍需显式选 `verl-gpu.yaml` 并按设备资源设置 workers。

## 看不到日志页面

`make status` 同时检查 PID 与 HTTP。远程服务器需要 SSH 转发。端口冲突时服务管理器会拒绝启动，不会关闭其他人的服务；显式指定备用端口即可。TensorBoard 读取 native、TRL 与 verl 回调写出的 event；Streamlit 直接读取 JSONL。

## 数据校验失败

先看 `scripts/verify_data.py` 提示的具体文件，按对应 prepare 脚本重建。JSONL 必须按 LF 分行，不能使用会把 U+2028 当换行的逻辑；这类字符可能出现在真实题目中。下载原始 schema 有变化时先核对固定 revision 和列名，不把 HTML 错误页或不完整下载当数据文件。

## verl 环境、Ray 与恢复

`backend: verl` 由 CLI 自动切换到 `.venv-verl` 或 `.venv-verl-gpu`，不要把 verl 的 numpy<2 依赖直接混入原环境。版本必须与 references/verl.json 一致；setup 脚本发现 revision 不一致会停止，不会覆盖源码修改。

本机 Ray 启动使用独立临时目录与 localhost，并绕过 HTTP 代理；多节点必须显式 `--ray-address`，各节点需相同环境和可访问的数据/模型路径。`verl_workers` 为总 rank 数；恢复时保持原数值，`--steps` 是大于已完成步数的累计目标。`--output` 必须用新目录。原生与 TRL CLI 暂无同样的 `--resume` 实现。

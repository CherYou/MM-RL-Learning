# 本地工具与环境

## 检索

`LocalSearch` 建立基于 title/text 的词频、逆文档频率与长度归一化 BM25，返回排名靠前的真实本地片段。无网络 API、无账户和远程计费；这个语料只是小型学习 corpus，不是完整 Wikipedia。模型查询与返回文本都进入日志，返回文本 mask=0。

## Python 数值工具

`python_tool()` 支持数值 Python 子集：算术、列表、循环、sum/range、部分 math 函数。它拒绝文件读取、任意 import、私有变量和任意对象属性；子进程在临时目录中运行，设置 CPU 2 秒、wall-clock 3 秒、256 MiB 地址空间和 1 MiB 输出文件上限，最多读取 8 KiB 输出。

这是一个受限的学习工具，不是完整 Python 科学计算环境，也不宣称拥有容器/内核级隔离。要增加 NumPy、SymPy 或任意脚本执行，应把工具后端换成独立容器服务，并保持 `code -> text observation` 接口不变。不要移除白名单后继续把同一实现称为沙箱。

## ALFWorld

文本版不依赖 THOR、MaskRCNN 或 GPU。资源由官方 release 中 JSON、PDDL 与 TW-PDDL 三组压缩包生成，已安装 `alfworld==0.4.2` 和 `textworld==1.7.0`。共享的 `alfworld_env.py` 保留上游 CPython/TextWorld 兼容处理、独立环境和正确关闭逻辑。

普通配置 `environment: alfworld` 使用真实游戏；`--smoke` 显式设为 `toy`，只做两步 fixture。`tests/test_mechanisms.py` 对真实 ALFWorld 的 reset/step 另作检查，避免用 toy 环境的成功代替真实依赖验证。

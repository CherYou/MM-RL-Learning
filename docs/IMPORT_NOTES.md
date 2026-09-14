# 发布与本地导入说明

本仓库保存 Agentic RL Lab 的代码、中文教程、概念图、参考资料、小型实验数据和已有验证报告。原有 Apache-2.0 许可证、NOTICE 及第三方内容的来源声明均予保留。

导入遵循源项目的 `.gitignore`：虚拟环境、下载的模型、训练输出、大型 ALFWorld 游戏数据、缓存和单独获取的 verl 源码不随本仓库发布。环境与数据准备方法见原 README、`docs/DATA.md` 和 `docs/VERL.md`。现有文档中的服务器绝对路径应替换为实际项目路径；训练环境仍以 Linux 和 Python 3.12 为目标。

为兼容默认不区分文件名大小写的 Windows，16 个与 `README.md` 冲突的短跳转文件 `readme.md` 改名为 `README.redirect.md`，正文和跳转文件内容均保留。

参考资料中缓存的 Hugging Face 网页 `references/papers/harness-rl-hf-page.html` 已清空 `captchaApiKey` 和 `apiKey` 两个站点配置字段。其余导入文件的内容经 SHA-256 与服务器导出清单核对一致。

本次导入未重新运行训练或复现实验；已有报告描述的是原环境中的执行结果。

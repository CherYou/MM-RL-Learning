# 代码、教程与素材的来源核对

本项目把“重新写的内容”和“带来源的复用”明确分开。仓库入口、章节组织与资料参考统一在[根 README 开头](../README.md#参考与感谢)致谢；章节里的参考仓库链接已经集中移走，原作者署名、许可证、参考快照和数据来源没有因此删除。

## 核对范围与结果

可重跑命令：

```bash
.venv/bin/python scripts/audit_provenance.py
```

完整清单与 SHA256 在 [code-provenance-audit.json](../reports/code-provenance-audit.json)。清单覆盖 `src/`、`scripts/`、`tests/`、环境配置目录及全部章节中的 Python/shell 文件；参考副本、虚拟环境、模型权重和运行输出不冒充本项目源码。

脚本对固定参考快照的所有 Python 文件建立 token 索引，忽略注释和空白，找出连续至少 **50 个完全相同 token** 的代码段，再记录双方文件、行号与长度。核对命中项后，长段相同内容均位于下表三个已注明来源的移植工具中，没有发现其他文件的未注明长段复用。shell 安装入口另外逐份查看，内容是本地环境建立、依赖同步与 revision 检查。

这项检测针对给定参考仓库，不能证明与全互联网所有代码都无相似内容。通用公式、数据类字段、官方 API 名和薄入口模板本来就可能相同；相似度不能替代来源判断，也不把改变量名当成独立实现。

## 三个明确保留来源的工具

| 本地文件 | 复用内容 | 本地修改与用途 |
| --- | --- | --- |
| [agentopsd.py](../src/agentic_rl/agentopsd.py) | 组优势、belief 累积、turn credit 与有界重塑工具 | 作为本地原生/verl 控制器的确定性数值组件 |
| [alfworld_data.py](../src/agentic_rl/alfworld_data.py) | 游戏元数据、split 发现、任务筛选与固定顺序取样 | 使用项目共享数据目录 |
| [alfworld_env.py](../src/agentic_rl/alfworld_env.py) | TextWorld 分组环境、信息字段、兼容与资源关闭逻辑 | 本地包引用、动作规范化和同步调用保护 |

来源为根 README 所列 KMnO4-zx 项目的固定提交 `d745e6e26f96485f7689da962630026723546cd5`，采用 Apache-2.0；文件头保留 `Ported from` 和修改说明。它们**不是本项目原创代码**。保留这份明确声明比删除来源、只调整表面写法后称为原创更准确。全部代码来源情况同时记录于 [NOTICE](../NOTICE)。

其余训练循环、native/TRL/verl 接口适配、工具调用、分区梯度路由、检查脚本和新增 `embodied/` 连续控制实现是在本地编写的。算法思想和数学表达依赖论文、公开机制与官方接口，不宣称这些思想由本项目提出。

## 教程文字和图片

20 篇 `TUTORIAL.md` 与 `preliminary/FOUNDATIONS.md` 采用重新组织的中文叙述、手算例子、代码定位、误区和练习答案。对固定参考仓库的 Markdown 执行长中文段落完全相同检测，未检出相同段落；短术语、公式与方法名称不纳入原创措辞保证。教程明确区分实际代码片段、机制伪代码、手算值与实测结果。

20 张概念图均由内置图像生成工具按照本教程描述生成，没有复制参考仓库图片。提示词在 [prompts.json](assets/algorithms/prompts.json)，修图指令在 [edits.json](assets/algorithms/edits.json)，人工查看记录、尺寸与 SHA256 在 [manifest.json](assets/algorithms/manifest.json)。生成工具的机器专属临时路径不随仓库发布。DAPO 全对组与 OPSD 相同 token 数的初稿细节已修正。插图辅助理解，机制以公式、代码和中文图注为准。

## 外部材料的单独归属

- `references/upstream/` 是未修改的参考源码/文档/图片快照，保留原文件和许可证，用于核对来源。
- `references/verl/` 与虚拟环境里的软件属于对应开源依赖，不计入“本地新写代码”。
- AgentOPSD 的 ALFWorld 技能文件通过参考快照复制；其中 [SOURCE.md](../data/skills/SOURCE.md) 进一步注明原始来源为 AgentOPSD 项目的提交 `0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b`、Apache-2.0。本地 `general.md` 是另外编写的短 fallback 提示，不将复制的技能正文描述为原创教程文字。
- 公开问题数据、图像、TextWorld 游戏和模型分别遵循来源条款，见 [DATA](DATA.md)。新 FetchReach 数据是本地程序控制器与随机策略采集的仿真经历，见 [EMBODIED](EMBODIED.md)。

后续如果引入别人的函数或文档段落，应同时更新文件头、NOTICE 和本表；不要等相似性扫描才补来源。开发 worklog 可以删除，本来源说明和实际验证证据保留。

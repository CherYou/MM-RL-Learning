# 代码与素材来源

参考项目与算法论文见[根 README](../README.md#参考与感谢)。本项目代码采用 [Apache-2.0](../LICENSE)，第三方材料的归属见 [NOTICE](../NOTICE)。

## 移植工具

以下工具来自 KMnO4-zx/agentic-rl-lab 的提交 `d745e6e26f96485f7689da962630026723546cd5`，保留 Apache-2.0 来源声明与修改说明。

| 本地文件 | 复用内容 | 本地适配 |
| --- | --- | --- |
| [agentopsd.py](../src/agentic_rl/agentopsd.py) | 组优势、belief 累积、turn credit 与有界重塑工具 | 原生/verl 控制器的数值组件 |
| [alfworld_data.py](../src/agentic_rl/alfworld_data.py) | 游戏元数据、split 发现、任务筛选与取样 | 项目共享数据目录 |
| [alfworld_env.py](../src/agentic_rl/alfworld_env.py) | TextWorld 分组环境、信息字段与资源关闭 | 包引用、动作规范化和同步调用 |

其余训练循环、后端适配与连续控制实现为本项目编写的教学代码；算法与数学依据在对应教程中引用。

## 教程与插图

算法教程采用中文讲解、手算例子、代码定位与练习。概念图由图像生成工具制作，辅助说明算法；公式与代码给出具体定义。图片尺寸与校验值见 [素材清单](assets/algorithms/manifest.json)。

## 第三方资料

- `references/upstream/` 保留参考源码、文档和许可证。
- ALFWorld 技能来自 AgentOPSD，具体版本与许可证见 [SOURCE.md](../data/skills/SOURCE.md)。
- 数据、图像、游戏和模型遵循各自来源条款，见 [数据说明](DATA.md)；FetchReach 数据的采集方法见 [具身实验指南](EMBODIED.md)。
- 大模型技术报告、译稿与配图的来源见 [Basic LLM](../Basic%20LLM/README.md) 中各模型的阅读入口。

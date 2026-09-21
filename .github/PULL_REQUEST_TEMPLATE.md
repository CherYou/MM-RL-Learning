# Pull Request

## 类型（勾选适用项）

- [ ] 教学表达 / 导航
- [ ] 实现逻辑（loss / 轨迹 / 后端）
- [ ] 数据或配置
- [ ] 检查脚本 / CI
- [ ] 文档与来源
- [ ] 其他

## 读者影响

改完之后，新手在哪一步会更容易理解或运行？若无，说明维护/修复原因。

## 验证

- [ ] `make docs`（或 CI docs job）通过
- [ ] 若改数学/loss：`examples/math/loss_walkthrough.py` 或相关 pytest 通过
- [ ] 若改训练入口：默认命令与章节后端一致（TRL / verl / embodied）
- [ ] 若含实验结论：已按 [docs/EVIDENCE.md](../docs/EVIDENCE.md) 标注 L0–L3，未夸大

## 证据层级（如涉及结果）

- 声称层级：L0 / L1 / L2 / L3 / 不涉及
- 运行命令与配置：
- 未验证部分：

## 来源

- [ ] 新增第三方内容已更新 NOTICE / CODE_PROVENANCE / DATA（如需要）
- [ ] 未删除既有归属与限制说明

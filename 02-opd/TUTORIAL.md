# 02｜Medical OPD、SAR 与 IDT：先学专长，再设计怎样保留通用能力

[学习路线](../docs/BEGINNER_GUIDE.md) · [General OPD 基础](general-opd/TUTORIAL.md) · [运行说明](README.md)

本章研究模型训练日程。假设一个通用学生跟随专科教师学习，专科能力可能改善，也可能改变原来的通用回答习惯。我们需要同时问“向谁学习”和“什么时候向谁学习”，不能只看最后一步用了哪种 loss。

![冻结的医疗和通用教师辅导同一学生；SAR 先后分段，IDT 交替出现](../docs/assets/algorithms/medical-opd.png)

青色代表医疗题与医疗教师，紫色代表通用题与 Base 教师。两条时间线都更新同一个学生。医学只是这里的数据领域，图和例子不构成诊疗建议，也不说明模型已经具有临床能力。

## 先画清三份权重

1. **Base**：原始预训练/指令模型，是学生的起点。
2. **Medical Teacher**：从 Base 出发，经过医疗 SFT 得到，然后冻结。
3. **Base Teacher**：保存 Base 的固定副本，在通用阶段作参照。

Student 从 fresh Base 开始，而不是直接继承 Medical Teacher 的 SFT checkpoint。训练中两位教师固定，学生参数连续变化。教师相同大小也能构成蒸馏，因为训练经历不同；“teacher”是一种角色，不保证参数更多。

## 第一步：SFT 怎样得到专科教师

SFT 有固定问题 x 与示范解答 $y^*$，最小化示范生成部分的负对数概率：

$$L_{SFT}=-\frac{\sum_{i,t}m_{i,t}\log\pi_\theta(y^*_{i,t}\mid x_i,y^*_{i,1:t-1})}{\sum_{i,t}m_{i,t}}.$$

如果某个示范 token 当前概率是 0.1，它贡献 $-\log0.1\approx2.3026$；提高到 0.2 后贡献约 1.6094。训练推动模型在这段示范历史后更容易选择示范 token。

[sft.yaml](sft.yaml)走真实 TRL `SFTTrainer`；[trl_backend.py](../src/agentic_rl/trl_backend.py)指定 `completion_only_loss=True`，使题目作为条件而非训练目标。医疗 SFT 数据来源可查 [数据发布页](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)，本地取样与字段见 [数据说明](../docs/DATA.md)。

## 第二步：Medical OPD 学的是教师概率

学生自己生成回答，Medical Teacher 对相同 token 复评。对每个有效 token：

$$d_t=\operatorname{sg}(\log\pi_{old}-\log\pi_{T_M}),\qquad
L_M=\operatorname{mean}_{m=1}\left[e^{\log\pi_\theta-\log\pi_{old}}d_t\right].$$

方向和 [General OPD](general-opd/TUTORIAL.md)一致。这里没有把 MedQA 答案正确率当成每 token 的 OPD 监督；任务评估和蒸馏信号是两条不同的测量渠道。

**SAR-OPD 与 IDT-OPD 是参考仓库的实验调度名称**，不是本教程声称的通用论文标准术语。OPD 基础来源可看 [GKD](https://arxiv.org/abs/2306.13649)与 [MiniLLM](https://arxiv.org/abs/2306.08543)，具体 SAR/IDT 命名以 [根目录的参考与感谢](../README.md#参考与感谢)为准。

## SAR：先专科，再恢复通用

设更新步编号 k 从 0 开始，边界为 M：

$$L_k=\begin{cases}
L_{OPD}(D_M,T_M),&k\lt M,\\
L_{OPD}(D_G,T_B),&k\ge M.
\end{cases}$$

$D_M,D_G$ 是医疗和通用题目池，$T_M,T_B$ 是医疗与 Base 教师。若总共 6 步、M=3，顺序为 `医 医 医 通 通 通`。后半段同时换数据与教师，不是继续在医疗题上加一项通用 KL。

“恢复”是实验意图，不是自动保证。用 Base Teacher 的分布约束学生，可能拉回某些通用行为，也可能抹去刚学到的专科变化，需要评估证明。

## IDT：把两类更新交错安排

$$L_k=\begin{cases}L_M,&k\text{ 为偶数},\\L_G,&k\text{ 为奇数}.\end{cases}$$

同样 6 步，顺序变成 `医 通 医 通 医 通`。每一步只用相应的题目和教师；它不是在同一 batch 中把两个教师 logprob 平均。两个日程即使各用三步医疗、三步通用，结果也不必相同，因为后续梯度在已经变化的学生参数上计算。

可以把它想成学乐器时安排“新曲专项练习”和“基础音阶复习”：总时长相同，先集中后复习与每日交错练习并不是同一种学习过程。这个类比帮助理解顺序效应，不是实验结论。

## 对照真实调度代码

[algorithms.py](../src/agentic_rl/verl_backend/algorithms.py) 的 `AlgorithmBatchBuilder.build` 在每步决定 `general_phase`，随后切换 `pool` 和 `teacher`。教学化摘录：

```python
general_phase = (
    (algorithm == "idt-opd" and step % 2 == 1)
    or (algorithm == "sar-opd" and step >= medical_steps)
)
pool = general_rows if general_phase else medical_rows
teacher = base_teacher if general_phase else medical_teacher
```

变量名在摘录中为易读而展开；实际分支就在上述函数。检查日志 `distill/general_phase`，SAR 应只切换一次，IDT 应按步交替。恢复训练时还要保留原来的阶段边界，不能因为增加总步数而重新把之前的通用阶段解释为医疗阶段。

[medical_pipeline.py](../scripts/medical_pipeline.py)负责先运行 SFT，再把 Teacher checkpoint 路径传给学生阶段；[teacher_logprobs](../src/agentic_rl/trainers.py)负责 token 对齐与无梯度教师复评。这里的共用 loss 很短，调度代码才决定实验究竟是哪一种。

## 怎样做一个可解释的实验

```bash
.venv/bin/python scripts/medical_pipeline.py --variant sar-opd --smoke
```

该命令只检查 CPU 微型链路。正式比较应保存 Base、Medical Teacher、医疗 OPD 中间学生、最终学生四个点，并在固定 MedQA 与通用 held-out 题目上分别评估。只看医疗得分可能漏掉遗忘，只看通用得分又可能掩盖专科根本没学到。

练习：总共 5 步、SAR 的 M=2，IDT 的医疗/通用步数一样吗？答案：SAR 是 2/3，IDT 是 3/2。比较日程时先统一预算，否则“调度优势”可能只是多训练了一类数据。

我的判断是先把“知识从谁来”和“哪些能力被保留”分别做成可观察量，再考虑复杂调度。教师 checkpoint、数据 split、阶段日志和双评估比一个总 loss 更能解释实验。当前仓库已验证机制与 CPU 管线，未验证医疗准确率改善或本章 GPU 训练效果。

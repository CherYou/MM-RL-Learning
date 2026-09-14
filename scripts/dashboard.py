"""Local Streamlit experiment viewer: metrics, trajectories, configuration, data."""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Agentic RL Lab · 本地实验", page_icon="🧪", layout="wide")
st.title("Agentic RL Lab · 本地实验")
st.caption("PyTorch · Transformers · TRL | 指标和轨迹均从本地 runs/ 读取")
files = sorted((ROOT / "runs").rglob("metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
if not files:
    st.info("运行一个章节后，这里会显示训练曲线。命令：arl train 01-grpo/config.yaml --smoke")
    st.stop()
options = {p.parent.relative_to(ROOT / "runs").as_posix(): p for p in files}
selected = st.sidebar.selectbox("选择实验", list(options))
if st.sidebar.button("刷新本地记录"):
    st.rerun()
file = options[selected]
run = file.parent
config = json.loads((run / "config.json").read_text())
if config.get("smoke_only"):
    st.info("这是 CPU smoke 检查：随机微型模型和调试奖励只用于验证执行链路，不代表算法能力。")
records = [json.loads(line) for line in file.read_text().split("\n") if line.strip()]
frame = pd.DataFrame(records)
tabs = st.tabs(["训练曲线", "轨迹与工具调用", "实验配置", "数据与验证"])
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("记录条数", len(frame))
    c2.metric("最后一步", int(frame["step"].max()))
    c3.metric("运行后端", config.get("backend", "native"))
    columns = [c for c in frame.columns if c not in {"step", "elapsed_seconds", "epoch"}]
    defaults = [c for c in columns if any(word in c for word in ["loss", "reward", "grad_norm"])][:4]
    chosen = st.multiselect("显示指标", columns, default=defaults or columns[:2])
    for name in chosen:
        st.write(name)
        st.line_chart(frame[["step", name]].dropna().set_index("step"), height=230)
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button("下载指标 JSONL", file.read_bytes(), file_name=f"{run.name}-metrics.jsonl")
with tabs[1]:
    trajectory = run / "trajectories.jsonl"
    if trajectory.exists():
        samples = [json.loads(line) for line in trajectory.read_text().split("\n") if line.strip()]
        index = st.selectbox(
            "选择生成样本",
            range(len(samples)),
            format_func=lambda i: f"#{i} · step {samples[i]['step']} · reward {samples[i].get('reward', 0):.3f}",
        )
        sample = samples[index]
        st.code(sample.get("text", ""), language=None)
        st.write("模型生成 token 数", sum(sample.get("loss_mask", [])))
        with st.expander("查看 mask、token 与多轮元数据"):
            st.json(sample)
    else:
        st.write("此 TRL 运行保存了 trainer_state 和指标；原生后端另外保存逐 token 轨迹。")
    interfaces = sorted(run.glob("interface-*.json"))
    if interfaces:
        selected_file = st.selectbox("Harness 接口记录", interfaces, format_func=lambda p: p.name)
        st.json(json.loads(selected_file.read_text()))
with tabs[2]:
    st.json(config)
    st.write("从仓库根目录运行：")
    st.code("source .venv/bin/activate\narl train 01-grpo/config.yaml --smoke", language="bash")
    status = run / "status.json"
    if status.exists():
        st.json(json.loads(status.read_text()))
with tabs[3]:
    for name in ["data-verification.json", "cpu-audit-latest.json", "doctor.json", "completion-audit.json"]:
        report = ROOT / "reports" / name
        if report.exists():
            with st.expander(name):
                st.json(json.loads(report.read_text()))
    st.write("完整教程与来源记录：仓库 README.md、docs/ 和 references/upstream/。")
st.caption(f"当前记录：{run.relative_to(ROOT).as_posix()}")

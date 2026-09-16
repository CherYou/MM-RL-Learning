#!/usr/bin/env python3
"""Start/status/stop this project's loopback-only TensorBoard and Streamlit."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs/local-services.json"


def owned(pid):
    try:
        command = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "args="], text=True, stderr=subprocess.DEVNULL
        )
        return str(ROOT) in command and ("streamlit" in command or "tensorboard" in command)
    except (FileNotFoundError, ProcessLookupError, PermissionError, subprocess.CalledProcessError):
        return False


def available(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def status(state):
    for name, data in state.items():
        data["live"] = owned(data["pid"])
        try:
            with urllib.request.urlopen(data["health"], timeout=2) as r:
                data["http_status"] = r.status
        except Exception:
            data["http_status"] = None
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["start", "status", "stop"])
    p.add_argument("--tensorboard-port", type=int, default=6006)
    p.add_argument("--dashboard-port", type=int, default=8501)
    a = p.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if a.action == "stop":
        for data in state.values():
            if owned(data["pid"]):
                os.kill(data["pid"], signal.SIGTERM)
        print(json.dumps(status(state), indent=2))
        return
    if a.action == "start":
        specs = {
            "tensorboard": (
                a.tensorboard_port,
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    str(ROOT / "runs"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(a.tensorboard_port),
                ],
                "/",
            ),
            "dashboard": (
                a.dashboard_port,
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ROOT / "scripts/dashboard.py"),
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(a.dashboard_port),
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                ],
                "/_stcore/health",
            ),
        }
        for name, (port, cmd, health) in specs.items():
            if name in state and owned(state[name]["pid"]):
                continue
            if not available(port):
                raise RuntimeError(f"Port {port} is occupied; choose a free port explicitly")
            log = ROOT / "runs" / f"{name}-server.log"
            with log.open("a") as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"},
                )
            state[name] = {
                "pid": process.pid,
                "url": f"http://127.0.0.1:{port}",
                "health": f"http://127.0.0.1:{port}{health}",
                "log": log.relative_to(ROOT).as_posix(),
            }
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=2) + "\n")
        for _ in range(20):
            report = status(state)
            if all(x["http_status"] == 200 for x in report.values()):
                break
            time.sleep(0.5)
    report = status(state)
    print(json.dumps(report, indent=2))
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/local-services.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(x["live"] and x["http_status"] == 200 for x in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

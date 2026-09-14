"""verl workers and algorithm extensions, imported only by the verl environment."""

SUPPORTED = frozenset(
    {
        "grpo",
        "ppo",
        "gspo",
        "dapo",
        "opd",
        "sar-opd",
        "idt-opd",
        "opsd",
        "search-r1",
        "retool",
        "alfworld",
        "agentopsd",
        "tempo",
        "vision-grpo",
        "harness-rl",
    }
)


def needs_environment_switch(config):
    import sys
    from pathlib import Path
    from agentic_rl.data import ROOT

    name = ".venv-verl" if config["device"] == "cpu" else ".venv-verl-gpu"
    return Path(sys.prefix).resolve() != (ROOT / name).resolve()


def run_verl(config):
    import json
    import os
    import subprocess
    import tempfile
    from agentic_rl.data import ROOT

    if not needs_environment_switch(config):
        from .trainer import run

        return run(config)
    name = ".venv-verl" if config["device"] == "cpu" else ".venv-verl-gpu"
    python = ROOT / name / "bin/python"
    if not python.exists():
        raise RuntimeError(f"Missing {python}; run scripts/setup_verl.sh for this device profile")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
        json.dump(config, file)
        file.flush()
        env = dict(os.environ)
        if config["device"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [str(python), "-m", "agentic_rl.verl_backend", "--config-json", file.name],
            cwd=ROOT,
            env=env,
            check=True,
        )
    return ROOT / config["output"]

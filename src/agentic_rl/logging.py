"""Local JSONL + TensorBoard; no hosted account or telemetry required."""

import json
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, path, config):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        self.writer = SummaryWriter(str(self.path / "tensorboard"))
        self.start = time.monotonic()

    def log(self, step, metrics):
        clean = {k: float(v) for k, v in metrics.items()}
        record = {"step": step, "elapsed_seconds": time.monotonic() - self.start, **clean}
        with (self.path / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        for k, v in clean.items():
            self.writer.add_scalar(k, v, step)
        self.writer.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def trajectories(self, step, records):
        with (self.path / "trajectories.jsonl").open("a") as f:
            for record in records:
                f.write(json.dumps({"step": step, **record}, ensure_ascii=False) + "\n")

    def close(self):
        self.writer.close()

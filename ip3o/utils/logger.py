from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List


class Logger:
    def __init__(self):
        self.metrics: DefaultDict[str, List[float]] = defaultdict(list)

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(float(value))

    def latest(self):
        return {k: v[-1] for k, v in self.metrics.items() if v}

    def to_csv(self, path: str) -> None:
        import csv

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.metrics.keys())
        rows = max((len(v) for v in self.metrics.values()), default=0)

        with output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(rows):
                writer.writerow([self.metrics[k][i] if i < len(self.metrics[k]) else "" for k in keys])

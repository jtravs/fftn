# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The results store: one JSON-lines file per run under bench-out/runs/ (DESIGN.md 16.4).

Record types, each with "type" and "run":
  run      once, first: arms (spec, describe, build identity), host, catalogue, settings
  round    once per round attempt: order, offsets, load before and after, valid or not
  timing   once per (round, arm, case): statistics, batches, accuracy, throughput
  summary  once, last: the band and the verdicts at the end of the run
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO / "bench-out"


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


class RunStore:
    def __init__(self, out_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = out_dir / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"
        self.work = self.dir / f"{run_id}.work"
        self.work.mkdir(exist_ok=True)

    def append(self, record: dict) -> None:
        record = {"run": self.run_id, **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":"), sort_keys=False) + "\n")


def find_run(out_dir: Path, run_id: str) -> Path:
    """The store file of a run: an exact ID, a unique prefix, "latest", or a path."""
    p = Path(run_id)
    if p.suffix == ".jsonl" and p.exists():
        return p
    runs = sorted((out_dir / "runs").glob("*.jsonl"))
    if run_id == "latest":
        if not runs:
            raise FileNotFoundError(f"no runs under {out_dir / 'runs'}")
        return max(runs, key=lambda q: q.stem)
    matches = [q for q in runs if q.stem.startswith(run_id)]
    if len(matches) != 1:
        what = "no run" if not matches else "several runs"
        raise FileNotFoundError(f"{what} matching {run_id!r} under {out_dir / 'runs'}")
    return matches[0]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

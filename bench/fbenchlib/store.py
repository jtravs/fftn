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
import re
import secrets
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO / "bench-out"


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def scrub(value: object, prefixes: list[tuple[str, str]]) -> object:
    """Replaces path prefixes by tokens in every string of a record (longest first).

    The store may be copied between machines; absolute paths would carry user names and
    directory layouts with it, and mean nothing elsewhere.
    """
    if isinstance(value, str):
        for prefix, token in prefixes:
            value = re.sub(re.escape(prefix) + r"(?=/|$|[\s\"'])", token, value)
        return value
    if isinstance(value, list):
        return [scrub(v, prefixes) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v, prefixes) for k, v in value.items()}
    return value


class RunStore:
    def __init__(self, out_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = out_dir / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"
        self.work = self.dir / f"{run_id}.work"
        self.work.mkdir(exist_ok=True)
        # Paths under the checkout are stored as <repo>/..., under the directory that holds
        # bench-out (the checkout, a build directory or a remote job's directory) as
        # <base>/..., and the rest of the home directory as ~/... (longest prefix wins).
        self.prefixes: list[tuple[str, str]] = []
        for path in {str(out_dir.absolute().parent), str(out_dir.resolve().parent)}:
            self.add_prefix(path, "<base>")
        self.add_prefix(str(REPO), "<repo>")
        for path in {str(Path.home()), str(Path.home().resolve())}:
            self.add_prefix(path, "~")

    def add_prefix(self, path: str, token: str) -> None:
        if path and path != "/" and all(p != path for p, _ in self.prefixes):
            self.prefixes.append((path.rstrip("/"), token))
            self.prefixes.sort(key=lambda p: -len(p[0]))

    def append(self, record: dict) -> None:
        record = scrub({"run": self.run_id, **record}, self.prefixes)
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

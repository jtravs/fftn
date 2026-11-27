#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X68's bandwidth model (dev/experiments/X68-free-space-stage.md): each arm's time
in sweeps of the field, against the sweeps its route makes.

    model.py RUN.jsonl [--copy C] [--cases REGEX] [--traffic CAMPAIGN.json]

fbench's arm C (STAGE=C) is one in-place sweep of the field (read and write V, the stored
field's bytes) at the run's thread count: its median time is the time of a sweep at the bandwidth
that width reaches. For every other arm and case this prints the arm's median time over C's
(paired round by round, as bench/x66/score.py pairs arms), the time a route bound by bandwidth
alone would take in sweeps, beside the sweeps the route makes by design (DESIGN.md 12.5: T 9; O
and Of 3 where a block of adjacent pencils over a plane fits a cache level and 5 where not, Of
one more read; G 3 where a slab fits a member's cache, else more). Time sweeps above the design's
mean the arm is bound by its arithmetic (the transforms, the propagator's cos and sin), not by
memory. On a machine whose counters cannot be read (the Mac), this is the traffic estimate. With
--traffic, the measured read sweeps of a campaign record (bench/x68/x68.py) for the same cases
are printed beside it, at the thread counts it measured.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbenchlib import store
from x66.score import collect

DESIGN = {"T": "9", "O": "3-5", "Of": "3-5 (+1 read)", "G": "3+"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run")
    p.add_argument("--copy", default="C", help="the label of the sweep arm")
    p.add_argument("--cases", default="")
    p.add_argument("--traffic", help="a campaign record of bench/x68/x68.py")
    args = p.parse_args()
    records = store.load(Path(args.run))
    values, _, cases = collect(records)
    copy = values.get(args.copy, {})
    traffic: dict[tuple[str, str], list[str]] = {}
    if args.traffic:
        for t in json.loads(Path(args.traffic).read_text()).get("traffic", []):
            if t.get("read_sweeps") is not None:
                traffic.setdefault((t["case"], t["arm"]), []).append(
                    f"{t['read_sweeps']:.2f}@{t['threads']}"
                )
    arms = [a for a in values if a != args.copy and not a.endswith("~")]
    print(f"{args.run}: time in sweeps (arm / {args.copy}); design sweeps; measured read sweeps")
    for c in cases:
        if not re.search(args.cases, c) or c not in copy:
            continue
        sweep_ms = statistics.median(copy[c].values()) / 1e6
        print(f"\n{c}: one sweep {sweep_ms:.2f} ms")
        for a in arms:
            v = values[a].get(c, {})
            paired = [math.log(v[r] / copy[c][r]) for r in sorted(v.keys() & copy[c].keys())]
            if not paired:
                continue
            ratio = math.exp(statistics.median(paired))
            ms = statistics.median(v.values()) / 1e6
            measured = " ".join(traffic.get((c, a), []))
            print(
                f"  {a:4s} {ms:10.1f} ms  {ratio:7.2f} sweeps of time  design "
                f"{DESIGN.get(a, '?'):14s} {measured}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

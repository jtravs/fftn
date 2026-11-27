#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired ratios between any two arms of one fbench run (experiment X66,
dev/experiments/X66-block-callbacks.md).

    score.py RUN.jsonl NUM/DEN [NUM/DEN ...] [--cases REGEX] [--differ A,B] [--notes ARM]

fbench reports every arm against its first; X66's rules compare other pairs (K/U, Ki/K2, O/S,
...). Arms of one run share their rounds, so any two are paired round by round as fbench pairs
an arm with its baseline. For each NUM/DEN this prints, per case, the median ratio over rounds
with its 95 % bootstrap interval and fbench's verdict against the run's noise band (the A/A
control's, as `fbench show` gives it), then the geometric mean over the cases with its
interval (rounds resampled as wholes). --cases keeps the cases whose ID matches; --differ A,B
keeps those whose plan note differs between arms A and B (the cases a forced setting changes:
X66's target sets); --notes ARM prints each kept case's note in that arm."""

from __future__ import annotations

import argparse
import math
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbenchlib import analysis, stats, store


def collect(records: list[dict]) -> tuple[dict, dict, list[str]]:
    """Per arm and case: {round: median_ns} over valid rounds, and the note."""
    run = next(r for r in records if r["type"] == "run")
    cases = [c["id"] for c in run["cases"]]
    valid = {r["round"] for r in records if r["type"] == "round" and r.get("valid")}
    values: dict[str, dict[str, dict[int, float]]] = {}
    notes: dict[str, dict[str, str]] = {}
    for r in records:
        if r["type"] != "timing" or r.get("status") != "ok":
            continue
        if not r.get("check", {}).get("pass", False):
            continue
        arm, case = r["arm"], r["case"]
        notes.setdefault(arm, {}).setdefault(case, r.get("note", ""))
        if r["round"] in valid and r.get("median_ns"):
            values.setdefault(arm, {}).setdefault(case, {})[r["round"]] = r["median_ns"]
    return values, notes, cases


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run")
    p.add_argument("pairs", nargs="+")
    p.add_argument("--cases", default="")
    p.add_argument("--differ", default="")
    p.add_argument("--notes", default="")
    args = p.parse_args()
    records = store.load(Path(args.run))
    band = analysis.analyse(records).band
    values, notes, cases = collect(records)
    keep = [c for c in cases if re.search(args.cases, c)]
    if args.differ:
        a, b = args.differ.split(",")
        keep = [c for c in keep if notes.get(a, {}).get(c) != notes.get(b, {}).get(c)]
    print(f"{args.run}: band +-{math.expm1(band) * 100:.2f} %, {len(keep)} cases")
    for pair in args.pairs:
        num, den = pair.split("/")
        per_round: dict[str, dict[int, float]] = {}
        print(f"\n{num} / {den}")
        for c in keep:
            vn, vd = values.get(num, {}).get(c, {}), values.get(den, {}).get(c, {})
            paired = {r: math.log(vn[r] / vd[r]) for r in sorted(vn.keys() & vd.keys())}
            if not paired:
                print(f"  {c:32s} no pairs")
                continue
            iv = stats.interval(list(paired.values()), random.Random(analysis._seed(pair + c, 0)))
            per_round[c] = paired
            verdict = stats.verdict(iv, band)
            ratio = f"{math.exp(iv.median):.3f} [{math.exp(iv.lo):.3f}, {math.exp(iv.hi):.3f}]"
            den_us = statistics.median(vd.values()) / 1e3
            print(f"  {c:32s} {ratio} {verdict:12s} {den_us:10.1f} us ({iv.pairs} rounds)")
        gm = stats.weighted_geomean_interval(
            per_round, dict.fromkeys(per_round, 1.0), random.Random(7)
        )
        if gm is not None:
            print(
                f"  geomean over {len(per_round)} cases: {math.exp(gm.median):.3f}"
                f" [{math.exp(gm.lo):.3f}, {math.exp(gm.hi):.3f}]"
            )
    if args.notes:
        print(f"\nnotes of {args.notes}")
        for c in keep:
            print(f"  {c}: {notes.get(args.notes, {}).get(c, '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

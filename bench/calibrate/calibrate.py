#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment P3's cost model for long rows: fit it to an fbench run and check its choices.

    calibrate.py fit RUN.jsonl [--machine NAME] [--json OUT]
    calibrate.py gnr RUN.jsonl              (rule 3: the Xeon's fit with Granite Rapids' caches)
    calibrate.py thresholds RUN.jsonl       (the fallback: the profile's long-row fields fitted)

The run times every case of bench/catalogues/p3.toml under the arms R (the rules), C (the
chain), G2 and G3 (the grid in two and three phases); each timing's note names the program
(radices, grid phases and their radices). The model (dev/experiments/P3-cost-model.md) prices
one row of a complex line of N points (a real row's half-length line, plus a pairing pass):

    chain pass of radix R:   N (a + b log2 R) m[level(W)], times 1 + k (R >= 16) or 1 + k8
                             (R < 16) beyond L1 when N's power-of-two part spans 16 KiB or
                             more (set conflicts of legs a power of two apart, X29 and X32)
    grid phase:              its passes on blocks in L1, N (a + b log2 R) each; a gather and
                             a scatter of N points at g[level(W)] each, unless the phase is
                             fused (row within L2/4 and radices <= 8, X12), when its first and
                             last passes are priced at m[level(W)] instead
    separate pairing pass:   N p m[level(W)]

with W = 2 N e bytes (the line and its scratch; e the complex size) and level 0, 1, 2, 3 for
L1, L2, L3 and beyond. Parameters (a, b, m1, m2, m3, k, k8, g1, g2, g3, p) are fitted per
precision by least squares on log time, with Nelder-Mead over their logarithms. Each case's
choice is predicted by a fit without that case (leave one out), and compared with the
fastest arm and with the rules. Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

# Cache sizes (bytes) by the host's CPU string: L1 data, L2 and L3 as one core sees them.
# Apple's cluster L2 is its last level; "L3" is its system cache, priced as memory.
MACHINES = {
    "Apple M5 Max": (128 * 1024, 16 * 2**20, 16 * 2**20),
    "AMD EPYC 7763": (32 * 1024, 512 * 1024, 32 * 2**20),
    "Xeon(R) Gold 6240": (32 * 1024, 1 * 2**20, 24 * 2**20 + 768 * 1024),
    # Granite Rapids in X31b's 8-vCPU VM: 48 KiB L1d, 2 MiB L2, 504 MiB L3 reported.
    "granite-rapids": (48 * 1024, 2 * 2**20, 504 * 2**20),
}

PARAMS = ["a", "b", "m1", "m2", "m3", "k", "k8", "g1", "g2", "g3", "p"]

# A chain pass beyond L1 conflicts in cache when the row's power-of-two part spans this many
# bytes (four 4 KiB pages): its time grows by k for a radix of 16 or more (16 streams a power
# of two apart, X32) and by k8 for smaller radices.
CONFLICT_BYTES = 16 * 1024


def machine_caches(cpu: str, name: str | None) -> tuple[int, int, int]:
    if name:
        return MACHINES[name]
    for key, caches in MACHINES.items():
        if key in cpu:
            return caches
    raise SystemExit(f"no cache sizes for {cpu!r}: give --machine")


# ---- Programs from notes --------------------------------------------------------------


def radices(text: str) -> list[int]:
    return [int(r) for r in text.split(" x ")]


def parse_program(note: str) -> dict:
    """The structure of the first axis line of a note."""
    line = note.split("; axis ")[0]
    prog: dict = {"pairing": "then a pairing pass" in line or "after a pairing pass" in line}
    g = re.search(r"grid ([\d x]+) \(decimation in frequency: phases of (.*?), each in blocks", line)
    if g:
        prog["kind"] = "grid"
        prog["phases"] = [radices(m) for m in re.findall(r"\(radices ([\d x]+)\)", g.group(2))]
        return prog
    c = re.search(r"radices ([\d x]+)", line)
    if not c:
        raise ValueError(f"no program in {note!r}")
    prog["kind"] = "chain"
    prog["radices"] = radices(c.group(1))
    return prog


def case_line(case: str) -> tuple[str, int, int]:
    """(kind, complex line length N, complex bytes) of a case id."""
    m = re.match(r"(c2c|r2c|c2r)-(\d+)-b1-op(-f32)?$", case)
    if not m:
        raise ValueError(case)
    kind, n = m.group(1), int(m.group(2))
    e = 8 if m.group(3) else 16
    return kind, (n if kind == "c2c" else n // 2), e


# ---- The model ------------------------------------------------------------------------


def level(w: int, caches: tuple[int, int, int]) -> int:
    l1, l2, l3 = caches
    return 0 if w <= l1 else 1 if w <= l2 else 2 if w <= l3 else 3


def predict(prm: dict, case: str, prog: dict, caches: tuple[int, int, int]) -> float:
    _, n, e = case_line(case)
    w = 2 * n * e
    lv = level(w, caches)
    m = [1.0, prm["m1"], prm["m2"], prm["m3"]][lv]
    g = [prm["g1"], prm["g1"], prm["g2"], prm["g3"]][lv]
    # The power-of-two part of N: legs that far apart map to few cache sets (X29: an odd
    # factor in every leg distance spreads them).
    part = n & -n
    conflicts = lv >= 1 and part * e >= CONFLICT_BYTES

    def pass_cost(r: int) -> float:
        return n * (prm["a"] + prm["b"] * math.log2(r))

    t = 0.0
    if prog["kind"] == "chain":
        for r in prog["radices"]:
            if conflicts:
                t += pass_cost(r) * m * (1 + (prm["k"] if r >= 16 else prm["k8"]))
            else:
                t += pass_cost(r) * m
    else:
        fused_size = n * e <= caches[1] // 4
        for rad in prog["phases"]:
            if fused_size and max(rad) <= 8:
                t += pass_cost(rad[0]) * m + pass_cost(rad[-1]) * m
                t += sum(pass_cost(r) for r in rad[1:-1])
            else:
                t += sum(pass_cost(r) for r in rad) + 2 * n * g
    if prog["pairing"]:
        t += n * prm["p"] * m
    return t


def loss(logp: list[float], rows: list[tuple], caches) -> float:
    prm = dict(zip(PARAMS, (math.exp(max(-30.0, min(30.0, v))) for v in logp)))
    s = 0.0
    for case, prog, t in rows:
        d = math.log(predict(prm, case, prog, caches)) - math.log(t)
        s += d * d
    return s


def nelder_mead(f, x0: list[float], steps: int = 4000) -> list[float]:
    n = len(x0)
    pts = [list(x0)] + [[x0[j] + (0.5 if j == i else 0.0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(steps):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[-1] - vals[0] < 1e-10:
            break
        centre = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        refl = [centre[j] + (centre[j] - pts[-1][j]) for j in range(n)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [centre[j] + 2 * (centre[j] - pts[-1][j]) for j in range(n)]
            fe = f(exp)
            pts[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            con = [centre[j] + 0.5 * (pts[-1][j] - centre[j]) for j in range(n)]
            fc = f(con)
            if fc < vals[-1]:
                pts[-1], vals[-1] = con, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(n)]
                    vals[i] = f(pts[i])
    return pts[min(range(n + 1), key=lambda i: vals[i])]


def fit(rows: list[tuple], caches, start: list[float] | None = None) -> dict:
    x0 = start or [math.log(v) for v in (0.2, 0.05, 1.5, 2.5, 4.0, 0.3, 0.1, 0.5, 1.0, 2.0, 0.5)]
    best = nelder_mead(lambda x: loss(x, rows, caches), x0)
    best = nelder_mead(lambda x: loss(x, rows, caches), best)  # a restart
    # Clamped as in loss(): a parameter the data cannot see may drift without bound.
    return dict(zip(PARAMS, (math.exp(max(-30.0, min(30.0, v))) for v in best)))


# ---- Data -----------------------------------------------------------------------------


def load(path: Path) -> tuple[str, dict, dict]:
    times: dict = collections.defaultdict(list)
    notes: dict = {}
    cpu = ""
    for line in path.open():
        r = json.loads(line)
        if r.get("type") == "run":
            cpu = r["host"]["cpu"]
        elif r.get("type") == "timing" and r.get("status") == "ok" and not r["arm"].endswith("~"):
            times[(r["arm"], r["case"])].append(float(r["median_ns"]))
            notes[(r["arm"], r["case"])] = r["note"]
    med = {k: statistics.median(v) for k, v in times.items()}
    return cpu, med, notes


def evaluate(path: Path, caches, out: Path | None) -> None:
    cpu, med, notes = load(path)
    arms = sorted({a for a, _ in med})
    cases = sorted({c for _, c in med})
    print(f"{path.name}: {cpu}; caches L1 {caches[0]} L2 {caches[1]} L3 {caches[2]}; arms {arms}")
    for prec in ("f64", "f32"):
        sel = [c for c in cases if (c.endswith("-f32")) == (prec == "f32")]
        rows = [(c, parse_program(notes[(a, c)]), med[(a, c)]) for c in sel for a in arms
                if (a, c) in med and a != "R"]
        if not rows:
            continue
        full = fit(rows, caches)
        within, ratios, rules = 0, [], []
        misses = []
        for c in sel:
            train = [r for r in rows if r[0] != c]
            prm = fit(train, caches, [math.log(full[p]) for p in PARAMS])
            cand = {a: med[(a, c)] for a in arms if (a, c) in med and a != "R"}
            pred = {a: predict(prm, c, parse_program(notes[(a, c)]), caches) for a in cand}
            choice = min(pred, key=pred.get)
            best = min(cand.values())
            ratio = cand[choice] / best
            ratios.append(ratio)
            rules.append(med[("R", c)] / best)
            if ratio <= 1.05:
                within += 1
            else:
                misses.append(f"{c} chose {choice} {ratio:.3f}")
        rms = math.sqrt(loss([math.log(full[p]) for p in PARAMS], rows, caches) / len(rows))
        print(f"  {prec}: {len(sel)} cases; fit rms {rms:.3f} (log); params "
              + ", ".join(f"{p} {full[p]:.3g}" for p in PARAMS))
        print(f"    model within 5 % of the fastest: {within} of {len(sel)} "
              f"({100 * within / len(sel):.1f} %); geomean {statistics.geometric_mean(ratios):.3f}; "
              f"rules {statistics.geometric_mean(rules):.3f}")
        for m in misses:
            print("      " + m)
        if out:
            out.write_text(json.dumps({"cpu": cpu, "precision": prec, "caches": caches, **full}) + "\n")


# ---- Calibrated thresholds (the fallback: the profile's own fields, fitted) ------------

SIZES = [2**k for k in range(13, 30)]  # candidate thresholds in bytes, 8 KiB to 512 MiB


def structure(t: dict, n: int, e: int) -> str:
    """The arm the profile's long-row fields choose for a complex line of n (plan_grid's rule:
    grid_min_bytes, chain_part_bytes and chain_mixed_bytes for mixed lengths, pow2_chain_bytes for powers of two,
    grid_three_bytes and grid_two_odd_part)."""
    b = n * e
    part = n & -n
    odd = n // part
    chain = (b <= t["grid_min"] or (odd > 1 and (part * e <= t["part"] or b <= t["mixed"]))
             or (odd == 1 and b <= t["pow2"]))
    if chain:
        return "C"
    three = b > t["three"] and (t["odd"] == 0 or odd < t["odd"])
    return "G3" if three else "G2"


def chosen(t: dict, c: str, med: dict) -> str:
    _, n, e = case_line(c)
    arm = structure(t, n, e)
    if (arm, c) not in med:
        arm = "G2" if arm == "G3" else "C"
    return arm


def fit_thresholds(cases: list[str], med: dict) -> dict:
    """Coordinate search over the six fields, minimising the summed log time of the choices."""

    def cost(t: dict) -> float:
        return sum(math.log(med[(chosen(t, c, med), c)]) for c in cases)

    # grid_min last: moved first, it would make every row a chain before the exceptions
    # (part, mixed, pow2) could keep the chains that win.
    choices = {
        "part": [0] + SIZES[:8],
        "mixed": [0] + SIZES[:8],
        "pow2": [0] + SIZES,
        "three": SIZES,
        "odd": [0, 3, 5, 15, 25, 125],
        "grid_min": SIZES,
    }
    best: dict | None = None
    for start in SIZES:  # one coordinate descent from every grid_min: the cost is not convex
        t = {"grid_min": start, "part": 0, "mixed": 0, "pow2": 0, "three": 2 * start, "odd": 0}
        for _ in range(3):
            for key, values in choices.items():
                t[key] = min(values, key=lambda v: cost({**t, key: v}))
        if best is None or cost(t) < cost(best):
            best = t
    return best


def thresholds(path: Path) -> None:
    cpu, med, _ = load(path)
    cases_all = sorted({c for _, c in med})
    print(f"{path.name}: {cpu}: the profile's long-row fields fitted to the fastest arms")
    for prec in ("f64", "f32"):
        cases = [c for c in cases_all if c.endswith("-f32") == (prec == "f32")]
        if not cases:
            continue
        full = fit_thresholds(cases, med)
        within, ratios, rules = 0, [], []
        rules_within = 0
        misses = []
        for c in cases:
            t = fit_thresholds([x for x in cases if x != c], med)
            arm = chosen(t, c, med)
            best = min(med[(a, c)] for a in ("C", "G2", "G3") if (a, c) in med)
            ratio = med[(arm, c)] / best
            ratios.append(ratio)
            rules.append(med[("R", c)] / best)
            rules_within += 1 if med[("R", c)] / best <= 1.05 else 0
            if ratio <= 1.05:
                within += 1
            else:
                misses.append(f"{c} chose {arm} {ratio:.3f}")
        print(f"  {prec}: fields {full}; the rules within 5 %: {rules_within} of {len(cases)}")
        print(f"    calibrated fields within 5 % of the fastest: {within} of {len(cases)} "
              f"({100 * within / len(cases):.1f} %); geomean {statistics.geometric_mean(ratios):.3f}; "
              f"rules {statistics.geometric_mean(rules):.3f}")
        for m in misses:
            print("      " + m)


def granite_rapids(path: Path) -> None:
    """Rule 3: the Xeon's fit, with Granite Rapids' cache sizes, chooses for X31b's rows."""
    cpu, med, notes = load(path)
    xeon = machine_caches(cpu, None)
    gnr = MACHINES["granite-rapids"]
    arms = sorted({a for a, _ in med} - {"R"})
    cases = sorted(c for c in {c for _, c in med} if not c.endswith("-f32"))
    rows = [(c, parse_program(notes[(a, c)]), med[(a, c)]) for c in cases for a in arms
            if (a, c) in med]
    prm = fit(rows, xeon)
    print(f"{path.name}: {cpu}'s fit priced with Granite Rapids' caches {gnr}")
    for c in cases:
        pred = {a: predict(prm, c, parse_program(notes[(a, c)]), gnr) for a in arms if (a, c) in med}
        here = {a: predict(prm, c, parse_program(notes[(a, c)]), xeon) for a in pred}
        print(f"  {c:24s} Granite Rapids: {min(pred, key=pred.get):3s}  Xeon: {min(here, key=here.get):3s}  "
              f"measured on the Xeon: {min(pred, key=lambda a: med[(a, c)])}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("run", type=Path)
    f.add_argument("--machine")
    f.add_argument("--json", type=Path)
    g = sub.add_parser("gnr")
    g.add_argument("run", type=Path)
    th = sub.add_parser("thresholds")
    th.add_argument("run", type=Path)
    args = p.parse_args(argv)
    if args.command == "thresholds":
        thresholds(args.run)
        return 0
    if args.command == "gnr":
        granite_rapids(args.run)
        return 0
    cpu, _, _ = load(args.run)
    evaluate(args.run, machine_caches(cpu, args.machine), args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

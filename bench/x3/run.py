#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X3's first measurement: radix 32 and 64 passes against the planner's choice.

    run.py [--out-dir D] [--build B] [--isa T[,T...]] [--lengths 512,1024] [--dry-run] [--fbench '...']
    run.py plan [--isa T]

One fbench run per length of bench/catalogues/x3.toml (c2c rows 2^9 to 2^16, batches 1 and
64, both precisions). Arm A is the FFTN backend as the public plan builds it (the planner's
chain or grid); the other arms, all on the same target (`--isa`):

- `cR1.R2...`: a chain forced to that radix sequence (RADICES): every sequence of codelet
  radices that contains 32 or 64 and has no more passes than the fewest a radix-16 chain
  needs, the best few by X2's cost estimate (bench/x2/radices.py) padded to each C;
- from 4096 up, `chain`: the planner's radix-16 chain (STRATEGY=chain), `grid`: the grid
  with its radix-16 phase rule (STRATEGY=grid), `g64`: the grid with phases of radix up to
  64 (GRID_PHASE_RADIX=64), and `gN1.N2[.N3]`: that phase rule on the splits with the
  fewest passes (GRID_FACTORS).

The method and the decision rule are dev/experiments/X3-big-radices.md.
"""

from __future__ import annotations

import argparse
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from fbenchlib import store  # noqa: E402

from x2 import radices  # noqa: E402

REPO = BENCH.parent
CATALOGUE = BENCH / "catalogues" / "x3.toml"
LENGTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
BIG = (32, 64)
RADICES = (*range(2, 17), *BIG)
CHAIN_TOP = 4  # per C, by the cost estimate
GRID_FROM = 4096
GRID_TOP = 3  # splits with the fewest passes


def sequences(n: int, max_passes: int) -> list[radices.Sequence]:
    out: list[radices.Sequence] = []

    def walk(rest: int, seq: tuple[int, ...]) -> None:
        if rest == 1:
            out.append(seq)
            return
        if len(seq) == max_passes:
            return
        for r in RADICES:
            if rest % r == 0:
                walk(rest // r, (*seq, r))

    walk(n, ())
    return out


def fewest_passes(n: int, largest: int) -> int:
    return math.ceil(math.log(n) / math.log(largest) - 1e-9)


def chain_candidates(n: int, lanes: tuple[int, ...]) -> list[radices.Sequence]:
    ops = radices.codelet_ops()
    seqs = [s for s in sequences(n, fewest_passes(n, 16)) if any(r in BIG for r in s)]
    out: list[radices.Sequence] = []
    for c in lanes or (None,):
        for s in sorted(seqs, key=lambda s: (radices.cost(s, ops, c), s))[:CHAIN_TOP]:
            if s not in out:
                out.append(s)
    return out


def phase_passes(f: int) -> int:
    return fewest_passes(f, 64) if f > 1 else 0


def grid_candidates(n: int) -> list[tuple[int, ...]]:
    """The two two-phase splits and the three-phase split of n with the fewest passes under
    the radix-64 phase rule; among equals the most even, the shorter first factor first."""
    splits: list[tuple[int, ...]] = []
    for a in range(4, n):
        if n % a or n // a < 4:
            continue
        splits.append((a, n // a))
        m = n // a
        for b in range(4, m):
            if m % b == 0 and m // b >= 4:
                splits.append((a, b, m // b))
    out: list[tuple[int, ...]] = []
    for phases, top in ((2, GRID_TOP - 1), (3, 1)):
        of = [s for s in splits if len(s) == phases]
        if not of:
            continue
        least = min(sum(phase_passes(f) for f in s) for s in of)
        best = [s for s in of if sum(phase_passes(f) for f in s) == least]
        best.sort(key=lambda s: (max(s) / min(s), s))
        out += best[:top]
    return out


def arms_of(n: int, isa: str | None) -> list[str]:
    lib = f"fftn+ISA={isa}" if isa else "fftn"
    lanes = radices.LANES[isa] if isa else ()
    arms = [f"A={lib}"]
    arms += [f"c{radices.label(s)}={lib}+RADICES={radices.label(s)}" for s in chain_candidates(n, lanes)]
    if n >= GRID_FROM:
        arms += [f"chain={lib}+STRATEGY=chain", f"grid={lib}+STRATEGY=grid"]
        arms.append(f"g64={lib}+STRATEGY=grid+GRID_PHASE_RADIX=64")
        for s in grid_candidates(n):
            f = radices.label(s)
            arms.append(f"g{f}={lib}+STRATEGY=grid+GRID_PHASE_RADIX=64+GRID_FACTORS={f}")
    return arms


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run.py", description=__doc__.split("\n")[0])
    p.add_argument("command", nargs="?", default="run", choices=("run", "plan"))
    p.add_argument("--out-dir", type=Path, default=store.DEFAULT_OUT)
    p.add_argument("--build", type=Path, default=REPO / "build" / "check-release")
    p.add_argument("--isa", help="targets, comma-separated: one run per target and length")
    p.add_argument("--lengths", help="comma-separated, from x3.toml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fbench", default="", help="more fbench run options")
    ctx = p.parse_args(argv)
    ns = [int(x) for x in ctx.lengths.split(",")] if ctx.lengths else list(LENGTHS)
    isas: list[str | None] = list(ctx.isa.split(",")) if ctx.isa else [None]
    for isa in isas:
        if isa is not None and isa not in radices.LANES:
            raise SystemExit(f"--isa {isa}: not one of {', '.join(sorted(radices.LANES))}")
    if ctx.command == "plan":
        for isa in isas:
            for n in ns:
                print(isa, n, " ".join(a.split("=", 1)[0] for a in arms_of(n, isa)))
        return 0
    failed = 0
    for isa, n in ((i, n) for i in isas for n in ns):
        ctx.isa = isa
        cmd = [
            sys.executable,
            str(BENCH / "fbench.py"),
            "--out-dir",
            str(ctx.out_dir),
            "run",
            *arms_of(n, ctx.isa),
            "--build",
            str(ctx.build),
            "--catalogue",
            str(CATALOGUE),
            "--tier",
            "full",
            "--filter",
            f"^c2c-{n}-b",
            "--budget",
            "1200",
            *shlex.split(ctx.fbench),
        ]
        if ctx.dry_run:
            print(" ".join(shlex.quote(x) for x in cmd))
            continue
        done = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=REPO)
        m = re.search(r"^run (\S+): ", done.stderr, re.MULTILINE)
        print(f"x3 n = {n} isa {ctx.isa or 'best'}: run {m[1] if m else '?'} exit {done.returncode}",
              flush=True)
        if done.returncode not in (0, 1) or not m:
            print(done.stderr[-3000:], file=sys.stderr, flush=True)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

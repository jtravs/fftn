#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X2's table fit: planner v1's pass costs per C from search campaigns.

    fit.py [--out-dir D] --name NAME[,NAME...] [--isa T] [--table PREFIX] [--out FILE]
           [--compare f64=avx2_double,f32=avx2_float] [--within 0.02] [--split]

Reads the campaigns' screening and refinement runs (bench-out/campaigns/x2-NAME.json and
bench-out/runs/), fits one cost table per precision to every candidate's median paired log
ratio to the baseline arm, adds the overrides, and writes the C++ rows in the form of
src/planner/rules.cpp (stdout or --out). It reports the reproduction rate: the share of
cases (length, batch, precision) above 16 where the search on the fitted table picks the
refinement's winner or a sequence measured within --within of it. --compare counts the
same for tables already in rules.cpp, by name per precision.

The method is dev/experiments/X2-radix-sets.md ("The fit"): the model of DESIGN.md 11.2
rule 4 plus a constant per transform; least squares on log(cost(candidate)) -
log(cost(baseline)) - observed, soft-L1 loss of scale 0.05, refinement weight 1 and
screening weight 0.3 for candidates not refined; unit costs bounded below by 0.4 x
(codelet operations + 2R), the middle radix-4 unit fixed at that to set the scale; x 10 and
rounded. It was first fitted with SciPy's trust-region least_squares; this is a
Levenberg-Marquardt with reweighting for the loss and bounds by projection, in the standard
library, so tables agree with that fit closely, not to the last unit.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from fbenchlib import store  # noqa: E402

from x2 import radices, search  # noqa: E402

RULES = search.REPO / "src" / "planner" / "rules.cpp"
LEAF = 16  # every radix up to it has a codelet (rules.cpp, largest_leaf)
BIG = radices.BIG_RADICES  # priced only by tables measured with them (experiment X3)
RADICES = (*range(2, LEAF + 1), *BIG)  # the columns of a table (rules.hpp, cost_column)
KINDS = 3  # first, middle, last (rules.hpp)
NR = len(RADICES)
COLUMN = {r: i for i, r in enumerate(RADICES)}
TAIL0 = KINDS * NR  # features: units, then tails, then passes, then the constant
PASS = 2 * KINDS * NR
CONST = PASS + 1
# --split (experiment X1's split runs priced, a table variant under test): units of the
# middle and last passes of a split run, and its conversions per point.
SPLIT0 = CONST + 1
CONVERT = SPLIT0 + 2 * NR
NF = CONVERT + 1
FIXED = 1 * NR + COLUMN[4]  # the middle radix-4 unit sets the scale
FLOOR = 0.4
LOSS_SCALE = 0.05
SCREEN_WEIGHT = 0.3
SCALE = 10  # the fitted costs are multiplied by this and rounded
MAX_OVERRIDE = 4  # radix_override holds up to four radices
SLOWER = math.log(1.02)  # an override is due where the choice is this much behind the baseline
NOT_SLOWER = math.log(1.01)  # and must be within fbench's least band of it in every case


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---- Campaign data -------------------------------------------------------------------------


@dataclass
class Case:
    n: int
    case: str  # c2c-N-b1-op, ...
    precision: str  # f64 or f32
    base: radices.Sequence  # the baseline arm's sequence
    winner: radices.Sequence
    refine: dict[radices.Sequence, float] = field(default_factory=dict)  # log ratio to base
    screen: dict[radices.Sequence, float] = field(default_factory=dict)

    @property
    def group(self) -> str:
        return self.case.split("-", 2)[2]


def arm_values(medians: dict[str, dict[str, float]], case: str, base: radices.Sequence) -> dict:
    out = {}
    for arm, per in medians.items():
        if case in per and arm != f"{search.V0}~":
            out[base if arm == search.V0 else radices.parse(arm)] = per[case]
    return out


def load_cases(out_dir: Path, names: list[str], isa: str | None) -> tuple[str, list[Case]]:
    """The campaigns' cases and their common ISA (from the campaigns, else `isa`)."""
    cases: list[Case] = []
    isas = set()
    for name in names:
        state = search.load_state(out_dir, name)
        if not state["lengths"]:
            raise SystemExit(f"no campaign {search.state_path(out_dir, name)}")
        isas.add(state.get("isa") or isa)
        for n_text, entry in sorted(state["lengths"].items(), key=lambda kv: int(kv[0])):
            if "refine" not in entry:
                continue
            screen_run = entry["screen"]["run"]
            refine_run = entry["refine"]["run"]
            screen = search.paired_medians(search.records_of(out_dir, screen_run))
            refine = search.paired_medians(search.records_of(out_dir, refine_run)) if refine_run else {}
            for case, win in sorted(entry["refine"]["winners"].items()):
                base = search.base_of(entry, case)
                cases.append(
                    Case(
                        n=int(n_text),
                        case=case,
                        precision="f32" if case.endswith("-f32") else "f64",
                        base=base,
                        winner=base if win == search.V0 else radices.parse(win),
                        refine=arm_values(refine, case, base),
                        screen=arm_values(screen, case, base),
                    )
                )
    cases = merge(cases)
    if len(isas) != 1 or None in isas:
        raise SystemExit(
            f"campaigns' ISA: {sorted(map(str, isas))}; give --isa for campaigns that predate "
            "it, and fit one target at a time"
        )
    return isas.pop(), cases


def merge(cases: list[Case]) -> list[Case]:
    """One case per id: campaigns that time different candidates of the same case against the
    same baseline (X3's big-radix campaign beside X2's) pool their timings; the winner is the
    best refined candidate of either."""
    by_id: dict[str, Case] = {}
    for c in cases:
        old = by_id.get(c.case)
        if old is None:
            by_id[c.case] = c
            continue
        if old.base != c.base:
            raise SystemExit(f"{c.case}: campaigns with different baselines ({old.base}, {c.base})")
        best = min((old, c), key=lambda k: k.refine.get(k.winner, 0.0))
        old.winner = best.winner
        old.refine = {**old.refine, **c.refine}
        old.screen = {**old.screen, **c.screen}
    return list(by_id.values())


# ---- The model (DESIGN.md 11.2 rule 4) -----------------------------------------------------


# The copies from which a split run may start (src/isa/vocab_*.inl split_block: avx512's W on
# both x86 targets), per precision; targets without split runs are absent.
SPLIT_BLOCK = {"avx2": (8, 16), "avx512": (8, 16)}


def split_from(seq: radices.Sequence, block: int) -> int:
    """strategies/chain.cpp's split_from for a contiguous line: the first pass of the split run,
    or len(seq) when there is none."""
    passes = len(seq)
    if not block or passes < 3 or not codelet_only(seq):
        return passes
    s, start = seq[0], passes
    for i in range(1, passes):
        if s % block == 0:
            start = i
            break
        s *= seq[i]
    if passes - start < 2 or any(r > LEAF for r in seq[start:]):
        return passes
    return start


def codelet_only(seq: radices.Sequence) -> bool:
    return all(r in COLUMN for r in seq)


def features(seq: radices.Sequence, lanes: int, block: int = 0) -> dict[int, float]:
    """Rule 4's counts of a sequence of codelet radices: units and tails per kind and radix,
    passes, and the constant per transform; with a split block, a split run's passes in
    their own columns and its conversions (n / C per run)."""
    n = math.prod(seq)
    f: dict[int, float] = {PASS: float(len(seq)), CONST: 1.0}
    start = split_from(seq, block)
    if start < len(seq):
        f[CONVERT] = n / lanes
    s = 1
    for i, r in enumerate(seq):
        m = n // (r * s)
        kind = 0 if i == 0 else (2 if i + 1 == len(seq) else 1)
        if i == 0:
            units, tails = -(-m // lanes), (1 if m % lanes else 0)
        else:
            units, tails = m * -(-s // lanes), (m if s % lanes else 0)
        j = kind * NR + COLUMN[r]
        if i >= start:
            j = SPLIT0 + (kind - 1) * NR + COLUMN[r]
        f[j] = f.get(j, 0.0) + units
        if tails:
            f[TAIL0 + j] = f.get(TAIL0 + j, 0.0) + tails
        s *= r
    return f


@dataclass
class Table:
    unit: list[list[int]]
    tail: list[list[int]]
    passes: int
    overrides: list[tuple[int, radices.Sequence]] = field(default_factory=list)
    split: list[list[int]] | None = None  # --split: middle and last units of a split run
    convert: int = 0  # --split: per C points of a split run
    block: int = 0  # --split: the split block of this precision


def pass_cost(t: Table, lanes: int, n: int, s: int, r: int, kind: int) -> int:
    m = n // (r * s)
    units = -(-m // lanes) if kind == 0 else m * -(-s // lanes)
    tails = (1 if m % lanes else 0) if kind == 0 else (m if s % lanes else 0)
    return t.unit[kind][COLUMN[r]] * units + t.tail[kind][COLUMN[r]] * tails + t.passes


def rule_radices(n: int, t: Table, lanes: int, overrides: bool = True) -> radices.Sequence:
    """rules.cpp's rule_radices, rule by rule (DESIGN.md 11.2)."""
    if n <= 1:
        return ()
    if n <= LEAF:
        return (n,)
    if overrides:
        for o_n, seq in t.overrides:
            if o_n == n:
                return seq
    smooth, big, divisors = 1, [], [1]
    for q, k in sorted(radices.factor(n).items()):
        if q <= LEAF:
            smooth *= q**k
            divisors = [d * q**i for d in divisors for i in range(k + 1)]
        else:
            big += [q] * k
    if smooth == 1:
        return tuple(big)
    divisors.sort()

    def kind(s: int, nxt: int) -> int:
        return 0 if s == 1 else (2 if nxt == smooth and not big else 1)

    if t.split is not None and not big:
        return split_radices(n, t, lanes, divisors, kind)

    def step(s: int, r: int) -> int:
        return pass_cost(t, lanes, n, s, r, kind(s, s * r)) + best[s * r]

    def allowed(s: int, r: int) -> bool:
        # A radix above 16 whose unit cost is 0 is not priced (rules.cpp, cheapest).
        return (smooth // s) % r == 0 and (r <= LEAF or t.unit[kind(s, s * r)][COLUMN[r]] != 0)

    best = {smooth: 0}
    for s in reversed(divisors[:-1]):
        best[s] = min(step(s, r) for r in RADICES if allowed(s, r))
    out: list[int] = []
    s = 1
    while s != smooth:
        r = next(r for r in RADICES if allowed(s, r) and step(s, r) == best[s])
        out.append(r)
        s *= r
    return (*out, *big)


def split_radices(n: int, t: Table, lanes: int, divisors: list[int], kind) -> radices.Sequence:
    """rule_radices with split runs priced (--split): a pass after prefix product s > 1 with
    s a multiple of the block is in the run; the run is split when it has two passes or more,
    all of radix 16 or below, and its passes then cost t.split's units (no tails: s is a
    multiple of C) plus the conversions; otherwise every pass is interleaved. The cheapest
    over both cases, ties by the lexicographically first sequence."""
    smooth = n
    block = t.block

    def allowed(s: int, r: int) -> bool:
        return (smooth // s) % r == 0 and (r <= LEAF or t.unit[kind(s, s * r)][COLUMN[r]] != 0)

    def cost(s: int, r: int, split: bool) -> int:
        k = kind(s, s * r)
        if split and s > 1 and s % block == 0:
            return t.split[k - 1][COLUMN[r]] * (n // (r * s)) * (s // lanes) + t.passes
        return pass_cost(t, lanes, n, s, r, k)

    # State after prefix s: (in-run passes so far, capped at 2; a big radix in the run).
    @functools.cache
    def best(s: int, count: int, bad: bool, split: bool) -> float:
        if s == smooth:
            valid = count >= 2 and not bad
            return 0 if valid == split else math.inf
        out = math.inf
        for r in RADICES:
            if allowed(s, r):
                out = min(out, cost(s, r, split) + best(s * r, *next_state(s, r, count, bad), split))
        return out

    def next_state(s: int, r: int, count: int, bad: bool) -> tuple[int, bool]:
        if s > 1 and s % block == 0:
            return min(count + 1, 2), bad or r > LEAF
        return count, bad

    convert = t.convert * (n // lanes)
    totals = {False: best(1, 0, False, False), True: best(1, 0, False, True) + convert}
    split = totals[True] < totals[False]
    out: list[int] = []
    s, count, bad = 1, 0, False
    while s != smooth:
        for r in RADICES:
            if not allowed(s, r):
                continue
            c2, b2 = next_state(s, r, count, bad)
            if cost(s, r, split) + best(s * r, c2, b2, split) == best(s, count, bad, split):
                out.append(r)
                s, count, bad = s * r, c2, b2
                break
    return tuple(out)


# ---- The fit -------------------------------------------------------------------------------


def priors(ops: dict[tuple[int, bool], int]) -> list[float]:
    p = [0.0] * NF
    for k in range(KINDS):
        for r in RADICES:
            p[k * NR + COLUMN[r]] = ops[(r, k < 2)] + 2.0 * r
            if k > 0:
                p[SPLIT0 + (k - 1) * NR + COLUMN[r]] = ops[(r, k < 2)] + 2.0 * r
    p[CONVERT] = 4.0
    return p


def rows_of(cases: list[Case], lanes: int, block: int = 0) -> list[tuple[dict, dict, float, float]]:
    """(candidate's features, baseline's features, observed log ratio, weight) of every
    timed candidate of the cases whose sequences are all codelet passes."""
    rows = []
    for c in cases:
        if c.n <= LEAF or not codelet_only(c.base):
            continue
        fb = features(c.base, lanes, block)
        for source, weight in ((c.refine, 1.0), (c.screen, SCREEN_WEIGHT)):
            for seq, y in source.items():
                if seq == c.base or not codelet_only(seq):
                    continue
                if source is c.screen and seq in c.refine:
                    continue
                rows.append((features(seq, lanes, block), fb, y, weight))
    return rows


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (the systems are at most NF wide)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda i: abs(m[i][col]))
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        for i in range(col + 1, n):
            f = m[i][col] / p
            if f:
                row, top = m[i], m[col]
                for j in range(col, n + 1):
                    row[j] -= f * top[j]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def fit(
    rows: list[tuple[dict, dict, float, float]],
    ops: dict,
    iterations: int = 300,
    base: Table | None = None,
) -> list[float]:
    """The fitted features. With `base` (X3's extension of a committed table), its costs of
    radices up to 16 and per pass stay as they are and only the big radices' costs and the
    constant per transform are fitted."""
    prior = priors(ops)
    lb = [FLOOR * v for v in prior]
    x = [max(1.5 * v, 1.0) for v in prior]
    fixed = {FIXED}
    x[FIXED] = lb[FIXED] = prior[FIXED]
    lb[CONVERT] = 0.0
    if base is not None:
        for k in range(KINDS):
            for r in range(2, LEAF + 1):
                j = k * NR + COLUMN[r]
                x[j] = lb[j] = base.unit[k][COLUMN[r]] / SCALE
                x[TAIL0 + j] = lb[TAIL0 + j] = base.tail[k][COLUMN[r]] / SCALE
                fixed |= {j, TAIL0 + j}
        x[PASS] = lb[PASS] = base.passes / SCALE
        fixed.add(PASS)
        for j in range(TAIL0, PASS):
            lb[j] = 0.0  # tails are not floored by the codelets' work
    seen = {j for fa, fb, _, _ in rows for j in (*fa, *fb)}
    movable = [j for j in range(NF) if j not in fixed and j in seen]

    def dot(f: dict, x: list[float]) -> float:
        return sum(v * x[j] for j, v in f.items())

    def cost(x: list[float]) -> float:
        total = 0.0
        for fa, fb, y, w in rows:
            r = w * (math.log(dot(fa, x)) - math.log(dot(fb, x)) - y)
            total += 2.0 * (math.sqrt(1.0 + (r / LOSS_SCALE) ** 2) - 1.0)
        return 0.5 * LOSS_SCALE**2 * total

    current = cost(x)
    lam = 1e-3
    for _ in range(iterations):
        # Gauss-Newton with the soft-L1 loss as weights (iteratively reweighted).
        g = [0.0] * NF
        h = [[0.0] * NF for _ in range(NF)]
        for fa, fb, y, w in rows:
            ax, bx = dot(fa, x), dot(fb, x)
            r = w * (math.log(ax) - math.log(bx) - y)
            rho = 1.0 / math.sqrt(1.0 + (r / LOSS_SCALE) ** 2)
            jac: dict[int, float] = {}
            for j, v in fa.items():
                jac[j] = jac.get(j, 0.0) + w * v / ax
            for j, v in fb.items():
                jac[j] = jac.get(j, 0.0) - w * v / bx
            items = [(j, v) for j, v in jac.items() if v]
            for j, v in items:
                g[j] += rho * r * v
                hj = h[j]
                for k, u in items:
                    hj[k] += rho * v * u
        free = [j for j in movable if not (x[j] <= lb[j] * (1 + 1e-12) and g[j] > 0)]
        accepted = False
        while lam < 1e12:
            a = [[h[j][k] for k in free] for j in free]
            for i, j in enumerate(free):
                a[i][i] += lam * max(h[j][j], 1e-12)
            d = solve(a, [-g[j] for j in free])
            trial = x[:]
            for i, j in enumerate(free):
                trial[j] = max(lb[j], x[j] + d[i])
            new = cost(trial)
            if new < current:
                done = (current - new) <= 1e-10 * current
                x, current, accepted = trial, new, True
                lam = max(lam / 3.0, 1e-9)
                break
            lam *= 4.0
        if not accepted or done:
            break
    return x


def table_of(
    x: list[float], cases: list[Case], ops: dict, base: Table | None = None, block: int = 0,
    lanes: int = 1,
) -> Table:
    unit = [[x[k * NR + COLUMN[r]] * SCALE for r in RADICES] for k in range(KINDS)]
    tail = [[x[TAIL0 + k * NR + COLUMN[r]] * SCALE for r in RADICES] for k in range(KINDS)]
    measured = {r for c in cases for seq in (*c.refine, *c.screen) for r in seq}
    if base is not None:
        for k in range(KINDS):
            for r in range(2, LEAF + 1):
                unit[k][COLUMN[r]] = base.unit[k][COLUMN[r]]
                tail[k][COLUMN[r]] = base.tail[k][COLUMN[r]]
    # A big radix no candidate had is not priced (unit cost 0: never chosen).
    seen = {(k, j) for c in cases for seq in (*c.refine, *c.screen) if codelet_only(seq)
            for j in features(seq, 1) if j < TAIL0 for k in (j // NR,)}
    for k in range(KINDS):
        for r in BIG:
            if (k, k * NR + COLUMN[r]) not in seen:
                unit[k][COLUMN[r]] = tail[k][COLUMN[r]] = 0
    # A radix no measured length has (11 and 13 below 1024) takes a neighbour's costs scaled
    # by the codelets' operation counts.
    for r, ref in ((11, 10), (13, 12)):
        if r not in measured and base is None:
            for k in range(KINDS):
                f = ops[(r, k < 2)] / ops[(ref, k < 2)]
                unit[k][COLUMN[r]] = unit[k][COLUMN[ref]] * f
                tail[k][COLUMN[r]] = tail[k][COLUMN[ref]] * f
    t = Table(
        unit=[[round(v) for v in row] for row in unit],
        tail=[[round(v) for v in row] for row in tail],
        passes=base.passes if base is not None else round(x[PASS] * SCALE),
    )
    if block:
        # A split column no candidate measured takes the interleaved unit of its kind.
        split_seen = {j for c in cases for seq in (*c.refine, *c.screen) if codelet_only(seq)
                      for j in features(seq, lanes, block) if SPLIT0 <= j < CONVERT}
        t.split = [
            [round(x[SPLIT0 + k * NR + COLUMN[r]] * SCALE) if SPLIT0 + k * NR + COLUMN[r] in split_seen
             else t.unit[k + 1][COLUMN[r]] for r in RADICES]
            for k in range(2)
        ]
        t.convert = round(x[CONVERT] * SCALE)
        t.block = block
    return t


# ---- Judging -------------------------------------------------------------------------------


def judge(c: Case, seq: radices.Sequence) -> float | None:
    """How far seq is behind the winner (log ratio, same run), or None if never timed with it."""
    if seq == c.winner:
        return 0.0
    for source in (c.refine, c.screen):
        if seq in source and c.winner in source:
            return source[seq] - source[c.winner]
    return None


def vs_base(c: Case, seq: radices.Sequence) -> float | None:
    for source in (c.refine, c.screen):
        if seq in source:
            return source[seq]
    return None


def overrides_for(
    cases: list[Case], t: Table, lanes: int, behind_winner: bool = False
) -> tuple[list, list[str]]:
    """X2's overrides: where the table's choice is more than 2 % behind the baseline in some
    case, the sequence refined in every case of that length (this precision) and measured
    no slower than the baseline in any (within 1 %), that is least behind the winner at worst.
    With behind_winner (X3's extensions, whose baseline is the radix-16 table's choice, so
    a choice can beat it and still miss the radix 32 or 64 winner widely): also where the
    choice is more than 2 % behind the winner in every case of that length."""
    out, notes = [], []
    by_n: dict[int, list[Case]] = {}
    for c in cases:
        by_n.setdefault(c.n, []).append(c)
    for n, group in sorted(by_n.items()):
        if n <= LEAF:
            continue
        choice = rule_radices(n, t, lanes, overrides=False)
        behind = [vs_base(c, choice) for c in group]
        gaps = [judge(c, choice) for c in group]
        missed = behind_winner and all(g is not None and g > SLOWER for g in gaps)
        if not missed and not any(v is not None and v > SLOWER for v in behind):
            continue
        common = set.intersection(*(set(c.refine) for c in group))
        options = []
        for seq in common:
            if all(c.refine[seq] <= NOT_SLOWER for c in group):
                options.append((max(c.refine[seq] - c.refine[c.winner] for c in group), seq))
        fits = sorted(o for o in options if len(o[1]) <= MAX_OVERRIDE)
        if not fits:
            notes.append(f"n = {n}: choice {radices.label(choice)} is slow, no override fits")
            continue
        out.append((n, fits[0][1]))
        worst = max(v for v in behind if v is not None)
        gap = f", {math.expm1(min(gaps)) * 100:+.1f} % or more behind the winner" if missed else ""
        notes.append(
            f"n = {n}: {radices.label(choice)} {math.expm1(worst) * 100:+.1f} % against the "
            f"baseline{gap}; override {radices.label(fits[0][1])}"
        )
    return out, notes


def reproduction(
    cases: list[Case], t: Table, lanes: int, within: float
) -> tuple[dict[str, list[int]], list[str]]:
    """Per group [reproduced, cases, untimed], and the misses."""
    per: dict[str, list[int]] = {}
    misses = []
    for c in cases:
        if c.n <= LEAF:
            continue
        seq = rule_radices(c.n, t, lanes)
        d = judge(c, seq)
        row = per.setdefault(c.group, [0, 0, 0])
        row[1] += 1
        if d is None:
            row[2] += 1
            misses.append(f"{c.case}: {radices.label(seq)} untimed")
        elif d <= math.log1p(within):
            row[0] += 1
        else:
            misses.append(
                f"{c.case}: {radices.label(seq)} {math.expm1(d) * 100:.1f} % behind "
                f"{radices.label(c.winner)}"
            )
    return per, misses


def show_reproduction(title: str, per: dict[str, list[int]], misses: list[str], verbose: bool):
    rep = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    untimed = sum(v[2] for v in per.values())
    groups = ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per.items()))
    print(f"// {title}: {rep} of {tot} cases reproduced ({100 * rep / max(tot, 1):.1f} %; {groups})")
    if untimed:
        print(f"//   {untimed} choices never timed")
    for m in misses if verbose else misses[:10]:
        print(f"//   miss {m}")
    if not verbose and len(misses) > 10:
        print(f"//   ... {len(misses) - 10} more (--verbose)")


# ---- rules.cpp -----------------------------------------------------------------------------


def parse_rules(text: str) -> dict[str, Table]:
    """The pass_costs tables of rules.cpp by name, with their overrides."""
    ovr = {}
    for m in re.finditer(r"std::array<radix_override, \d+> (\w+)\{\{(.*?)\}\};", text, re.S):
        ovr[m[1]] = [
            (int(o[1]), tuple(int(r) for r in o[2].split(",") if int(r)))
            for o in re.finditer(r"\.n = (\d+), \.radices = \{([\d, ]+)\}", m[2])
        ]

    def rows(block: str) -> list[list[int]]:
        out = [[int(v) for v in r.split(",")] for r in re.findall(r"\{([\d, ]+)\}", block)]
        return [r + [0] * (NR - len(r)) for r in out]

    out = {}
    for m in re.finditer(
        r"constexpr pass_costs (\w+)\{\s*\.unit = \{\{(.*?)\}\},\s*\.tail = \{\{(.*?)\}\},"
        r"\s*\.pass = (\d+),\s*\.overrides = (\w+),",
        text,
        re.S,
    ):
        out[m[1]] = Table(rows(m[2]), rows(m[3]), int(m[4]), ovr.get(m[5], []))
    return out


def cpp(name: str, t: Table) -> str:
    def rows(table: list[list[int]]) -> str:
        return "\n".join("        {" + ", ".join(map(str, r)) + "}," for r in table)

    if t.overrides:
        entries = "\n".join(
            f"    {{.n = {n}, .radices = {{{', '.join(map(str, (*seq, 0, 0, 0)[:MAX_OVERRIDE]))}}}}},"
            for n, seq in t.overrides
        )
        ovr = f"constexpr std::array<radix_override, {len(t.overrides)}> {name}_overrides{{{{\n{entries}\n}}}};\n"
    else:
        ovr = f"constexpr std::array<radix_override, 0> {name}_overrides{{}};\n"
    return (
        f"{ovr}\nconstexpr pass_costs {name}{{\n    .unit = {{{{\n{rows(t.unit)}\n    }}}},\n"
        f"    .tail = {{{{\n{rows(t.tail)}\n    }}}},\n    .pass = {t.passes},\n"
        f"    .overrides = {name}_overrides,\n}};\n"
    )


# ---- Main ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fit.py", description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=store.DEFAULT_OUT, help="bench-out")
    p.add_argument("--name", required=True, help="campaign names, comma-separated")
    p.add_argument("--isa", choices=sorted(radices.LANES), help="for campaigns without one")
    p.add_argument("--table", help="C++ name prefix (default: the ISA)")
    p.add_argument("--out", type=Path, help="write the C++ here instead of stdout")
    p.add_argument("--within", type=float, default=0.02, help="reproduced within this of the winner")
    p.add_argument("--compare", default="", help="rules.cpp tables per precision, f64=NAME,f32=NAME")
    p.add_argument("--verbose", action="store_true", help="every miss")
    p.add_argument(
        "--base",
        default="",
        help="X3: extend rules.cpp tables per precision (f32=NAME): their costs up to radix 16 "
        "stay, the radix 32 and 64 columns are fitted; only these precisions are fitted",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="X1: price split runs (their own middle and last units and a conversion cost); "
        "the tables are a variant under test, not rules.cpp rows",
    )
    ctx = p.parse_args(argv)

    isa, cases = load_cases(ctx.out_dir, ctx.name.split(","), ctx.isa)
    ops = radices.codelet_ops()
    prefix = ctx.table or isa
    compare = dict(x.split("=", 1) for x in ctx.compare.split(",") if x)
    bases = dict(x.split("=", 1) for x in ctx.base.split(",") if x)
    committed = parse_rules(RULES.read_text()) if compare or bases else {}
    text = []
    out = sys.stdout if ctx.out is None else ctx.out.open("w")
    print(f"// fit.py --name {ctx.name}: ISA {isa}, {len(cases)} cases", file=out)
    for i, precision in enumerate(radices.PRECISIONS):
        if bases and precision not in bases:
            continue
        base = committed[bases[precision]] if precision in bases else None
        lanes = radices.LANES[isa][i]
        mine = [c for c in cases if c.precision == precision]
        block = SPLIT_BLOCK.get(isa, (0, 0))[i] if ctx.split else 0
        rows = rows_of(mine, lanes, block)
        log(f"fit {precision} (C = {lanes}): {len(rows)} timings of {len(mine)} cases")
        x = fit(rows, ops, base=base)
        t = table_of(x, mine, ops, base, block, lanes)
        t.overrides, notes = overrides_for(mine, t, lanes, behind_winner=base is not None)
        if base is not None:
            # The base table's overrides stay; new ones only where the extension's choice
            # is slow.
            new = {n for n, _ in t.overrides}
            t.overrides = sorted([o for o in base.overrides if o[0] not in new] + t.overrides)
        name = f"{prefix}_{'double' if precision == 'f64' else 'float'}"
        with contextlib.redirect_stdout(out):
            print(f"// {name}: C = {lanes}, constant per transform {x[CONST] * SCALE:.0f}")
            if t.split is not None:
                print(f"//   split block {block}, conversion {t.convert} per C points")
                for k, row in zip(("middle", "last"), t.split):
                    print(f"//   split {k}: {row}")
            for note in notes:
                print(f"//   {note}")
            show_reproduction(name, *reproduction(mine, t, lanes, ctx.within), ctx.verbose)
            if precision in compare:
                old = committed[compare[precision]]
                show_reproduction(
                    f"rules.cpp {compare[precision]} on these cases",
                    *reproduction(mine, old, lanes, ctx.within),
                    ctx.verbose,
                )
                ns = sorted({c.n for c in mine if c.n > LEAF})
                same = sum(rule_radices(n, t, lanes) == rule_radices(n, old, lanes) for n in ns)
                print(f"//   same choice as {compare[precision]} for {same} of {len(ns)} lengths")
        text.append(cpp(name, t))
    print("\n" + "\n".join(text), file=out)
    rows = ", ".join(
        f"{{target::{isa}, {radices.LANES[isa][i]}, &{prefix}_{w}}}"
        for i, w in enumerate(("double", "float"))
    )
    print(f"// measured_tables rows: {rows}", file=out)
    if ctx.out is not None:
        out.close()
        log(f"wrote {ctx.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

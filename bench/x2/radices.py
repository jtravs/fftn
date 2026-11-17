# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X2's candidates: radix sequences per length, their cost estimate, the cases.

The method is dev/experiments/X2-radix-sets.md; the numbers below are its numbers.
"""

from __future__ import annotations

import math
import re
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODELETS = REPO / "src" / "gen" / "codelets.hpp"
DENSE_LIMIT = 16  # every radix up to it has a codelet; prime factors above it are generic
CODELET_RADICES = tuple(range(2, DENSE_LIMIT + 1))  # X2's candidates (codelets.hpp adds 32, 64)
BIG_RADICES = (32, 64)  # the codelet radices above DENSE_LIMIT (experiment X3)
MAX_PASSES = 8
SMOOTH_LIMIT = 1024  # every 7-smooth length up to here is searched exhaustively
LONG_TOP = 20  # above it, this many best by the cost estimate, plus v0's choice
BATCHES = (1, 64)
LONG_BATCH_POINTS = 1 << 22  # batch 64 only while n * 64 stays within this
PRECISIONS = ("f64", "f32")
# C, complex numbers per vector register, in the order of PRECISIONS, per ISA= target
# (src/isa/: C<double> and C<float> of each vocabulary).
LANES = {"scalar": (1, 1), "neon": (1, 2), "neon_fcma": (1, 2), "avx2": (2, 4), "avx512": (4, 8)}

Sequence = tuple[int, ...]


def factor(n: int) -> dict[int, int]:
    out: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def smooth(n: int, limit: int = 7) -> bool:
    return all(p <= limit for p in factor(n))


def sequences(
    n: int, max_passes: int = MAX_PASSES, codelets: tuple[int, ...] = CODELET_RADICES
) -> list[Sequence]:
    """Every radix sequence of n in pass order: codelet radices (2 to 16, or `codelets`) and,
    for prime factors above 16, those primes as single generic passes; at most max_passes
    passes. Each multiset of radices appears in every distinct order exactly once."""
    big = tuple(sorted(p for p in factor(n) if p > DENSE_LIMIT))
    allowed = tuple(codelets) + big

    @cache
    def tails(rest: int, budget: int) -> tuple[Sequence, ...]:
        if rest == 1:
            return ((),)
        if budget == 0:
            return ()
        out: list[Sequence] = []
        for r in allowed:
            if rest % r == 0:
                out.extend((r, *t) for t in tails(rest // r, budget - 1))
        return tuple(out)

    return [] if n < 2 else list(tails(n, max_passes))


def codelet_ops(path: Path = CODELETS) -> dict[tuple[int, bool], int]:
    """(radix, twiddled) -> vocabulary operations of the forward codelet, loads and stores
    apart: the plain variant for untwiddled passes, the DIF variant for twiddled ones (the
    chain's passes, DESIGN.md 8.2)."""
    entry = re.compile(
        r"\{(\d+), variant::(\w+), (-?1), lowering::\w+, (\d+), (\d+), (\d+), (\d+), (\d+), "
        r"(\d+), (\d+)\}"
    )
    ops: dict[tuple[int, bool], int] = {}
    for m in entry.finditer(path.read_text()):
        radix, kind, sign, vops = int(m[1]), m[2], int(m[3]), int(m[9])
        if sign != -1 or kind == "dit":
            continue
        ops[(radix, kind == "dif")] = vops
    if not ops:
        raise RuntimeError(f"no codelet table in {path}")
    return ops


def generic_ops(radix: int, twiddled: bool) -> int:
    # The generic pass is O(R^2) per butterfly (DESIGN.md 8.2): one complex multiply-add per
    # input and output, counted as two vocabulary operations, plus the twiddles.
    return 2 * radix * radix + (radix - 1 if twiddled else 0)


def cost(
    seq: Sequence, ops: dict[tuple[int, bool], int], lanes: int | None = None
) -> tuple[int, int]:
    """The X2 cost estimate: (passes, vocabulary operations per transform). Every pass but
    the last applies twiddles; a single radix is a leaf (plain codelet) or one generic pass.
    With `lanes` (C), each codelet pass's butterflies are padded to whole units of C as the
    planner counts them (DESIGN.md 11.2 rule 4): the first pass runs ceil(m / C) units, the
    others m ceil(s / C)."""
    n = math.prod(seq)
    total = 0
    s = 1
    for i, r in enumerate(seq):
        twiddled = i + 1 < len(seq)
        per = ops[(r, twiddled)] if (r, twiddled) in ops else generic_ops(r, twiddled)
        butterflies = n // r
        if lanes is not None and (r, twiddled) in ops and len(seq) > 1:
            m = n // (r * s)
            units = -(-m // lanes) if i == 0 else m * -(-s // lanes)
            butterflies = units * lanes
        total += butterflies * per
        s *= r
    return (len(seq), total)


def shortlist(
    n: int,
    v0: Sequence | None,
    ops: dict[tuple[int, bool], int],
    top: int = LONG_TOP,
    lanes: tuple[int, ...] = (),
    codelets: tuple[int, ...] = CODELET_RADICES,
    require: tuple[int, ...] = (),
) -> list[Sequence]:
    """The candidates above SMOOTH_LIMIT: the `top` best by the cost estimate, then v0's
    choice if it is not among them (it may have more than MAX_PASSES passes). With `lanes`
    (C per precision of a forced target), the union of the `top` best padded to each C.
    `codelets` and `require` as for candidates()."""
    seqs = [s for s in sequences(n, codelets=codelets) if wanted(s, require)]
    out: list[Sequence] = []
    for c in lanes or (None,):
        for s in sorted(seqs, key=lambda s: (cost(s, ops, c), s))[:top]:
            if s not in out:
                out.append(s)
    if v0 is not None and v0 not in out:
        out.append(v0)
    return out


def wanted(seq: Sequence, require: tuple[int, ...]) -> bool:
    # With `require`, chains only: a single radix is a leaf, which X8 decided.
    return not require or (len(seq) > 1 and any(r in require for r in seq))


def candidates(
    n: int,
    v0: Sequence | None,
    ops: dict[tuple[int, bool], int],
    lanes: tuple[int, ...] = (),
    codelets: tuple[int, ...] = CODELET_RADICES,
    require: tuple[int, ...] = (),
) -> list[Sequence]:
    """The protocol's candidates of length n, v0's choice included. `codelets`: the codelet
    radices of the sequences (X3 adds 32 and 64); `require`: if given, only sequences with
    one of these radices (X3 times the sequences with 32 or 64 that X2 did not)."""
    if n <= SMOOTH_LIMIT:
        out = [s for s in sequences(n, codelets=codelets) if wanted(s, require)]
        if v0 is not None and v0 not in out:
            out.append(v0)
        return out
    return shortlist(n, v0, ops, lanes=lanes, codelets=codelets, require=require)


def parse_radix_set(text: str) -> tuple[int, ...]:
    """'2-16,32,64' -> (2, ..., 16, 32, 64)."""
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return tuple(sorted(set(out)))


def label(seq: Sequence) -> str:
    """The arm label and RADICES value of a sequence: radices joined by dots."""
    return ".".join(str(r) for r in seq)


def parse(text: str) -> Sequence:
    return tuple(int(x) for x in text.split("."))


# The radices may be followed by the chain's split run (strategies/chain.hpp, experiment X1),
# or end in a Bluestein stage of the product of the primes above 16 (DESIGN.md 11.2 rule 3).
NOTE = re.compile(
    r"\(n = (\d+)\): (?:c2c \w+, )?radices ((?:\d+ x )*\d+)?"
    r"(?:(?: x )?bluestein (\d+) \(M [^)]*\))?"
    r"(?:, split from pass \d+)?$"
)


def from_note(note: str) -> tuple[int, Sequence]:
    """The length and radices of a c2c row from the FFTN backend's case note; a Bluestein
    stage of Q counts as the passes of Q's prime factors, ascending (rule_radices' form)."""
    m = NOTE.search(note)
    if not m or not (m[2] or m[3]):
        raise ValueError(f"not a c2c row's note: {note!r}")
    head = tuple(int(x) for x in m[2].split(" x ")) if m[2] else ()
    tail = tuple(sorted(q for q, k in factor(int(m[3])).items() for _ in range(k))) if m[3] else ()
    return int(m[1]), head + tail


# ---- The cases -----------------------------------------------------------------------------


def smooth_lengths() -> list[int]:
    """7-smooth n <= SMOOTH_LIMIT with a choice to make (more than one sequence)."""
    return [n for n in range(2, SMOOTH_LIMIT + 1) if smooth(n) and len(sequences(n)) > 1]


def long_lengths(standard: list[tuple[str, tuple[int, ...]]]) -> list[int]:
    """The standard catalogue's c2c row lengths above SMOOTH_LIMIT that have a choice among
    every codelet radix: not primes (one generic pass); 5^5 and 5^6 have one since radix 25
    (experiment X42). `standard`: (kind, dims) of its cases."""
    lengths = {d[0] for kind, d in standard if kind == "c2c" and len(d) == 1}
    every = tuple(sorted({r for r, _ in codelet_ops()}))
    return sorted(n for n in lengths if n > SMOOTH_LIMIT and len(sequences(n, codelets=every)) > 1)


def catalogue_text(smooth_ns: list[int], long_ns: list[int]) -> str:
    """bench/catalogues/x2.toml: the cases of X2 (generated; `search.py catalogue` writes it)."""
    head = """\
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Experiment X2's cases (dev/experiments/X2-radix-sets.md): c2c rows, forward, out of place,
# double and single precision. Generated by bench/x2/search.py catalogue from bench/x2/
# radices.py and the standard catalogue; a test keeps them equal. Lengths with a single
# radix sequence (primes) are left out: nothing to choose.

[catalogue]
name = "x2"
"""

    def group(name: str, note: str, ns: list[int], batch: list[int]) -> str:
        rows = []
        line = "  "
        for n in ns:
            item = f"{n}, "
            if len(line) + len(item) > 92:
                rows.append(line.rstrip())
                line = "  "
            line += item
        rows.append(line.rstrip())
        dims = "\n".join(rows)
        return (
            f'\n[[group]]\nname = "{name}"\nnote = "{note}"\nkind = "c2c"\n'
            f"dims = [\n{dims}\n]\nbatch = {batch}\n"
            f'precision = ["f64", "f32"]\n'
        )

    small = [n for n in long_ns if n * BATCHES[1] <= LONG_BATCH_POINTS]
    big = [n for n in long_ns if n * BATCHES[1] > LONG_BATCH_POINTS]
    return (
        head
        + group("x2-smooth", "every 7-smooth length up to 1024", smooth_ns, list(BATCHES))
        + group("x2-long", "standard catalogue rows above 1024", small, list(BATCHES))
        + group(
            "x2-long-single",
            f"standard catalogue rows whose batch of {BATCHES[1]} would exceed "
            f"{LONG_BATCH_POINTS} points",
            big,
            [1],
        )
    )


def length_filter(n: int) -> str:
    """fbench --filter selecting the X2 cases of length n."""
    return f"^c2c-{n}-b"

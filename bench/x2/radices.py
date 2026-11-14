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
CODELET_RADICES = tuple(range(2, 17))  # every radix the generator emits (codelets.hpp)
MAX_PASSES = 8
SMOOTH_LIMIT = 1024  # every 7-smooth length up to here is searched exhaustively
LONG_TOP = 20  # above it, this many best by the cost estimate, plus v0's choice
BATCHES = (1, 64)
LONG_BATCH_POINTS = 1 << 22  # batch 64 only while n * 64 stays within this
PRECISIONS = ("f64", "f32")

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


def sequences(n: int, max_passes: int = MAX_PASSES) -> list[Sequence]:
    """Every radix sequence of n in pass order: codelet radices 2 to 16 and, for prime factors
    above 16, those primes as single generic passes; at most max_passes passes. Each
    multiset of radices appears in every distinct order exactly once."""
    big = tuple(sorted(p for p in factor(n) if p > CODELET_RADICES[-1]))
    allowed = CODELET_RADICES + big

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


def cost(seq: Sequence, ops: dict[tuple[int, bool], int]) -> tuple[int, int]:
    """The X2 cost estimate: (passes, vocabulary operations per transform). Every pass but
    the last applies twiddles; a single radix is a leaf (plain codelet) or one generic pass."""
    n = math.prod(seq)
    total = 0
    for i, r in enumerate(seq):
        twiddled = i + 1 < len(seq)
        per = ops[(r, twiddled)] if (r, twiddled) in ops else generic_ops(r, twiddled)
        total += (n // r) * per
    return (len(seq), total)


def shortlist(
    n: int, v0: Sequence | None, ops: dict[tuple[int, bool], int], top: int = LONG_TOP
) -> list[Sequence]:
    """The candidates above SMOOTH_LIMIT: the `top` best by the cost estimate, then v0's
    choice if it is not among them (it may have more than MAX_PASSES passes)."""
    ranked = sorted(sequences(n), key=lambda s: (cost(s, ops), s))
    out = ranked[:top]
    if v0 is not None and v0 not in out:
        out.append(v0)
    return out


def candidates(n: int, v0: Sequence | None, ops: dict[tuple[int, bool], int]) -> list[Sequence]:
    """The protocol's candidates of length n, v0's choice included."""
    if n <= SMOOTH_LIMIT:
        out = sequences(n)
        if v0 is not None and v0 not in out:
            out.append(v0)
        return out
    return shortlist(n, v0, ops)


def label(seq: Sequence) -> str:
    """The arm label and RADICES value of a sequence: radices joined by dots."""
    return ".".join(str(r) for r in seq)


def parse(text: str) -> Sequence:
    return tuple(int(x) for x in text.split("."))


NOTE = re.compile(r"\(n = (\d+)\): (?:c2c \w+, )?radices ([\d x]+)$")


def from_note(note: str) -> tuple[int, Sequence]:
    """The length and radices of a c2c row from the FFTN backend's case note."""
    m = NOTE.search(note)
    if not m:
        raise ValueError(f"not a c2c row's note: {note!r}")
    return int(m[1]), tuple(int(x) for x in m[2].split(" x "))


# ---- The cases -----------------------------------------------------------------------------


def smooth_lengths() -> list[int]:
    """7-smooth n <= SMOOTH_LIMIT with a choice to make (more than one sequence)."""
    return [n for n in range(2, SMOOTH_LIMIT + 1) if smooth(n) and len(sequences(n)) > 1]


def long_lengths(standard: list[tuple[str, tuple[int, ...]]]) -> list[int]:
    """The standard catalogue's c2c row lengths above SMOOTH_LIMIT that have a choice: not
    primes (one generic pass) nor powers like 5^5 (radix 25 is no codelet, so only 5s).
    `standard`: (kind, dims) of its cases."""
    lengths = {d[0] for kind, d in standard if kind == "c2c" and len(d) == 1}
    return sorted(n for n in lengths if n > SMOOTH_LIMIT and len(sequences(n)) > 1)


def catalogue_text(smooth_ns: list[int], long_ns: list[int]) -> str:
    """bench/catalogues/x2.toml: the cases of X2 (generated; `search.py catalogue` writes it)."""
    head = """\
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Experiment X2's cases (dev/experiments/X2-radix-sets.md): c2c rows, forward, out of place,
# double and single precision. Generated by bench/x2/search.py catalogue from bench/x2/
# radices.py and the standard catalogue; a test keeps them equal. Lengths with a single
# radix sequence (primes, or 5^5 with no radix-25 codelet) are left out: nothing to choose.

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

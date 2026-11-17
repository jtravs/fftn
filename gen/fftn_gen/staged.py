# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Two-level codelets staged through memory, for targets with 16 vector registers.

A flat codelet keeps every value in registers: radix 32 peaks at 35 live values and 64 at
67 (gen/README.md, "Subtleties"), so on a 16-register target the compiler spills them
wherever its allocator decides, and X38 found FFTW's two-pass plans with a radix-32 or 64
pass slower than three radix-16 passes in FFTN (dev/experiments/X38-epyc-mid.md). The
two-level form is the Cooley-Tukey split R = n1 n2 written out with its own staging:

  level A, for each j2 < n2: the n1-point DFT of x[n2 j1 + j2] over j1 (inputs times their
          twiddles first, for dit), each output k1 times w^(j2 k1), w = exp(sign 2 pi i / R),
          stored to slot k1 n2 + j2 of a local array;
  level B, for each k1 < n1: the n2-point DFT of slots k1 n2 + j2 over j2, output k2 being
          X[k1 + n1 k2] (times its twiddle after, for dif).

Each level is built, put in real-coefficient form, lowered and ordered on its own, so each
keeps few values live (radix 64 as 8 x 8: at most 13); the stores of level A, a compiler
fence on the array and the loads of level B are emitted between them (emit_cpp.py), so the
compiler cannot fold the array back into registers and spill on its own. A slot holds its
value up to a rotation, which level B undoes for free (lower(..., residual)). Every input is
loaded in level A and every output stored in level B, so in and out may still alias.

The whole operation list, both levels and the slot traffic, is verified like a flat
codelet's (verify.py evaluates the stage_store and stage_load operations).
"""

from __future__ import annotations

from .algorithms import dft
from .codelet import Codelet, count
from .ir import Graph, rotate
from .lower import Op, lower
from .realify import realify
from .schedule import peak_live, schedule
from .verify import verify


def _renumber(ops: list[Op], base: int) -> tuple[list[Op], int]:
    """ops with value numbers moved to base, base + 1, ...; and the next free number."""
    names: dict[int, int] = {}
    out = []
    for op in ops:
        args = tuple(names[a] for a in op.args)
        dst = -1
        if op.dst >= 0:
            dst = names[op.dst] = base + len(names)
        out.append(Op(op.kind, dst, args, op.const, op.index))
    return out, base + len(names)


def build_staged_ops(radix: int, n1: int, n2: int, variant: str, sign: int) -> list[Op]:
    """The two-level operations, level A then level B, each in its own order. Level A is
    lowered both as built and in real-coefficient form, and the form with fewer vocabulary
    operations is kept (as built wins where the internal twiddles are many: 25 and 64)."""
    forms = [_build(radix, n1, n2, variant, sign, real_a) for real_a in (True, False)]
    return min(forms, key=lambda ops: count(ops).vector_ops)


def _build(radix: int, n1: int, n2: int, variant: str, sign: int, real_a: bool) -> list[Op]:
    assert n1 * n2 == radix
    conj = sign > 0
    ga = Graph(radix, twiddles=radix - 1)
    x = [ga.input(j) for j in range(radix)]
    if variant == "dit":
        x = [x[0]] + [ga.tw_mul(x[j], ga.twiddle(j), conj) for j in range(1, radix)]
    slots = [x[0]] * radix
    for j2 in range(n2):
        sub = dft(ga, [x[n2 * j1 + j2] for j1 in range(n1)], sign)
        for k1 in range(n1):
            slots[k1 * n2 + j2] = ga.root_mul(sub[k1], -sign * j2 * k1, radix)
    if real_a:
        ga, slots = realify(ga, slots, radix)
    residual: list[int] = []
    level_a = schedule(lower(ga, slots, residual))

    gb = Graph(radix, twiddles=radix - 1)
    s = [rotate(gb.input(j), residual[j]) for j in range(radix)]
    y = [s[0]] * radix
    for k1 in range(n1):
        col = dft(gb, [s[k1 * n2 + j2] for j2 in range(n2)], sign)
        for k2 in range(n2):
            y[k1 + n1 * k2] = col[k2]
    if variant == "dif":
        y = [y[0]] + [gb.tw_mul(y[k], gb.twiddle(k), conj) for k in range(1, radix)]
    gb, y = realify(gb, y, radix)
    level_b = schedule(lower(gb, y))

    # Level A's stores and level B's loads address the slots, not the accessors.
    level_a = [
        Op("stage_store", -1, op.args, op.const, op.index) if op.kind == "store" else op
        for op in level_a
    ]
    level_b = [
        Op("stage_load", op.dst, op.args, op.const, op.index) if op.kind == "load" else op
        for op in level_b
    ]
    a, next_value = _renumber(level_a, 0)
    b, _ = _renumber(level_b, next_value)
    return a + b


def build_staged(radix: int, n1: int, n2: int, variant: str, sign: int) -> Codelet:
    """Builds and verifies one two-level codelet; live is the larger level's peak."""
    ops = build_staged_ops(radix, n1, n2, variant, sign)
    verify(ops, radix, variant, sign)
    cut = next(i for i, op in enumerate(ops) if op.kind == "stage_load")
    live = max(peak_live(ops[:cut]), peak_live(ops[cut:]))
    return Codelet(radix, variant, sign, tuple(ops), count(ops), live)

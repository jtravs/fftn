# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Complex-register lowering (DESIGN.md 7.2, step 4): DAG nodes to vocabulary operations.

Each node becomes at most one operation. What is decided here:

- Rotations. A node's register holds value * (-i)^m for a rotation m chosen per node
  ("materialised rotation"). An Add can take either operand first, which gives two
  choices of m; a Scale or a twiddle multiply keeps its operand's m; loads have m = 0.
  The choice is driven backwards from the stores, which need m equal to the output's
  pending rotation, so that no output needs an extra operation. Where that fails,
  lower() raises ForcedRotation: no codelet so far needs one, and an operation that
  rotates a register (neg, or swap and conjugate) would be added only when a lowering
  does, with a test that compiles it.
- Fused real multiply-adds. acc +- c * x becomes fmadd_r(acc, x, +-c) when the scaled
  node has no other use. A scale used twice stays one multiply (the unfused count of the
  algorithm does not grow); fusing it into both users would trade one multiply for an
  extra multiply on unfused targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath

from .ir import Add, Graph, Input, Scale, Term, Twiddle, TwMul

# Add-type operation for a + (-i)^r b.
ADD_OPS = ("add", "subj", "sub", "addj")


@dataclass(slots=True)
class Op:
    """One vocabulary operation. dst and args are value numbers (-1 for a store)."""

    kind: str  # load, load_tw, add, sub, addj, subj, scale, fmadd_r, mul, mulc, store
    dst: int
    args: tuple[int, ...] = ()
    const: mpmath.mpf | None = None
    index: int = -1  # load / store / load_tw position


class ForcedRotation(Exception):
    pass


def lower(g: Graph, outputs: list[Term], residual: list[int] | None = None) -> list[Op]:
    """Operations computing outputs[k] into store k, in topological order.

    residual: for a store whose reader can rotate for free (the first level of a two-level
    codelet, staged.py), a list that receives, per output, the rotation q such that the
    output is the stored register times (-i)^q; without it a store that needs a rotation
    raises ForcedRotation."""
    live = g.live(outputs)
    uses: dict[int, int] = dict.fromkeys(live, 0)
    for n in live:
        for c in g.children(n):
            uses[c] += 1
    for t in outputs:
        if t.node != 0:
            uses[t.node] += 1

    # Wanted rotations, by distance from a store (smaller is stronger).
    want: dict[int, dict[int, int]] = {n: {} for n in live}

    def wish(n: int, m: int, dist: int) -> None:
        w = want[n]
        if m not in w or w[m] > dist:
            w[m] = dist

    for t in outputs:
        if t.node != 0:
            wish(t.node, t.rot, 0)
    for n in reversed(live):
        for m, dist in list(want[n].items()):
            match g.nodes[n]:
                case Add(a, b, r):
                    wish(a, m % 4, dist + 1)
                    wish(b, (m + r) % 4, dist + 1)
                case Scale(a, _) | TwMul(a, _, _):
                    wish(a, m, dist + 1)

    mat: dict[int, int] = {}  # materialised rotation per node
    ops: list[Op] = []
    fused: set[int] = set()  # Scale nodes folded into an fmadd_r
    plan: dict[int, tuple] = {}

    def fusable(n: int, r: int) -> bool:
        return isinstance(g.nodes[n], Scale) and uses[n] == 1 and r % 2 == 0

    for n in live:
        match g.nodes[n]:
            case Input() | Twiddle():
                mat[n] = 0
            case Scale(a, _) | TwMul(a, _, _):
                mat[n] = mat[a]
            case Add(a, b, r):
                # Option A: a first, m = m_a; option B: b first, m = m_b - r.
                options = []
                for swapped in (False, True):
                    if swapped:  # b + (-i)^r_op a
                        first, second, m = b, a, (mat[b] - r) % 4
                        r_op = (m - mat[a]) % 4
                    else:  # a + (-i)^r_op b
                        first, second, m = a, b, mat[a]
                        r_op = (m + r - mat[b]) % 4
                    dist = want[n].get(m, 1 << 30)
                    options.append(
                        (dist, not fusable(second, r_op), swapped, first, second, m, r_op)
                    )
                options.sort()
                _, _, _, first, second, m, r_op = options[0]
                mat[n] = m
                plan[n] = (first, second, r_op)
                if fusable(second, r_op):
                    fused.add(second)

    for n in live:
        match g.nodes[n]:
            case Input(j):
                ops.append(Op("load", n, index=j))
            case Twiddle(k):
                ops.append(Op("load_tw", n, index=k))
            case Scale(a, key):
                if n not in fused:
                    ops.append(Op("scale", n, (a,), g.consts[key]))
            case TwMul(a, w, conj):
                ops.append(Op("mulc" if conj else "mul", n, (a, w)))
            case Add():
                first, second, r_op = plan[n]
                if second in fused:
                    sc = g.nodes[second]
                    c = g.consts[sc.key] * (1 if r_op == 0 else -1)
                    ops.append(Op("fmadd_r", n, (first, sc.a), c))
                else:
                    ops.append(Op(ADD_OPS[r_op], n, (first, second)))

    for k, t in enumerate(outputs):
        if t.node == 0:
            raise ValueError(f"output {k} is identically zero")
        if residual is not None:
            residual.append((t.rot - mat[t.node]) % 4)
        elif (t.rot - mat[t.node]) % 4:
            raise ForcedRotation(
                f"output {k} needs its register rotated by (-i)^{(t.rot - mat[t.node]) % 4}"
            )
        ops.append(Op("store", -1, (t.node,), index=k))
    return ops

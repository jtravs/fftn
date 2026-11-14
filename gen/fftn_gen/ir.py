# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The intermediate representation: a hash-consed DAG of complex-valued nodes.

A Term is a reference to a node with a pending rotation: Term(n, k) stands for
value(n) * (-i)^k. Multiplying by -i, -1 or i therefore only changes k and costs nothing;
the rotation is absorbed by the add, sub, addj or subj that eventually combines the term
(DESIGN.md 7.2, step 2).

Nodes are interned twice. Structurally: an Add or Scale with the same children and
constant is the same node. By value: every node is evaluated at a few random inputs in
high precision, and a node whose values equal those of an existing node times (-i)^q is
that node, rotated by q. The second rule is common-subexpression elimination modulo
commutativity, associativity, distributivity and sign or rotation (value numbering); it
makes a - b and b - a one node, and a + (-i)b and b + i a one node.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

import mpmath

# Working precision of the values used for value numbering and of the constants. Equal
# values computed along different paths agree to about 2^-250 relative; keys keep 2^-150,
# so a spurious split (a missed merge) needs a rounding boundary within 2^-100.
PRECISION = 256
_KEY_BITS = 150
_ZERO_BITS = 200

mpmath.mp.prec = PRECISION


@dataclass(frozen=True, slots=True)
class Term:
    node: int
    rot: int = 0  # pending factor (-i)^rot, 0..3


@dataclass(frozen=True, slots=True)
class Zero:
    pass


@dataclass(frozen=True, slots=True)
class Input:
    index: int


@dataclass(frozen=True, slots=True)
class Twiddle:
    index: int


@dataclass(frozen=True, slots=True)
class Add:
    """value(a) + (-i)^rot * value(b), with a <= b (node ids)."""

    a: int
    b: int
    rot: int


@dataclass(frozen=True, slots=True)
class Scale:
    """c * value(a), c > 0 and c != 1. `key` identifies c; the value is in Graph.consts."""

    a: int
    key: str


@dataclass(frozen=True, slots=True)
class TwMul:
    """value(a) * w or value(a) * conj(w), w a Twiddle node."""

    a: int
    w: int
    conj: bool


type Node = Zero | Input | Twiddle | Add | Scale | TwMul

ZERO = Term(0, 0)

# i^q as multipliers: _ROT[k] = (-i)^k.
_ROT = (mpmath.mpc(1, 0), mpmath.mpc(0, -1), mpmath.mpc(-1, 0), mpmath.mpc(0, 1))


def rotate(t: Term, k: int) -> Term:
    """t * (-i)^k."""
    if t.node == 0:
        return ZERO
    return Term(t.node, (t.rot + k) % 4)


def neg(t: Term) -> Term:
    return rotate(t, 2)


def const_key(c: mpmath.mpf) -> str:
    return mpmath.nstr(c, 45, strip_zeros=False)


class Graph:
    """One codelet's DAG. Node 0 is Zero; ids grow in creation order."""

    def __init__(self, inputs: int, twiddles: int = 0, samples: int = 3, seed: int = 1) -> None:
        rng = random.Random(seed)

        def rand() -> mpmath.mpc:
            return mpmath.mpc(
                mpmath.mpf(rng.getrandbits(64)) / 2**64 - 0.5,
                mpmath.mpf(rng.getrandbits(64)) / 2**64 - 0.5,
            )

        self.samples = samples
        self.input_values = [[rand() for _ in range(inputs)] for _ in range(samples)]
        self.twiddle_values = [[rand() for _ in range(twiddles + 1)] for _ in range(samples)]
        self.nodes: list[Node] = []
        self.values: list[tuple[mpmath.mpc, ...]] = []
        self.consts: dict[str, mpmath.mpf] = {}
        self._by_structure: dict[Node, Term] = {}
        self._by_value: dict[tuple[int, ...], tuple[int, int]] = {}
        self._new(Zero(), tuple(mpmath.mpc(0) for _ in range(samples)))

    # Construction ---------------------------------------------------------------------

    def input(self, j: int) -> Term:
        vals = tuple(s[j] for s in self.input_values)
        return self._intern(Input(j), vals)

    def twiddle(self, k: int) -> Term:
        vals = tuple(s[k] for s in self.twiddle_values)
        return self._intern(Twiddle(k), vals)

    def add(self, x: Term, y: Term) -> Term:
        """x + y."""
        if x.node == 0:
            return y
        if y.node == 0:
            return x
        if x.node > y.node:
            x, y = y, x
        r = (y.rot - x.rot) % 4
        a, b = x.node, y.node
        if a == b and r == 2:
            return ZERO
        if a == b and r == 0:
            return rotate(self.scale(Term(a), mpmath.mpf(2)), x.rot)
        vals = tuple(
            va + _ROT[r] * vb for va, vb in zip(self.values[a], self.values[b], strict=True)
        )
        return rotate(self._intern(Add(a, b, r), vals), x.rot)

    def sub(self, x: Term, y: Term) -> Term:
        return self.add(x, neg(y))

    def scale(self, x: Term, c: mpmath.mpf) -> Term:
        """c * x for a real constant c."""
        if x.node == 0 or c == 0:
            return ZERO
        rot = x.rot
        if c < 0:
            c, rot = -c, rot + 2
        node = self.nodes[x.node]
        a = x.node
        if isinstance(node, Scale):  # constant folding: c * (c2 * y)
            c, a = c * self.consts[node.key], node.a
        if abs(c - 1) < mpmath.mpf(2) ** -_ZERO_BITS:
            return Term(a, rot % 4)
        key = const_key(c)
        c = self.consts.setdefault(key, c)
        vals = tuple(c * v for v in self.values[a])
        return rotate(self._intern(Scale(a, key), vals), rot)

    def tw_mul(self, x: Term, w: Term, conj: bool) -> Term:
        """x * w, or x * conj(w); w is a twiddle input."""
        if x.node == 0:
            return ZERO
        wv = self.values[w.node]
        vals = tuple(
            v * (mpmath.conj(t) if conj else t)
            for v, t in zip(self.values[x.node], wv, strict=True)
        )
        return rotate(self._intern(TwMul(x.node, w.node, conj), vals), x.rot)

    def root_mul(self, x: Term, e: int | Fraction, n: int) -> Term:
        """x * exp(-2 pi i e / n), a constant root of unity.

        The angle is reduced exactly (rationally) to a quarter turn, which becomes a
        pending rotation, and then to the first octant: the eighth turn is (1 - i) sqrt(1/2)
        (one add and one scale), every other angle cos(t) -+ i sin(t) with 0 < t < pi/4.
        """
        f = Fraction(e, n) % 1
        q = int(f * 4)
        f -= Fraction(q, 4)
        if f == 0:
            return rotate(x, q)
        if f == Fraction(1, 8):
            y = self.add(x, rotate(x, 1))
            return rotate(self.scale(y, mpmath.sqrt(mpmath.mpf(1) / 2)), q)
        sign = 1  # x * (c - i s)
        if f > Fraction(1, 8):
            f, q, sign = Fraction(1, 4) - f, q + 1, 3  # (-i) * (c + i s)
        angle = 2 * mpmath.pi * f.numerator / f.denominator
        c, s = mpmath.cos(angle), mpmath.sin(angle)
        y = self.add(self.scale(x, c), rotate(self.scale(x, s), sign))
        return rotate(y, q)

    # Interning ------------------------------------------------------------------------

    def _new(self, node: Node, vals: tuple[mpmath.mpc, ...]) -> int:
        self.nodes.append(node)
        self.values.append(vals)
        return len(self.nodes) - 1

    def _intern(self, node: Node, vals: tuple[mpmath.mpc, ...]) -> Term:
        hit = self._by_structure.get(node)
        if hit is not None:
            return hit
        term = self._number(node, vals)
        self._by_structure[node] = term
        return term

    def _number(self, node: Node, vals: tuple[mpmath.mpc, ...]) -> Term:
        tiny = mpmath.mpf(2) ** -_ZERO_BITS
        if all(abs(v) < tiny for v in vals):
            return ZERO
        # vals = (-i)^q * u with u[0] in the sector |arg| < pi/4.
        v0 = next(v for v in vals if abs(v) >= tiny)
        q = next(
            q
            for q in range(4)
            if (v0 * _ROT[(4 - q) % 4]).real > abs((v0 * _ROT[(4 - q) % 4]).imag)
        )
        u = [v * _ROT[(4 - q) % 4] for v in vals]
        scale = mpmath.mpf(2) ** _KEY_BITS
        key = tuple(int(mpmath.nint(p * scale)) for v in u for p in (v.real, v.imag))
        hit = self._by_value.get(key)
        if hit is not None:
            existing, q_existing = hit
            return Term(existing, (q - q_existing) % 4)
        nid = self._new(node, vals)
        self._by_value[key] = (nid, q)
        return Term(nid, 0)

    # Queries --------------------------------------------------------------------------

    def children(self, nid: int) -> tuple[int, ...]:
        match self.nodes[nid]:
            case Add(a, b, _):
                return (a, b)
            case Scale(a, _):
                return (a,)
            case TwMul(a, w, _):
                return (a, w)
            case _:
                return ()

    def live(self, roots: list[Term]) -> list[int]:
        """Node ids reachable from roots, in creation (topological) order."""
        seen: set[int] = set()
        stack = [t.node for t in roots if t.node != 0]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self.children(n))
        return sorted(seen)

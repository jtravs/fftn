# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Split lowering (DESIGN.md 7.2, step 4): the complex DAG as real-register operations.

Every complex node becomes a real and an imaginary node in a second, real-valued DAG
(RealGraph), which is hash-consed and value-numbered like the complex one but modulo sign
only: a rotation by -i, -1 or i exchanges and negates parts, which costs nothing, and the
value numbering then shares the parts that coincide up to sign (the real and imaginary
parts of conjugate-pair outputs, for instance). The operations are those of the real
vocabulary (src/isa/vocabulary.inl, namespace real), on W reals per register: no shuffle
anywhere, and every codelet call transforms W lines at once.

What is decided here:

- FMA shaping. a +- c*x and a +- y*z become one fused operation (fmadd, fnmadd or fmsub,
  with a constant or with two values) when the product has no other user; a product with
  two users stays one multiply, as in the complex-register lowering.
- Signs. A node's register holds its value or its negation, chosen per node: a sum can be
  formed as a + b, a - b or b - a, and a fused form's constant may carry either sign, so
  the choice is driven backwards from the stores, which need the output itself. Where that
  fails a `neg` is emitted (counted in the metadata; no codelet so far needs one).
- Direction. Only forward codelets are generated: the backward transform is the forward
  one on the input with its real and imaginary parts exchanged, and the output exchanged
  back (swap(z) = i conj(z), so swap(F swap(x)) = conj(F conj(x))). That holds for the DIF
  variant too, with the same forward twiddles: swap(a w) = swap(a) conj(w). The passes
  exchange the parts in their accessors (src/passes/split.inl).
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath

from .ir import Add, Graph, Input, Scale, Term, Twiddle, TwMul, const_key
from .lower import Op

_KEY_BITS = 150
_ZERO_BITS = 200


@dataclass(frozen=True, slots=True)
class RZero:
    pass


@dataclass(frozen=True, slots=True)
class RIn:
    index: int
    part: int  # 0 real, 1 imaginary


@dataclass(frozen=True, slots=True)
class RTw:
    index: int
    part: int


@dataclass(frozen=True, slots=True)
class RAdd:
    """value(a) + value(b), or value(a) - value(b) when neg; a <= b (node ids)."""

    a: int
    b: int
    neg: bool


@dataclass(frozen=True, slots=True)
class RScale:
    """c * value(a), c > 0 and c != 1, c in RealGraph.consts[key]."""

    a: int
    key: str


@dataclass(frozen=True, slots=True)
class RMul:
    """value(a) * value(b), a <= b: a product of two variables (twiddle multiplies)."""

    a: int
    b: int


type RNode = RZero | RIn | RTw | RAdd | RScale | RMul


@dataclass(frozen=True, slots=True)
class RTerm:
    """A reference to a real node, negated when neg."""

    node: int
    neg: bool = False


RZERO = RTerm(0)


def negate(t: RTerm) -> RTerm:
    return t if t.node == 0 else RTerm(t.node, not t.neg)


class RealGraph:
    """The real-valued DAG of one codelet. Node 0 is zero; ids grow in creation order.

    Sample values are the real and imaginary parts of the complex graph's samples, so
    value numbering here sees the same inputs.
    """

    def __init__(self, g: Graph) -> None:
        self.samples = g.samples
        self.input_values = g.input_values
        self.twiddle_values = g.twiddle_values
        self.nodes: list[RNode] = []
        self.values: list[tuple[mpmath.mpf, ...]] = []
        self.consts: dict[str, mpmath.mpf] = {}
        self._by_structure: dict[RNode, RTerm] = {}
        self._by_value: dict[tuple[int, ...], tuple[int, bool]] = {}
        self._new(RZero(), tuple(mpmath.mpf(0) for _ in range(self.samples)))

    def input(self, j: int, part: int) -> RTerm:
        vals = tuple(s[j].imag if part else s[j].real for s in self.input_values)
        return self._intern(RIn(j, part), vals)

    def twiddle(self, k: int, part: int) -> RTerm:
        vals = tuple(s[k].imag if part else s[k].real for s in self.twiddle_values)
        return self._intern(RTw(k, part), vals)

    def add(self, x: RTerm, y: RTerm) -> RTerm:
        """x + y."""
        if x.node == 0:
            return y
        if y.node == 0:
            return x
        if x.node > y.node:
            x, y = y, x
        neg = x.neg != y.neg  # x + y = sx (a + sx sy b)
        a, b = x.node, y.node
        if a == b:
            return RZERO if neg else self.scale(x, mpmath.mpf(2))
        vals = tuple(
            va - vb if neg else va + vb
            for va, vb in zip(self.values[a], self.values[b], strict=True)
        )
        t = self._intern(RAdd(a, b, neg), vals)
        return RTerm(t.node, t.neg != x.neg)

    def sub(self, x: RTerm, y: RTerm) -> RTerm:
        return self.add(x, negate(y))

    def scale(self, x: RTerm, c: mpmath.mpf) -> RTerm:
        """c * x for a real constant c."""
        if x.node == 0 or c == 0:
            return RZERO
        neg = x.neg
        if c < 0:
            c, neg = -c, not neg
        a = x.node
        node = self.nodes[a]
        if isinstance(node, RScale):  # constant folding: c * (c2 * y)
            c, a = c * self.consts[node.key], node.a
        if abs(c - 1) < mpmath.mpf(2) ** -_ZERO_BITS:
            return RTerm(a, neg)
        key = const_key(c)
        c = self.consts.setdefault(key, c)
        t = self._intern(RScale(a, key), tuple(c * v for v in self.values[a]))
        return RTerm(t.node, t.neg != neg)

    def mul(self, x: RTerm, y: RTerm) -> RTerm:
        """x * y for two variables."""
        if x.node == 0 or y.node == 0:
            return RZERO
        a, b = sorted((x.node, y.node))
        vals = tuple(va * vb for va, vb in zip(self.values[a], self.values[b], strict=True))
        t = self._intern(RMul(a, b), vals)
        return RTerm(t.node, t.neg != (x.neg != y.neg))

    def _new(self, node: RNode, vals: tuple[mpmath.mpf, ...]) -> int:
        self.nodes.append(node)
        self.values.append(vals)
        return len(self.nodes) - 1

    def _intern(self, node: RNode, vals: tuple[mpmath.mpf, ...]) -> RTerm:
        hit = self._by_structure.get(node)
        if hit is not None:
            return hit
        term = self._number(node, vals)
        self._by_structure[node] = term
        return term

    def _number(self, node: RNode, vals: tuple[mpmath.mpf, ...]) -> RTerm:
        tiny = mpmath.mpf(2) ** -_ZERO_BITS
        if all(abs(v) < tiny for v in vals):
            return RZERO
        v0 = next(v for v in vals if abs(v) >= tiny)
        neg = v0 < 0  # the key is of the value with its first nonzero sample positive
        scale = mpmath.mpf(2) ** _KEY_BITS
        key = tuple(int(mpmath.nint((-v if neg else v) * scale)) for v in vals)
        hit = self._by_value.get(key)
        if hit is not None:
            existing, neg_existing = hit
            return RTerm(existing, neg != neg_existing)
        nid = self._new(node, vals)
        self._by_value[key] = (nid, neg)
        return RTerm(nid)

    def children(self, nid: int) -> tuple[int, ...]:
        match self.nodes[nid]:
            case RAdd(a, b, _) | RMul(a, b):
                return (a, b)
            case RScale(a, _):
                return (a,)
            case _:
                return ()

    def live(self, roots: list[RTerm]) -> list[int]:
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


def _rotate(parts: tuple[RTerm, RTerm], r: int) -> tuple[RTerm, RTerm]:
    """(re, im) of the value times (-i)^r."""
    re, im = parts
    match r % 4:
        case 0:
            return re, im
        case 1:  # -i (re + i im) = im - i re
            return im, negate(re)
        case 2:
            return negate(re), negate(im)
        case _:  # i (re + i im) = -im + i re
            return negate(im), re


def expand(g: Graph, outputs: list[Term]) -> tuple[RealGraph, list[tuple[RTerm, RTerm]]]:
    """The real DAG of the complex one, and the (re, im) parts of each output."""
    h = RealGraph(g)
    parts: dict[int, tuple[RTerm, RTerm]] = {}
    for n in g.live(outputs):
        match g.nodes[n]:
            case Input(j):
                parts[n] = (h.input(j, 0), h.input(j, 1))
            case Twiddle(k):
                parts[n] = (h.twiddle(k, 0), h.twiddle(k, 1))
            case Add(a, b, r):
                ar, ai = parts[a]
                br, bi = _rotate(parts[b], r)
                parts[n] = (h.add(ar, br), h.add(ai, bi))
            case Scale(a, key):
                c = g.consts[key]
                ar, ai = parts[a]
                parts[n] = (h.scale(ar, c), h.scale(ai, c))
            case TwMul(a, w, conj):
                ar, ai = parts[a]
                wr, wi = parts[w]
                if conj:  # (ar wr + ai wi, ai wr - ar wi)
                    re = h.add(h.mul(ar, wr), h.mul(ai, wi))
                    im = h.sub(h.mul(ai, wr), h.mul(ar, wi))
                else:  # (ar wr - ai wi, ar wi + ai wr)
                    re = h.sub(h.mul(ar, wr), h.mul(ai, wi))
                    im = h.add(h.mul(ar, wi), h.mul(ai, wr))
                parts[n] = (re, im)
    return h, [_rotate(parts[t.node], t.rot) for t in outputs]


# Operation kinds of the split lowering (the real vocabulary), with their meaning:
#   load_re j, load_im j, load_twr k, load_twi k     the parts of input j and twiddle k
#   add a b, sub a b, mul a b                        a + b, a - b, a * b
#   mulk a (c)                                       c * a
#   fmaddk x acc (c), fnmaddk x acc (c), fmsubk x acc (c)   c x + acc, acc - c x, c x - acc
#   fmadd y z acc, fnmadd y z acc, fmsub y z acc     y z + acc, acc - y z, y z - acc
#   neg a                                            -a
#   store re im (index k)                            output k
SPLIT_LOADS = ("load_re", "load_im", "load_twr", "load_twi")
SPLIT_FUSED = ("fmaddk", "fnmaddk", "fmsubk", "fmadd", "fnmadd", "fmsub")


def lower_split(h: RealGraph, outputs: list[tuple[RTerm, RTerm]]) -> list[Op]:
    """Operations computing the parts of outputs[k] into store k, in topological order."""
    roots = [t for pair in outputs for t in pair]
    live = h.live(roots)
    uses: dict[int, int] = dict.fromkeys(live, 0)
    for n in live:
        for c in h.children(n):
            uses[c] += 1
    for t in roots:
        if t.node == 0:
            raise ValueError("an output part is identically zero")
        uses[t.node] += 1

    users: dict[int, list[int]] = {n: [] for n in live}
    for n in live:
        for c in h.children(n):
            users[c].append(n)
    stored = {t.node for t in roots}

    # Which product each sum absorbs (at most one). A product is fused into every one of its
    # users or into none: fused into all of u sums it costs u operations against 1 + u
    # unfused, so a shared product is fused too (on an unfused target each fused form is a
    # multiply and an add, so it costs u - 1 more multiplies there: such targets would need
    # their own text, DESIGN.md 6.1). Products used once go first, then shared ones; a
    # product whose users cannot all take it stays a multiply.
    fuse: dict[int, int] = {}
    fused: set[int] = set()
    products = [n for n in live if isinstance(h.nodes[n], (RScale, RMul)) and n not in stored]
    for p in sorted(products, key=lambda n: (uses[n], n)):
        us = users[p]
        if us and all(
            isinstance(h.nodes[u], RAdd) and u not in fuse and h.nodes[u].a != h.nodes[u].b
            for u in us
        ):
            for u in us:
                fuse[u] = p
            fused.add(p)

    # Wanted signs (+1 the value, -1 its negation), by distance from a store.
    want: dict[int, dict[int, int]] = {n: {} for n in live}

    def wish(n: int, mu: int, dist: int) -> None:
        w = want[n]
        if mu not in w or w[mu] > dist:
            w[mu] = dist

    for t in roots:
        wish(t.node, -1 if t.neg else 1, 0)
    for n in reversed(live):
        for mu, dist in list(want[n].items()):
            match h.nodes[n]:
                case RAdd(a, b, neg):
                    wish(a, mu, dist + 1)
                    wish(b, -mu if neg else mu, dist + 1)
                case RScale(a, _):
                    wish(a, mu, dist + 1)
                case RMul(a, b):
                    wish(a, mu, dist + 1)
                    wish(b, 1, dist + 1)

    mat: dict[int, int] = {}  # the sign each emitted register holds
    plan: dict[int, tuple] = {}

    def product_sign(p: int) -> int:
        """pi with value(p) = pi * Q, Q the product the fused operation computes."""
        match h.nodes[p]:
            case RScale(x, _):
                return mat[x]
            case RMul(y, z):
                return mat[y] * mat[z]
        raise AssertionError("not a product")

    for n in live:
        node = h.nodes[n]
        options: list[tuple] = []  # (distance, prefer, mu, how)
        match node:
            case RIn() | RTw():
                options.append((0, 0, 1, None))
            case RMul(a, b):
                options.append((0, 0, mat[a] * mat[b], None))
            case RScale(a, _):
                options.append((0, 0, mat[a], None))
            case RAdd(a, b, neg) if n in fuse:
                p = fuse[n]
                acc = b if p == a else a
                s1 = -1 if (neg and acc == b) else 1  # value = s1 acc + s2 P
                s2 = -1 if (neg and p == b) else 1
                for mu in (1, -1):
                    alpha = mu * s1 * mat[acc]
                    beta = mu * s2 * product_sign(p)
                    if isinstance(h.nodes[p], RScale):
                        # The constant's sign is free: beta = -1 on alpha = -1 negates it.
                        kind, csign = {
                            (1, 1): ("fmaddk", 1),
                            (1, -1): ("fnmaddk", 1),
                            (-1, 1): ("fmsubk", 1),
                            (-1, -1): ("fmsubk", -1),
                        }[(alpha, beta)]
                        options.append((0, int(csign < 0), mu, (kind, p, acc, csign)))
                    elif (alpha, beta) != (-1, -1):
                        kind = {(1, 1): "fmadd", (1, -1): "fnmadd", (-1, 1): "fmsub"}[
                            (alpha, beta)
                        ]
                        options.append((0, 0, mu, (kind, p, acc, 1)))
            case RAdd(a, b, neg):
                s = -1 if neg else 1
                for mu in (1, -1):
                    alpha, beta = mu * mat[a], mu * s * mat[b]
                    match (alpha, beta):
                        case (1, 1):
                            options.append((0, 0, mu, ("add", a, b)))
                        case (1, -1):
                            options.append((0, 0, mu, ("sub", a, b)))
                        case (-1, 1):
                            options.append((0, 0, mu, ("sub", b, a)))
        if not options:
            raise AssertionError(f"node {n} has no realisable sign")
        options = [
            (want[n].get(mu, 1 << 30), prefer, -mu, mu, how)
            for (_, prefer, mu, how) in options
        ]
        options.sort(key=lambda o: o[:3])
        _, _, _, mu, how = options[0]
        mat[n] = mu
        plan[n] = how

    ops: list[Op] = []
    for n in live:
        if n in fused:
            continue
        match h.nodes[n]:
            case RIn(j, part):
                ops.append(Op("load_im" if part else "load_re", n, index=j))
            case RTw(k, part):
                ops.append(Op("load_twi" if part else "load_twr", n, index=k))
            case RMul(a, b):
                ops.append(Op("mul", n, (a, b)))
            case RScale(a, key):
                ops.append(Op("mulk", n, (a,), h.consts[key]))
            case RAdd():
                how = plan[n]
                if how[0] in ("add", "sub"):
                    ops.append(Op(how[0], n, (how[1], how[2])))
                else:
                    kind, p, acc, csign = how
                    match h.nodes[p]:
                        case RScale(x, key):
                            ops.append(Op(kind, n, (x, acc), csign * h.consts[key]))
                        case RMul(y, z):
                            ops.append(Op(kind, n, (y, z, acc)))

    # Stores: each part must hold the output itself; a register holding its negation is
    # negated once (shared if both parts of several outputs need the same one).
    negs: dict[int, int] = {}
    next_id = len(h.nodes)

    def value_of(t: RTerm) -> int:
        nonlocal next_id
        want_sign = -1 if t.neg else 1
        if mat[t.node] == want_sign:
            return t.node
        if t.node not in negs:
            negs[t.node] = next_id
            ops.append(Op("neg", next_id, (t.node,)))
            next_id += 1
        return negs[t.node]

    for k, (re, im) in enumerate(outputs):
        ops.append(Op("store", -1, (value_of(re), value_of(im)), index=k))
    return ops

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Real-coefficient form: the DAG rewritten so that the factors of i wait for the outputs.

Every node of a DFT's DAG is a linear combination of the atoms (the inputs, and the
products by twiddles) with complex coefficients. Writing each node v as R(v) + i I(v),
with R(v) and I(v) the combinations of the same atoms by the real and the imaginary parts
of v's coefficients, gives a second DAG in which every node is a real combination of the
atoms, up to a pending rotation:

  Add(a, b, r)  R = R(a) + R((-i)^r b),  I = I(a) + I((-i)^r b), where
                (-i) b = I(b) - i R(b), so a rotation only exchanges and negates parts;
  Scale(a, c)   R = c R(a),  I = c I(a);
  atoms         R = the atom, I = 0.

An output X = R(X) + i I(X) is then one add-type operation, and an output pair X_k,
X_{n-k} of a DFT shares both parts (R(X_{n-k}) = R(X_k), I(X_{n-k}) = -I(X_k)), so its two
operations are addj and subj of the same two values: one multiplication by i (a register
swap on x86, gen/README.md "Subtleties") per output pair, n/2 - 1 or (n - 1)/2 per
codelet, where the direct form had them inside every complex-coefficient node (19 at
radix 16 against 7). Value numbering does the rest: the parts that coincide up to sign
(for split radix, the twiddled quarter transforms of k and q - k) become one node, so the
real-coefficient form costs no more operations than the direct one on any codelet
generated (fewer on fused targets: the scales it leaves single-use fuse into fmadd_r) and
the same unfused operation count; a test holds both. Twiddle products are atoms: a
product's operand is formed whole, R + i I, before the multiply.

The two forms compute the same values exactly; they round differently.
"""

from __future__ import annotations

from .ir import ZERO, Add, Graph, Input, Scale, Term, Twiddle, TwMul, rotate


def realify(g: Graph, outputs: list[Term], radix: int) -> tuple[Graph, list[Term]]:
    """A new graph computing the same outputs in real-coefficient form, and its outputs."""
    h = Graph(radix, twiddles=radix - 1)
    parts: dict[int, tuple[Term, Term]] = {}

    def part(t: Term) -> tuple[Term, Term]:
        """(R, I) of value(t.node) * (-i)^t.rot."""
        re, im = parts[t.node]
        match t.rot % 4:
            case 0:
                return re, im
            case 1:  # -i (R + i I) = I - i R
                return im, rotate(re, 2)
            case 2:
                return rotate(re, 2), rotate(im, 2)
            case _:  # i (R + i I) = -I + i R
                return rotate(im, 2), re

    def whole(t: Term) -> Term:
        re, im = part(t)
        return h.add(re, rotate(im, 3))  # R + i I

    for n in g.live(outputs):
        match g.nodes[n]:
            case Input(j):
                parts[n] = (h.input(j), ZERO)
            case Twiddle(k):
                parts[n] = (h.twiddle(k), ZERO)
            case Add(a, b, r):
                ra, ia = part(Term(a))
                rb, ib = part(Term(b, r))
                parts[n] = (h.add(ra, rb), h.add(ia, ib))
            case Scale(a, key):
                c = g.consts[key]
                ra, ia = parts[a]
                parts[n] = (h.scale(ra, c), h.scale(ia, c))
            case TwMul(a, w, conj):
                parts[n] = (h.tw_mul(whole(Term(a)), parts[w][0], conj), ZERO)
    return h, [whole(t) for t in outputs]

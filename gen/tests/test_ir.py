# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The DAG's rules: rotations, hash-consing, value numbering, constant folding."""

from __future__ import annotations

import unittest

import mpmath
from fftn_gen.ir import ZERO, Add, Graph, Scale, Term, neg, rotate


class Rotations(unittest.TestCase):
    def test_rotation_is_free_and_modular(self) -> None:
        g = Graph(2)
        x = g.input(0)
        before = len(g.nodes)
        self.assertEqual(rotate(x, 5), Term(x.node, 1))
        self.assertEqual(neg(neg(x)), x)
        self.assertEqual(len(g.nodes), before, "a rotation creates no node")

    def test_root_mul_reduces_quarter_turns_to_rotations(self) -> None:
        g = Graph(1)
        x = g.input(0)
        before = len(g.nodes)
        for q in range(8):
            self.assertEqual(g.root_mul(x, 4 * q, 16), rotate(x, q))
        self.assertEqual(len(g.nodes), before)

    def test_eighth_turn_costs_one_add_and_one_scale(self) -> None:
        g = Graph(1)
        x = g.input(0)
        y = g.root_mul(x, 1, 8)
        live = g.live([y])
        kinds = sorted(type(g.nodes[n]).__name__ for n in live)
        self.assertEqual(kinds, ["Add", "Input", "Scale"])
        self.assertTrue(_equal(g, y, x, mpmath.expjpi(mpmath.mpf(-1) / 4)))

    def test_combining_rotated_terms_absorbs_the_rotation(self) -> None:
        g = Graph(2)
        a, b = g.input(0), g.input(1)
        for k in range(4):
            s = g.add(a, rotate(b, k))
            node = g.nodes[s.node]
            self.assertIsInstance(node, Add)
            self.assertTrue(_equal(g, s, a, 1, b, (-1j) ** k))


class Interning(unittest.TestCase):
    def test_structurally_equal_nodes_are_one(self) -> None:
        g = Graph(2)
        a, b = g.input(0), g.input(1)
        self.assertEqual(g.add(a, b), g.add(a, b))
        self.assertEqual(g.add(a, b), g.add(b, a))

    def test_a_minus_b_and_b_minus_a_share_a_node(self) -> None:
        g = Graph(2)
        a, b = g.input(0), g.input(1)
        d1, d2 = g.sub(a, b), g.sub(b, a)
        self.assertEqual(d1.node, d2.node)
        self.assertEqual((d1.rot - d2.rot) % 4, 2)

    def test_value_numbering_merges_rotated_equal_values(self) -> None:
        # a - i b and b + i a = i (a - i b) are one node, up to rotation.
        g = Graph(2)
        a, b = g.input(0), g.input(1)
        u = g.add(a, rotate(b, 1))
        v = g.add(b, rotate(a, 3))
        self.assertEqual(u.node, v.node)
        self.assertEqual((v.rot - u.rot) % 4, 3)

    def test_value_numbering_sees_through_associativity(self) -> None:
        g = Graph(3)
        a, b, c = g.input(0), g.input(1), g.input(2)
        left = g.add(g.add(a, b), c)
        right = g.add(a, g.add(b, c))
        self.assertEqual(left, right)

    def test_cancellation_folds_to_zero(self) -> None:
        g = Graph(2)
        a, b = g.input(0), g.input(1)
        self.assertEqual(g.sub(g.add(a, b), g.add(b, a)), ZERO)
        self.assertEqual(g.add(a, ZERO), a)
        self.assertEqual(g.scale(ZERO, mpmath.mpf(3)), ZERO)


class Constants(unittest.TestCase):
    def test_negative_constant_becomes_a_rotation(self) -> None:
        g = Graph(1)
        x = g.input(0)
        p = g.scale(x, mpmath.mpf(3))
        m = g.scale(x, mpmath.mpf(-3))
        self.assertEqual(p.node, m.node)
        self.assertEqual((m.rot - p.rot) % 4, 2)
        node = g.nodes[p.node]
        self.assertIsInstance(node, Scale)
        self.assertGreater(g.consts[node.key], 0)

    def test_scales_fold_and_unit_scale_vanishes(self) -> None:
        g = Graph(1)
        x = g.input(0)
        y = g.scale(g.scale(x, mpmath.mpf(2)), mpmath.mpf(3))
        self.assertEqual(y, g.scale(x, mpmath.mpf(6)))
        self.assertEqual(g.scale(g.scale(x, mpmath.mpf(4)), mpmath.mpf(1) / 4), x)


def _equal(g: Graph, t: Term, *pairs: object) -> bool:
    """value(t) == sum coefficient * value(term) over (term, coefficient) pairs, sample 0."""
    want = mpmath.mpc(0)
    for term, coeff in zip(pairs[0::2], pairs[1::2], strict=True):
        want += mpmath.mpc(coeff) * g.values[term.node][0] * (-1j) ** term.rot
    got = g.values[t.node][0] * (-1j) ** t.rot
    return abs(got - want) < mpmath.mpf(2) ** -200


if __name__ == "__main__":
    unittest.main()

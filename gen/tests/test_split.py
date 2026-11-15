# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The split lowering (split.py) over every generated split codelet."""

from __future__ import annotations

import dataclasses
import functools
import unittest

import mpmath
from fftn_gen.catalogue import SPLIT_RADICES, all_split_codelets
from fftn_gen.codelet import SPLIT_VARIANTS, build_split_ops
from fftn_gen.ir import Graph, rotate
from fftn_gen.schedule import peak_live, schedule
from fftn_gen.split import RealGraph, expand, lower_split
from fftn_gen.verify import VerificationError, verify_split


@functools.cache
def split_codelets() -> tuple:
    return tuple(all_split_codelets())


class SplitCodelets(unittest.TestCase):
    def test_every_radix_and_variant_is_generated_once(self) -> None:
        keys = [(c.radix, c.variant) for c in split_codelets()]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(set(keys), {(r, v) for r in SPLIT_RADICES for v in SPLIT_VARIANTS})

    def test_no_codelet_needs_a_negation(self) -> None:
        for c in split_codelets():
            self.assertEqual(c.counts.negs, 0, (c.radix, c.variant))

    def test_arithmetic_is_the_complex_forms(self) -> None:
        # Unfused real additions of the plain codelets equal the published split-radix
        # counts where the complex form reaches them (radix 16: 144 additions), and the
        # FMA-shaped forms never do more operations than additions plus multiplies.
        by_key = {(c.radix, c.variant): c for c in split_codelets()}
        self.assertEqual(by_key[(16, "plain")].counts.flops_adds, 144)
        self.assertEqual(by_key[(8, "plain")].counts.flops_adds, 52)
        for c in split_codelets():
            k = c.counts
            self.assertLessEqual(k.vector_ops, k.flops_adds + k.flops_muls, (c.radix, c.variant))

    def test_twiddle_variant_adds_four_operations_per_leg_at_most(self) -> None:
        by_key = {(c.radix, c.variant): c for c in split_codelets()}
        for r in SPLIT_RADICES:
            plain, dif = by_key[(r, "plain")], by_key[(r, "dif")]
            self.assertLessEqual(dif.counts.vector_ops - plain.counts.vector_ops, 4 * (r - 1), r)

    def test_ordering_is_valid_and_recorded(self) -> None:
        for c in split_codelets():
            seen: set[int] = set()
            for op in c.ops:
                for a in op.args:
                    self.assertIn(a, seen, (c.radix, c.variant, op))
                if op.dst >= 0:
                    seen.add(op.dst)
            self.assertEqual(c.live, peak_live(c.ops))

    def test_a_seeded_wrong_constant_is_caught(self) -> None:
        ops = schedule(build_split_ops(8, "plain"))
        i = next(i for i, op in enumerate(ops) if op.const is not None)
        bad = list(ops)
        bad[i] = dataclasses.replace(ops[i], const=ops[i].const * (1 + mpmath.mpf(2) ** -60))
        with self.assertRaises(VerificationError):
            verify_split(bad, 8, "plain")

    def test_exchanged_parts_are_caught(self) -> None:
        # A store with its real and imaginary parts swapped: the backward transform's trick
        # applied where it does not belong.
        ops = schedule(build_split_ops(4, "dif"))
        i = next(i for i, op in enumerate(ops) if op.kind == "store" and op.index == 1)
        bad = list(ops)
        bad[i] = dataclasses.replace(ops[i], args=tuple(reversed(ops[i].args)))
        with self.assertRaises(VerificationError):
            verify_split(bad, 4, "dif")

    def test_a_missing_twiddle_load_is_caught(self) -> None:
        ops = schedule(build_split_ops(3, "dif"))
        bad = [op for op in ops if not (op.kind == "load_twi" and op.index == 2)]
        with self.assertRaises((VerificationError, KeyError)):
            verify_split(bad, 3, "dif")

    def test_value_numbering_shares_parts_up_to_sign(self) -> None:
        g = Graph(2)
        h = RealGraph(g)
        a, b = h.input(0, 0), h.input(1, 0)
        d1, d2 = h.sub(a, b), h.sub(b, a)
        self.assertEqual(d1.node, d2.node)
        self.assertNotEqual(d1.neg, d2.neg)

    def test_rotations_cost_nothing(self) -> None:
        # x0 - i x1: the parts are exchanged and negated, no operation beyond the adds.
        g = Graph(2)
        x0, x1 = g.input(0), g.input(1)
        y = g.add(x0, g.add(x1, x1))  # x0 + 2 x1: one scale
        out = [g.add(x0, rotate(x1, 1)), y]
        h, parts = expand(g, out)
        ops = lower_split(h, parts)
        kinds = [op.kind for op in ops]
        self.assertEqual(kinds.count("neg"), 0)
        self.assertNotIn("mul", kinds)


if __name__ == "__main__":
    unittest.main()

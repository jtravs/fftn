# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Algorithms, lowering, ordering and verification, over every generated codelet."""

from __future__ import annotations

import dataclasses
import functools
import unittest

import mpmath
from fftn_gen.algorithms import choose, dft
from fftn_gen.catalogue import RADICES, all_codelets
from fftn_gen.codelet import build_ops, count
from fftn_gen.ir import Graph, rotate
from fftn_gen.lower import ForcedRotation, lower
from fftn_gen.schedule import creation_order, greedy_order, peak_live, schedule
from fftn_gen.verify import VerificationError, verify


@functools.cache
def codelets() -> tuple:
    return tuple(all_codelets())


class Algorithms(unittest.TestCase):
    def test_choice_by_size(self) -> None:
        self.assertEqual(choose(16)[0], "split_radix")
        self.assertEqual(choose(13)[0], "prime")
        self.assertEqual(choose(9), ("cooley_tukey", 3, 3))
        self.assertEqual(choose(12), ("good_thomas", 4, 3))
        self.assertEqual(choose(15), ("good_thomas", 3, 5))

    def test_good_thomas_needs_no_internal_multiply_beyond_its_parts(self) -> None:
        # 6 = 2 x 3: three 2-point and two 3-point DFTs, no twiddle: 2 x 2 real multiplies.
        c = next(c for c in codelets() if (c.radix, c.variant, c.sign) == (6, "plain", -1))
        self.assertEqual(c.counts.flops_muls, 2 * 4)


class Codelets(unittest.TestCase):
    def test_every_radix_variant_and_direction_is_generated(self) -> None:
        keys = {(c.radix, c.variant, c.sign) for c in codelets()}
        self.assertEqual(len(keys), len(codelets()))
        self.assertEqual(len(keys), len(RADICES) * 3 * 2)

    def test_an_output_that_needs_a_forced_rotation_is_refused(self) -> None:
        # A rotated load cannot be fixed by the choice of operand order: its register
        # holds the input itself. Every generated codelet built, so none needed one.
        g = Graph(2)
        x0, x1 = g.input(0), g.input(1)
        with self.assertRaises(ForcedRotation):
            lower(g, [g.add(x0, x1), rotate(x0, 1)])

    def test_directions_cost_the_same(self) -> None:
        by_key = {(c.radix, c.variant, c.sign): c for c in codelets()}
        for (r, v, s), c in by_key.items():
            if s < 0:
                self.assertEqual(c.counts, by_key[(r, v, 1)].counts, (r, v))

    def test_real_coefficient_form_has_one_rotation_per_output_pair(self) -> None:
        # realify.py: the multiplications by i (operands of addj and subj, a register swap
        # each on x86) are one per output pair, floor((R - 1) / 2), at no more vector
        # operations and the same unfused arithmetic as the DAG lowered as built.
        def rotated(ops: list) -> int:
            return len({op.args[1] for op in ops if op.kind in ("addj", "subj")})

        for radix in RADICES:
            for variant in ("plain", "dit", "dif"):
                for sign in (-1, 1):
                    key = (radix, variant, sign)
                    real = build_ops(radix, variant, sign)
                    direct = build_ops(radix, variant, sign, real=False)
                    self.assertEqual(rotated(real), (radix - 1) // 2, key)
                    self.assertLessEqual(rotated(real), rotated(direct), key)
                    c, d = count(real), count(direct)
                    self.assertLessEqual(c.vector_ops, d.vector_ops, key)
                    self.assertEqual((c.flops_adds, c.cmuls), (d.flops_adds, d.cmuls), key)
                    self.assertLessEqual(c.flops_muls, d.flops_muls, key)

    def test_twiddle_variants_add_one_complex_multiply_per_leg(self) -> None:
        by_key = {(c.radix, c.variant, c.sign): c for c in codelets()}
        for (r, v, s), c in by_key.items():
            plain = by_key[(r, "plain", s)].counts
            if v != "plain":
                self.assertEqual(c.counts.cmuls, r - 1)
                self.assertEqual(c.counts.flops_adds, plain.flops_adds + 2 * (r - 1))
                self.assertEqual(c.counts.flops_muls, plain.flops_muls + 4 * (r - 1))

    def test_stores_follow_their_producers_and_loads_precede_first_use(self) -> None:
        for c in codelets():
            ops = list(c.ops)
            pos = {op.dst: i for i, op in enumerate(ops) if op.dst >= 0}
            for i, op in enumerate(ops):
                if op.kind == "store":
                    self.assertEqual(
                        pos[op.args[0]], i - 1 - _stores_between(ops, pos[op.args[0]], i)
                    )
                if op.kind in ("load", "load_tw"):
                    first = min(j for j, o in enumerate(ops) if op.dst in o.args)
                    between = ops[i + 1 : first]
                    self.assertTrue(all(o.kind in ("load", "load_tw") for o in between))

    def test_order_keeps_no_more_values_live_than_the_alternatives(self) -> None:
        # Every codelet, both directions, all variants; every order gets the same load and
        # store placement, so only the order of the arithmetic differs (schedule.py).
        orders = {"greedy": greedy_order, "creation": creation_order, "chosen": schedule}
        totals = dict.fromkeys(orders, 0)
        for radix in RADICES:
            for variant in ("plain", "dit", "dif"):
                for sign in (-1, 1):
                    ops = build_ops(radix, variant, sign)
                    live = {name: peak_live(f(list(ops))) for name, f in orders.items()}
                    best = min(live["greedy"], live["creation"])
                    self.assertEqual(live["chosen"], best, (radix, variant, sign))
                    for name, value in live.items():
                        totals[name] += value
        self.assertLess(totals["greedy"], totals["creation"], totals)


def _with(ops: list, i: int, **changes: object) -> list:
    return [*ops[:i], dataclasses.replace(ops[i], **changes), *ops[i + 1 :]]


def _insert(ops: list, i: int, op: object) -> list:
    return [*ops[:i], op, *ops[i:]]


def _copy(op: object, dst: int) -> object:
    """The same access into a new value, which nothing reads."""
    return dataclasses.replace(op, dst=dst)


def _stores_between(ops: list, start: int, end: int) -> int:
    return sum(1 for o in ops[start + 1 : end] if o.kind == "store")


class Verification(unittest.TestCase):
    def test_a_seeded_wrong_constant_is_caught(self) -> None:
        c = next(c for c in codelets() if (c.radix, c.variant, c.sign) == (5, "plain", -1))
        ops = list(c.ops)
        i = next(i for i, op in enumerate(ops) if op.const is not None)
        ops[i] = dataclasses.replace(ops[i], const=ops[i].const * (1 + mpmath.mpf(2) ** -60))
        with self.assertRaises(VerificationError):
            verify(ops, 5, "plain", -1)

    def test_a_swapped_rotation_is_caught(self) -> None:
        c = max((c for c in codelets() if (c.variant, c.sign) == ("dit", 1)), key=lambda c: c.radix)
        ops = list(c.ops)
        i = next(i for i, op in enumerate(ops) if op.kind == "addj")
        ops[i] = dataclasses.replace(ops[i], kind="subj")
        with self.assertRaises(VerificationError):
            verify(ops, c.radix, "dit", 1)

    def test_a_wrong_direction_or_variant_is_caught(self) -> None:
        c = next(c for c in codelets() if (c.radix, c.variant, c.sign) == (7, "dif", -1))
        with self.assertRaises(VerificationError):
            verify(c.ops, 7, "dif", 1)
        with self.assertRaises(VerificationError):
            verify(c.ops, 7, "dit", -1)

    def test_a_load_of_twiddle_zero_is_caught(self) -> None:
        # A DIT builder that also multiplies leg 0 computes a correct DFT of its inputs
        # times twiddles, but the tables hold no twiddle for leg 0.
        g = Graph(4, twiddles=3)
        x = [g.tw_mul(g.input(j), g.twiddle(j), False) for j in range(4)]
        ops = schedule(lower(g, dft(g, x, -1)))
        self.assertIn(0, [op.index for op in ops if op.kind == "load_tw"])
        with self.assertRaisesRegex(VerificationError, "load_tw indices"):
            verify(ops, 4, "dit", -1)

    def test_out_of_range_or_repeated_accesses_are_caught(self) -> None:
        c = next(c for c in codelets() if (c.radix, c.variant, c.sign) == (5, "dit", -1))
        ops = list(c.ops)
        fresh = max(op.dst for op in ops) + 1
        first_load = next(i for i, op in enumerate(ops) if op.kind == "load")
        first_tw = next(i for i, op in enumerate(ops) if op.kind == "load_tw")
        first_store = next(i for i, op in enumerate(ops) if op.kind == "store")
        seeded = {
            "input out of range": _with(ops, first_load, index=5),
            "input loaded twice": _insert(ops, first_load, _copy(ops[first_load], fresh)),
            "twiddle out of range": _with(ops, first_tw, index=5),
            "twiddle loaded twice": _insert(ops, first_tw, _copy(ops[first_tw], fresh)),
            "output out of range": _with(ops, first_store, index=5),
            "output stored twice": [*ops, ops[first_store]],
        }
        for name, bad in seeded.items():
            with self.subTest(name), self.assertRaisesRegex(VerificationError, "indices"):
                verify(bad, 5, "dit", -1)

    def test_a_twiddle_load_in_a_plain_codelet_is_caught(self) -> None:
        dit = next(c for c in codelets() if (c.radix, c.variant, c.sign) == (3, "dit", -1))
        with self.assertRaisesRegex(VerificationError, "load_tw indices"):
            verify(dit.ops, 3, "plain", -1)

    def test_a_missing_store_is_caught(self) -> None:
        c = next(c for c in codelets() if (c.radix, c.variant, c.sign) == (3, "plain", -1))
        ops = [op for op in c.ops if not (op.kind == "store" and op.index == 2)]
        with self.assertRaises(VerificationError):
            verify(ops, 3, "plain", -1)


if __name__ == "__main__":
    unittest.main()

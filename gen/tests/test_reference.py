# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Operation counts against the reference table (DESIGN.md 7.3)."""

from __future__ import annotations

import unittest
from math import log2

from fftn_gen.catalogue import RADICES
from fftn_gen.codelet import build
from fftn_gen.reference import (
    GAPS,
    PLAIN,
    MissingReference,
    Reference,
    compare,
    good_thomas,
    reference,
    table,
)

from .test_codelets import codelets


class ReferenceCounts(unittest.TestCase):
    def test_every_codelet_reaches_the_reference_or_its_recorded_gap(self) -> None:
        problems = [p for c in codelets() if (p := compare(c))]
        self.assertEqual(problems, [])

    def test_every_radix_has_an_entry_with_a_source(self) -> None:
        for r in RADICES:
            self.assertIn(r, PLAIN)
            self.assertRegex(PLAIN[r].source, r"^(published|derived|none): ")
            self.assertEqual(PLAIN[r].counts is None, PLAIN[r].source.startswith("none"))

    def test_split_radix_formula(self) -> None:
        # 4n log2 n - 6n + 8 real operations in all (Johnson and Frigo 2007).
        for n in (2, 4, 8, 16, 32, 64):
            adds, muls = PLAIN[n].counts
            self.assertEqual(adds + muls, 4 * n * int(log2(n)) - 6 * n + 8, n)

    def test_split_radix_counts_exactly(self) -> None:
        for c in codelets():
            if c.variant == "plain" and c.radix in (2, 4, 8, 16, 32, 64):
                self.assertEqual((c.counts.flops_adds, c.counts.flops_muls), PLAIN[c.radix].counts)

    def test_recorded_gaps_are_real_gaps(self) -> None:
        for r, (ours, _) in GAPS.items():
            ref = reference(r, "plain")
            self.assertTrue(ours[0] > ref[0] or ours[1] > ref[1], r)


class UnpublishedRadices(unittest.TestCase):
    """A radix with no published count (18 = 2 x 9 here) needs an explicit entry."""

    def setUp(self) -> None:
        self.c = build(18, "plain", -1)

    def test_a_missing_entry_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(MissingReference, "radix 18 has no entry"):
            compare(self.c)
        with self.assertRaises(MissingReference):
            table([self.c])

    def test_a_derived_entry_is_the_sum_of_the_parts_and_is_checked(self) -> None:
        plain = {**PLAIN, 18: good_thomas(2, 9)}
        self.assertEqual(plain[18].counts, (9 * 4 + 2 * 80, 9 * 0 + 2 * 40))
        self.assertIsNone(compare(self.c, plain))
        self.assertIn("derived: Good-Thomas 2 x 9", table([self.c], plain))
        tight = {**PLAIN, 18: Reference((150, 60), "published: a seeded lower count")}
        self.assertIsNotNone(compare(self.c, tight))

    def test_an_entry_marked_none_is_reported_as_unchecked(self) -> None:
        plain = {**PLAIN, 18: Reference(None, "none: no published count")}
        self.assertIsNone(reference(18, "dit", plain))
        self.assertIsNone(compare(self.c, plain))
        self.assertIn("| none | none: no published count |", table([self.c], plain))


if __name__ == "__main__":
    unittest.main()

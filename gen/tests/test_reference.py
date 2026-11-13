# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Operation counts against the reference table (DESIGN.md 7.3)."""

from __future__ import annotations

import unittest

from fftn_gen.catalogue import RADICES
from fftn_gen.reference import GAPS, PLAIN, compare, reference

from .test_codelets import codelets


class ReferenceCounts(unittest.TestCase):
    def test_every_codelet_reaches_the_reference_or_its_recorded_gap(self) -> None:
        problems = [p for c in codelets() if (p := compare(c))]
        self.assertEqual(problems, [])

    def test_every_radix_has_a_reference(self) -> None:
        for r in RADICES:
            self.assertIn(r, PLAIN)

    def test_split_radix_counts_exactly(self) -> None:
        # Johnson and Frigo 2007: 2^k sizes with a 4-multiply complex product.
        for c in codelets():
            if c.variant == "plain" and c.radix in (2, 4, 8, 16):
                self.assertEqual((c.counts.flops_adds, c.counts.flops_muls), PLAIN[c.radix])

    def test_recorded_gaps_are_real_gaps(self) -> None:
        for r, (ours, _) in GAPS.items():
            ref = reference(r, "plain")
            self.assertTrue(ours[0] > ref[0] or ours[1] > ref[1], r)


if __name__ == "__main__":
    unittest.main()

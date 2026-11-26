# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The pruned codelets (task 5.1e): dif_head and plain_head of the even radices."""

from __future__ import annotations

import functools
import unittest

from fftn_gen.catalogue import PRUNED_RADICES, RADICES, all_pruned_codelets
from fftn_gen.codelet import PRUNED_VARIANTS, build_ops, head
from fftn_gen.reference import FULL_OF, memory_accesses
from fftn_gen.schedule import schedule
from fftn_gen.verify import VerificationError, verify

from .test_codelets import codelets


@functools.cache
def pruned() -> tuple:
    return tuple(all_pruned_codelets())


class Pruned(unittest.TestCase):
    def test_every_even_radix_variant_and_direction_is_generated(self) -> None:
        self.assertEqual(PRUNED_RADICES, tuple(r for r in RADICES if r % 2 == 0))
        keys = {(c.radix, c.variant, c.sign) for c in pruned()}
        self.assertEqual(len(keys), len(pruned()))
        self.assertEqual(
            keys, {(r, v, s) for r in PRUNED_RADICES for v in PRUNED_VARIANTS for s in (-1, 1)}
        )

    def test_legs_read_and_written(self) -> None:
        for c in pruned():
            with self.subTest(radix=c.radix, variant=c.variant, sign=c.sign):
                loads = sorted(op.index for op in c.ops if op.kind == "load")
                stores = sorted(op.index for op in c.ops if op.kind == "store")
                h = head(c.radix)
                self.assertEqual(loads, list(range(h if c.variant == "dif_head" else c.radix)))
                self.assertEqual(stores, list(range(h if c.variant == "plain_head" else c.radix)))
                self.assertEqual(c.twiddles, c.radix - 1 if c.variant == "dif_head" else 0)

    def test_never_costs_more_than_the_full_codelet(self) -> None:
        # Halves of even radices only: pruning outputs of an odd radix can cost more than the
        # whole codelet (9 and 25 lose the output pairs that the real-coefficient form shares).
        full = {(c.radix, c.variant, c.sign): c for c in codelets()}
        for c in pruned():
            with self.subTest(radix=c.radix, variant=c.variant, sign=c.sign):
                f = full[(c.radix, FULL_OF[c.variant], c.sign)]
                self.assertLess(c.counts.vector_ops, f.counts.vector_ops)
                self.assertLessEqual(c.counts.flops_adds, f.counts.flops_adds)
                self.assertLessEqual(c.counts.flops_muls, f.counts.flops_muls)
                self.assertEqual(4 * memory_accesses(c), 3 * memory_accesses(f))

    def test_a_full_codelet_is_not_a_pruned_one(self) -> None:
        # The accessor contract refuses a codelet that reads a leg known zero or stores an
        # output not wanted, and the values refuse a pruned codelet passed off as full.
        for radix in (2, 8, 12):
            dif = schedule(build_ops(radix, "dif", -1))
            plain = schedule(build_ops(radix, "plain", 1))
            with self.assertRaises(VerificationError):
                verify(dif, radix, "dif_head", -1)
            with self.assertRaises(VerificationError):
                verify(plain, radix, "plain_head", 1)
            with self.assertRaises(VerificationError):
                verify(schedule(build_ops(radix, "dif_head", -1)), radix, "dif", -1)

    def test_every_input_load_precedes_every_store(self) -> None:
        # So that in and out may be the same memory (gen/README.md, "Calling convention").
        for c in pruned():
            kinds = [op.kind for op in c.ops]
            last_load = max(i for i, k in enumerate(kinds) if k == "load")
            self.assertLess(last_load, kinds.index("store"), (c.radix, c.variant, c.sign))


if __name__ == "__main__":
    unittest.main()

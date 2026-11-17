# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Two-level codelets (staged.py): verified, register-fitting, emitted behind a fence."""

from __future__ import annotations

import dataclasses
import functools
import unittest

from fftn_gen.catalogue import STAGED, STAGED_VARIANTS, all_codelets, all_staged_codelets, generate
from fftn_gen.codelet import SIGNS
from fftn_gen.lower import Op
from fftn_gen.schedule import peak_live
from fftn_gen.verify import VerificationError, verify


@functools.cache
def staged() -> tuple:
    return tuple(all_staged_codelets())


class Staged(unittest.TestCase):
    def test_every_staged_radix_variant_and_direction_is_built(self) -> None:
        keys = {(c.radix, c.variant, c.sign) for c in staged()}
        want = {(r, v, s) for r in STAGED for v in STAGED_VARIANTS for s in SIGNS}
        self.assertEqual(keys, want)

    def test_each_level_fits_sixteen_registers(self) -> None:
        # The point of the form: the flat codelets peak at 29, 35 and 67 live values.
        flat = {(c.radix, c.variant, c.sign): c.live for c in all_codelets()}
        for c in staged():
            self.assertLessEqual(c.live, 16, (c.radix, c.variant, c.sign))
            self.assertGreater(flat[(c.radix, c.variant, c.sign)], 16)

    def test_slots_are_stored_once_before_any_is_loaded(self) -> None:
        for c in staged():
            kinds = [op.kind for op in c.ops]
            cut = kinds.index("stage_load")
            self.assertNotIn("stage_store", kinds[cut:])
            self.assertNotIn("load", kinds[cut:], "every input in level A: in place stays safe")
            self.assertNotIn("store", kinds[:cut])
            stored = sorted(op.index for op in c.ops if op.kind == "stage_store")
            loaded = sorted(op.index for op in c.ops if op.kind == "stage_load")
            self.assertEqual(stored, list(range(c.radix)))
            self.assertEqual(loaded, list(range(c.radix)))
            self.assertEqual(c.live, max(peak_live(c.ops[:cut]), peak_live(c.ops[cut:])))

    def test_a_wrong_slot_is_caught(self) -> None:
        c = next(c for c in staged() if (c.radix, c.variant, c.sign) == (32, "dif", -1))
        ops = list(c.ops)
        i = next(i for i, op in enumerate(ops) if op.kind == "stage_load")
        j = next(j for j, op in enumerate(ops) if op.kind == "stage_load" and j > i)
        ops[i], ops[j] = (
            dataclasses.replace(ops[i], index=ops[j].index),
            dataclasses.replace(ops[j], index=ops[i].index),
        )
        with self.assertRaises(VerificationError):
            verify(ops, 32, "dif", -1)

    def test_a_slot_stored_twice_is_caught(self) -> None:
        c = next(c for c in staged() if (c.radix, c.variant, c.sign) == (25, "plain", 1))
        ops = list(c.ops)
        i = next(i for i, op in enumerate(ops) if op.kind == "stage_store")
        with self.assertRaises(VerificationError):
            verify([*ops[: i + 1], ops[i], *ops[i + 1 :]], 25, "plain", 1)

    def test_emitted_behind_a_fence_on_sixteen_register_targets(self) -> None:
        text = generate()["complex_codelets.inl"]
        self.assertIn('asm("" : "+m"(st));', text)
        for r in STAGED:
            body = text.split(f"template <> struct cdft<{r}> {{")[1].split("};")[0]
            # plain and dif, each direction: one array and one fence each.
            self.assertEqual(body.count("gen_fence(st);"), 4, r)
            self.assertEqual(body.count("if constexpr (codelet_registers <= 16) {"), 2, r)
        self.assertIsInstance(staged()[0].ops[0], Op)


if __name__ == "__main__":
    unittest.main()

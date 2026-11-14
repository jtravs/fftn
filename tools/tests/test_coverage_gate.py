# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""coverage_gate's choice of the target files a processor cannot run."""

from __future__ import annotations

import unittest

from coverage_gate import target_files, unrunnable


class Unrunnable(unittest.TestCase):
    def test_an_avx2_machine_cannot_run_avx512(self) -> None:
        self.assertEqual(unrunnable("avx2"), ["avx512"])
        self.assertEqual(
            target_files(unrunnable("avx2")), ("src/isa/vocab_avx512.inl", "src/isa/avx512.cpp")
        )

    def test_the_best_of_a_family_runs_everything_below_it(self) -> None:
        self.assertEqual(unrunnable("avx512"), [])
        self.assertEqual(unrunnable("neon_fcma"), [])

    def test_a_neon_machine_cannot_run_neon_fcma(self) -> None:
        self.assertEqual(unrunnable("neon"), ["neon_fcma"])

    def test_scalar_or_unknown_leaves_nothing_out(self) -> None:
        # A scalar best target says nothing about which family the processor belongs to; the
        # other family's files are not compiled there anyway.
        self.assertEqual(unrunnable("scalar"), [])
        self.assertEqual(unrunnable(""), [])


if __name__ == "__main__":
    unittest.main()

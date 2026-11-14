# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X2's candidates, cost estimate, cases and stage bookkeeping (bench/x2/)."""

import contextlib
import io
import math
import re
import unittest
from collections import Counter

from fbenchlib import catalogue
from x2 import fit, radices, search


def multinomial(seq: tuple[int, ...]) -> int:
    out = math.factorial(len(seq))
    for count in Counter(seq).values():
        out //= math.factorial(count)
    return out


class Enumeration(unittest.TestCase):
    def test_small_lengths_by_hand(self):
        self.assertEqual(radices.sequences(1), [])
        self.assertEqual(radices.sequences(7), [(7,)])
        self.assertEqual(sorted(radices.sequences(8)), [(2, 2, 2), (2, 4), (4, 2), (8,)])
        self.assertEqual(
            sorted(radices.sequences(12)),
            [(2, 2, 3), (2, 3, 2), (2, 6), (3, 2, 2), (3, 4), (4, 3), (6, 2), (12,)],
        )
        self.assertEqual(len(radices.sequences(16)), 8)
        self.assertEqual(sorted(radices.sequences(25)), [(5, 5)])

    def test_products_and_radices(self):
        for n in (60, 64, 210, 720, 1024, 2431, 65535):
            for seq in radices.sequences(n):
                self.assertEqual(math.prod(seq), n)
                for r in seq:
                    self.assertTrue(2 <= r <= 16 or (r > 16 and radices.factor(r) == {r: 1}))

    def test_every_distinct_order_once(self):
        for n in (48, 360, 720, 960):
            seqs = radices.sequences(n)
            self.assertEqual(len(seqs), len(set(seqs)))
            by_multiset = Counter(tuple(sorted(s)) for s in seqs)
            for multiset, orders in by_multiset.items():
                self.assertEqual(orders, multinomial(multiset), multiset)

    def test_at_most_eight_passes(self):
        unlimited = radices.sequences(1024, max_passes=10)
        limited = radices.sequences(1024)
        self.assertEqual(max(len(s) for s in limited), 8)
        self.assertEqual(set(limited), {s for s in unlimited if len(s) <= 8})
        self.assertIn((2,) * 10, unlimited)
        self.assertEqual(radices.sequences(2**9, max_passes=2), [])

    def test_primes_above_sixteen_are_generic_passes(self):
        self.assertEqual(sorted(radices.sequences(34)), [(2, 17), (17, 2)])
        self.assertEqual(sorted(radices.sequences(4097)), [(17, 241), (241, 17)])

    def test_candidates_include_v0(self):
        ops = radices.codelet_ops()
        v0 = (4,) * 12  # 2^24 on a target without rule 3: more passes than the search allows
        long = radices.candidates(1 << 24, v0, ops)
        self.assertEqual(len(long), radices.LONG_TOP + 1)
        self.assertEqual(long[-1], v0)
        self.assertEqual(long[0], (16,) * 6)
        small = radices.candidates(64, (4, 16), ops)
        self.assertEqual(small, radices.sequences(64))


class Cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = radices.codelet_ops()

    def test_codelet_table(self):
        self.assertEqual(set(r for r, _ in self.ops), set(range(2, 17)) | {32, 64})
        # The radix-16 forward codelets of src/gen/codelets.hpp: plain and DIF.
        self.assertEqual(self.ops[(16, False)], 80)
        self.assertEqual(self.ops[(16, True)], 95)

    def test_estimate(self):
        # A leaf is one plain codelet; every pass but the last applies twiddles.
        self.assertEqual(radices.cost((16,), self.ops), (1, 80))
        self.assertEqual(radices.cost((4, 16), self.ops), (2, 16 * 11 + 4 * 80))
        self.assertEqual(radices.cost((2, 17), self.ops), (2, 17 * 3 + 2 * 2 * 17 * 17))
        self.assertEqual(
            radices.cost((17, 2), self.ops), (2, 2 * (2 * 17 * 17 + 16) + 17 * self.ops[(2, False)])
        )
        # Fewer passes rank first, then fewer operations.
        ranked = sorted(radices.sequences(4096), key=lambda s: radices.cost(s, self.ops))
        self.assertEqual(ranked[0], (16, 16, 16))
        self.assertTrue(all(len(s) >= 3 for s in ranked))

    def test_shortlist_is_the_best_twenty(self):
        top = radices.shortlist(46080, None, self.ops)
        self.assertEqual(len(top), radices.LONG_TOP)
        worst = max(radices.cost(s, self.ops) for s in top)
        rest = set(radices.sequences(46080)) - set(top)
        self.assertTrue(all(radices.cost(s, self.ops) >= worst for s in rest))


class Cases(unittest.TestCase):
    def test_catalogue_is_generated(self):
        std = catalogue.load("standard")
        text = radices.catalogue_text(
            radices.smooth_lengths(), radices.long_lengths([(c.kind, c.dims) for c in std.cases])
        )
        self.assertEqual(search.CATALOGUE.read_text(), text, "run bench/x2/search.py catalogue")

    def test_case_set(self):
        cases = search.x2_cases()
        ns = search.lengths_of(cases)
        self.assertNotIn(7, ns)  # one sequence only
        self.assertNotIn(3125, ns)  # 5^5: radix 25 is no codelet
        self.assertIn(1024, ns)
        self.assertIn(16777216, ns)
        self.assertTrue(all(radices.smooth(n) for n in ns if n <= 1024))
        self.assertEqual({c.kind for c in cases}, {"c2c"})
        self.assertEqual({c.sign for c in cases}, {-1})
        self.assertEqual({c.precision for c in cases}, {"f64", "f32"})
        self.assertEqual(search.batches_of(960, cases), [1, 64])
        self.assertEqual(search.batches_of(1 << 20, cases), [1])
        ids = [c.id for c in cases if re.search(radices.length_filter(64), c.id)]
        self.assertEqual(
            sorted(ids),
            ["c2c-64-b1-op", "c2c-64-b1-op-f32", "c2c-64-b64-op", "c2c-64-b64-op-f32"],
        )

    def test_lengths_option(self):
        ns = [4, 60, 64, 1024, 2048, 1 << 20]
        self.assertEqual(search.select_lengths(ns, "-1024"), [4, 60, 64, 1024])
        self.assertEqual(search.select_lengths(ns, "1025-"), [2048, 1 << 20])
        self.assertEqual(search.select_lengths(ns, "60,2048"), [60, 2048])
        with self.assertRaises(SystemExit):
            search.select_lengths(ns, "61")

    def test_labels_and_notes(self):
        self.assertEqual(radices.label((4, 8, 8)), "4.8.8")
        self.assertEqual(radices.parse("4.8.8"), (4, 8, 8))
        self.assertEqual(radices.from_note("axis 0 (n = 256): radices 4 x 8 x 8"), (256, (4, 8, 8)))
        self.assertEqual(radices.from_note("axis 0 (n = 16): radices 16"), (16, (16,)))
        with self.assertRaises(ValueError):
            radices.from_note("axis 0 (n = 100): r2c through a half-length complex chain")


class Stages(unittest.TestCase):
    @staticmethod
    def records(times: dict[tuple[str, str], list[float]], valid=(0, 1)) -> list[dict]:
        out = [{"type": "round", "round": r, "valid": True} for r in valid]
        for (arm, case), values in times.items():
            for rnd, ns in enumerate(values):
                out.append(
                    {
                        "type": "timing",
                        "round": rnd,
                        "arm": arm,
                        "case": case,
                        "status": "ok",
                        "check": {"pass": True},
                        "median_ns": ns,
                    }
                )
        return out

    def test_paired_medians_and_ranking(self):
        recs = self.records(
            {
                ("v0", "c"): [100.0, 120.0],
                ("v0~", "c"): [101.0, 119.0],
                ("4.16", "c"): [90.0, 108.0],
                ("16.4", "c"): [110.0, 130.0],
                ("8.8", "c"): [80.0],  # missing a round: not ranked
            }
        )
        med = search.paired_medians(recs)
        self.assertAlmostEqual(med["4.16"]["c"], math.log(0.9))
        self.assertNotIn("8.8", med)
        order = [arm for _, arm in search.ranked(med, "c")]
        self.assertEqual(order, ["4.16", "v0", "16.4"])

    def test_wrong_output_is_never_a_winner(self):
        recs = self.records({("v0", "c"): [100.0, 100.0], ("2.32", "c"): [50.0, 50.0]})
        recs[-1]["check"] = {"pass": False}
        self.assertNotIn("2.32", search.paired_medians(recs))

    def test_fbench_command(self):
        cmd = search.fbench_command(
            search.REPO / "bench-out", search.DEFAULT_BUILD, 64, ["4.16"], search.SCREEN, []
        )
        self.assertIn("v0=fftn", cmd)
        self.assertIn("4.16=fftn+RADICES=4.16", cmd)
        self.assertEqual(cmd[cmd.index("--filter") + 1], "^c2c-64-b")
        self.assertEqual(cmd[cmd.index("--tier") + 1], "full")
        self.assertEqual(cmd[cmd.index("--max-rounds") + 1], str(search.SCREEN_ROUNDS))


class Isa(unittest.TestCase):
    def test_every_arm_is_forced(self):
        cmd = search.fbench_command(
            search.REPO / "bench-out", search.DEFAULT_BUILD, 64, ["4.16"], search.SCREEN, [], "avx512"
        )
        self.assertIn("v0=fftn+ISA=avx512", cmd)
        self.assertIn("4.16=fftn+ISA=avx512+RADICES=4.16", cmd)
        self.assertEqual(search.isa_options("avx512"), "ISA=avx512")
        self.assertEqual(search.isa_options(None), "")

    def test_estimate_padded_to_c(self):
        ops = radices.codelet_ops()
        self.assertEqual(radices.cost((2, 16), ops, 1), radices.cost((2, 16), ops))
        # n = 32, C = 8: the first pass's 16 butterflies are 2 units; the last pass has
        # s = 2, m = 1, so its 2 butterflies fill one unit of 8.
        self.assertEqual(
            radices.cost((2, 16), ops, 8), (2, 16 * ops[(2, True)] + 8 * ops[(16, False)])
        )
        one = radices.shortlist(46080, None, ops, lanes=(4,))
        both = radices.shortlist(46080, None, ops, lanes=(4, 8))
        self.assertEqual(both[: len(one)], one)
        self.assertTrue(radices.LONG_TOP <= len(both) <= 2 * radices.LONG_TOP)

    def test_campaign_keeps_its_isa(self):
        state = {"name": "x", "lengths": {"64": {}}, "isa": "avx2"}
        search.check_isa(state, "avx2")
        for other in ("avx512", None):
            with self.assertRaises(SystemExit):
                search.check_isa(state, other)
        search.check_isa({"name": "x", "lengths": {"64": {}}}, None)  # before --isa existed
        search.check_isa({"name": "x", "lengths": {}}, "avx512")  # a new campaign

    def test_remote_lines(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            search.main(["remote", "xeon", "--isa", "avx512"])
        lines = [x for x in out.getvalue().splitlines() if not x.startswith("#")]
        self.assertEqual(len(lines), 2)
        self.assertIn("--name xeon-avx512-smooth --lengths 2-1024 --isa avx512", lines[0])
        self.assertIn("--name xeon-avx512-long", lines[1])


class Fit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = fit.parse_rules(fit.RULES.read_text())

    def test_search_is_the_planners(self):
        """fit.py's port of rule_radices on rules.cpp's AVX2 tables gives the recorded
        radices (tests/core/data/rule_radices.txt) for C = 2 and C = 4."""
        self.assertEqual(len(self.tables["avx2_double"].overrides), 6)
        path = search.REPO / "tests" / "core" / "data" / "rule_radices.txt"
        checked = 0
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            n, *cols = (x.split() for x in line.split("|"))
            n = int(n[0])
            for c, name, col in ((2, "avx2_double", cols[1]), (4, "avx2_float", cols[2])):
                got = fit.rule_radices(n, self.tables[name], c)
                self.assertEqual(got, tuple(int(r) for r in col), f"n = {n}, C = {c}")
            checked += 1
        self.assertEqual(checked, 5208)

    def test_fit_recovers_a_synthetic_table(self):
        """Timings made from a known table (rules.cpp's float one at C = 4, plus a constant
        per transform): the fit picks every length's best sequence again."""
        truth = self.tables["avx2_float"]
        truth = fit.Table(truth.unit, truth.tail, truth.passes)
        ops = radices.codelet_ops()

        def cost(seq):
            f = fit.features(seq, 4)
            x = [0.0] * fit.NF
            for k in range(fit.KINDS):
                for r in fit.RADICES:
                    x[k * fit.NR + r - 2] = truth.unit[k][r - 2]
                    x[fit.TAIL0 + k * fit.NR + r - 2] = truth.tail[k][r - 2]
            x[fit.PASS], x[fit.CONST] = truth.passes, 5000.0
            return sum(v * x[j] for j, v in f.items())

        cases = []
        for n in (32, 48, 60, 64, 72, 96, 100, 120, 128, 144, 160, 192, 240, 256):
            seqs = [s for s in radices.sequences(n) if len(s) > 1]
            base = seqs[0]
            ratio = {s: math.log(cost(s) / cost(base)) for s in seqs}
            best = min(seqs, key=lambda s: (ratio[s], s))
            cases.append(fit.Case(n, f"c2c-{n}-b1-op-f32", "f32", base, best, refine=ratio))
        x = fit.fit(fit.rows_of(cases, 4), ops)
        table = fit.table_of(x, cases, ops)
        per, misses = fit.reproduction(cases, table, 4, 0.0)
        self.assertEqual(per["b1-op-f32"][:2], [len(cases), len(cases)], misses)


if __name__ == "__main__":
    unittest.main()

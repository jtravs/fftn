# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import random
import statistics
import unittest

from fbenchlib import stats
from fbenchlib.stats import Interval


class WilliamsOrders(unittest.TestCase):
    def test_balance(self):
        for k in range(1, 8):
            rows = stats.williams_orders(k)
            self.assertEqual(len(rows), k if k % 2 == 0 or k == 1 else 2 * k)
            for row in rows:
                self.assertEqual(sorted(row), list(range(k)))
            # Every arm in every position equally often.
            for pos in range(k):
                counts = [sum(1 for r in rows if r[pos] == a) for a in range(k)]
                self.assertEqual(len(set(counts)), 1, (k, pos, counts))
            # Every ordered pair adjacent equally often (first-order carry-over).
            pairs = {p: 0 for p in itertools.permutations(range(k), 2)}
            for r in rows:
                for a, b in itertools.pairwise(r):
                    pairs[(a, b)] += 1
            self.assertLessEqual(len(set(pairs.values())), 1, (k, pairs))

    def test_rejects_no_arms(self):
        with self.assertRaises(ValueError):
            stats.williams_orders(0)


class Pairing(unittest.TestCase):
    def test_only_common_rounds_in_order(self):
        base = {0: 100.0, 1: 200.0, 3: 50.0}
        other = {3: 100.0, 0: 110.0, 2: 1.0}
        self.assertEqual(stats.paired_log_ratios(base, other), [math.log(1.1), math.log(2.0)])

    def test_pairing_removes_a_shared_round_effect(self):
        # A machine state shared by both arms of a round (here up to 2x) cancels exactly.
        rng = random.Random(3)
        base, other = {}, {}
        for r in range(10):
            machine = 1.0 + rng.random()
            base[r] = 100.0 * machine
            other[r] = 105.0 * machine
        for v in stats.paired_log_ratios(base, other):
            self.assertAlmostEqual(v, math.log(1.05))


class Bootstrap(unittest.TestCase):
    def test_deterministic_and_contains_the_median(self):
        rng = random.Random(1)
        xs = [rng.gauss(0.0, 1.0) for _ in range(25)]
        a = stats.bootstrap_ci(xs, random.Random(7))
        b = stats.bootstrap_ci(xs, random.Random(7))
        self.assertEqual(a, b)
        self.assertLess(a[0], statistics.median(xs))
        self.assertGreater(a[1], statistics.median(xs))

    def test_narrows_with_more_rounds(self):
        rng = random.Random(2)
        few = [rng.gauss(0.0, 1.0) for _ in range(8)]
        many = [rng.gauss(0.0, 1.0) for _ in range(200)]
        lo1, hi1 = stats.bootstrap_ci(few, random.Random(0))
        lo2, hi2 = stats.bootstrap_ci(many, random.Random(0))
        self.assertLess(hi2 - lo2, hi1 - lo1)

    def test_coverage(self):
        # The 95 % interval of the median covers the true median in most repetitions.
        rng = random.Random(4)
        covered = 0
        trials = 200
        for _ in range(trials):
            xs = [rng.gauss(0.0, 1.0) for _ in range(15)]
            lo, hi = stats.bootstrap_ci(xs, rng, resamples=400)
            covered += lo <= 0.0 <= hi
        self.assertGreater(covered / trials, 0.85)

    def test_degenerate(self):
        self.assertEqual(stats.bootstrap_ci([0.5], random.Random(0)), (0.5, 0.5))
        with self.assertRaises(ValueError):
            stats.bootstrap_ci([], random.Random(0))


class BandAndVerdict(unittest.TestCase):
    def test_band_floor_and_quantile(self):
        floor = math.log1p(0.01)
        self.assertEqual(stats.noise_band([], 0.01), floor)
        tiny = [Interval(0.001, -0.01, 0.01, 10)] * 4
        self.assertEqual(stats.noise_band(tiny, 0.01), floor)
        # The 95th percentile of |median| over cases, signs ignored.
        spread = [Interval(m, m - 0.1, m + 0.1, 10) for m in (0.01, -0.02, 0.03, -0.05, 0.09)]
        self.assertAlmostEqual(stats.noise_band(spread, 0.0), 0.05 + 0.8 * 0.04)
        self.assertAlmostEqual(stats.noise_band(spread, 0.0, q=0.75), 0.05)
        # Intervals from too few rounds do not set the band.
        self.assertEqual(stats.noise_band([Interval(1.0, 0.5, 1.5, 2)], 0.01), floor)

    def test_verdicts(self):
        band = 0.02
        self.assertEqual(stats.verdict(Interval(0.05, 0.03, 0.07, 8), band), stats.SLOWER)
        self.assertEqual(stats.verdict(Interval(-0.05, -0.07, -0.03, 8), band), stats.FASTER)
        # The interval clears zero but not the band: not enough.
        self.assertEqual(stats.verdict(Interval(0.025, 0.01, 0.04, 8), band), stats.NO_EVIDENCE)
        self.assertEqual(stats.verdict(Interval(0.5, 0.4, 0.6, 2), band), stats.NO_EVIDENCE)

    def test_resolved(self):
        band = 0.02
        self.assertTrue(stats.resolved(Interval(0.05, 0.03, 0.07, 8), band))
        self.assertTrue(stats.resolved(Interval(0.0, -0.01, 0.015, 8), band))
        self.assertFalse(stats.resolved(Interval(0.01, -0.01, 0.03, 8), band))
        self.assertFalse(stats.resolved(Interval(0.0, 0.0, 0.0, 2), band))

    def test_quantile(self):
        self.assertEqual(stats.quantile([3.0], 0.75), 3.0)
        self.assertAlmostEqual(stats.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.75), 4.0)
        self.assertAlmostEqual(stats.quantile([0.0, 1.0], 0.25), 0.25)


class GeomeanInterval(unittest.TestCase):
    def test_weights_and_interval(self):
        per_round = {
            "a": {r: math.log(1.10) for r in range(6)},
            "b": {r: math.log(1.00) for r in range(6)},
        }
        g = stats.weighted_geomean_interval(per_round, {"a": 3.0, "b": 1.0}, random.Random(0))
        self.assertIsNotNone(g)
        self.assertAlmostEqual(g.median, 0.75 * math.log(1.10))
        self.assertAlmostEqual(g.lo, g.median)
        self.assertAlmostEqual(g.hi, g.median)
        self.assertIsNone(stats.weighted_geomean_interval({}, {}, random.Random(0)))


if __name__ == "__main__":
    unittest.main()

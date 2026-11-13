# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The verdict pipeline on synthetic runs with known truth.

Noise model: each process sees a per-round machine factor shared by all arms (up to tens of
percent, which pairing must remove) and its own lognormal process-to-process noise.
"""

import math
import random
import unittest

from fbenchlib import analysis, stats

ARMS = ["A", "A~", "B"]


def synth(
    factor_b: float,
    seed: int,
    cases: int = 20,
    rounds: int = 12,
    sigma: float = 0.03,
    round_sigma: float = 0.10,
) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = [
        {
            "type": "run",
            "run": "t",
            "settings": {"seed": seed},
            "arms": [{"label": a, "control": a == "A~"} for a in ARMS],
            "cases": [{"id": f"c{i}", "weight": 1.0} for i in range(cases)],
        }
    ]
    base = [10.0 ** rng.uniform(1, 6) for _ in range(cases)]
    for r in range(rounds):
        records.append({"type": "round", "round": r, "valid": True})
        machine = math.exp(rng.gauss(0.0, round_sigma))
        for arm in ARMS:
            factor = factor_b if arm == "B" else 1.0
            for i in range(cases):
                t = base[i] * factor * machine * math.exp(rng.gauss(0.0, sigma))
                records.append(timing(r, arm, f"c{i}", t))
    return records


def timing(r: int, arm: str, case: str, t: float, ok: bool = True) -> dict:
    return {
        "type": "timing",
        "round": r,
        "arm": arm,
        "case": case,
        "status": "ok",
        "median_ns": t,
        "min_ns": t * 0.99,
        "check": {"pass": ok, "err_rms": 1e-16 if ok else 1e-6},
    }


def verdicts(result: analysis.Analysis) -> set[str]:
    return {r.verdict for r in result.results["B"].values()}


class Verdicts(unittest.TestCase):
    def test_identical_arms_give_no_evidence(self):
        for seed in range(5):
            result = analysis.analyse(synth(1.0, seed), min_band=0.01, seed=seed)
            self.assertEqual(verdicts(result), {stats.NO_EVIDENCE}, seed)
            self.assertGreaterEqual(result.band, math.log1p(0.01))

    def test_five_percent_slowdown_is_detected(self):
        covered = total = 0
        for seed in range(5):
            result = analysis.analyse(synth(1.05, seed, rounds=30, sigma=0.02), seed=seed)
            self.assertEqual(verdicts(result), {stats.SLOWER}, seed)
            for r in result.results["B"].values():
                covered += math.exp(r.interval.lo) <= 1.05 <= math.exp(r.interval.hi)
                total += 1
        # The 95 % intervals contain the true ratio in most cases.
        self.assertGreater(covered / total, 0.85)

    def test_speedup_is_detected(self):
        result = analysis.analyse(synth(0.95, 1, rounds=30, sigma=0.02), seed=1)
        self.assertEqual(verdicts(result), {stats.FASTER})

    def test_geomean(self):
        result = analysis.analyse(synth(1.05, 2, rounds=30, sigma=0.02), seed=2)
        g = result.geomean["B"]
        self.assertLess(math.exp(g.lo), 1.05)
        self.assertGreater(math.exp(g.hi), 1.05)

    def test_noisier_control_widens_the_band(self):
        quiet = analysis.analyse(synth(1.0, 3, sigma=0.01), seed=3)
        noisy = analysis.analyse(synth(1.0, 3, sigma=0.08), seed=3)
        self.assertGreater(noisy.band, quiet.band)

    def test_without_control_the_floor_is_the_band(self):
        records = [r for r in synth(1.0, 4) if r.get("arm") != "A~"]
        records[0]["arms"] = [a for a in records[0]["arms"] if not a["control"]]
        result = analysis.analyse(records, min_band=0.02, seed=4)
        self.assertIsNone(result.control)
        self.assertAlmostEqual(result.band, math.log1p(0.02))

    def test_invalid_rounds_are_ignored(self):
        records = synth(1.0, 5)
        records.append({"type": "round", "round": 99, "valid": False})
        records += [timing(99, "B", f"c{i}", 1e12) for i in range(20)]
        result = analysis.analyse(records, seed=5)
        self.assertEqual(verdicts(result), {stats.NO_EVIDENCE})
        self.assertEqual(result.invalid_rounds, 1)


class Guard(unittest.TestCase):
    def test_wrong_output_in_any_round_is_flagged(self):
        records = synth(1.0, 6)
        records.append({"type": "round", "round": 50, "valid": False})
        records.append(timing(50, "B", "c3", 100.0, ok=False))
        result = analysis.analyse(records, seed=6)
        self.assertEqual(result.results["B"]["c3"].verdict, analysis.WRONG)
        self.assertEqual(result.results["B"]["c4"].verdict, stats.NO_EVIDENCE)

    def test_wrong_baseline_is_flagged(self):
        records = synth(1.0, 7)
        records.append(timing(0, "A", "c1", 100.0, ok=False))
        result = analysis.analyse(records, seed=7)
        self.assertEqual(result.results["B"]["c1"].verdict, analysis.BASE_WRONG)

    def test_unsupported_is_reported(self):
        records = synth(1.0, 8)
        records = [
            r
            for r in records
            if not (r["type"] == "timing" and r["arm"] == "B" and r["case"] == "c2")
        ]
        records.append(
            {"type": "timing", "round": 0, "arm": "B", "case": "c2", "status": "unsupported"}
        )
        result = analysis.analyse(records, seed=8)
        self.assertEqual(result.results["B"]["c2"].verdict, "unsupported")


class Stopping(unittest.TestCase):
    def test_unresolved_until_enough_rounds(self):
        few = analysis.analyse(synth(1.03, 9, rounds=4, sigma=0.04), seed=9)
        self.assertTrue(few.unresolved({f"c{i}" for i in range(20)}))
        self.assertEqual(few.unresolved(set()), [])
        many = analysis.analyse(synth(1.05, 9, rounds=30, sigma=0.01), seed=9)
        self.assertEqual(many.unresolved({f"c{i}" for i in range(20)}), [])


if __name__ == "__main__":
    unittest.main()

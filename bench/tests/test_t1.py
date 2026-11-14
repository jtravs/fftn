# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path

# The T1 driver is a script beside its harness (bench/t1/), not part of fbenchlib.
_spec = importlib.util.spec_from_file_location(
    "t1_driver", Path(__file__).resolve().parent.parent / "t1" / "t1.py"
)
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)


def probe(name: str, threads: int, median: float, variant: str = "hot") -> dict:
    stats = {"median": median, "p10": median, "p90": median, "min": median, "max": median}
    return {
        "type": "probe",
        "probe": name,
        "executor": "thread_pool",
        "threads": threads,
        "group": 0,
        "variant": variant,
        "reps": 10,
        "metrics": {"round_trip": stats},
    }


class Summary(unittest.TestCase):
    def test_median_over_rounds_and_spread(self):
        processes = [
            {"binary": "t1_costs", "lines": [probe("bulk", 8, m), probe("bulk", 16, 2 * m)]}
            for m in (1000.0, 1200.0, 1100.0)
        ]
        rows = t1.summarise(processes)
        self.assertEqual(len(rows), 2)
        by_threads = {r["threads"]: r for r in rows}
        self.assertEqual(by_threads[8]["median"], 1100.0)
        self.assertEqual(by_threads[8]["rounds"], 3)
        self.assertEqual((by_threads[8]["low"], by_threads[8]["high"]), (1000.0, 1200.0))
        self.assertAlmostEqual(by_threads[8]["spread"], 200.0 / 1100.0)
        self.assertEqual(by_threads[16]["median"], 2200.0)
        # Rows sort by team size numerically, not as text.
        self.assertEqual([r["threads"] for r in rows], [8, 16])

    def test_binaries_and_variants_kept_apart(self):
        processes = [
            {"binary": "t1_costs", "lines": [probe("bulk", 4, 10.0, "hot")]},
            {"binary": "t1_costs_omp", "lines": [probe("bulk", 4, 20.0, "hot")]},
            {"binary": "t1_costs", "lines": [probe("bulk", 4, 30.0, "parked")]},
        ]
        self.assertEqual(len(t1.summarise(processes)), 3)

    def test_complete(self):
        lines = [{"type": "describe"}] + [
            probe(p, 2, 1.0) for p in ("work", "bulk", "team", "phase", "claim", "wake")
        ]
        good = [{"binary": "t1_costs", "lines": lines}]
        self.assertEqual(t1.check_complete(good, {"t1_costs": Path("x")}), [])
        partial = [{"binary": "t1_costs", "lines": lines[:-1]}]
        self.assertIn("wake", t1.check_complete(partial, {"t1_costs": Path("x")})[0])
        # The OpenMP binary must also report OpenMP executors.
        omp = [{"binary": "t1_costs_omp", "lines": lines}]
        problems = t1.check_complete(omp, {"t1_costs_omp": Path("x")})
        self.assertEqual(problems, ["t1_costs_omp: no OpenMP probes"])


if __name__ == "__main__":
    unittest.main()

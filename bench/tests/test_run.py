# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""`run` end to end on a stand-in backend: load is recorded rather than refused, strict mode
still refuses it, the overload pause is bounded, and a process timeout costs only the cases
it reached."""

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fbenchlib import analysis, cli, machine, store

# A stand-in backend speaking the harness protocol. Option HANG=<case id> makes it stall on
# that case (so the driver's process timeout ends it); every case takes 1000 ns.
FAKE = """#!{python}
import json, sys, time
desc = {{"type": "describe", "library": "fake", "version": "1", "isa": "none",
         "harness": {{"build_type": "Release", "compiler": "none", "commit": "0"}}}}
if sys.argv[1] == "describe":
    print(json.dumps(desc))
    sys.exit(0)
args = sys.argv[2:]
opts = dict(t.split("=", 1) for t in args[args.index("--options") + 1].split())
lines = open(args[args.index("--cases") + 1]).read().splitlines()
print(json.dumps(desc), flush=True)
for line in filter(None, lines):
    f = dict(t.split("=", 1) for t in line.split())
    if opts.get("HANG") == f["id"]:
        time.sleep(60)
    check = {{"pass": True, "err_rms": 0, "err_max": 0, "tol": 1}}
    print(json.dumps({{"type": "case", "id": f["id"], "status": "ok", "median_ns": 1000.0,
                      "min_ns": 1000.0, "check": check}}), flush=True)
print(json.dumps({{"type": "end"}}))
"""

CATALOGUE = """[catalogue]
name = "test"

[[group]]
name = "rows"
dims = [16, 32, 64, 128]
tiers = ["quick"]
"""

QUIET = {"load1": 0.0, "busy": 0.0}


@unittest.skipIf(os.name == "nt", "the stand-in backend is a #! script")
class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        exe = self.root / "build" / "bench" / "fbench-fake"
        exe.parent.mkdir(parents=True)
        exe.write_text(FAKE.format(python=sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        (self.root / "test.toml").write_text(CATALOGUE)
        self.out = self.root / "bench-out"

    def tearDown(self):
        self.tmp.cleanup()

    def run_fbench(self, arms: list[str], *extra: str, states=None) -> tuple[list[dict], str]:
        """Runs `fbench run` with load_state returning `states` in turn (the last repeats)."""
        states = list(states or [QUIET])

        def load_state():
            return dict(states.pop(0) if len(states) > 1 else states[0])

        argv = [
            "--out-dir",
            str(self.out),
            "run",
            *arms,
            "--catalogue",
            str(self.root / "test.toml"),
            "--build",
            str(self.root / "build"),
            "--min-rounds",
            "2",
            "--max-rounds",
            "2",
            "--seed",
            "1",
            "--busy-wait",
            "0",
            *extra,
        ]
        stdout = io.StringIO()
        with (
            mock.patch.object(machine, "load_state", load_state),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            status = cli.main(argv)
        self.assertEqual(status, 0)
        return store.load(store.find_run(self.out, "latest")), stdout.getvalue()

    @staticmethod
    def rounds(records: list[dict]) -> list[dict]:
        return [r for r in records if r["type"] == "round"]

    def test_loaded_round_is_kept_and_recorded(self):
        loaded = {"load1": 1000.0, "busy": 0.0}  # far above --max-load, no overload
        records, report = self.run_fbench(["A=fake", "B=fake+X=1"], states=[loaded])
        rounds = self.rounds(records)
        self.assertEqual(len(rounds), 2)
        for r in rounds:
            self.assertTrue(r["valid"])
            self.assertTrue(r["loaded"])
            self.assertFalse(r["overloaded"])
            self.assertEqual(r["load_before"]["load1"], 1000.0)
        self.assertIn("load average median 1000.00, max 1000.00", report)
        self.assertIn("rounds under load: 2 of 2", report)
        self.assertEqual(records[-1]["valid_rounds"], 2)

    def test_quiet_round_is_not_loaded(self):
        records, report = self.run_fbench(["A=fake", "B=fake+X=1"])
        self.assertFalse(any(r["loaded"] for r in self.rounds(records)))
        self.assertIn("rounds under load: 0 of 2", report)

    def test_strict_mode_still_invalidates(self):
        loaded = {"load1": 1000.0, "busy": 0.0}
        # Busy for the first check, then quiet.
        records, report = self.run_fbench(
            ["A=fake", "B=fake+X=1"], "--strict-quiet", states=[loaded, QUIET]
        )
        rounds = self.rounds(records)
        self.assertFalse(rounds[0]["valid"])
        self.assertIn("load average", rounds[0]["reason"])
        self.assertEqual([r["valid"] for r in rounds[1:]], [True, True])
        self.assertIn("1 invalid", report)
        self.assertIn("strict quiet mode", report)

    def test_overload_runs_anyway_after_the_wait(self):
        overload = {"load1": 1000.0, "busy": 1e6}
        records, _ = self.run_fbench(
            ["A=fake", "B=fake+X=1"], "--overload-wait", "0", states=[overload]
        )
        rounds = self.rounds(records)
        self.assertEqual(len(rounds), 2)
        self.assertTrue(all(r["valid"] and r["overloaded"] and r["loaded"] for r in rounds))

    def test_timed_out_case_is_excluded_and_the_run_completes(self):
        hang = "c2c-64-b1-op"
        records, report = self.run_fbench(
            ["A=fake", f"B=fake+HANG={hang}"], "--process-timeout", "1.5"
        )
        rounds = self.rounds(records)
        self.assertEqual([r["valid"] for r in rounds], [True, True])
        self.assertTrue(all(r["timeouts"] == ["B"] for r in rounds))
        lost = [
            r for r in records if r["type"] == "timing" and r["arm"] == "B" and r["status"] != "ok"
        ]
        self.assertEqual({r["case"] for r in lost if r["status"] == analysis.TIMED_OUT}, {hang})
        summary = records[-1]
        self.assertEqual(summary["type"], "summary")
        self.assertEqual(summary["verdicts"]["B"][hang], analysis.TIMEOUT)
        self.assertIn("process timeouts: B in 2 rounds", report)
        self.assertIn("timed out (arm)", report)


class OverloadWait(unittest.TestCase):
    def wait(self, states: list[float], max_wait: float):
        now = [0.0]
        slept: list[float] = []

        def sleep(s: float) -> None:
            slept.append(s)
            now[0] += s

        def read() -> dict:
            return {"load1": 0.0, "busy": states.pop(0) if len(states) > 1 else states[0]}

        result = machine.wait_for_capacity(
            4.0, max_wait, poll=5.0, read=read, sleep=sleep, clock=lambda: now[0]
        )
        return result, slept

    def test_bounded(self):
        (state, waited, overloaded), slept = self.wait([10.0], 60.0)
        self.assertTrue(overloaded)
        self.assertEqual(waited, 60.0)
        self.assertAlmostEqual(sum(slept), 60.0)
        self.assertEqual(state["busy"], 10.0)

    def test_ends_when_the_load_drops(self):
        (state, waited, overloaded), _ = self.wait([10.0, 10.0, 2.0], 60.0)
        self.assertFalse(overloaded)
        self.assertEqual(waited, 10.0)
        self.assertEqual(state["busy"], 2.0)

    def test_no_wait_below_the_limit(self):
        (_, waited, overloaded), slept = self.wait([3.0], 60.0)
        self.assertEqual((waited, overloaded, slept), (0.0, False, []))


class OldRuns(unittest.TestCase):
    def test_report_of_a_run_stored_before_load_was_kept(self):
        # Rounds without "loaded": recomputed from the stored readings and thresholds.
        run = {
            "type": "run",
            "run": "old",
            "catalogue": {"name": "t"},
            "settings": {"mode": "hot", "seed": 0, "max_load": 4.0, "max_busy": 1.0},
            "host": {"cpu": "x", "logical_cpus": 8, "os": "y", "host_id": "z"},
            "arms": [
                {"label": "A", "spec": "A=f", "control": False, "describe": {}},
                {"label": "B", "spec": "B=f", "control": False, "describe": {}},
            ],
            "cases": [{"id": "c", "points": 16, "batch": 1, "flops": 320.0}],
        }
        records = [run]
        for i, load in enumerate((1.0, 6.0)):
            state = {"load1": load, "busy": 0.5}
            records.append(
                {
                    "type": "round",
                    "round": i,
                    "valid": True,
                    "load_before": state,
                    "load_after": state,
                }
            )
            for arm in ("A", "B"):
                records.append(
                    {
                        "type": "timing",
                        "round": i,
                        "arm": arm,
                        "case": "c",
                        "status": "ok",
                        "median_ns": 100.0,
                        "check": {"pass": True, "err_rms": 0},
                    }
                )
        text = cli.report(json.loads(json.dumps(records)))
        self.assertIn("rounds under load: 1 of 2", text)


if __name__ == "__main__":
    unittest.main()

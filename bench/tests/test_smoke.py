# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from fbenchlib import catalogue, cli

# A stand-in backend: passes the case kinds listed in $FAKE_PASS (in double precision, and
# in single precision when f32 is listed too) and calls every other case unsupported, with
# the harness's output lines.
FAKE = """#!{python}
import json, os, sys
args = sys.argv[1:]
cases = open(args[args.index("--cases") + 1]).read().split("\\n")
ok = set(os.environ.get("FAKE_PASS", "").split(","))
print(json.dumps({{"type": "describe"}}))
for line in filter(None, cases):
    f = dict(t.split("=", 1) for t in line.split())
    if f["kind"] in ok and f.get("prec", "f64") in ok | {{"f64"}}:
        check = {{"pass": "--perturb" not in args, "err_rms": 0, "err_max": 0, "tol": 1}}
        print(json.dumps({{"type": "case", "id": f["id"], "status": "ok", "check": check}}))
    else:
        print(json.dumps({{"type": "case", "id": f["id"], "status": "unsupported"}}))
print(json.dumps({{"type": "end"}}))
"""


class SmokeKinds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        if os.name == "nt":
            # Windows runs no script by its #! line; a batch file passes the arguments on.
            script = self.work / "fake.py"
            script.write_text(FAKE.format(python=sys.executable))
            self.exe = self.work / "fbench-fake.bat"
            self.exe.write_text(f'@"{sys.executable}" "{script}" %*\n')
        else:
            self.exe = self.work / "fbench-fake"
            self.exe.write_text(FAKE.format(python=sys.executable))
            self.exe.chmod(self.exe.stat().st_mode | stat.S_IXUSR)
        self.cases = [
            catalogue.Case("c2c", (16,), 1, "out", -1),
            catalogue.Case("c2c", (17,), 1, "out", -1),
            catalogue.Case("r2c", (16,), 1, "out", -1),
            catalogue.Case("c2c", (16,), 1, "out", -1, precision="f32"),
        ]
        (self.work / "cases.txt").write_text("".join(c.line() + "\n" for c in self.cases))

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("FAKE_PASS", None)

    def smoke(self, info: dict, passing: str, require_partial: bool = True) -> list[str]:
        os.environ["FAKE_PASS"] = passing
        failures: list[str] = []
        cli.smoke_one(
            self.exe,
            "fake",
            info,
            {},
            self.cases,
            self.work,
            60,
            failures,
            require_partial=require_partial,
        )
        return failures

    def test_unsupported_case_of_a_supported_kind_fails(self):
        failures = self.smoke({"supports": ["c2c", "r2c"]}, "r2c")
        self.assertTrue(any("declares c2c supported" in f for f in failures), failures)

    def test_partial_kind_needs_one_pass(self):
        self.assertEqual(self.smoke({"supports": [], "partial": ["c2c", "r2c"]}, "c2c,r2c"), [])
        failures = self.smoke({"supports": [], "partial": ["c2c", "r2c"]}, "c2c")
        self.assertEqual(failures, ["fake: declares r2c but passed no r2c case"])

    def test_partial_kinds_need_no_pass_outside_the_standard_smoke_tier(self):
        info = {"supports": [], "partial": ["c2c", "r2c"]}
        self.assertEqual(self.smoke(info, "c2c", require_partial=False), [])

    def test_single_precision_counts_only_when_declared(self):
        self.assertEqual(self.smoke({"supports": ["c2c", "r2c"]}, "c2c,r2c"), [])
        info = {"supports": ["c2c", "r2c"], "precisions": ["f64", "f32"]}
        failures = self.smoke(info, "c2c,r2c")
        self.assertTrue(any("c2c-16-b1-op-f32: unsupported" in f for f in failures), failures)
        self.assertEqual(self.smoke(info, "c2c,r2c,f32"), [])

    def test_undeclared_kinds_may_be_unsupported(self):
        self.assertEqual(self.smoke({"supports": []}, ""), [])


if __name__ == "__main__":
    unittest.main()

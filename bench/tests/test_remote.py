# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The Slurm runner's job script (scripts/remote/fbench_remote.py), with git stubbed."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).resolve().parents[2] / "scripts" / "remote" / "fbench_remote.py"
spec = importlib.util.spec_from_file_location("fbench_remote", PATH)
remote = importlib.util.module_from_spec(spec)
sys.modules["fbench_remote"] = remote  # dataclasses look their module up there
spec.loader.exec_module(remote)

HEAD = "a" * 40
OTHER = "b" * 40


def fake_git(*args):
    if args[:2] == ("rev-parse", "HEAD"):
        return HEAD
    if args[:2] == ("rev-parse", "--verify"):
        rev = args[2].removesuffix("^{commit}")
        if rev in ("main", OTHER):
            return OTHER
        raise remote.subprocess.CalledProcessError(1, "git")
    raise AssertionError(args)


PROFILE = remote.Profile(
    name="test",
    ssh="user@example",
    workdir="/scratch/fbench/",
    partition="compute",
    fftw_root="/opt/fftw",
    setup=["source /opt/env.sh"],
    sbatch=["--time=01:00:00"],
)


class JobScript(unittest.TestCase):
    def plan(self, arms, args=()):
        with mock.patch.object(remote, "git", fake_git):
            return remote.make_plan(PROFILE, arms, list(args), "job1")

    def test_arms_get_their_own_builds_and_commits(self):
        plan = self.plan(["A=fftw", "B=fftn:main+DELAY=0.05", "C=fftn"], ["--tier", "quick"])
        self.assertEqual(plan.jobdir, "/scratch/fbench/job1")
        self.assertEqual(plan.arm_commits, {"A": HEAD, "B": OTHER, "C": HEAD})
        self.assertEqual(plan.commits, {HEAD: ["fftw", "fftn"], OTHER: ["fftn"]})
        self.assertEqual(
            plan.arms,
            [
                f"A=fftw:/scratch/fbench/job1/build-{HEAD[:12]}",
                f"B=fftn:/scratch/fbench/job1/build-{OTHER[:12]}+DELAY=0.05",
                f"C=fftn:/scratch/fbench/job1/build-{HEAD[:12]}",
            ],
        )

    def test_script(self):
        plan = self.plan(["A=fftw", "B=fftn:main"], ["--tier", "quick"])
        text = remote.sbatch_script(PROFILE, plan, after="123")
        for needed in (
            "#SBATCH --exclusive",
            "#SBATCH --partition=compute",
            "#SBATCH --dependency=afterok:123",
            "#SBATCH --time=01:00:00",
            "set -euo pipefail",
            "source /opt/env.sh",
            "export FFTN_BENCH_FFTW_ROOT=/opt/fftw",
            f"tar -xzf src-{OTHER[:12]}.tar.gz -C src-{OTHER[:12]}",
            f"-DFBENCH_SOURCE_COMMIT={OTHER}",
            "--target fbench-fftw",
            f"--build /scratch/fbench/job1/build-{HEAD[:12]}",
            "--tier quick",
        ):
            self.assertIn(needed, text)
        # Every commit is unpacked into a fresh directory of its own and built from scratch.
        self.assertIn(f"rm -rf src-{HEAD[:12]} build-{HEAD[:12]}", text)

    def test_never_rsync(self):
        # Copies go by content (scp of git archive output): rsync -a keeps mtimes.
        source = PATH.read_text()
        self.assertNotIn('"rsync"', source)
        plan = self.plan(["A=fftw"])
        self.assertNotIn("rsync", remote.sbatch_script(PROFILE, plan))

    def test_bad_arm_build(self):
        with self.assertRaises(SystemExit):
            self.plan(["A=fftn:build/check-release"])


class JobNames(unittest.TestCase):
    # Submissions in the same second once collided and the second one was lost. Uniqueness
    # comes from claiming the directory, not from the random suffix (an earlier test that
    # drew 200 suffixes and expected no repeat failed about once in 800 runs).

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(remote, "STATE", Path(self.tmp.name) / "remote")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(remote.time, "strftime", return_value="20261113-120000")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_name_format(self):
        name = remote.job_name(HEAD)
        self.assertRegex(name, rf"^fbench-20261113-120000-{HEAD[:8]}-[0-9a-f]{{6}}$")

    def test_repeated_suffix_is_drawn_again(self):
        suffixes = iter(["aaaaaa", "aaaaaa", "bbbbbb"])
        with mock.patch.object(remote.secrets, "token_hex", lambda _n: next(suffixes)):
            first, first_dir = remote.claim_job(HEAD)
            second, second_dir = remote.claim_job(HEAD)
        self.assertTrue(first.endswith("-aaaaaa"))
        self.assertTrue(second.endswith("-bbbbbb"))
        self.assertTrue(first_dir.is_dir())
        self.assertTrue(second_dir.is_dir())

    def test_explicit_name_is_never_redrawn(self):
        remote.claim_job(HEAD, "mine")
        with self.assertRaises(SystemExit):
            remote.claim_job(HEAD, "mine")


class Config(unittest.TestCase):
    def test_file_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "remote.toml"
            path.write_text(
                '[profile.x]\nssh = "u@h"\nworkdir = "/w"\nsetup = ["a", "b"]\n'
                'sbatch = ["--time=1:00:00"]\n'
            )
            with mock.patch.dict(os.environ, {"FFTN_REMOTE_FFTW_ROOT": "/f"}, clear=False):
                p = remote.load_profile("x", path)
            self.assertEqual(
                (p.ssh, p.workdir, p.fftw_root, p.setup), ("u@h", "/w", "/f", ["a", "b"])
            )
            env = {k: "" for k in remote.ENV.values()}
            with mock.patch.dict(os.environ, env, clear=False), self.assertRaises(SystemExit):
                remote.load_profile("missing", path)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The Slurm runner's job script (scripts/remote/fbench_remote.py), with git stubbed."""

import importlib.util
import os
import shutil
import subprocess
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


CACHE = "/scratch/fbench/cache"


class JobScript(unittest.TestCase):
    def plan(self, arms, args=(), campaign=None):
        with mock.patch.object(remote, "git", fake_git):
            return remote.make_plan(PROFILE, arms, list(args), "job1", campaign)

    def key(self, sha):
        return remote.build_key(PROFILE, sha)

    def test_arms_use_the_shared_cache(self):
        plan = self.plan(["A=fftw", "B=fftn:main+DELAY=0.05", "C=fftn"], ["--tier", "quick"])
        self.assertEqual(plan.jobdir, "/scratch/fbench/job1")
        self.assertEqual(plan.cache, CACHE)
        self.assertEqual(plan.arm_commits, {"A": HEAD, "B": OTHER, "C": HEAD})
        self.assertEqual(
            plan.commits, {HEAD: ["fbench-fftw", "fbench-fftn"], OTHER: ["fbench-fftn"]}
        )
        self.assertEqual(
            plan.arms,
            [
                f"A=fftw:{CACHE}/{self.key(HEAD)}/build",
                f"B=fftn:{CACHE}/{self.key(OTHER)}/build+DELAY=0.05",
                f"C=fftn:{CACHE}/{self.key(HEAD)}/build",
            ],
        )

    def test_head_is_built_only_when_used(self):
        # `fbench run` gets a build per arm; only a campaign driver needs the head's.
        self.assertEqual(list(self.plan(["B=fftn:main"]).commits), [OTHER])
        plan = self.plan(["B=fftn:main"], campaign="bench/fbench.py")
        self.assertEqual(plan.commits, {OTHER: ["fbench-fftn"], HEAD: ["fbench-fftn"]})

    def test_key_covers_what_changes_the_binaries(self):
        key = self.key(HEAD)
        self.assertRegex(key, rf"^{HEAD[:12]}-[0-9a-f]{{10}}$")
        self.assertEqual(key, remote.build_key(PROFILE, HEAD))
        for change in (
            {"cxx": "clang++"},
            {"setup": ["source /opt/other-env.sh"]},
            {"fftw_root": "/opt/fftw-avx2"},
        ):
            other = remote.Profile(**{**PROFILE.__dict__, **change})
            self.assertNotEqual(remote.build_key(other, HEAD), key, change)
        # The allocation and the queue do not change the binaries.
        other = remote.Profile(**{**PROFILE.__dict__, "sbatch": ["-c 4"], "partition": "x"})
        self.assertEqual(remote.build_key(other, HEAD), key)

    def test_build_job(self):
        plan = self.plan(["A=fftw", "B=fftn:main"])
        profile = remote.Profile(**{**PROFILE.__dict__, "sbatch": ["--time=06:00:00", "-c 8"]})
        text = remote.build_script(profile, plan, [OTHER])
        for needed in (
            "#SBATCH --job-name=job1-build",
            "#SBATCH --output=/scratch/fbench/job1/slurm-build-%j.log",
            "#SBATCH -c 32",
            "#SBATCH --time=02:00:00",
            "source /opt/env.sh",
            "export FFTN_BENCH_FFTW_ROOT=/opt/fftw",
            f"cache={CACHE}",
            "flock -x 9",
            f"build_entry {self.key(OTHER)} {OTHER} fbench-fftn",
        ):
            self.assertIn(needed, text)
        self.assertNotIn("-c 8", text)
        self.assertNotIn("06:00:00", text)
        self.assertNotIn(f"build_entry {self.key(HEAD)}", text)  # cached: not asked for
        self.assertNotIn("fbench.py", text)
        # A target's marker is written only after the build command, under set -e.
        self.assertLess(text.index("cmake --build"), text.index('touch "$dir/done-$t"'))

    def test_timing_job_waits_for_the_build_and_reads_the_cache(self):
        plan = self.plan(["A=fftw", "B=fftn:main"], ["--tier", "quick"])
        text = remote.sbatch_script(PROFILE, plan, after=["900", "123"])
        for needed in (
            "#SBATCH --exclusive",
            "#SBATCH --partition=compute",
            "#SBATCH --dependency=afterok:900:123",
            "#SBATCH --kill-on-invalid-dep=yes",
            "#SBATCH --time=01:00:00",
            "set -euo pipefail",
            f"use_entry {self.key(HEAD)} fbench-fftw",
            f"use_entry {self.key(OTHER)} fbench-fftn",
            "flock -s",
            f"tar -xzf src-{HEAD[:12]}.tar.gz -C src-{HEAD[:12]}",
            f"cd src-{HEAD[:12]}",
            f"A=fftw:{CACHE}/{self.key(HEAD)}/build",
            "--tier quick",
        ):
            self.assertIn(needed, text)
        self.assertNotIn("build_entry", text)
        self.assertNotIn("cmake", text)

    def test_timing_job_can_build_itself(self):
        plan = self.plan(["B=fftn:main"])
        text = remote.sbatch_script(PROFILE, plan, build_here=[OTHER])
        build = f"build_entry {self.key(OTHER)} {OTHER} fbench-fftn"
        self.assertLess(text.index(build), text.index(f"use_entry {self.key(OTHER)} "))
        self.assertNotIn("--dependency", text)

    def test_serial(self):
        plan = self.plan(["B=fftn:main"])
        text = remote.sbatch_script(PROFILE, plan, after=["900"], serial=True)
        self.assertIn("#SBATCH --job-name=fbench-serial", text)
        self.assertIn("#SBATCH --comment=job1", text)
        self.assertIn("#SBATCH --dependency=afterok:900,singleton", text)
        text = remote.sbatch_script(PROFILE, plan, serial=True)
        self.assertIn("#SBATCH --dependency=singleton", text)

    def test_missing_check(self):
        plan = self.plan(["A=fftw", "B=fftn:main", "C=fftn"])
        text = remote.missing_script(plan)
        entry = f"{CACHE}/{self.key(HEAD)}"
        self.assertIn(
            f"[ -e {entry}/done-fbench-fftw ] && [ -e {entry}/done-fbench-fftn ] || echo {HEAD}",
            text,
        )

    def test_never_rsync(self):
        # Copies go by content (scp of git archive output): rsync -a keeps mtimes.
        source = PATH.read_text()
        self.assertNotIn('"rsync"', source)
        plan = self.plan(["A=fftw"])
        self.assertNotIn("rsync", remote.sbatch_script(PROFILE, plan))
        self.assertNotIn("rsync", remote.build_script(PROFILE, plan, [HEAD]))

    def test_campaign_runs_its_driver_with_the_head_build(self):
        plan = self.plan(["fftn"], ["all", "--stage", "screen"], "bench/fbench.py")
        with self.assertRaises(SystemExit):
            self.plan(["fftn"], [], "../outside.py")
        text = remote.sbatch_script(PROFILE, plan)
        self.assertIn(
            "uv run --frozen python bench/fbench.py --out-dir /scratch/fbench/job1/bench-out "
            f"--build {CACHE}/{self.key(HEAD)}/build all --stage screen",
            text,
        )
        self.assertNotIn(" run fftn", text)
        longer = remote.sbatch_script(PROFILE, plan, time_limit="12:00:00")
        self.assertIn("#SBATCH --time=12:00:00", longer)
        self.assertNotIn("--time=01:00:00", longer)

    def test_bad_arm_build(self):
        with self.assertRaises(SystemExit):
            self.plan(["A=fftn:build/check-release"])


BASH = shutil.which("bash")
FAKE_CMAKE = """\
#!/bin/sh
# Records each call; "builds" a target by writing a file (slowly with SLOW); fails with FAIL.
echo "$*" >> "$CALLS"
[ -z "${FAIL:-}" ] || exit 1
case "$1" in
  -S) mkdir -p "$4" ;;
  --build) [ -z "${SLOW:-}" ] || sleep 1; touch "$2/built" ;;
esac
"""


@unittest.skipUnless(BASH and shutil.which("flock"), "needs bash 4 and flock (util-linux)")
class CacheShell(unittest.TestCase):
    """The cache's shell functions, run for real with a stand-in for cmake."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cache = self.root / "cache"
        (self.root / "bin").mkdir()
        cmake = self.root / "bin" / "cmake"
        cmake.write_text(FAKE_CMAKE)
        cmake.chmod(0o755)
        (self.root / "tree").mkdir()
        (self.root / "tree" / "CMakeLists.txt").write_text("")
        subprocess.run(
            ["tar", "-czf", str(self.root / f"src-{HEAD[:12]}.tar.gz"), "-C", "tree", "."],
            cwd=self.root,
            check=True,
        )
        self.calls = self.root / "calls"

    def run_shell(self, body, fail=False, slow=False):
        script = "\n".join(
            [
                "set -euo pipefail",
                f"cache={self.cache}",
                "gen=Ninja; jobs=2",
                'mkdir -p "$cache/locks"',
                remote.BUILD_ENTRY,
                remote.USE_ENTRY,
                body,
            ]
        )
        env = {
            **os.environ,
            "PATH": f"{self.root / 'bin'}:{os.environ['PATH']}",
            "CALLS": str(self.calls),
            **({"FAIL": "1"} if fail else {}),
            **({"SLOW": "1"} if slow else {}),
        }
        return subprocess.run(
            [BASH, "-c", script], cwd=self.root, env=env, capture_output=True, text=True
        )

    def builds(self):
        if not self.calls.exists():
            return 0
        return sum(line.startswith("--build") for line in self.calls.read_text().splitlines())

    def test_build_once_then_reuse(self):
        done = self.run_shell(f"build_entry k1 {HEAD} fbench-fftn")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue((self.cache / "k1" / "done-fbench-fftn").exists())
        self.assertTrue((self.cache / "k1" / "src" / "CMakeLists.txt").exists())
        done = self.run_shell(f"build_entry k1 {HEAD} fbench-fftn\nuse_entry k1 fbench-fftn")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("cached: k1", done.stdout)
        self.assertEqual(self.builds(), 1)
        # A new target is added to the entry without configuring it again.
        done = self.run_shell(f"build_entry k1 {HEAD} fbench-fftn fbench-fftw")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.builds(), 2)
        self.assertIn("--target fbench-fftw", self.calls.read_text().splitlines()[-1])
        self.assertNotIn("fbench-fftn", self.calls.read_text().splitlines()[-1])

    def test_concurrent_jobs_build_once(self):
        body = f"""
build_entry k1 {HEAD} fbench-fftn & a=$!
build_entry k1 {HEAD} fbench-fftn & b=$!
wait $a; wait $b
"""
        done = self.run_shell(body, slow=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.builds(), 1)
        self.assertIn("cached: k1", done.stdout)

    def test_failed_build_leaves_no_marker(self):
        done = self.run_shell(f"build_entry k1 {HEAD} fbench-fftn", fail=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse((self.cache / "k1" / "done-fbench-fftn").exists())
        done = self.run_shell("use_entry k1 fbench-fftn")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no finished fbench-fftn", done.stderr)

    def test_prune_keeps_recent_and_locked_entries(self):
        for key in ("old", "busy", "new"):
            self.assertEqual(self.run_shell(f"build_entry {key} {HEAD} fbench-fftn").returncode, 0)
        os.utime(self.cache / "old" / "used", (0, 0))
        # A running timing job holds busy's use lock while prune runs.
        prune = remote.prune_script(str(self.cache), 1)
        body = 'use_entry busy fbench-fftn\ntouch -t 200001010000 "$cache/busy/used"\n'
        done = self.run_shell(body + prune)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("pruned old", done.stdout)
        self.assertIn("in use, kept: busy", done.stdout)
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()), ["busy", "locks", "new"])


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

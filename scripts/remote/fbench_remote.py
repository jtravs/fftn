#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Run fbench on a Slurm machine with an exclusive node (bench/README.md, "Remote runs").

    fbench_remote.py submit PROFILE ARM... [--after JOBID] [--dry-run] -- [fbench run options]
    fbench_remote.py submit PROFILE ARM... --campaign SCRIPT -- [SCRIPT options]
    fbench_remote.py status JOB
    fbench_remote.py log JOB [--lines N]
    fbench_remote.py fetch JOB
    fbench_remote.py jobs

Sources go by content: every commit an arm needs is packed locally with `git archive` and
unpacked on the remote into a directory of its own, then built into a build directory of
its own inside the job's directory, so no two jobs share a build and no stale object can
survive a copy (rsync -a keeps mtimes, which once left stale builds). Arms run from those
builds; the job directory and each arm's commit are recorded locally under
bench-out/remote/<job>/ and in fbench's own run record.

With --campaign, the job runs a campaign driver from the checkout instead of one fbench
run (for example bench/x2/x2.py, which runs many): the arms only name the builds it needs,
and it is called as `SCRIPT --out-dir <job>/bench-out --build <job>/build-<head> ...`.
Campaign drivers keep their own state in bench-out/campaigns/, which fetch copies too.

Hosts, partitions and paths are private: they come from an untracked TOML file
($FFTN_REMOTE_CONFIG, default ~/.config/fftn/remote.toml) or the environment
(FFTN_REMOTE_SSH, _WORKDIR, _PARTITION, _FFTW_ROOT, _SETUP), never from this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench"))

from fbenchlib import arms as arm_grammar  # noqa: E402

STATE = REPO / "bench-out" / "remote"
ENV = {
    "ssh": "FFTN_REMOTE_SSH",
    "workdir": "FFTN_REMOTE_WORKDIR",
    "partition": "FFTN_REMOTE_PARTITION",
    "fftw_root": "FFTN_REMOTE_FFTW_ROOT",
}


@dataclass
class Profile:
    name: str
    ssh: str
    workdir: str
    partition: str = ""
    fftw_root: str = ""
    setup: list[str] = field(default_factory=list)  # shell lines run first in the job
    sbatch: list[str] = field(default_factory=list)  # extra #SBATCH options
    cxx: str = ""
    uv: str = "uv"


def config_path() -> Path:
    return Path(os.environ.get("FFTN_REMOTE_CONFIG", "~/.config/fftn/remote.toml")).expanduser()


def load_profile(name: str, path: Path | None = None) -> Profile:
    path = path or config_path()
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text()).get("profile", {}).get(name, {})
    values = {k: v for k, v in data.items() if k in Profile.__dataclass_fields__}
    for key, var in ENV.items():
        if os.environ.get(var):
            values[key] = os.environ[var]
    if os.environ.get("FFTN_REMOTE_SETUP"):
        values["setup"] = [os.environ["FFTN_REMOTE_SETUP"]]
    missing = [k for k in ("ssh", "workdir") if not values.get(k)]
    if missing:
        raise SystemExit(
            f"profile {name!r}: no {', '.join(missing)} in {path} or the environment "
            f"({', '.join(ENV[k] for k in missing)})"
        )
    return Profile(name=name, **values)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


@dataclass
class Plan:
    """What one job builds and runs; written locally before anything is sent."""

    job: str
    jobdir: str
    head: str
    commits: dict[str, list[str]]  # sha -> libraries to build there
    arms: list[str]  # rewritten for the remote build directories
    arm_commits: dict[str, str]  # label -> sha
    fbench_args: list[str]
    campaign: str | None = None  # a driver script to run instead of `fbench.py run`


def make_plan(
    profile: Profile,
    arm_specs: list[str],
    fbench_args: list[str],
    job: str,
    campaign: str | None = None,
) -> Plan:
    head = git("rev-parse", "HEAD")
    jobdir = f"{profile.workdir.rstrip('/')}/{job}"
    commits: dict[str, list[str]] = {}
    rewritten: list[str] = []
    arm_commits: dict[str, str] = {}
    for spec in arm_specs:
        arm = arm_grammar.parse(spec)
        sha = head
        if arm.build is not None:
            try:
                sha = git("rev-parse", "--verify", f"{arm.build}^{{commit}}")
            except subprocess.CalledProcessError:
                raise SystemExit(f"arm {arm.label}: {arm.build!r} is not a commit") from None
        commits.setdefault(sha, [])
        if arm.library not in commits[sha]:
            commits[sha].append(arm.library)
        arm_commits[arm.label] = sha
        build = f"{jobdir}/build-{sha[:12]}"
        options = "".join(f"+{k}={v}" for k, v in arm.options)
        rewritten.append(f"{arm.label}={arm.library}:{build}{options}")
    commits.setdefault(head, [])
    if campaign is not None:
        path = Path(campaign)
        if path.is_absolute() or ".." in path.parts or not (REPO / path).is_file():
            raise SystemExit(f"--campaign {campaign}: not a script of this checkout")
    return Plan(job, jobdir, head, commits, rewritten, arm_commits, fbench_args, campaign)


def sbatch_script(profile: Profile, plan: Plan, after: str | None = None) -> str:
    q = shlex.quote
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={plan.job}",
        "#SBATCH --exclusive",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --output={plan.jobdir}/slurm-%j.log",
    ]
    if profile.partition:
        lines.append(f"#SBATCH --partition={profile.partition}")
    if after:
        lines.append(f"#SBATCH --dependency=afterok:{after}")
    lines += [f"#SBATCH {opt}" for opt in profile.sbatch]
    lines += [
        "set -euo pipefail",
        f"cd {q(plan.jobdir)}",
        *profile.setup,
        f'echo "fbench job {plan.job}: $(uname -sm), $(date -u +%FT%TZ)"',
    ]
    if profile.fftw_root:
        lines.append(f"export FFTN_BENCH_FFTW_ROOT={q(profile.fftw_root)}")
    if profile.cxx:
        lines.append(f"export CXX={q(profile.cxx)}")
    lines += [
        'gen="Unix Makefiles"; if command -v ninja > /dev/null; then gen=Ninja; fi',
        'jobs="${SLURM_CPUS_ON_NODE:-4}"',
    ]
    for sha, libraries in plan.commits.items():
        short = sha[:12]
        targets = " ".join(f"--target fbench-{lib}" for lib in libraries) or "--target fbench-fftn"
        lines += [
            f"# {sha}",
            f"rm -rf src-{short} build-{short}",
            f"mkdir src-{short}",
            f"tar -xzf src-{short}.tar.gz -C src-{short}",
            f'cmake -S src-{short} -B build-{short} -G "$gen" -DCMAKE_BUILD_TYPE=Release '
            f"-DFFTN_BUILD_TESTS=OFF -DFFTN_BUILD_BENCH=ON -DFBENCH_SOURCE_COMMIT={sha}",
            f'cmake --build build-{short} -j "$jobs" {targets}',
        ]
    build = f"{plan.jobdir}/build-{plan.head[:12]}"
    out_dir = f"{plan.jobdir}/bench-out"
    if plan.campaign is None:
        run = [
            profile.uv,
            "run",
            "--frozen",
            "python",
            "bench/fbench.py",
            "--out-dir",
            out_dir,
            "run",
            *plan.arms,
            "--build",
            build,
            *plan.fbench_args,
        ]
    else:
        run = [profile.uv, "run", "--frozen", "python", plan.campaign]
        run += ["--out-dir", out_dir, "--build", build, *plan.fbench_args]
    lines += [
        f"cd src-{plan.head[:12]}",
        " ".join(q(x) for x in run),
        f'echo "fbench job {plan.job}: done"',
    ]
    return "\n".join(lines) + "\n"


def job_name(head: str) -> str:
    """Time, commit and a random suffix, so names rarely repeat; claim_job makes them unique."""
    return f"fbench-{time.strftime('%Y%m%d-%H%M%S')}-{head[:8]}-{secrets.token_hex(3)}"


def claim_job(head: str, name: str | None = None) -> tuple[str, Path]:
    """A job name with its local directory, created exclusively.

    Two submissions in the same second once shared a name and the second was lost; a random
    suffix alone only makes that unlikely (200 names in one second collide about once in 800
    runs), so a name whose directory exists is drawn again. An explicit name is never redrawn.
    """
    for _ in range(100):
        job = name or job_name(head)
        local = STATE / job
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            local.mkdir()
        except FileExistsError:
            if name:
                raise SystemExit(f"job {name} exists already: {local}") from None
            continue
        return job, local
    raise SystemExit(f"could not find a free job name under {STATE}")


def ssh(profile: Profile, command: str, check: bool = True) -> str:
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", profile.ssh, command],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and done.returncode != 0:
        raise SystemExit(f"ssh failed: {done.stderr.strip()}")
    return done.stdout


def cmd_submit(args: argparse.Namespace) -> int:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(
            "the working tree has uncommitted changes: commit them first (AGENTS.md 6)"
        )
    profile = load_profile(args.profile)
    job, local = claim_job(git("rev-parse", "HEAD"), args.name)
    plan = make_plan(profile, args.arms, args.fbench_args, job, args.campaign)
    for sha in plan.commits:
        out = local / f"src-{sha[:12]}.tar.gz"
        subprocess.run(
            ["git", "archive", "--format=tar.gz", "-o", str(out), sha], cwd=REPO, check=True
        )
    (local / "job.sbatch").write_text(sbatch_script(profile, plan, args.after))
    manifest = {
        "job": job,
        "profile": profile.name,
        "jobdir": plan.jobdir,
        "head": plan.head,
        "arms": args.arms,
        "remote_arms": plan.arms,
        "arm_commits": plan.arm_commits,
        "fbench_args": plan.fbench_args,
        "campaign": plan.campaign,
        "after": args.after,
        "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if args.dry_run:
        (local / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"dry run: job files in {local}")
        return 0
    ssh(profile, f"mkdir -p {shlex.quote(plan.jobdir)}")
    files = [str(p) for p in sorted(local.iterdir())]
    # scp copies file contents; nothing is synchronised by timestamp.
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", *files, f"{profile.ssh}:{plan.jobdir}/"], check=True
    )
    out = ssh(profile, f"cd {shlex.quote(plan.jobdir)} && sbatch job.sbatch")
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise SystemExit(f"sbatch said: {out.strip()}")
    manifest["jobid"] = m.group(1)
    (local / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"submitted {job} as Slurm job {m.group(1)}")
    print(f"check its log once it starts: fbench_remote.py log {job}")
    if args.after:
        print(
            "this job waits on afterok: if the first job fails, this one never runs and "
            "Slurm says so only in squeue (DependencyNeverSatisfied): check the first log"
        )
    return 0


def manifest_of(job: str) -> tuple[dict, Profile]:
    path = STATE / job / "manifest.json"
    if not path.exists():
        raise SystemExit(f"no job {job!r} under {STATE}")
    manifest = json.loads(path.read_text())
    return manifest, load_profile(manifest["profile"])


def cmd_status(args: argparse.Namespace) -> int:
    manifest, profile = manifest_of(args.job)
    jobid = manifest["jobid"]
    print(
        ssh(profile, f"squeue -h -j {jobid} -o '%T %M %r' 2>/dev/null || true", check=False)
        or ssh(profile, f"sacct -n -X -j {jobid} -o State,Elapsed,ExitCode", check=False)
    )
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    manifest, profile = manifest_of(args.job)
    log = f"{manifest['jobdir']}/slurm-{manifest['jobid']}.log"
    print(ssh(profile, f"tail -n {int(args.lines)} {shlex.quote(log)}", check=False), end="")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest, profile = manifest_of(args.job)
    runs = REPO / "bench-out" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    remote = f"{manifest['jobdir']}/bench-out/runs/*.jsonl"
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", f"{profile.ssh}:{remote}", str(runs)], check=True
    )
    log = f"{manifest['jobdir']}/slurm-{manifest['jobid']}.log"
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", f"{profile.ssh}:{log}", str(STATE / args.job)],
        check=False,
    )
    if manifest.get("campaign"):
        campaigns = REPO / "bench-out" / "campaigns"
        campaigns.mkdir(parents=True, exist_ok=True)
        remote_campaigns = f"{manifest['jobdir']}/bench-out/campaigns/*"
        subprocess.run(
            ["scp", "-q", "-r", "-o", "BatchMode=yes", f"{profile.ssh}:{remote_campaigns}"]
            + [str(campaigns)],
            check=False,
        )
        print(f"campaign state copied into {campaigns}")
    listing = ssh(profile, f"ls {shlex.quote(manifest['jobdir'])}/bench-out/runs/", check=False)
    ids = sorted(Path(x).stem for x in listing.split() if x.endswith(".jsonl"))
    for run_id in ids:
        print(f"fetched {run_id}: uv run --frozen python bench/fbench.py show {run_id}")
    return 0


def cmd_jobs(_: argparse.Namespace) -> int:
    for path in sorted(STATE.glob("*/manifest.json")):
        m = json.loads(path.read_text())
        print(
            f"{m['job']}  profile {m['profile']}  job {m.get('jobid', '(dry run)')}  "
            f"arms {' '.join(m['arms'])}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fbench_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, fbench_args = argv[:cut], argv[cut + 1 :]
    p = argparse.ArgumentParser(prog="fbench_remote.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit", help="pack, copy, build and run fbench in one Slurm job")
    s.add_argument("profile")
    s.add_argument("arms", nargs="+", help="as for fbench run; builds must be commits")
    s.add_argument("--name", help="job name (default fbench-<time>-<commit>)")
    s.add_argument("--after", help="Slurm job ID this job waits for (afterok)")
    s.add_argument("--dry-run", action="store_true", help="write the job files only")
    s.add_argument(
        "--campaign",
        help="run this campaign driver of the checkout (e.g. bench/x2/x2.py) instead of fbench",
    )
    s.set_defaults(func=cmd_submit)
    for name, func in (("status", cmd_status), ("log", cmd_log), ("fetch", cmd_fetch)):
        sp = sub.add_parser(name)
        sp.add_argument("job")
        if name == "log":
            sp.add_argument("--lines", type=int, default=40)
        sp.set_defaults(func=func)
    sub.add_parser("jobs", help="jobs submitted from here").set_defaults(func=cmd_jobs)
    args = p.parse_args(argv)
    args.fbench_args = fbench_args
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Run fbench on a Slurm machine (bench/README.md, "Remote runs").

    fbench_remote.py submit PROFILE ARM... [--after JOBID] [--serial] [--no-build-job]
                     [--dry-run] -- [fbench run options]
    fbench_remote.py submit PROFILE ARM... --campaign SCRIPT [--time T] -- [SCRIPT options]
    fbench_remote.py status JOB
    fbench_remote.py log JOB [--lines N] [--build]
    fbench_remote.py fetch JOB
    fbench_remote.py jobs
    fbench_remote.py prune PROFILE [--days N] [--dry-run]

Sources go by content: every commit an arm needs is packed locally with `git archive`
(rsync -a keeps mtimes, which once left stale builds) and built on the remote into a build
cache shared by all jobs of the machine, <cache>/<sha12>-<hash>/, where the hash covers
the compiler, the profile's setup lines, the FFTW root and the CMake options. A commit
whose entry is finished is neither packed nor built again. Missing builds are made by a
short build job with more cores; the timing job waits for it (afterok) and only reads the
cache. The job directory, each arm's commit and the Slurm job IDs are recorded locally
under bench-out/remote/<job>/ and in fbench's own run record.

With --campaign, the job runs a campaign driver from the checkout instead of one fbench
run (for example bench/x2/x2.py, which runs many): the arms only name the builds it needs,
and it is called as `SCRIPT --out-dir <job>/bench-out --build <head's build> ...`.
Campaign drivers keep their own state in bench-out/campaigns/, which fetch copies too.

Hosts, partitions and paths are private: they come from an untracked TOML file
($FFTN_REMOTE_CONFIG, default ~/.config/fftn/remote.toml) or the environment
(FFTN_REMOTE_SSH, _WORKDIR, _PARTITION, _FFTW_ROOT, _SETUP), never from this repository.
"""

from __future__ import annotations

import argparse
import hashlib
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
    # The whole node by default. false shares it (a shared machine whose other jobs must keep
    # running): fbench's interleaved rounds and A/A band then carry the noise.
    exclusive: bool = True
    cache_dir: str = ""  # the builds shared by every job of the machine (<workdir>/cache)
    build_cpus: int = 32  # the build job's allocation
    build_time: str = "02:00:00"  # the build job's time limit


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


# The configure line of every cached build; part of the cache key.
CMAKE_OPTIONS = "-DCMAKE_BUILD_TYPE=Release -DFFTN_BUILD_TESTS=OFF -DFFTN_BUILD_BENCH=ON"


def cache_dir(profile: Profile) -> str:
    return profile.cache_dir or f"{profile.workdir.rstrip('/')}/cache"


def build_key(profile: Profile, sha: str) -> str:
    """The cache entry of a commit: everything that can change its binaries is hashed.

    The whole setup block goes in (PATH, library roots and the like choose the compiler and
    what the backends link), not only the lines that look relevant. Targets are not part of
    the key: each one gets its own completion marker inside the entry.
    """
    config = json.dumps([profile.cxx, profile.fftw_root, profile.setup, CMAKE_OPTIONS])
    return f"{sha[:12]}-{hashlib.sha256(config.encode()).hexdigest()[:10]}"


@dataclass
class Plan:
    """What one job builds and runs; written locally before anything is sent."""

    job: str
    jobdir: str
    head: str
    cache: str
    commits: dict[str, list[str]]  # sha -> fbench targets it needs
    keys: dict[str, str]  # sha -> cache entry
    arms: list[str]  # rewritten for the cached build directories
    arm_commits: dict[str, str]  # label -> sha
    fbench_args: list[str]
    campaign: str | None = None  # a driver script to run instead of `fbench.py run`

    def build(self, sha: str) -> str:
        return f"{self.cache}/{self.keys[sha]}/build"


def make_plan(
    profile: Profile,
    arm_specs: list[str],
    fbench_args: list[str],
    job: str,
    campaign: str | None = None,
) -> Plan:
    head = git("rev-parse", "HEAD")
    jobdir = f"{profile.workdir.rstrip('/')}/{job}"
    cache = cache_dir(profile)
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
        targets = commits.setdefault(sha, [])
        if f"fbench-{arm.library}" not in targets:
            targets.append(f"fbench-{arm.library}")
        arm_commits[arm.label] = sha
        build = f"{cache}/{build_key(profile, sha)}/build"
        options = "".join(f"+{k}={v}" for k, v in arm.options)
        rewritten.append(f"{arm.label}={arm.library}:{build}{options}")
    if campaign is not None:
        path = Path(campaign)
        if path.is_absolute() or ".." in path.parts or not (REPO / path).is_file():
            raise SystemExit(f"--campaign {campaign}: not a script of this checkout")
        # A driver is given the head's build; `fbench run` needs none (every arm names one).
        targets = commits.setdefault(head, [])
        if not targets:
            targets.append("fbench-fftn")
    keys = {sha: build_key(profile, sha) for sha in commits}
    return Plan(
        job, jobdir, head, cache, commits, keys, rewritten, arm_commits, fbench_args, campaign
    )


# Shell functions of the job scripts. Both lock files live outside the entries, so prune can
# delete an entry while holding them, and a job that waited on them then finds it missing.
BUILD_ENTRY = """\
# build_entry KEY SHA TARGET...: build into the cache whatever of KEY is not finished. The
# lock makes a second job wait, then reuse the build; a done-<target> marker is written only
# after its target built, so a failed or interrupted build is never used.
build_entry() {
  local key=$1 sha=$2; shift 2
  (
    flock -x 9
    dir="$cache/$key"
    missing=()
    for t in "$@"; do [ -e "$dir/done-$t" ] || missing+=(--target "$t"); done
    if [ ${#missing[@]} -eq 0 ]; then echo "cached: $key"; exit 0; fi
    start=$SECONDS
    if [ ! -e "$dir/configured" ]; then
      rm -rf "$dir"
      mkdir -p "$dir/src"
      tar -xzf "src-${sha:0:12}.tar.gz" -C "$dir/src"
      cmake -S "$dir/src" -B "$dir/build" -G "$gen" @CMAKE@ -DFBENCH_SOURCE_COMMIT="$sha"
      touch "$dir/configured"
    fi
    cmake --build "$dir/build" -j "$jobs" "${missing[@]}"
    for t in "$@"; do touch "$dir/done-$t"; done
    touch "$dir/used"
    echo "built $key (${missing[*]}) in $((SECONDS - start)) s"
  ) 9> "$cache/locks/$key.build"
}
""".replace("@CMAKE@", CMAKE_OPTIONS)

USE_ENTRY = """\
# use_entry KEY TARGET...: hold KEY's use lock (shared) until the job ends, so prune leaves
# the entry alone, and check that every target finished building.
use_entry() {
  local key=$1 fd; shift
  exec {fd}>> "$cache/locks/$key.use"
  flock -s "$fd"
  for t in "$@"; do
    if [ ! -e "$cache/$key/done-$t" ]; then
      echo "no finished $t in $cache/$key: did the build job run?" >&2
      exit 1
    fi
  done
  touch "$cache/$key/used"
}
"""


def script_head(profile: Profile, plan: Plan, name: str, log: str, options: list[str]) -> list[str]:
    """The #SBATCH block and the environment shared by the build job and the timing job."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        *(["#SBATCH --exclusive"] if profile.exclusive else []),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --output={plan.jobdir}/{log}",
    ]
    if profile.partition:
        lines.append(f"#SBATCH --partition={profile.partition}")
    lines += [f"#SBATCH {opt}" for opt in options]
    lines += [
        "set -euo pipefail",
        f"cd {shlex.quote(plan.jobdir)}",
        *profile.setup,
        f'echo "fbench job {name}: $(uname -sm), $(date -u +%FT%TZ)"',
    ]
    if profile.fftw_root:
        lines.append(f"export FFTN_BENCH_FFTW_ROOT={shlex.quote(profile.fftw_root)}")
    if profile.cxx:
        lines.append(f"export CXX={shlex.quote(profile.cxx)}")
    lines += [
        'gen="Unix Makefiles"; if command -v ninja > /dev/null; then gen=Ninja; fi',
        'jobs="${SLURM_CPUS_ON_NODE:-4}"',
        f"cache={shlex.quote(plan.cache)}",
        'mkdir -p "$cache/locks"',
    ]
    return lines


def replace_option(options: list[str], prefixes: tuple[str, ...], new: str) -> list[str]:
    return [o for o in options if not o.startswith(prefixes)] + [new]


def build_lines(plan: Plan, shas: list[str]) -> list[str]:
    return [BUILD_ENTRY] + [
        f"build_entry {plan.keys[sha]} {sha} {' '.join(plan.commits[sha])}" for sha in shas
    ]


def build_script(profile: Profile, plan: Plan, shas: list[str]) -> str:
    """A short, wide job that builds the missing commits into the cache and nothing else."""
    options = replace_option(profile.sbatch, ("-c", "--cpus-per-task"), f"-c {profile.build_cpus}")
    options = replace_option(options, ("--time", "-t"), f"--time={profile.build_time}")
    lines = script_head(profile, plan, f"{plan.job}-build", "slurm-build-%j.log", options)
    lines += build_lines(plan, shas)
    lines.append(f'echo "fbench job {plan.job}-build: done"')
    return "\n".join(lines) + "\n"


def sbatch_script(
    profile: Profile,
    plan: Plan,
    after: list[str] | None = None,
    time_limit: str | None = None,
    build_here: list[str] | None = None,
    serial: bool = False,
) -> str:
    """The timing job; it builds only the commits in build_here (when there is no build job)."""
    q = shlex.quote
    options = list(profile.sbatch)
    if time_limit:
        # Replaces the profile's limit: a campaign runs for hours, a single A/B for minutes.
        options = replace_option(options, ("--time", "-t"), f"--time={time_limit}")
    name = plan.job
    depend = [f"afterok:{':'.join(after)}"] if after else []
    if serial:
        # Slurm runs one job of a given name and user at a time, so timing jobs that share
        # this name queue behind each other instead of disturbing each other's rounds.
        name = "fbench-serial"
        depend.append("singleton")
        options.append(f"--comment={plan.job}")
    if depend:
        # A failed dependency cancels this job instead of leaving it pending for ever.
        options += [f"--dependency={','.join(depend)}", "--kill-on-invalid-dep=yes"]
    lines = script_head(profile, plan, name, "slurm-%j.log", options)
    if build_here:
        lines += build_lines(plan, build_here)
    lines.append(USE_ENTRY)
    lines += [f"use_entry {plan.keys[sha]} {' '.join(t)}" for sha, t in plan.commits.items()]
    out_dir = f"{plan.jobdir}/bench-out"
    if plan.campaign is None:
        run = [profile.uv, "run", "--frozen", "python", "bench/fbench.py", "--out-dir", out_dir]
        run += ["run", *plan.arms, *plan.fbench_args]
    else:
        run = [profile.uv, "run", "--frozen", "python", plan.campaign]
        run += ["--out-dir", out_dir, "--build", plan.build(plan.head), *plan.fbench_args]
    # fbench runs from the job's own copy of the head: uv keeps its environment there.
    src = f"src-{plan.head[:12]}"
    lines += [
        f"rm -rf {src}",
        f"mkdir {src}",
        f"tar -xzf {src}.tar.gz -C {src}",
        f"cd {src}",
        " ".join(q(x) for x in run),
        f'echo "fbench job {plan.job}: done"',
    ]
    return "\n".join(lines) + "\n"


def missing_script(plan: Plan) -> str:
    """Prints the commits whose cache entry lacks a finished target."""
    lines = []
    for sha, targets in plan.commits.items():
        tests = " && ".join(f"[ -e {q_entry(plan, sha)}/done-{t} ]" for t in targets)
        lines.append(f"{tests} || echo {sha}")
    return "\n".join(lines) + "\n"


def q_entry(plan: Plan, sha: str) -> str:
    return shlex.quote(f"{plan.cache}/{plan.keys[sha]}")


def prune_script(cache: str, days: int, dry_run: bool = False) -> str:
    """Deletes the cache entries unused for more than `days` days whose locks are free."""
    action = 'echo "would prune $1"' if dry_run else 'rm -rf -- "$1" && echo "pruned $1"'
    return f"""\
cd {shlex.quote(cache)} 2> /dev/null || exit 0
for d in */; do
  d=${{d%/}}
  [ "$d" = locks ] && continue
  stamp="$d/used"; [ -e "$stamp" ] || stamp="$d"
  [ -n "$(find "$stamp" -maxdepth 0 -mtime +{int(days)})" ] || continue
  flock -n "locks/$d.use" flock -n "locks/$d.build" sh -c '{action}' sh "$d" \\
    || echo "in use, kept: $d"
done
"""


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
    if args.dry_run:
        missing = list(plan.commits)
    else:
        ssh(profile, f"mkdir -p {shlex.quote(plan.jobdir)}")
        found = ssh(profile, missing_script(plan)).split()
        missing = [sha for sha in plan.commits if sha in found]
    # Only the commits to build are packed, and the head, which fbench runs from.
    for sha in dict.fromkeys([*missing, plan.head]):
        out = local / f"src-{sha[:12]}.tar.gz"
        subprocess.run(
            ["git", "archive", "--format=tar.gz", "-o", str(out), sha], cwd=REPO, check=True
        )
    build_job = args.build_job and bool(missing)
    if build_job:
        (local / "build.sbatch").write_text(build_script(profile, plan, missing))
    after = [args.after] if args.after else []
    manifest = {
        "job": job,
        "profile": profile.name,
        "jobdir": plan.jobdir,
        "head": plan.head,
        "arms": args.arms,
        "remote_arms": plan.arms,
        "arm_commits": plan.arm_commits,
        "cache_keys": plan.keys,
        "built": missing,
        "fbench_args": plan.fbench_args,
        "campaign": plan.campaign,
        "time": args.time,
        "after": args.after,
        "serial": args.serial,
        "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    def write_timing(depends: list[str]) -> None:
        build_here = [] if build_job else missing
        text = sbatch_script(profile, plan, depends, args.time, build_here, args.serial)
        (local / "job.sbatch").write_text(text)

    def save() -> None:
        (local / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.dry_run:
        write_timing((["BUILD_JOB"] if build_job else []) + after)
        save()
        print(f"dry run: job files in {local}")
        return 0
    # scp copies file contents; nothing is synchronised by timestamp.
    copy(profile, sorted(local.iterdir()), plan.jobdir)
    if build_job:
        manifest["build_jobid"] = submit_script(profile, plan.jobdir, "build.sbatch")
        save()
        after.insert(0, manifest["build_jobid"])
        print(f"submitted the build of {len(missing)} commit(s) as Slurm job {after[0]}")
    else:
        print("every build is cached: no build job" if not missing else "building in the job")
    write_timing(after)
    copy(profile, [local / "job.sbatch"], plan.jobdir)
    manifest["jobid"] = submit_script(profile, plan.jobdir, "job.sbatch")
    save()
    print(f"submitted {job} as Slurm job {manifest['jobid']}")
    print(f"check its log once it starts: fbench_remote.py log {job}")
    if after:
        print(
            "it waits on afterok: if a job it waits for fails, Slurm cancels it "
            "(fbench_remote.py log --build shows a failed build)"
        )
    return 0


def copy(profile: Profile, files: list[Path], remote_dir: str) -> None:
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", *map(str, files), f"{profile.ssh}:{remote_dir}/"],
        check=True,
    )


def submit_script(profile: Profile, jobdir: str, script: str) -> str:
    out = ssh(profile, f"cd {shlex.quote(jobdir)} && sbatch {script}")
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise SystemExit(f"sbatch said: {out.strip()}")
    return m.group(1)


def manifest_of(job: str) -> tuple[dict, Profile]:
    path = STATE / job / "manifest.json"
    if not path.exists():
        raise SystemExit(f"no job {job!r} under {STATE}")
    manifest = json.loads(path.read_text())
    return manifest, load_profile(manifest["profile"])


def slurm_jobs(manifest: dict) -> list[tuple[str, str, str]]:
    """(role, Slurm job ID, log file) of a job's build job, if any, and its timing job."""
    jobdir = manifest["jobdir"]
    out = []
    if manifest.get("build_jobid"):
        jobid = manifest["build_jobid"]
        out.append(("build", jobid, f"{jobdir}/slurm-build-{jobid}.log"))
    out.append(("timing", manifest["jobid"], f"{jobdir}/slurm-{manifest['jobid']}.log"))
    return out


def cmd_status(args: argparse.Namespace) -> int:
    manifest, profile = manifest_of(args.job)
    for role, jobid, _ in slurm_jobs(manifest):
        state = ssh(
            profile, f"squeue -h -j {jobid} -o '%T %M %r' 2>/dev/null || true", check=False
        ) or ssh(profile, f"sacct -n -X -j {jobid} -o State,Elapsed,ExitCode", check=False)
        print(f"{role} {jobid}: {state.strip()}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    manifest, profile = manifest_of(args.job)
    logs = {role: log for role, _, log in slurm_jobs(manifest)}
    role = "build" if args.build else "timing"
    if role not in logs:
        raise SystemExit(f"job {args.job} had no build job")
    tail = f"tail -n {int(args.lines)} {shlex.quote(logs[role])}"
    print(ssh(profile, tail, check=False), end="")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest, profile = manifest_of(args.job)
    runs = REPO / "bench-out" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    remote = f"{manifest['jobdir']}/bench-out/runs/*.jsonl"
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", f"{profile.ssh}:{remote}", str(runs)], check=True
    )
    for _, _, log in slurm_jobs(manifest):
        subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", f"{profile.ssh}:{log}", str(STATE / args.job)],
            check=False,
        )
    if manifest.get("campaign"):
        campaigns = REPO / "bench-out" / "campaigns"
        campaigns.mkdir(parents=True, exist_ok=True)
        source = f"{profile.ssh}:{manifest['jobdir']}/bench-out/campaigns/*"
        subprocess.run(
            ["scp", "-q", "-r", "-o", "BatchMode=yes", source, str(campaigns)], check=False
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
        build = f"  build {m['build_jobid']}" if m.get("build_jobid") else ""
        print(
            f"{m['job']}  profile {m['profile']}  job {m.get('jobid', '(dry run)')}{build}  "
            f"arms {' '.join(m['arms'])}"
        )
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    print(ssh(profile, prune_script(cache_dir(profile), args.days, args.dry_run)), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fbench_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, fbench_args = argv[:cut], argv[cut + 1 :]
    p = argparse.ArgumentParser(prog="fbench_remote.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit", help="pack, copy, build (unless cached) and run fbench")
    s.add_argument("profile")
    s.add_argument("arms", nargs="+", help="as for fbench run; builds must be commits")
    s.add_argument("--name", help="job name (default fbench-<time>-<commit>)")
    s.add_argument("--after", help="Slurm job ID this job waits for (afterok)")
    s.add_argument(
        "--build-job",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build missing commits in a separate, wider job first (default); "
        "--no-build-job builds them in the timing job",
    )
    s.add_argument(
        "--serial",
        action="store_true",
        help="queue behind every other --serial timing job of yours on the machine",
    )
    s.add_argument("--dry-run", action="store_true", help="write the job files only")
    s.add_argument("--time", help="Slurm time limit of this job (replaces the profile's)")
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
            sp.add_argument("--build", action="store_true", help="the build job's log")
        sp.set_defaults(func=func)
    sub.add_parser("jobs", help="jobs submitted from here").set_defaults(func=cmd_jobs)
    sp = sub.add_parser("prune", help="delete cached builds unused for DAYS days")
    sp.add_argument("profile")
    sp.add_argument("--days", type=int, default=14)
    sp.add_argument("--dry-run", action="store_true", help="list what would go")
    sp.set_defaults(func=cmd_prune)
    args = p.parse_args(argv)
    args.fbench_args = fbench_args
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

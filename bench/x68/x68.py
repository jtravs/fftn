#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X68's campaign (dev/experiments/X68-free-space-stage.md): the free-space stage's
arms timed by fbench at each thread count, and their DRAM traffic counted.

    x68.py --out-dir DIR --build BUILD [--threads 1,8,18] [--place xeon|slurm|none]
           [--rounds N] [--budget S] [--filter REGEX] [--arms T,O,Of,G,C]
           [--traffic perf|pmu|none] [--traffic-threads 1,18] [--timing/--no-timing]
           [--profile-threads 1,8]

It is a campaign driver for scripts/remote/fbench_remote.py (`submit PROFILE fftn:SHA --campaign
bench/x68/x68.py -- ...`), and runs anywhere fbench runs.

Timing: one `fbench run` per thread count over catalogue x68, the arms STAGE=T (the baseline),
O, Of, G and C of the FFTN OpenMP backend (fbench-fftn_omp: the client application's runtime),
with STAGE_TIME giving the real time lines of the catalogue's 433-bin cases, three batches per
process (a stage is long) and THREADS for the threaded counts.

Placement (--place): `xeon` confines each run to one socket of a two-socket machine whose
scheduler binds nothing (the socket with the fewest busy cores over three seconds, its idlest
cores first: taskset, numactl --membind); `slurm` keeps the allocation's own binding and, for
more threads than one NUMA domain holds, interleaves memory over the allocation's nodes; `none`
does neither. Threaded runs set OMP_PLACES=cores and OMP_PROC_BIND=close.

Traffic (--traffic): each arm and case once more, in its own harness process, its counters over
one execute of the stage after a warm one:
- `perf`: `perf stat` around the process (every thread, kernel included), started disabled and
  enabled by the harness over the counted execute (--perf-ctl). The events are the core's
  offcore responses that miss the L3 (Intel Skylake-SP / Cascade Lake encoding): all data reads
  (demand, L1 and L2 prefetches) and all RFOs, each a 64-byte line from memory. Uncore memory
  controller counters need perf_event_paranoid 0 or the sampling driver's group, which a shared
  machine's user may not have; these core events need neither.
- `pmu`: the harness's own user-mode counters (--pmu) on its thread, so serial runs only: AMD Zen
  3's data cache fills from local or remote memory (PMCx044, umask 0x48) and its L2 prefetches
  that miss the L3 (PMCx072, umask 0xff), each a 64-byte line.
Writes are not counted by either: every sweep of these arms writes what it reads, so the report
gives the lines read (and read for ownership) in sweeps, bytes / V with V the stored field's
bytes, and the total estimated as twice that.

Profile (--profile-threads, a diagnosis): each arm and case checked once by the harness with
STAGE_PROFILE=1, whose note starts with the mean time of each of the arm's steps over three runs.

Results: the fbench runs in DIR/runs, and DIR/campaigns/x68/<stamp>.json with the run IDs, the
placement and load of each run, and the traffic per arm and case.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench"))

from fbenchlib import catalogue  # noqa: E402

# The catalogue's real time lines: 433 stored bins of a 1728-point real line from bin 1 (the
# client application's Cartesian filament rows); the other cases are envelopes.
STAGE_TIME = "433.1728.1"
LIBRARY = "fftn_omp"

PERF_EVENTS = [
    "cpu/event=0xb7,umask=0x1,offcore_rsp=0x3FFC400491,name=rd_l3_miss/",
    "cpu/event=0xbb,umask=0x1,offcore_rsp=0x3FFC400122,name=rfo_l3_miss/",
]
PMU_EVENTS = "fill_dram=0x4844,l2pf_miss_l3=0xff72"


def busy_cores(seconds: float = 3.0) -> dict[int, float]:
    """Each CPU's busy fraction over `seconds` (Linux /proc/stat)."""

    def sample() -> dict[int, tuple[int, int]]:
        out = {}
        for line in Path("/proc/stat").read_text().splitlines():
            m = re.match(r"cpu(\d+)\s+(.*)", line)
            if m:
                v = [int(x) for x in m.group(2).split()]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                out[int(m.group(1))] = (sum(v), idle)
        return out

    a = sample()
    time.sleep(seconds)
    b = sample()
    busy = {}
    for cpu, (total, idle) in b.items():
        dt = total - a[cpu][0]
        di = idle - a[cpu][1]
        busy[cpu] = 0.0 if dt <= 0 else 1.0 - di / dt
    return busy


def topology() -> dict[int, list[tuple[int, list[int]]]]:
    """Per package: its physical cores as (core's first CPU, the core's CPUs)."""
    packages: dict[int, dict[int, list[int]]] = {}
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        cpu = int(cpu_dir.name[3:])
        topo = cpu_dir / "topology"
        if not topo.exists():
            continue
        pkg = int((topo / "physical_package_id").read_text())
        core = int((topo / "core_id").read_text())
        packages.setdefault(pkg, {}).setdefault(core, []).append(cpu)
    return {
        p: sorted((min(cpus), sorted(cpus)) for cpus in cores.values())
        for p, cores in packages.items()
    }


def numa_node_of(cpu: int) -> int:
    for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
        if (node / f"cpu{cpu}").exists():
            return int(node.name[4:])
    return 0


def placement(place: str, threads: int) -> tuple[list[str], dict]:
    """The command prefix confining a run, and what it chose."""
    if place == "none":
        return [], {"place": "none"}
    if place == "slurm":
        prefix: list[str] = []
        nodes = sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
        allowed = os.sched_getaffinity(0)
        cpus_per_node = max(1, len(allowed) // max(1, len(nodes)))
        if threads > 16 and shutil.which("numactl"):
            prefix = ["numactl", "--interleave=all"]
        return prefix, {
            "place": "slurm",
            "allowed_cpus": len(allowed),
            "interleave": bool(prefix),
            "cpus_per_node_estimate": cpus_per_node,
        }
    busy = busy_cores()
    best = None
    for pkg, cores in topology().items():
        # A core is as busy as its busiest CPU (a sibling's work slows it too).
        scored = sorted((max(busy.get(c, 0.0) for c in cpus), first) for first, cpus in cores)
        load = sum(s for s, _ in scored)
        if best is None or load < best[0]:
            best = (load, pkg, scored)
    assert best is not None
    load, pkg, scored = best
    chosen = sorted(first for _, first in scored[:threads])
    node = numa_node_of(chosen[0])
    prefix = ["taskset", "-c", ",".join(str(c) for c in chosen)]
    if shutil.which("numactl"):
        prefix += ["numactl", f"--membind={node}"]
    return prefix, {
        "place": "xeon",
        "package": pkg,
        "node": node,
        "cpus": chosen,
        "package_busy_cores": round(load, 2),
        "chosen_busy": round(sum(s for s, _ in scored[:threads]), 2),
    }


def arm_specs(build: str, arms: list[str], threads: int) -> list[str]:
    extra = f"+THREADS={threads}" if threads > 1 else ""
    return [f"{a}={LIBRARY}:{build}+STAGE={a}+STAGE_TIME={STAGE_TIME}{extra}" for a in arms]


def cpuset() -> dict:
    """This process's CPUs and memory nodes (Linux), recorded with every run."""
    out: dict = {}
    with contextlib.suppress(AttributeError, OSError):
        out["cpus"] = sorted(os.sched_getaffinity(0))
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Mems_allowed_list:"):
                out["mems"] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return out


def load_average() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return float("nan")


def run_timing(args, threads: int, prefix: list[str], env: dict) -> dict:
    out = Path(args.out_dir)
    before = {p.name for p in (out / "runs").glob("*")} if (out / "runs").exists() else set()
    cmd = [
        *prefix,
        sys.executable,
        str(REPO / "bench" / "fbench.py"),
        "--out-dir",
        str(out),
        "run",
        *arm_specs(args.build, args.arms, threads),
        "--catalogue",
        "x68",
        "--tier",
        "full",
        "--batches",
        "3",
        "--warmup",
        "0",
        "--min-rounds",
        str(args.rounds),
        "--max-rounds",
        str(args.max_rounds or args.rounds),
        "--budget",
        str(args.budget),
        "--process-timeout",
        str(args.process_timeout),
    ]
    if args.filter:
        cmd += ["--filter", args.filter]
    print("+", " ".join(cmd), flush=True)
    started = time.time()
    done = subprocess.run(cmd, env=env, check=False)
    after = {p.name for p in (out / "runs").glob("*")} if (out / "runs").exists() else set()
    return {
        "threads": threads,
        "returncode": done.returncode,
        "seconds": round(time.time() - started, 1),
        "new_run_files": sorted(after - before),
    }


def harness(build: str) -> Path:
    return Path(build) / "bench" / f"fbench-{LIBRARY}"


def stage_options(arm: str, threads: int) -> str:
    opts = f"STAGE={arm} STAGE_TIME={STAGE_TIME}"
    return opts + (f" THREADS={threads}" if threads > 1 else "")


def field_bytes(case: catalogue.Case) -> int:
    n = 1
    for d in case.dims:
        n *= d
    return n * case.batch * 16


def harness_args(case_file: Path, arm: str, threads: int) -> list[str]:
    return [
        "run",
        "--cases",
        str(case_file),
        "--options",
        stage_options(arm, threads),
        "--batches",
        "1",
        "--batch-time",
        "1e-9",
        "--warmup",
        "0",
    ]


def parse_case_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "case":
            return d
    return {}


def traffic_perf(args, case, arm, threads, prefix, env, tmp: Path) -> dict:
    case_file = tmp / "case.txt"
    case_file.write_text(case.line() + "\n")
    ctl = tmp / "ctl.fifo"
    ack = tmp / "ack.fifo"
    for f in (ctl, ack):
        if f.exists():
            f.unlink()
        os.mkfifo(f)
    out = tmp / "perf.csv"
    cmd = [
        *prefix,
        "perf",
        "stat",
        "-x",
        ",",
        "--no-big-num",
        "-o",
        str(out),
        "-D",
        "-1",
        "--control",
        f"fifo:{ctl},{ack}",
        "-e",
        ",".join(PERF_EVENTS),
        "--",
        str(harness(args.build)),
        *harness_args(case_file, arm, threads),
        "--perf-ctl",
        f"{ctl},{ack}",
    ]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    result = parse_case_result(done.stdout)
    counts = {}
    if out.exists():
        for line in out.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 3 and not line.startswith("#"):
                try:
                    counts[parts[2]] = float(parts[0])
                except ValueError:
                    counts[parts[2]] = None
    calls = (result.get("pmu") or {}).get("calls")
    lines = sum(v for v in counts.values() if v is not None) if counts else None
    return finish_traffic(case, arm, threads, result, counts, calls, lines, done)


def traffic_pmu(args, case, arm, threads, prefix, env, tmp: Path) -> dict:
    case_file = tmp / "case.txt"
    case_file.write_text(case.line() + "\n")
    cmd = [
        *prefix,
        str(harness(args.build)),
        *harness_args(case_file, arm, threads),
        "--pmu",
        PMU_EVENTS,
    ]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    result = parse_case_result(done.stdout)
    pmu = result.get("pmu") or {}
    counts = {k: v for k, v in pmu.items() if k in ("fill_dram", "l2pf_miss_l3")}
    calls = pmu.get("calls")
    # The harness reports its counts per call already.
    lines = sum(counts.values()) * calls if counts and calls else None
    return finish_traffic(case, arm, threads, result, counts, calls, lines, done, pmu)


def finish_traffic(case, arm, threads, result, counts, calls, lines, done, extra=None) -> dict:
    v = field_bytes(case)
    sweeps = lines * 64 / v / calls if lines is not None and calls else None
    return {
        "case": case.id,
        "arm": arm,
        "threads": threads,
        "status": result.get("status"),
        "check": (result.get("check") or {}).get("pass"),
        "calls": calls,
        "counts": counts,
        "field_bytes": v,
        "read_sweeps": None if sweeps is None else round(sweeps, 3),
        "total_sweeps_estimate": None if sweeps is None else round(2 * sweeps, 3),
        "load": load_average(),
        "returncode": done.returncode,
        "stderr_tail": done.stderr[-400:] if done.returncode != 0 else "",
        "pmu": extra,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--threads", default="1")
    p.add_argument("--place", choices=("xeon", "slurm", "none"), default="none")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--max-rounds", type=int)
    p.add_argument("--budget", type=float, default=4 * 3600.0)
    p.add_argument("--process-timeout", type=float, default=3600.0)
    p.add_argument("--filter")
    p.add_argument("--arms", default="T,O,Of,G,C")
    p.add_argument("--traffic", choices=("perf", "pmu", "none"), default="none")
    p.add_argument("--traffic-threads", default="1")
    p.add_argument("--timing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile-threads", default="")
    args = p.parse_args(argv)
    args.arms = args.arms.split(",")
    args.build = str(Path(args.build).resolve())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    record: dict = {
        "stamp": stamp,
        "build": args.build,
        "argv": sys.argv[1:],
        "timing": [],
        "traffic": [],
    }
    state = Path(args.out_dir) / "campaigns" / "x68"
    state.mkdir(parents=True, exist_ok=True)
    path = state / f"{stamp}.json"

    def save() -> None:
        path.write_text(json.dumps(record, indent=1) + "\n")

    if args.timing:
        for threads in [int(t) for t in args.threads.split(",")]:
            prefix, chosen = placement(args.place, threads)
            env = dict(os.environ)
            if threads > 1:
                env.update(OMP_PLACES="cores", OMP_PROC_BIND="close")
            entry = {"placement": chosen, "cpuset": cpuset(), "load_before": load_average()}
            entry.update(run_timing(args, threads, prefix, env))
            entry["load_after"] = load_average()
            record["timing"].append(entry)
            save()
    if args.traffic != "none":
        cases = catalogue.load("x68").select(None, args.filter)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            for threads in [int(x) for x in args.traffic_threads.split(",")]:
                prefix, chosen = placement(args.place, threads)
                env = dict(os.environ)
                if threads > 1:
                    env.update(OMP_PLACES="cores", OMP_PROC_BIND="close")
                for case in cases:
                    for arm in args.arms:
                        if args.traffic == "perf":
                            r = traffic_perf(args, case, arm, threads, prefix, env, tmp)
                        else:
                            r = traffic_pmu(args, case, arm, threads, prefix, env, tmp)
                        r["placement"] = chosen
                        print(
                            json.dumps(
                                {
                                    k: r[k]
                                    for k in ("case", "arm", "threads", "read_sweeps", "check")
                                }
                            ),
                            flush=True,
                        )
                        record["traffic"].append(r)
                        save()
    if args.profile_threads:
        record["profile"] = []
        cases = catalogue.load("x68").select(None, args.filter)
        with tempfile.TemporaryDirectory() as t:
            case_file = Path(t) / "case.txt"
            for threads in [int(x) for x in args.profile_threads.split(",")]:
                prefix, chosen = placement(args.place, threads)
                env = dict(os.environ)
                if threads > 1:
                    env.update(OMP_PLACES="cores", OMP_PROC_BIND="close")
                for case in cases:
                    case_file.write_text(case.line() + "\n")
                    for arm in args.arms:
                        cmd = [
                            *prefix,
                            str(harness(args.build)),
                            "run",
                            "--cases",
                            str(case_file),
                            "--options",
                            stage_options(arm, threads) + " STAGE_PROFILE=1",
                            "--check-only",
                        ]
                        done = subprocess.run(
                            cmd, env=env, capture_output=True, text=True, check=False
                        )
                        note = parse_case_result(done.stdout).get("note", "")
                        head = note.split(" " + arm + " (")[0] if note else done.stderr[-300:]
                        print(f"profile {case.id} {arm} threads {threads}: {head}", flush=True)
                        record["profile"].append(
                            {
                                "case": case.id,
                                "arm": arm,
                                "threads": threads,
                                "profile": head,
                                "placement": chosen,
                                "load": load_average(),
                            }
                        )
                        save()
    save()
    print(f"x68: campaign record {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

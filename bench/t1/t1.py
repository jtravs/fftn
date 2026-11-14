#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Driver of experiment T1's harness (dev/experiments/T1-dispatch-wait-costs.md).

Runs the harness binaries (t1_costs, and t1_costs_omp where built) in process rounds, in
rotating order, on a quiet machine; stores every process's output with the host fingerprint
under <out-dir>/t1/runs/<run-id>.jsonl; and reports, per probe, team size, variant and metric,
the median over rounds of the in-process medians with the range over rounds.

uv run --frozen python bench/t1/t1.py run --build build/check-release --rounds 10
uv run --frozen python bench/t1/t1.py show latest
uv run --frozen python bench/t1/t1.py schedule --build build/check-release \\
    --catalogue threads --tier quick -- --threads 8,16,64 --group 8 --cpus 128
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbenchlib import arms, catalogue, machine, store

BINARIES = ("t1_costs", "t1_costs_omp")
# Environment variables that change how threads are placed and wait: recorded with the run.
ENV_KEYS = (
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OMP_WAIT_POLICY",
    "OMP_NUM_THREADS",
    "GOMP_SPINCOUNT",
    "KMP_BLOCKTIME",
    "KMP_AFFINITY",
)
KEY_FIELDS = ("binary", "probe", "executor", "threads", "group", "variant", "metric")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_binary(exe: Path, args: list[str], timeout: float) -> tuple[list[dict], str]:
    """One harness process: its JSON lines, and an error message if it failed."""
    try:
        done = subprocess.run(
            [str(exe), *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return [], f"timed out after {timeout:.0f} s"
    lines = []
    for text in done.stdout.splitlines():
        try:
            lines.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    if done.returncode != 0:
        return lines, f"exit {done.returncode}: {done.stderr.strip()[-2000:]}"
    if not lines or lines[-1].get("type") != "end":
        return lines, "no end line"
    return lines, ""


def summarise(processes: list[dict]) -> list[dict]:
    """Per (binary, probe, executor, threads, group, variant, metric): the median over
    rounds of each process's median, the lowest and highest of them, and their spread."""
    per_key: dict[tuple, list[float]] = {}
    for proc in processes:
        for line in proc["lines"]:
            if line.get("type") != "probe":
                continue
            for metric, values in line["metrics"].items():
                key = (
                    proc["binary"],
                    line["probe"],
                    line["executor"],
                    line["threads"],
                    line["group"],
                    line["variant"],
                    metric,
                )
                per_key.setdefault(key, []).append(values["median"])
    rows = []
    for key, medians in per_key.items():
        mid = statistics.median(medians)
        low, high = min(medians), max(medians)
        rows.append(
            {
                **dict(zip(KEY_FIELDS, key, strict=True)),
                "rounds": len(medians),
                "median": mid,
                "low": low,
                "high": high,
                "spread": (high - low) / mid if mid > 0 else 0.0,
            }
        )
    rows.sort(key=lambda r: tuple(str(r[k]).zfill(8) for k in KEY_FIELDS))
    return rows


def print_rows(rows: list[dict]) -> None:
    print(
        f"{'binary':<13} {'probe':<6} {'executor':<14} {'thr':>3} {'grp':>3} "
        f"{'variant':<26} {'metric':<20} {'median':>11} {'low':>11} {'high':>11} spread"
    )
    for r in rows:
        print(
            f"{r['binary']:<13} {r['probe']:<6} {r['executor']:<14} {r['threads']:>3} "
            f"{r['group']:>3} {r['variant']:<26} {r['metric']:<20} {r['median']:>11.4g} "
            f"{r['low']:>11.4g} {r['high']:>11.4g} {100 * r['spread']:5.1f}%"
        )


def cmd_run(args: argparse.Namespace) -> int:
    build = Path(args.build)
    names = args.binaries.split(",") if args.binaries else list(BINARIES)
    exes = {n: build / "bench" / f"{n}{arms.EXE_SUFFIX}" for n in names}
    missing = [n for n, e in exes.items() if not e.is_file()]
    if args.binaries and missing:
        log(f"t1: not built: {', '.join(missing)}")
        return 2
    exes = {n: e for n, e in exes.items() if e.is_file()}
    if not exes:
        log(f"t1: no harness binaries in {build / 'bench'}")
        return 2
    harness = list(args.harness)
    if harness and harness[0] == "--":
        harness = harness[1:]
    if args.quick:
        harness.append("--quick")
    run_id = store.new_run_id()
    st = store.RunStore(Path(args.out_dir) / "t1", run_id)
    limits = machine.QuietLimits(
        args.max_load if args.max_load is not None else machine.QuietLimits.defaults().max_load,
        args.max_busy if args.max_busy is not None else machine.QuietLimits.defaults().max_busy,
    )
    st.append(
        {
            "type": "run",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": sys.argv[1:],
            "host": machine.fingerprint(),
            "thermal": machine.thermal_note(),
            "source": arms.source_identity(),
            "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
            "binaries": sorted(exes),
            "harness_args": harness,
            "quiet_check": not args.no_quiet_check,
        }
    )
    processes: list[dict] = []
    failures: list[str] = []
    order = sorted(exes)
    for rnd in range(args.rounds):
        if not args.no_quiet_check:
            for attempt in range(args.busy_retries + 1):
                state = machine.load_state()
                quiet, why = machine.is_quiet(state, limits)
                if quiet:
                    break
                log(f"t1: round {rnd}: machine busy ({why}); waiting {args.busy_wait:.0f} s")
                if attempt == args.busy_retries:
                    log("t1: the machine stayed busy; stopping")
                    return 3
                time.sleep(args.busy_wait)
        # Rotate the order between rounds, so no binary always runs first.
        k = rnd % len(order)
        for name in order[k:] + order[:k]:
            lines, error = run_binary(exes[name], harness, args.process_timeout)
            record = {"type": "process", "round": rnd, "binary": name, "lines": lines}
            if error:
                record["error"] = error
                failures.append(f"{name} round {rnd}: {error}")
            st.append(record)
            if not error:
                processes.append(record)
        log(f"t1: round {rnd + 1} of {args.rounds} done")
    rows = summarise(processes)
    st.append({"type": "summary", "rows": rows, "failures": failures})
    print_rows(rows)
    print(f"\nrun {run_id}: {st.path}")
    for f in failures:
        print(f"t1 FAILURE: {f}")
    if args.strict:
        problems = check_complete(processes, exes)
        for p in problems:
            print(f"t1 FAILURE: {p}")
        failures += problems
    return 1 if failures else 0


def check_complete(processes: list[dict], exes: dict[str, Path]) -> list[str]:
    """Every binary ran, described itself, and reported every probe."""
    problems = []
    for name in exes:
        procs = [p for p in processes if p["binary"] == name]
        if not procs:
            problems.append(f"{name}: no successful process")
            continue
        lines = procs[0]["lines"]
        if not lines or lines[0].get("type") != "describe":
            problems.append(f"{name}: no describe line")
        probes = {x["probe"] for x in lines if x.get("type") == "probe"}
        want = {"work", "bulk", "team", "phase", "claim", "wake"}
        if not want <= probes:
            problems.append(f"{name}: probes missing: {sorted(want - probes)}")
        executors = {x["executor"] for x in lines if x.get("type") == "probe"}
        if name.endswith("_omp") and not {"omp::executor", "omp-region"} <= executors:
            problems.append(f"{name}: no OpenMP probes")
    return problems


def cmd_show(args: argparse.Namespace) -> int:
    path = store.find_run(Path(args.out_dir) / "t1", args.run_id)
    records = store.load(path)
    run = next(r for r in records if r["type"] == "run")
    print(f"run {run['run']}  {run['host']['cpu']}  env {run.get('env', {})}")
    processes = [r for r in records if r["type"] == "process" and "error" not in r]
    print_rows(summarise(processes))
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    exe = Path(args.build) / "bench" / f"t1_costs{arms.EXE_SUFFIX}"
    cat = catalogue.load(args.catalogue)
    cases = cat.select(args.tier, args.filter)
    work = Path(args.out_dir) / "t1"
    work.mkdir(parents=True, exist_ok=True)
    cases_file = work / "schedule-cases.txt"
    cases_file.write_text("".join(c.line() + "\n" for c in cases))
    harness = list(args.harness)
    if harness and harness[0] == "--":
        harness = harness[1:]
    lines, error = run_binary(exe, ["schedule", "--cases", str(cases_file), *harness], 600)
    if error:
        log(f"t1 schedule: {error}")
        return 1
    rows = [x for x in lines if x.get("type") == "schedule"]
    if len(rows) == 0 or len({r["id"] for r in rows}) != len(cases):
        log("t1 schedule: not every case was scheduled")
        return 1
    print(
        f"{'case':<26} {'thr':>3} {'grp':>3} {'tasks':>6} {'phases':>6} {'widest':>6} "
        f"{'task_us':>9} inline groups dealt whole split cap"
    )
    for r in rows:
        print(
            f"{r['id']:<26} {r['threads']:>3} {r['group']:>3} {r['tasks']:>6} "
            f"{r['phases']:>6} {r['widest']:>6} {r['task_ns'] / 1000:>9.2f} "
            f"{'yes' if r['inline'] else 'no':>6} {r['groups']:>6} {r['lanes_dealt']:>5} "
            f"{r['whole']:>5} {r['split']:>5} {r['cap']:>3}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="t1", description="experiment T1's harness driver")
    p.add_argument("--out-dir", default=str(store.DEFAULT_OUT), help="results root (bench-out)")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process rounds of the harness binaries")
    r.add_argument("--build", default=str(store.REPO / "build" / "check-release"))
    r.add_argument("--binaries", help=f"comma-separated, default those built of {BINARIES}")
    r.add_argument("--rounds", type=int, default=10)
    r.add_argument("--quick", action="store_true", help="the harness's short mode (smoke)")
    r.add_argument("--strict", action="store_true", help="exit 1 unless every probe reported")
    r.add_argument("--no-quiet-check", action="store_true")
    r.add_argument("--max-load", type=float)
    r.add_argument("--max-busy", type=float)
    r.add_argument("--busy-wait", type=float, default=10.0)
    r.add_argument("--busy-retries", type=int, default=30)
    r.add_argument("--process-timeout", type=float, default=3600.0)
    r.add_argument("harness", nargs=argparse.REMAINDER, help="-- then harness arguments")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("show", help="report of a stored run")
    s.add_argument("run_id", help="run ID, unique prefix, 'latest' or a .jsonl path")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("schedule", help="what the scheduler decides for catalogue cases")
    c.add_argument("--build", default=str(store.REPO / "build" / "check-release"))
    c.add_argument("--catalogue", default="threads")
    c.add_argument("--tier", default="quick", choices=catalogue.TIERS)
    c.add_argument("--filter", help="regex on case ids")
    c.add_argument("harness", nargs=argparse.REMAINDER, help="-- then schedule arguments")
    c.set_defaults(func=cmd_schedule)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log(f"t1: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

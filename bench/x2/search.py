#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X2's search driver: the best radix sequence per length, then v0 against it.

    search.py [--out-dir D] [--build B] plan      [--lengths L] [--isa T] [--v0]
    search.py [--out-dir D] [--build B] all       --name NAME [--lengths L] [--isa T] [--dry-run]
    search.py [--out-dir D] [--build B] screen|refine|confirm --name NAME [--lengths L] [--isa T]
    search.py [--out-dir D]             report    --name NAME[,NAME...]
    search.py                           remote    PROFILE... [--isa T]
    search.py                           catalogue [--check]
    search.py              [--build B]  selftest

The method, stages and settings are dev/experiments/X2-radix-sets.md ("Method as built");
the candidates and the cost estimate are radices.py. Every timing is an fbench run on the
cases of bench/catalogues/x2.toml, one run per length and stage, stored as usual under
bench-out/runs/; the campaign's state (which run is which stage of which length, the
winners) is bench-out/campaigns/x2-NAME.json, written after every run.

--isa T forces every arm, the baseline included, to the fbench FFTN backend's ISA=T, and
prices the cost estimate (the shortlist above 1024) with T's C per precision; the campaign
records T and refuses to resume under another. Without it each arm runs the machine's best
target and the estimate is unpadded, as in X2's first campaigns.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from fbenchlib import analysis, arms, catalogue, machine, store  # noqa: E402

from x2 import radices  # noqa: E402

REPO = BENCH.parent
CATALOGUE = BENCH / "catalogues" / "x2.toml"
DEFAULT_BUILD = REPO / "build" / "check-release"
V0 = "v0"  # the baseline arm's label: planner v0's rules, no option

# fbench run settings per stage (the method's numbers).
SCREEN_ROUNDS = 2
SCREEN = [
    *("--min-rounds", str(SCREEN_ROUNDS), "--max-rounds", str(SCREEN_ROUNDS)),
    *("--batches", "3", "--batch-time", "0.001", "--warmup", "0.005"),
    *("--check-budget", "65536", "--budget", "86400"),
]
REFINE_TOP = 4  # per case, from the screening
REFINE_ROUNDS = 6
REFINE = [
    *("--min-rounds", str(REFINE_ROUNDS), "--max-rounds", str(REFINE_ROUNDS)),
    *("--budget", "86400"),
]
CONFIRM = ["--min-rounds", "6", "--max-rounds", "30", "--budget", "1800"]
SCREEN_BUDGET_S = 900.0  # estimated screening time per length above which candidates go
DECISION = 0.03  # the decision rule's effect size


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---- Lengths and candidates ----------------------------------------------------------------


def x2_cases() -> list[catalogue.Case]:
    return list(catalogue.load(str(CATALOGUE)).cases)


def lengths_of(cases: list[catalogue.Case]) -> list[int]:
    return sorted({c.dims[0] for c in cases})


def select_lengths(all_ns: list[int], spec: str | None) -> list[int]:
    """--lengths: comma-separated lengths and ranges a-b, a- or -b (inclusive)."""
    if not spec:
        return all_ns
    keep: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, _, hi = part.partition("-")
            keep |= {n for n in all_ns if int(lo or 0) <= n <= int(hi or sys.maxsize)}
        elif int(part) in all_ns:
            keep.add(int(part))
        else:
            raise SystemExit(f"--lengths: {part} is not an X2 length")
    return [n for n in all_ns if n in keep]


def fbench_fftn(build: Path) -> Path:
    exe = arms.executable(build, "fftn")
    if not exe.exists():
        raise SystemExit(f"no {exe}: build the fbench-fftn target first")
    return exe


def isa_options(isa: str | None) -> str:
    return f"ISA={isa}" if isa else ""


def harness_check(
    exe: Path, cases: list[catalogue.Case], options: str = ""
) -> tuple[list[dict], str]:
    """One check-only process over the cases: the case lines and any process error."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cases.txt"
        path.write_text("".join(c.line() + "\n" for c in cases))
        done = subprocess.run(
            [str(exe), "run", "--cases", str(path), "--options", options, "--check-only"],
            capture_output=True,
            text=True,
            check=False,
        )
    lines = [json.loads(x) for x in done.stdout.splitlines() if x.startswith("{")]
    error = next((x["message"] for x in lines if x.get("type") == "error"), "")
    if done.returncode != 0 and not error:
        error = done.stderr.strip() or f"exit {done.returncode}"
    return [x for x in lines if x.get("type") == "case"], error


def v0_choices(
    build: Path, ns: list[int], isa: str | None = None, precision: str = "f64"
) -> dict[int, radices.Sequence]:
    """The baseline's sequence per length on this machine's target (or `isa`) in one
    precision, from the backend's notes."""
    cases = [catalogue.Case("c2c", (n,), 1, "out", -1, precision=precision) for n in ns]
    results, error = harness_check(fbench_fftn(build), cases, isa_options(isa))
    if error:
        raise SystemExit(f"fbench-fftn: {error}")
    out = {}
    for res in results:
        n, seq = radices.from_note(res.get("note", ""))
        out[n] = seq
    return out


def base_of(entry: dict, case: str) -> radices.Sequence:
    """The baseline's sequence for a case: "v0" holds double's; campaigns from X3 on also
    record single precision's as "v0_f32" (where the two differ, earlier campaigns name
    double's sequence for both)."""
    if case.endswith("-f32") and "v0_f32" in entry:
        return tuple(entry["v0_f32"])
    return tuple(entry["v0"])


# ---- Time estimate -------------------------------------------------------------------------


def process_seconds(n: int, batches_of: list[int], gflops: float, screen: bool) -> float:
    """A rough estimate of one harness process over the X2 cases of length n."""
    batch_time, warmup, timed = (0.001, 0.005, 3) if screen else (0.002, 0.02, 7)
    total = 0.03  # start-up and describe
    for batch in batches_of:
        for scale in (1.0, 0.5):  # double, then single precision at about twice the speed
            flops = 5.0 * n * math.log2(n) * batch
            execute = flops / (gflops * 1e9) * scale
            per_batch = max(batch_time, execute)
            plan_and_check = 0.002 + 3e-8 * n * batch + (0.001 if screen else 0.01)
            total += plan_and_check + warmup + per_batch * (timed + 2)
    return total


def batches_of(n: int, cases: list[catalogue.Case]) -> list[int]:
    return sorted({c.batch for c in cases if c.dims[0] == n})


# ---- Campaign state ------------------------------------------------------------------------


def state_path(out_dir: Path, name: str) -> Path:
    return out_dir / "campaigns" / f"x2-{name}.json"


def load_state(out_dir: Path, name: str) -> dict:
    path = state_path(out_dir, name)
    if path.exists():
        return json.loads(path.read_text())
    return {"name": name, "experiment": "X2", "lengths": {}}


def check_isa(state: dict, isa: str | None) -> None:
    """A campaign's timings are of one target: refuse to resume it under another."""
    if state["lengths"] and state.get("isa") != isa:
        raise SystemExit(
            f"campaign {state['name']} ran with --isa {state.get('isa') or '(none)'}, "
            f"not {isa or '(none)'}: use another --name"
        )


def save_state(out_dir: Path, state: dict) -> None:
    path = state_path(out_dir, state["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


# ---- Running fbench ------------------------------------------------------------------------


def fbench_command(
    out_dir: Path,
    build: Path,
    n: int,
    labels: list[str],
    settings: list[str],
    extra: list[str],
    isa: str | None = None,
) -> list[str]:
    lib = f"fftn+ISA={isa}" if isa else "fftn"
    arm_specs = [f"{V0}={lib}"] + [f"{lab}={lib}+RADICES={lab}" for lab in labels]
    return [
        sys.executable,
        str(BENCH / "fbench.py"),
        "--out-dir",
        str(out_dir),
        "run",
        *arm_specs,
        "--build",
        str(build),
        "--catalogue",
        str(CATALOGUE),
        "--tier",
        "full",
        "--filter",
        radices.length_filter(n),
        *settings,
        *extra,
    ]


def run_fbench(cmd: list[str], dry_run: bool) -> str | None:
    """Runs one fbench run and returns its run ID (None for a dry run)."""
    if dry_run:
        shown = [*cmd[:12], "..."] if len(cmd) > 40 else cmd
        print(" ".join(shlex.quote(x) for x in shown), f"({len(cmd)} words)")
        return None
    done = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=REPO)
    m = re.search(r"^run (\S+): ", done.stderr, re.MULTILINE)
    if done.returncode not in (0, 1) or not m:
        raise RuntimeError(f"fbench failed (exit {done.returncode}):\n{done.stderr[-4000:]}")
    if done.returncode == 1:
        log(f"run {m[1]}: some output was WRONG; see fbench show {m[1]}")
    return m[1]


def records_of(out_dir: Path, run_id: str) -> list[dict]:
    return store.load(store.find_run(out_dir, run_id))


def complete_records(out_dir: Path, run_id: str, rounds: int) -> list[dict]:
    """A screening or refining run's records; stops the campaign if the run has fewer valid
    rounds than it was given (fbench stops a run at a crash or a refused option), since
    ranking what is left would hide the failure."""
    records = records_of(out_dir, run_id)
    valid = sum(1 for r in records if r["type"] == "round" and r.get("valid"))
    if valid < rounds:
        reasons = {r.get("reason") for r in records if r["type"] == "round" and not r["valid"]}
        raise RuntimeError(
            f"run {run_id}: {valid} of {rounds} valid rounds ({'; '.join(map(str, reasons))})"
        )
    return records


def paired_medians(records: list[dict]) -> dict[str, dict[str, float]]:
    """arm -> case -> median over valid rounds of log(arm's median ns / v0's, same round).
    Arms with a wrong output or no timing in some valid round are left out of that case."""
    valid = {r["round"] for r in records if r["type"] == "round" and r.get("valid")}
    ns: dict[tuple[str, str], dict[int, float]] = {}
    bad: set[tuple[str, str]] = set()
    for r in records:
        if r["type"] != "timing" or r["round"] not in valid:
            continue
        key = (r["arm"], r["case"])
        if r.get("status") != "ok" or not r.get("check", {}).get("pass") or not r.get("median_ns"):
            bad.add(key)
            continue
        ns.setdefault(key, {})[r["round"]] = r["median_ns"]
    out: dict[str, dict[str, float]] = {}
    for (arm, case), per_round in ns.items():
        base = ns.get((V0, case), {})
        common = sorted(set(per_round) & set(base))
        if (arm, case) in bad or not common or len(common) < len(valid):
            continue
        logs = [math.log(per_round[k] / base[k]) for k in common]
        out.setdefault(arm, {})[case] = statistics.median(logs)
    return out


def ranked(medians: dict[str, dict[str, float]], case: str) -> list[tuple[float, str]]:
    """(log ratio to v0, arm) of every arm with a value for the case, best first; the
    identical-arm control is not a candidate."""
    rows = [(per[case], arm) for arm, per in medians.items() if case in per and arm != f"{V0}~"]
    return sorted(rows)


# ---- Stages --------------------------------------------------------------------------------


def stage_screen(ctx: argparse.Namespace, state: dict, n: int, cases: list[str]) -> None:
    entry = state["lengths"].setdefault(str(n), {})
    if "screen" in entry:
        return
    v0 = tuple(entry["v0"]) if "v0" in entry else None
    ops = radices.codelet_ops()
    lanes = radices.LANES[ctx.isa] if ctx.isa else ()
    v0s = {v0, tuple(entry.get("v0_f32", v0 or ()))}
    cands = [
        s for s in radices.candidates(n, v0, ops, lanes, ctx.codelets, ctx.require) if s not in v0s
    ]
    per_process = process_seconds(n, batches_of(n, ctx.cases), ctx.gflops, screen=True)
    fits = int(ctx.screen_budget / (SCREEN_ROUNDS * per_process)) - 2
    dropped = 0
    if len(cands) > fits:
        # Over the budget: keep the best by the cost estimate (v0 is the baseline anyway).
        # Priced at the double-precision C of a forced target (the narrower one).
        c = lanes[0] if lanes else None
        keep = sorted(cands, key=lambda s: (radices.cost(s, ops, c), s))[: max(fits, 1)]
        dropped = len(cands) - len(keep)
        kept = set(keep)
        cands = [s for s in cands if s in kept]
    labels = [radices.label(s) for s in cands]
    log(f"x2 screen n = {n}: {len(labels)} candidates besides v0, {dropped} dropped")
    cmd = fbench_command(ctx.out_dir, ctx.build, n, labels, SCREEN, ctx.fbench_extra, ctx.isa)
    run_id = run_fbench(cmd, ctx.dry_run)
    if run_id is None:
        return
    medians = paired_medians(complete_records(ctx.out_dir, run_id, SCREEN_ROUNDS))
    top = {c: [arm for _, arm in ranked(medians, c)[:REFINE_TOP]] for c in cases}
    timed = {lab for lab in labels if lab in medians}
    entry["screen"] = {
        "run": run_id,
        "candidates": len(labels),
        "dropped": dropped,
        "untimed": sorted(set(labels) - timed),
        "top": top,
    }


def stage_refine(ctx: argparse.Namespace, state: dict, n: int, cases: list[str]) -> None:
    entry = state["lengths"][str(n)]
    if "refine" in entry or "screen" not in entry:
        return
    labels = sorted({a for c in cases for a in entry["screen"]["top"].get(c, []) if a != V0})
    if not labels:
        entry["refine"] = {"run": None, "winners": dict.fromkeys(cases, V0)}
        return
    log(f"x2 refine n = {n}: {len(labels)} candidates besides v0")
    cmd = fbench_command(ctx.out_dir, ctx.build, n, labels, REFINE, ctx.fbench_extra, ctx.isa)
    run_id = run_fbench(cmd, ctx.dry_run)
    if run_id is None:
        return
    medians = paired_medians(complete_records(ctx.out_dir, run_id, REFINE_ROUNDS))
    winners = {}
    for c in cases:
        best = ranked(medians, c)
        winners[c] = best[0][1] if best and best[0][0] < 0 else V0
    entry["refine"] = {"run": run_id, "winners": winners}


def stage_confirm(ctx: argparse.Namespace, state: dict, n: int, cases: list[str]) -> None:
    entry = state["lengths"][str(n)]
    if "confirm" in entry or "refine" not in entry:
        return
    labels = sorted({w for w in entry["refine"]["winners"].values() if w != V0})
    if not labels:
        entry["confirm"] = {"run": None}
        return
    log(f"x2 confirm n = {n}: v0 against {', '.join(labels)}")
    cmd = fbench_command(ctx.out_dir, ctx.build, n, labels, CONFIRM, ctx.fbench_extra, ctx.isa)
    entry["confirm"] = {"run": run_fbench(cmd, ctx.dry_run)}


STAGES = {"screen": stage_screen, "refine": stage_refine, "confirm": stage_confirm}


def cmd_stages(ctx: argparse.Namespace, names: list[str]) -> int:
    ctx.cases = x2_cases()
    ns = select_lengths(lengths_of(ctx.cases), ctx.lengths)
    state = load_state(ctx.out_dir, ctx.name)
    check_isa(state, ctx.isa)
    state["isa"] = ctx.isa
    radix_set = {"codelets": list(ctx.codelets), "require": list(ctx.require)}
    if state["lengths"] and state.get("radix_set", radix_set) != radix_set:
        raise SystemExit(
            f"campaign {state['name']} ran with {state['radix_set']}: use another --name"
        )
    if ctx.require or ctx.codelets != radices.CODELET_RADICES:
        state["radix_set"] = radix_set
    if "host" not in state:
        state["host"] = machine.fingerprint()
        state["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state["source"] = arms.source_identity()
    missing = [n for n in ns if "v0" not in state["lengths"].get(str(n), {})]
    if missing:
        for n, seq in v0_choices(ctx.build, missing, ctx.isa).items():
            state["lengths"].setdefault(str(n), {})["v0"] = list(seq)
        for n, seq in v0_choices(ctx.build, missing, ctx.isa, "f32").items():
            state["lengths"][str(n)]["v0_f32"] = list(seq)
        if not ctx.dry_run:
            save_state(ctx.out_dir, state)
    # Stage by stage over all lengths, so that a stage's timings share one stretch of the
    # machine's life rather than being interleaved with the next stage's.
    for stage in names:
        for n in ns:
            ids = [c.id for c in ctx.cases if c.dims[0] == n]
            STAGES[stage](ctx, state, n, ids)
            if not ctx.dry_run:
                save_state(ctx.out_dir, state)
    if not ctx.dry_run:
        print(f"x2 campaign state: {state_path(ctx.out_dir, ctx.name)}")
    return 0


# ---- plan ----------------------------------------------------------------------------------


def cmd_plan(ctx: argparse.Namespace) -> int:
    cases = x2_cases()
    ns = select_lengths(lengths_of(cases), ctx.lengths)
    v0 = v0_choices(ctx.build, ns, ctx.isa) if ctx.v0 else {}
    ops = radices.codelet_ops()
    lanes = radices.LANES[ctx.isa] if ctx.isa else ()
    total_cands = 0
    screen_s = 0.0
    later_s = 0.0
    dropped_total = 0
    print(f"{'n':>9} {'cand.':>6} {'kept':>6} {'screen s':>9}  v0")
    for n in ns:
        seq = v0.get(n)
        cands = [
            s for s in radices.candidates(n, seq, ops, lanes, ctx.codelets, ctx.require) if s != seq
        ]
        per = process_seconds(n, batches_of(n, cases), ctx.gflops, screen=True)
        fits = max(int(ctx.screen_budget / (SCREEN_ROUNDS * per)) - 2, 1)
        kept = min(len(cands), fits)
        dropped_total += len(cands) - kept
        total_cands += len(cands)
        s = SCREEN_ROUNDS * (kept + 2) * per
        screen_s += s
        full = process_seconds(n, batches_of(n, cases), ctx.gflops, screen=False)
        # Refine: at most REFINE_TOP per case; confirm: about 10 rounds of a few arms.
        refine_arms = min(kept, REFINE_TOP * len(batches_of(n, cases)) * 2)
        later_s += REFINE_ROUNDS * (refine_arms + 2) * full + 10 * 6 * full
        shown = radices.label(seq) if seq else "?"
        print(f"{n:>9} {len(cands):>6} {kept:>6} {s:>9.1f}  {shown}")
    print(
        f"{len(ns)} lengths, {total_cands} candidates, {dropped_total} dropped by the "
        f"screening budget of {ctx.screen_budget:.0f} s per length"
    )
    print(
        f"estimated time at {ctx.gflops:g} GFLOP/s (double): screening "
        f"{screen_s / 3600:.1f} h, refine and confirm {later_s / 3600:.1f} h, plus analysis"
    )
    return 0


# ---- report --------------------------------------------------------------------------------


def cmd_report(ctx: argparse.Namespace) -> int:
    rows = []
    for name in ctx.name.split(","):
        state = load_state(ctx.out_dir, name)
        if not state["lengths"]:
            raise SystemExit(f"no campaign {state_path(ctx.out_dir, name)}")
        host = state.get("host", {})
        if not ctx.json:
            print(
                f"campaign {name}: host {host.get('cpu')} (id {host.get('host_id')}), "
                f"ISA {state.get('isa') or 'best'}"
            )
        for n_text, entry in sorted(state["lengths"].items(), key=lambda kv: int(kv[0])):
            winners = entry.get("refine", {}).get("winners", {})
            run = entry.get("confirm", {}).get("run")
            res = None
            if run:
                res = analysis.analyse(records_of(ctx.out_dir, run))
            for case, winner in sorted(winners.items()):
                row = {
                    "campaign": name,
                    "n": int(n_text),
                    "case": case,
                    "v0": radices.label(base_of(entry, case)),
                    "best": radices.label(base_of(entry, case)) if winner == V0 else winner,
                    "screen_run": entry.get("screen", {}).get("run"),
                    "refine_run": entry.get("refine", {}).get("run"),
                    "confirm_run": run,
                    "ratio": 1.0,
                    "lo": 1.0,
                    "hi": 1.0,
                    "band": None,
                    "verdict": "v0 best",
                }
                if res is not None and winner != V0:
                    r = res.results[winner][case]
                    row["band"] = math.expm1(res.band)
                    row["verdict"] = r.verdict
                    if r.interval:
                        # time(v0) / time(best) is the reciprocal of fbench's B/A ratio.
                        row["ratio"] = math.exp(-r.interval.median)
                        row["lo"] = math.exp(-r.interval.hi)
                        row["hi"] = math.exp(-r.interval.lo)
                rows.append(row)
    if ctx.json:
        print(json.dumps(rows, indent=1))
        return 0
    print(
        f"{'case':<24} {'v0':<16} {'winner':<16} {'v0/win':>7} {'95 % interval':>15} "
        f"{'band':>6} {'rule':<5} verdict"
    )
    for r in rows:
        clears = r["ratio"] > 1 + DECISION and r["verdict"] == "faster"
        band = f"{r['band'] * 100:5.1f}%" if r["band"] is not None else "     -"
        print(
            f"{r['case']:<24} {r['v0']:<16} {r['best']:<16} {r['ratio']:>7.3f} "
            f"[{r['lo']:.3f}, {r['hi']:.3f}] {band} {'yes' if clears else '':<5} {r['verdict']}"
        )
    groups: dict[str, list[float]] = {}
    for r in rows:
        key = r["case"].split("-", 2)[2]  # b1-op, b64-op-f32, ...
        groups.setdefault(key, []).append(r["ratio"])
    for key, ratios in sorted(groups.items()):
        geo = math.exp(statistics.fmean(math.log(x) for x in ratios))
        print(f"geomean time(v0)/time(winner), {key}: {geo:.3f} over {len(ratios)} lengths")
    print(
        "rule: the winner beats v0 by more than 3 % with fbench's verdict 'faster' (clear of "
        "the band); a winner the confirmation finds slower shows a ratio below 1"
    )
    return 0


# ---- remote, catalogue, selftest -----------------------------------------------------------


def cmd_remote(ctx: argparse.Namespace) -> int:
    cases = x2_cases()
    ns = lengths_of(cases)
    split = [("smooth", f"2-{radices.SMOOTH_LIMIT}"), ("long", f"{radices.SMOOTH_LIMIT + 1}-")]
    isa = ["--isa", ctx.isa] if ctx.isa else []
    for profile in ctx.profiles:
        for part, spec in split:
            count = len(select_lengths(ns, spec))
            name = f"{profile}-{ctx.isa}-{part}" if ctx.isa else f"{profile}-{part}"
            args = ["all", "--name", name, "--lengths", spec, *isa, *ctx.args]
            cmd = [
                "uv",
                "run",
                "--frozen",
                "python",
                "scripts/remote/fbench_remote.py",
                "submit",
                profile,
                "fftn",
                "--campaign",
                "bench/x2/search.py",
                *(["--time", ctx.time] if ctx.time else []),
                "--",
                *args,
            ]
            print(f"# {profile}: {count} lengths ({part})")
            print(" ".join(shlex.quote(x) for x in cmd))
    return 0


def cmd_catalogue(ctx: argparse.Namespace) -> int:
    std = catalogue.load("standard")
    text = radices.catalogue_text(
        radices.smooth_lengths(), radices.long_lengths([(c.kind, c.dims) for c in std.cases])
    )
    if ctx.check:
        if CATALOGUE.read_text() != text:
            print(f"{CATALOGUE} differs from what radices.py generates: run catalogue")
            return 1
        return 0
    CATALOGUE.write_text(text)
    print(f"wrote {CATALOGUE}")
    return 0


def cmd_selftest(ctx: argparse.Namespace) -> int:
    """The RADICES option end to end: every candidate of a few lengths is planned, checked
    and reported with exactly its radices, in both precisions; invalid ones are refused."""
    exe = fbench_fftn(ctx.build)
    ops = radices.codelet_ops()
    failures: list[str] = []
    checked = 0
    for n in (16, 48, 64, 210):
        cases = [
            catalogue.Case("c2c", (n,), b, "out", s, precision=p)
            for b in (1, 3)
            for s in (-1, 1)
            for p in radices.PRECISIONS
        ]
        for seq in radices.candidates(n, None, ops):
            results, error = harness_check(exe, cases, f"RADICES={radices.label(seq)}")
            if error or len(results) != len(cases):
                failures.append(f"RADICES={radices.label(seq)}: {error or 'missing cases'}")
                continue
            for res in results:
                ok = res.get("status") == "ok" and res["check"]["pass"]
                if not ok or radices.from_note(res.get("note", "")) != (n, seq):
                    failures.append(f"RADICES={radices.label(seq)} {res.get('id')}: {res}")
            checked += 1
    refusals = [
        ("RADICES=2.8", [catalogue.Case("c2c", (32,), 1, "out", -1)], "multiplies to 16"),
        ("RADICES=4.8", [catalogue.Case("r2c", (32,), 1, "out", -1)], "not the half length 16"),
        ("RADICES=4.8", [catalogue.Case("c2r", (33,), 1, "out", 1)], "real rows of even length"),
        ("RADICES=4.8", [catalogue.Case("c2c", (4, 8), 1, "out", -1)], "rows only"),
    ]
    # A real row of even length takes radices of its half length (experiment X72).
    for kind, sign in (("r2c", -1), ("c2r", 1)):
        case = catalogue.Case(kind, (64,), 1, "out", sign)
        results, error = harness_check(exe, [case], "RADICES=4.8")
        ok = bool(results) and results[0].get("status") == "ok" and results[0]["check"]["pass"]
        if error or not ok:
            failures.append(f"RADICES=4.8 on {kind} 64 was not planned: {results or error}")
    for options, cases, expect in refusals:
        results, error = harness_check(exe, cases, options)
        message = results[0].get("message", "") if results else error
        if not results or results[0].get("status") != "error" or expect not in message:
            failures.append(f"{options} on {cases[0].id} was not refused: {results or error}")
    for options in ("RADICES=4.1", "RADICES=4..8", "RADICES=", "RADIX=4.8"):
        _, error = harness_check(exe, [catalogue.Case("c2c", (32,), 1, "out", -1)], options)
        if not error:
            failures.append(f"{options} was accepted")
    for f in failures:
        print(f"x2 selftest FAILURE: {f}")
    print(f"x2 selftest: {checked} forced sequences checked, {len(failures)} failures")
    return 1 if failures else 0


# ---- Arguments -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="search.py", description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=store.DEFAULT_OUT, help="bench-out")
    p.add_argument("--build", type=Path, default=DEFAULT_BUILD, help="build with fbench-fftn")
    sub = p.add_subparsers(dest="command", required=True)

    def isa(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--isa", choices=sorted(radices.LANES), help="force every arm to ISA=T (FFTN backend)"
        )

    def lengths(sp: argparse.ArgumentParser) -> None:
        isa(sp)
        sp.add_argument("--lengths", help="e.g. 2-1024 or 1025- or 60,64")
        sp.add_argument(
            "--radices",
            dest="codelets",
            type=radices.parse_radix_set,
            default=radices.CODELET_RADICES,
            help="codelet radices of the candidates (default 2-16; X3: 2-16,32,64)",
        )
        sp.add_argument(
            "--require",
            type=radices.parse_radix_set,
            default=(),
            help="only candidates with one of these radices (X3: 32,64)",
        )
        sp.add_argument("--gflops", type=float, default=8.0, help="assumed double-precision speed")
        sp.add_argument(
            "--screen-budget",
            type=float,
            default=SCREEN_BUDGET_S,
            help="estimated screening seconds per length beyond which candidates go",
        )

    sp = sub.add_parser("plan", help="candidate counts and an estimate of the time")
    lengths(sp)
    sp.add_argument("--v0", action="store_true", help="ask the build for v0's choices")
    for name in ("all", "screen", "refine", "confirm"):
        sp = sub.add_parser(name, help="run the stages" if name == "all" else f"run {name}")
        lengths(sp)
        sp.add_argument("--name", required=True, help="campaign name (one per machine)")
        sp.add_argument("--dry-run", action="store_true", help="print the fbench commands")
        sp.add_argument(
            "--fbench",
            dest="fbench_extra",
            default="",
            help="further fbench run options, e.g. '--max-load 2'",
        )
    sp = sub.add_parser("report", help="v0 against the best, per case")
    sp.add_argument("--name", required=True, help="campaign names, comma-separated")
    sp.add_argument("--json", action="store_true")
    sp = sub.add_parser("remote", help="the fbench_remote submit lines for Slurm machines")
    sp.add_argument("profiles", nargs="+", help="profiles of ~/.config/fftn/remote.toml")
    sp.add_argument("--time", default="12:00:00", help="Slurm --time per job")
    isa(sp)
    sp.add_argument("--args", nargs=argparse.REMAINDER, default=[], help="more for `all`")
    sp = sub.add_parser("catalogue", help="write bench/catalogues/x2.toml")
    sp.add_argument("--check", action="store_true", help="only compare")
    sub.add_parser("selftest", help="RADICES end to end on a few lengths (CTest)")

    ctx = p.parse_args(argv)
    if ctx.command in STAGES or ctx.command == "all":
        ctx.fbench_extra = shlex.split(ctx.fbench_extra)
        return cmd_stages(ctx, list(STAGES) if ctx.command == "all" else [ctx.command])
    return {
        "plan": cmd_plan,
        "report": cmd_report,
        "remote": cmd_remote,
        "catalogue": cmd_catalogue,
        "selftest": cmd_selftest,
    }[ctx.command](ctx)


if __name__ == "__main__":
    sys.exit(main())

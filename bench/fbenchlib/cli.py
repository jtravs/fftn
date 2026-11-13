# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The fbench command line (bench/README.md has the guide)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from . import analysis, arms, catalogue, machine, stats, store

DEFAULT_BUILD = store.REPO / "build" / "check-release"
# Option sets a library is smoke-tested with (correctness only, so the cheapest planning);
# a library with several code paths behind an option gets one run per path.
SMOKE_OPTIONS = {
    "fftw": [{"PLANNER": "ESTIMATE"}],
    "accelerate": [{"API": "dft"}, {"API": "interleaved"}],
}
# The relative output error the guard's self-check injects.
SMOKE_PERTURB = "1e-6"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---- Running one process ------------------------------------------------------------------


def run_process(
    exe: Path, harness_args: list[str], timeout: float
) -> tuple[dict | None, list[dict], str]:
    """Runs one harness process; returns its describe, its case lines and an error, if any."""
    try:
        done = subprocess.run(
            [str(exe), "run", *harness_args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, [], f"timed out after {timeout:.0f} s"
    lines = []
    for text in done.stdout.splitlines():
        try:
            lines.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    desc = next((x for x in lines if x.get("type") == "describe"), None)
    cases = [x for x in lines if x.get("type") == "case"]
    error = ""
    if done.returncode != 0:
        first = next((x for x in lines if x.get("type") == "error"), None)
        error = (first or {}).get("message") or done.stderr.strip() or f"exit {done.returncode}"
    elif not any(x.get("type") == "end" for x in lines):
        error = "the process ended early (crash?)"
    return desc, cases, error


def options_text(options: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in options.items())


# ---- run ----------------------------------------------------------------------------------


def wisdom_options(arm: arms.Arm, binary: arms.Binary, work: Path) -> dict[str, str]:
    """FFTW wisdom per campaign: fresh at the start of a run, shared by the processes of
    arms with the same binary and planner (so every round uses the same plans)."""
    if arm.library != "fftw":
        if arm.option("WISDOM") is not None:
            raise ValueError(f"arm {arm.label}: WISDOM applies to FFTW only")
        return {}
    mode = arm.option("WISDOM") or "run"
    if mode == "none":
        return {}
    if mode != "run":
        return {"WISDOM_IN": str(Path(mode).resolve())}
    planner = arm.backend_options().get("PLANNER", "MEASURE")
    key = hashlib.sha256(f"{binary.path}|{planner}".encode()).hexdigest()[:12]
    path = work / f"fftw-{key}.wisdom"
    return {"WISDOM_IN": str(path), "WISDOM_OUT": str(path)}


def cmd_run(args: argparse.Namespace) -> int:
    cat = catalogue.load(args.catalogue)
    cases = cat.select(args.tier, args.filter)
    if not cases:
        log("no cases selected")
        return 2
    targets = {c.id for c in cases if not args.target or re.search(args.target, c.id)}
    arm_list = [arms.parse(s) for s in args.arms]
    labels = [a.label for a in arm_list]
    if len(set(labels)) != len(labels):
        log(f"arm labels must differ: {labels}")
        return 2
    if not args.no_control:
        arm_list.insert(1, arm_list[0].as_control())

    out_dir = Path(args.out_dir)
    run_id = store.new_run_id()
    st = store.RunStore(out_dir, run_id)
    binaries: dict[str, arms.Binary] = {}
    roots: list[str] = []
    for arm in arm_list:
        binary = arms.resolve(arm, Path(args.build), out_dir, log)
        binary.describe = arms.describe(binary, arm.backend_options())
        problem = arms.library_problem(binary.describe, binary.path)
        if problem:
            log(f"arm {arm.label}: refused: {problem}")
            return 2
        problem = arms.timing_problem(binary.describe)
        if problem and not args.allow_debug_build:
            log(f"arm {arm.label}: refused for timing: {problem} (--allow-debug-build overrides)")
            return 2
        binaries[arm.label] = binary
        root = binary.describe.get("expected_root")
        if root and root not in roots:
            # Library installations are stored as <fftw-root>, <fftw-root-2>, ...
            roots.append(root)
            st.add_prefix(root, f"<{arm.library}-root{'' if len(roots) == 1 else len(roots)}>")
    configs: dict[str, str] = {}
    for arm in arm_list:
        if arm.control:
            continue
        config = arms.configuration(binaries[arm.label], arm)
        for other, other_config in configs.items():
            other_arm = next(a for a in arm_list if a.label == other)
            meant_to_differ = (other_arm.library, other_arm.build, other_arm.options) != (
                arm.library,
                arm.build,
                arm.options,
            )
            if meant_to_differ and other_config == config and not args.allow_identical:
                log(
                    f"arms {other} and {arm.label} are meant to differ but report identical "
                    "configurations (same binary identity and options); --allow-identical "
                    "overrides"
                )
                return 2
        configs[arm.label] = config

    host = machine.fingerprint()
    pool = args.pool_bytes or max(4 * host["llc_bytes"], 256 * 1024 * 1024)
    limits = machine.QuietLimits(
        args.max_load if args.max_load is not None else machine.QuietLimits.defaults().max_load,
        args.max_busy if args.max_busy is not None else machine.QuietLimits.defaults().max_busy,
    )
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 31)
    cases_file = st.work / "cases.txt"
    st.append(
        {
            "type": "run",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": sys.argv[1:],
            "python": platform.python_version(),
            "host": host,
            "thermal": machine.thermal_note(),
            "source": arms.source_identity(),
            "catalogue": {"name": cat.name, "path": str(cat.path), "sha256": cat.sha256},
            "tier": args.tier,
            "filter": args.filter,
            "settings": {
                "mode": args.mode,
                "stat": args.stat,
                "batches": args.batches,
                "batch_time": args.batch_time,
                "warmup": args.warmup,
                "pool_bytes": pool,
                "check_budget": args.check_budget,
                "min_band": args.min_band,
                "min_rounds": args.min_rounds,
                "max_rounds": args.max_rounds,
                "budget_s": args.budget,
                "seed": seed,
                "max_load": limits.max_load,
                "max_busy": limits.max_busy,
            },
            "arms": [
                {
                    "label": a.label,
                    "spec": a.spec,
                    "library": a.library,
                    "control": a.control,
                    "binary": str(binaries[a.label].path),
                    "commit": binaries[a.label].commit,
                    "describe": binaries[a.label].describe,
                }
                for a in arm_list
            ],
            "cases": [
                {
                    "id": c.id,
                    "weight": c.weight,
                    "points": c.points,
                    "batch": c.batch,
                    "flops": c.flops,
                    "group": c.group,
                }
                for c in cases
            ],
        }
    )
    by_id = {c.id: c for c in cases}
    orders = stats.williams_orders(len(arm_list))
    log(
        f"run {run_id}: {len(cases)} cases, arms {', '.join(a.label for a in arm_list)}; "
        f"store {st.path}"
    )
    started = time.monotonic()
    attempt = 0
    valid_rounds = 0
    result = None
    while valid_rounds < args.max_rounds and time.monotonic() - started < args.budget:
        before = machine.load_state()
        quiet, why = machine.is_quiet(before, limits)
        if not quiet:
            st.append({"type": "round", "round": attempt, "valid": False, "reason": why})
            log(f"round {attempt}: machine busy ({why}); waiting {args.busy_wait:.0f} s")
            attempt += 1
            time.sleep(args.busy_wait)
            continue
        rng = random.Random(seed * 7919 + attempt)
        order = [arm_list[i] for i in orders[valid_rounds % len(orders)]]
        offsets = (rng.randrange(64) * 64, rng.randrange(64) * 64)
        shuffled = list(cases)
        rng.shuffle(shuffled)
        cases_file.write_text("".join(c.line() + "\n" for c in shuffled))
        failures = []
        for arm in order:
            binary = binaries[arm.label]
            backend = {**arm.backend_options(), **wisdom_options(arm, binary, st.work)}
            harness_args = [
                "--cases",
                str(cases_file),
                "--options",
                options_text(backend),
                "--mode",
                args.mode,
                "--pool-bytes",
                str(pool),
                "--batches",
                str(args.batches),
                "--batch-time",
                str(args.batch_time),
                "--warmup",
                str(args.warmup),
                "--offset-in",
                str(offsets[0]),
                "--offset-out",
                str(offsets[1]),
                "--check-budget",
                str(args.check_budget),
                "--seed",
                str(seed),
                *arm.harness_args(),
            ]
            _, results, error = run_process(binary.path, harness_args, args.process_timeout)
            if error:
                failures.append(f"{arm.label}: {error}")
            for res in results:
                case = by_id.get(res.get("id", ""))
                rec = {"type": "timing", "round": attempt, "arm": arm.label, "case": res.get("id")}
                rec.update({k: v for k, v in res.items() if k not in ("type", "id")})
                if case is not None and res.get("median_ns"):
                    rec["ns_per_point"] = res["median_ns"] / (case.points * case.batch)
                    rec["mflops"] = case.flops / res["median_ns"] * 1e3
                st.append(rec)
        after = machine.load_state()
        quiet_after, why_after = machine.is_quiet(after, limits)
        valid = quiet_after and not failures
        st.append(
            {
                "type": "round",
                "round": attempt,
                "valid": valid,
                "order": [a.label for a in order],
                "offsets": list(offsets),
                "load_before": before,
                "load_after": after,
                "reason": "; ".join(failures) or why_after,
            }
        )
        attempt += 1
        if failures:
            # A crash or a refused option is a bug to fix, not noise to average over.
            log(f"round {attempt - 1}: harness failure ({'; '.join(failures)}); stopping")
            break
        if not valid:
            log(f"round {attempt - 1}: invalid ({why_after})")
            continue
        valid_rounds += 1
        if valid_rounds >= args.min_rounds:
            result = analysis.analyse(
                store.load(st.path), args.stat, args.min_band, seed, resamples=1000
            )
            left = result.unresolved(targets)
            log(
                f"round {attempt - 1}: valid ({valid_rounds}); band "
                f"{math.expm1(result.band) * 100:.2f} %; {len(left)} unresolved"
            )
            if not left:
                break
        else:
            log(f"round {attempt - 1}: valid ({valid_rounds})")

    records = store.load(st.path)
    result = analysis.analyse(records, args.stat, args.min_band, seed)
    st.append(
        {
            "type": "summary",
            "elapsed_s": round(time.monotonic() - started, 1),
            "band": result.band,
            "valid_rounds": len(result.valid_rounds),
            "invalid_rounds": result.invalid_rounds,
            "verdicts": {
                arm: {c: r.verdict for c, r in cases_.items()}
                for arm, cases_ in result.results.items()
            },
        }
    )
    print(report(store.load(st.path), args.stat, args.min_band))
    wrong = any(
        r.verdict in (analysis.WRONG, analysis.BASE_WRONG)
        for cases_ in result.results.values()
        for r in cases_.values()
    )
    return 1 if wrong else 0


# ---- Reports ------------------------------------------------------------------------------


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def report(records: list[dict], stat: str | None = None, min_band: float | None = None) -> str:
    run = next(r for r in records if r["type"] == "run")
    seed = run["settings"].get("seed", 0)
    stat = stat or run["settings"].get("stat", "median")
    min_band = min_band if min_band is not None else run["settings"].get("min_band", 0.01)
    res = analysis.analyse(records, stat, min_band, seed)
    cases = {c["id"]: c for c in run["cases"]}
    host = run["host"]
    rounds = [r for r in records if r["type"] == "round" and r.get("valid")]
    loads = [
        x["load1"]
        for r in rounds
        for x in (r.get("load_before", {}), r.get("load_after", {}))
        if x.get("load1") is not None
    ]
    busy = [
        x["busy"]
        for r in rounds
        for x in (r.get("load_before", {}), r.get("load_after", {}))
        if x.get("busy") is not None
    ]
    out = [
        f"run {run['run']}  catalogue {run['catalogue']['name']}, tier {run.get('tier')}"
        + (f", filter {run['filter']!r}" if run.get("filter") else "")
        + f", {len(cases)} cases, {run['settings']['mode']}, statistic {stat}",
        f"host {host['cpu']}, {host['logical_cpus']} CPUs, {host['os']} (id {host['host_id']})",
    ]
    if loads:
        line = f"load average {min(loads):.2f}-{max(loads):.2f}"
        if busy:
            line += f"; other processes {min(busy):.2f}-{max(busy):.2f} cores"
        out.append(line)
    src = run.get("source", {})
    out.append(
        f"source {str(src.get('commit'))[:12]}{' (dirty)' if src.get('dirty') else ''}; "
        f"rounds {len(res.valid_rounds)} valid, {res.invalid_rounds} invalid"
    )
    out.append("arms:")
    for a in run["arms"]:
        d = a["describe"]
        what = "identical-arm control of " + res.baseline if a["control"] else a["spec"]
        built = d.get("harness", {})
        out.append(
            f"  {a['label']:<8} {what}  [{d.get('library')} {d.get('version')}, isa "
            f"{d.get('isa')}, {built.get('compiler')} {built.get('build_type')}, commit "
            f"{str(a.get('commit') or built.get('commit'))[:12]}]"
        )
    if res.control:
        out.append(
            f"noise band (A/A control): +-{math.expm1(res.band) * 100:.2f} % "
            f"(floor {min_band * 100:.1f} %)"
        )
    else:
        out.append(f"noise band: +-{math.expm1(res.band) * 100:.2f} % (no control arm)")
    for arm, per_case in res.results.items():
        out.append("")
        out.append(f"{arm} vs {res.baseline}")
        out.append(
            f"{'case':<28} {'ratio':>7} {'95 % interval':>17}  {'verdict':<14} "
            f"{'A ns/pt':>9} {'B ns/pt':>9} {'B mflops':>9} {'B err':>8}"
        )
        counts: dict[str, int] = {}
        for cid, r in per_case.items():
            c = cases[cid]
            pts = c["points"] * c["batch"]
            iv = r.interval
            ratio = _fmt(math.exp(iv.median) if iv else None, ".3f")
            ci = f"[{math.exp(iv.lo):.3f}, {math.exp(iv.hi):.3f}]" if iv else "-"
            verdict = r.verdict + (f" ({r.note})" if r.note else "")
            if r.verdict in (analysis.WRONG, analysis.BASE_WRONG):
                verdict = r.verdict.upper()
            counts[r.verdict] = counts.get(r.verdict, 0) + 1
            out.append(
                f"{cid:<28} {ratio:>7} {ci:>17}  {verdict:<14} "
                f"{_fmt(r.base_ns / pts if r.base_ns else None, '9.3f')} "
                f"{_fmt(r.arm_ns / pts if r.arm_ns else None, '9.3f')} "
                f"{_fmt(c['flops'] / r.arm_ns * 1e3 if r.arm_ns else None, '9.0f')} "
                f"{_fmt(r.err_rms, '8.1e')}"
            )
        g = res.geomean.get(arm)
        if g:
            out.append(
                f"{'weighted geomean':<28} {math.exp(g.median):>7.3f} "
                f"[{math.exp(g.lo):.3f}, {math.exp(g.hi):.3f}]"
            )
        out.append("verdicts: " + ", ".join(f"{n} {v}" for v, n in sorted(counts.items())))
    return "\n".join(out)


def cmd_show(args: argparse.Namespace) -> int:
    path = store.find_run(Path(args.out_dir), args.run_id)
    print(report(store.load(path), args.stat, args.min_band))
    return 0


# ---- list ---------------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    cat = catalogue.load(args.catalogue)
    cases = cat.select(args.tier, args.filter)
    print(f"# {cat.name} ({cat.path.name}), tier {args.tier or 'full'}: {len(cases)} cases")
    for c in cases:
        tiers = ",".join(t for t in catalogue.TIERS if t in c.tiers) or "full"
        print(f"{c.id:<32} weight {c.weight:<6g} {tiers:<26} {c.group}")
    return 0


# ---- smoke --------------------------------------------------------------------------------


def smoke_one(
    exe: Path,
    library: str,
    info: dict,
    opts: dict[str, str],
    cases: list[catalogue.Case],
    work: Path,
    timeout: float,
    failures: list[str],
    require_partial: bool = True,
) -> None:
    """One backend with one option set over the smoke cases, then the guard's self-check."""
    name = "+".join([library, *(f"{k}={v}" for k, v in opts.items())])
    # Unsupported cases fail for the kinds a backend says it supports. A kind it supports
    # only partly (some lengths or ranks) must still pass somewhere, so that a backend that
    # turns everything down is not counted as correct; that needs cases chosen to contain
    # such lengths (the standard smoke tier), so other selections skip it.
    supported = set(info.get("supports", []))
    partial = set(info.get("partial", []))
    kind_of = {c.id: c.kind for c in cases}
    options = options_text(opts)
    base = ["--cases", str(work / "cases.txt"), "--options", options, "--check-only"]
    _, results, error = run_process(exe, base, timeout)
    if error:
        failures.append(f"{name}: {error}")
    counts = {"pass": 0, "fail": 0, "unsupported": 0, "error": 0}
    passed_kinds: set[str] = set()
    first_ok = None
    for res in results:
        if res.get("status") == "ok":
            if res["check"]["pass"]:
                counts["pass"] += 1
                passed_kinds.add(kind_of.get(res["id"], ""))
                first_ok = first_ok or res["id"]
            else:
                counts["fail"] += 1
                failures.append(
                    f"{name} {res['id']}: wrong output (rms {res['check']['err_rms']:.2e},"
                    f" max {res['check']['err_max']:.2e}, tol {res['check']['tol']:.2e})"
                )
        elif res.get("status") == "unsupported":
            counts["unsupported"] += 1
            if kind_of.get(res.get("id", "")) in supported:
                failures.append(
                    f"{name} {res['id']}: unsupported, but {library} declares "
                    f"{kind_of[res['id']]} supported"
                )
        else:
            counts["error"] += 1
            failures.append(f"{name} {res.get('id', '?')}: {res.get('message')}")
    missing = len(cases) - len(results)
    if missing and not error:
        failures.append(f"{name}: {missing} cases produced no result")
    required = supported | partial if require_partial else supported
    for kind in sorted((required & set(kind_of.values())) - passed_kinds):
        failures.append(f"{name}: declares {kind} but passed no {kind} case")
    print(
        f"smoke {name} ({info.get('version')}, isa {info.get('isa')}): "
        f"{counts['pass']} pass, {counts['fail']} wrong, {counts['unsupported']} "
        f"unsupported, {counts['error']} errors of {len(cases)}"
    )
    if first_ok:
        # The guard's self-check: the same case with a perturbed output must fail.
        one = work / f"guard-{library}.txt"
        one.write_text(next(c.line() for c in cases if c.id == first_ok) + "\n")
        guard = ["--cases", str(one), "--options", options, "--check-only"]
        _, res, error = run_process(exe, [*guard, "--perturb", SMOKE_PERTURB], 60)
        flagged = bool(res) and res[0].get("status") == "ok" and not res[0]["check"]["pass"]
        print(f"smoke {name}: guard {'flags' if flagged else 'MISSES'} a perturbed output")
        if not flagged:
            failures.append(f"{name}: the correctness guard missed a perturbed output")


def cmd_smoke(args: argparse.Namespace) -> int:
    """Correctness only, every backend in the build, plus a self-check of the guard."""
    build = Path(args.build)
    exes = sorted((build / "bench").glob("fbench-*"))
    exes = [e for e in exes if e.is_file() and not e.suffix]
    if not exes:
        log(f"smoke: no fbench backends in {build / 'bench'}")
        return 1 if args.strict else 0
    if not any(e.name == "fbench-fftw" for e in exes):
        print("smoke: FFTW backend not built (FFTN_BENCH_FFTW_ROOT unset at configure); skipped")
    cat = catalogue.load(args.catalogue)
    cases = cat.select(args.tier, args.filter)
    work = Path(args.out_dir) / "smoke"
    work.mkdir(parents=True, exist_ok=True)
    cases_file = work / "cases.txt"
    cases_file.write_text("".join(c.line() + "\n" for c in cases))
    standard_smoke = (
        cat.path.resolve() == catalogue.resolve("standard").resolve()
        and args.tier == "smoke"
        and not args.filter
    )
    failures: list[str] = []
    for exe in exes:
        library = exe.name.removeprefix("fbench-")
        binary = arms.Binary(exe)
        try:
            info = arms.describe(binary, {})
        except RuntimeError as e:
            failures.append(f"{library}: {e}")
            continue
        problem = arms.library_problem(info, exe)
        if problem:
            failures.append(f"{library}: {problem}")
            continue
        # The harness's own self-test: the reference, the check and describe.
        selftest = subprocess.run(
            [str(exe), "selftest"], capture_output=True, text=True, timeout=120, check=False
        )
        if selftest.returncode != 0:
            failures.append(f"{library}: {(selftest.stdout + selftest.stderr).strip()}")
        for opts in SMOKE_OPTIONS.get(library, [{}]):
            smoke_one(
                exe,
                library,
                info,
                opts,
                cases,
                work,
                args.process_timeout,
                failures,
                require_partial=standard_smoke,
            )
    for f in failures:
        print(f"smoke FAILURE: {f}")
    return 1 if failures and args.strict else 0


# ---- Arguments ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fbench", description="FFTN's benchmark rig")
    p.add_argument("--out-dir", default=str(store.DEFAULT_OUT), help="results root (bench-out)")
    sub = p.add_subparsers(dest="command", required=True)

    def selection(sp: argparse.ArgumentParser, tier: str | None) -> None:
        sp.add_argument("--catalogue", default="standard", help="name or path of a catalogue")
        sp.add_argument("--tier", default=tier, choices=catalogue.TIERS)
        sp.add_argument("--filter", help="regex on case ids")

    r = sub.add_parser("run", help="A/B run: process rounds, paired intervals, A/A control")
    r.add_argument("arms", nargs="+", help="label=library[:build][+OPTION=value...]; first is A")
    selection(r, "quick")
    r.add_argument("--build", default=str(DEFAULT_BUILD), help="default build directory")
    r.add_argument("--target", help="regex: cases that must resolve before stopping (all)")
    r.add_argument("--mode", choices=("hot", "cold"), default="hot")
    r.add_argument("--stat", choices=("median", "min"), default="median")
    r.add_argument("--min-rounds", type=int, default=6)
    r.add_argument("--max-rounds", type=int, default=40)
    r.add_argument("--budget", type=float, default=900.0, help="seconds")
    r.add_argument("--min-band", type=float, default=0.01, help="noise band floor (fraction)")
    r.add_argument("--batches", type=int, default=7)
    r.add_argument("--batch-time", type=float, default=0.002, help="seconds")
    r.add_argument("--warmup", type=float, default=0.02, help="seconds")
    r.add_argument("--pool-bytes", type=int, default=0, help="cold pool (0: from the LLC)")
    r.add_argument("--check-budget", type=int, default=1 << 20)
    r.add_argument("--seed", type=int)
    r.add_argument("--no-control", action="store_true", help="skip the A/A control arm")
    r.add_argument("--max-load", type=float)
    r.add_argument("--max-busy", type=float)
    r.add_argument("--busy-wait", type=float, default=10.0)
    r.add_argument("--process-timeout", type=float, default=3600.0)
    r.add_argument("--allow-identical", action="store_true")
    r.add_argument("--allow-debug-build", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("smoke", help="correctness of every built backend (the gate's stage)")
    s.add_argument("--build", default=str(DEFAULT_BUILD))
    selection(s, "smoke")
    s.add_argument("--strict", action="store_true", help="exit 1 on any failure")
    s.add_argument("--process-timeout", type=float, default=600.0)
    s.set_defaults(func=cmd_smoke)

    ls = sub.add_parser("list", help="cases of a catalogue")
    selection(ls, None)
    ls.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="report of a stored run")
    sh.add_argument("run_id", help="run ID, unique prefix, 'latest' or a .jsonl path")
    sh.add_argument("--stat", choices=("median", "min"), help="default: the run's")
    sh.add_argument("--min-band", type=float, help="default: the run's")
    sh.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log(f"fbench: {e}")
        return 2

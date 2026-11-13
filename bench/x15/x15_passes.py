#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X15, pass level: x15-passes (bench/x15/passes.cpp) in process rounds.

uv run --frozen python bench/x15/x15_passes.py --binary build/check-release/bench/x15-passes \
    --rounds 12 --label A

Each round launches the binary once, the variants in a Williams order (fbenchlib.stats), after
fbench's quiet-machine check (fbenchlib.machine); a round on a busy machine is repeated. The run
is stored as bench-out/x15/runs/<run-id>.jsonl: a run record (host fingerprint, the binary's
describe, the settings), the rounds, every timing line, and a summary line per variant,
precision, m and level: the median over rounds of each process's median ns per butterfly, and
the median paired ratio to the reference variant (default scalar) and to the hand vocabulary
(avx2 or neon_fcma) with 95 % bootstrap intervals over rounds. The noise band is fbench's, from
the identical pair scalar~ / scalar. Experiment code: deleted with X15's verdict.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbenchlib import machine, stats, store

HAND = ("avx2", "neon_fcma")


def _sysctl(name: str) -> int:
    done = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
    text = done.stdout.strip()
    return int(text) if done.returncode == 0 and text.isdigit() else 0


def cache_sizes() -> tuple[int, int]:
    """L1 data and L2 of the core the harness runs on (the performance cores on Apple)."""
    if platform.system() == "Darwin":
        l1 = _sysctl("hw.perflevel0.l1dcachesize") or _sysctl("hw.l1dcachesize")
        l2 = _sysctl("hw.perflevel0.l2cachesize") or _sysctl("hw.l2cachesize")
        return l1, l2
    l1 = l2 = 0
    for index in sorted(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*")):
        try:
            level = (index / "level").read_text().strip()
            kind = (index / "type").read_text().strip()
            text = (index / "size").read_text().strip()
        except OSError:
            continue
        mult = {"K": 1024, "M": 1024 * 1024}.get(text[-1:], 1)
        size = int(text.rstrip("KM")) * mult
        if level == "1" and kind == "Data":
            l1 = size
        elif level == "2":
            l2 = size
    return l1, l2


def run_round(
    binary: str,
    args: argparse.Namespace,
    variants: list[str],
    order: list[int],
    l1: int,
    l2: int,
    seed: int,
) -> list[dict]:
    cmd = [
        binary,
        "--l1-bytes", str(l1),
        "--l2-bytes", str(l2),
        "--variants", ",".join(variants),
        "--order", ",".join(str(i) for i in order),
        "--m", args.m,
        "--precisions", args.precisions,
        "--levels", args.levels,
        "--batch-time", str(args.batch_time),
        "--batches", str(args.batches),
        "--warmup", str(args.warmup),
        "--seed", str(seed),
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"x15-passes failed ({done.returncode}): {done.stderr.strip()}")
    lines = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
    return [r for r in lines if r.get("type") == "timing"]


def summarise(
    timings: dict[tuple, dict[int, float]],
    variants: list[str],
    reference: str,
    min_band: float,
    seed: int,
) -> tuple[float, list[dict]]:
    rng = random.Random(seed)
    cases = sorted({key[1:] for key in timings}, key=lambda c: (c[0], c[2], c[1]))
    control = []
    for case in cases:
        base = timings.get((reference, *case))
        twin = timings.get(("scalar~", *case))
        if base and twin:
            control.append(stats.interval(stats.paired_log_ratios(base, twin), rng))
    band = stats.noise_band(control, min_band)
    hand = next((h for h in HAND if h in variants), None)
    rows = []
    for case in cases:
        base = timings.get((reference, *case), {})
        hand_vals = timings.get((hand, *case), {}) if hand else {}
        for v in variants:
            vals = timings.get((v, *case))
            if not vals:
                continue
            row: dict = {
                "type": "summary_case",
                "variant": v,
                "precision": case[0],
                "m": case[1],
                "level": case[2],
                "rounds": len(vals),
                "ns_per_butterfly": statistics.median(vals.values()),
            }
            for name, other in ((reference, base), (hand, hand_vals)):
                if not name or v == name or not other:
                    continue
                pairs = stats.paired_log_ratios(other, vals)
                if len(pairs) >= 1:
                    iv = stats.interval(pairs, rng)
                    row[f"vs_{name}"] = {
                        "ratio": math.exp(iv.median),
                        "lo": math.exp(iv.lo),
                        "hi": math.exp(iv.hi),
                        "verdict": stats.verdict(iv, band),
                    }
            rows.append(row)
    return band, rows


def ratio_cell(r: dict | None) -> str:
    return f"{r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if r else "-"


def print_table(band: float, rows: list[dict], reference: str) -> None:
    print(f"noise band (scalar~ / scalar, 95th percentile, floor applied): ±{math.expm1(band):.2%}")
    hand = next((h for r in rows for h in HAND if f"vs_{h}" in r), None)
    head = f"{'prec':4} {'m':>5} {'level':5} {'variant':22} {'ns/bfly':>8}  x {reference:24}"
    print(head + (f"  x {hand}" if hand else ""))
    for r in rows:
        ref = r.get(f"vs_{reference}")
        hd = r.get(f"vs_{hand}") if hand else None
        cells = [
            f"{r['precision']:4} {r['m']:>5} {r['level']:5} {r['variant']:22}",
            f"{r['ns_per_butterfly']:8.3f}",
        ]
        cells.append(" " + ratio_cell(ref))
        if hand:
            cells.append(" " + ratio_cell(hd))
        print(" ".join(cells))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--binary", required=True, help="x15-passes of the build under test")
    p.add_argument("--label", default="", help="the arm, for the record (A, B, E, ...)")
    p.add_argument("--rounds", type=int, default=12, help="valid process rounds")
    p.add_argument("--variants", help="comma list (default: every variant the binary has)")
    p.add_argument("--reference", default="scalar")
    p.add_argument("--m", default="16,32,64,128,256,512,1024,2048,4096")
    p.add_argument("--precisions", default="f64,f32")
    p.add_argument("--levels", default="l1,l2,hot")
    p.add_argument("--batch-time", type=float, default=0.002)
    p.add_argument("--batches", type=int, default=7)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--min-band", type=float, default=0.01)
    p.add_argument("--max-load", type=float)
    p.add_argument("--max-busy", type=float)
    p.add_argument("--busy-wait", type=float, default=10.0)
    p.add_argument("--max-attempts", type=int, default=0, help="round attempts (0: 4 x rounds)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default=str(store.DEFAULT_OUT / "x15"))
    args = p.parse_args(argv)

    binary = str(Path(args.binary).resolve())
    describe = json.loads(
        subprocess.run([binary, "--describe"], capture_output=True, text=True, check=True).stdout
    )
    variants = args.variants.split(",") if args.variants else list(describe["variants"])
    if args.reference not in variants or "scalar~" not in variants:
        raise SystemExit(f"need {args.reference} and scalar~ among the variants")
    l1, l2 = cache_sizes()
    if not l1 or not l2:
        raise SystemExit("could not read the L1 and L2 sizes")
    defaults = machine.QuietLimits.defaults()
    limits = machine.QuietLimits(
        max_load=args.max_load if args.max_load is not None else defaults.max_load,
        max_busy=args.max_busy if args.max_busy is not None else defaults.max_busy,
    )
    run_id = store.new_run_id()
    out = store.RunStore(Path(args.out_dir), run_id)
    out.add_prefix(str(Path(binary).parent.parent.parent), "<build>")
    out.append(
        {
            "type": "run",
            "experiment": "X15",
            "label": args.label,
            "binary": binary,
            "describe": describe,
            "host": machine.fingerprint(),
            "l1_bytes": l1,
            "l2_bytes": l2,
            "thermal": machine.thermal_note(),
            "settings": {k: v for k, v in vars(args).items() if k != "binary"},
            "variants": variants,
        }
    )
    orders = stats.williams_orders(len(variants))
    timings: dict[tuple, dict[int, float]] = {}
    valid = attempts = 0
    max_attempts = args.max_attempts or 4 * args.rounds
    while valid < args.rounds and attempts < max_attempts:
        attempts += 1
        before = machine.load_state()
        quiet, why = machine.is_quiet(before, limits)
        if not quiet:
            out.append(
                {
                    "type": "round",
                    "attempt": attempts,
                    "valid": False,
                    "why": why,
                    "load_before": before,
                }
            )
            print(
                f"round {valid + 1}: busy machine ({why}); waiting {args.busy_wait:.0f} s",
                flush=True,
            )
            time.sleep(args.busy_wait)
            continue
        order = orders[valid % len(orders)]
        lines = run_round(binary, args, variants, order, l1, l2, args.seed + valid)
        after = machine.load_state()
        quiet, why = machine.is_quiet(after, limits)
        out.append(
            {
                "type": "round",
                "attempt": attempts,
                "round": valid,
                "order": order,
                "valid": quiet,
                "why": why,
                "load_before": before,
                "load_after": after,
            }
        )
        if not quiet:
            print(f"round {valid + 1}: machine became busy ({why}); repeating", flush=True)
            time.sleep(args.busy_wait)
            continue
        for t in lines:
            out.append({"round": valid, **t})
            key = (t["variant"], t["precision"], t["m"], t["level"])
            timings.setdefault(key, {})[valid] = t["ns_per_butterfly"]
        valid += 1
        print(f"round {valid}/{args.rounds} done", flush=True)
    band, rows = summarise(timings, variants, args.reference, args.min_band, args.seed)
    for row in rows:
        out.append(row)
    out.append(
        {
            "type": "summary",
            "valid_rounds": valid,
            "attempts": attempts,
            "band": band,
            "complete": valid >= args.rounds,
        }
    )
    with contextlib.suppress(OSError):
        out.work.rmdir()
    print(
        f"\nX15 pass level, run {run_id} ({args.label or 'unlabelled'}), "
        f"{valid} valid rounds; switches {describe.get('x15_switches')}, "
        f"{describe.get('compiler')}"
    )
    print_table(band, rows, args.reference)
    print(f"stored: {out.path}")
    return 0 if valid >= args.rounds else 1


if __name__ == "__main__":
    sys.exit(main())

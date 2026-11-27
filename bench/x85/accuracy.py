#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Experiment X85, rule D1: the accuracy of the folded DCT-I and DST-I against the r2r tests' bound.

Runs an FFTN fbench backend with --check-only on DCT-I and DST-I rows of odd lengths (2^k + 1
and 3 2^k + 1 up to 65535), in both precisions, with the given backend options (the folded
route: R2R_FOLDED_MAX=65535; the extensions: none), and prints each case's relative RMS error
against the harness's double-double reference over two rows, beside the rms tolerance the r2r
tests apply to these kinds (tests/api/r2r_test.cpp, dht_tolerance with dct_ops): eps (3 sqrt(L)
+ 11), L = log2 N with N = 2(n - 1) (DCT-I) or 2(n + 1) (DST-I). N_acc is the largest swept n
such that every swept n up to it, both kinds and precisions, is within half that tolerance.

    uv run --frozen python bench/x85/accuracy.py BUILD/bench/fbench-fftn [OPTION=VALUE ...]
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

EPS = {"f64": 2.0**-52, "f32": 2.0**-23}


def lengths() -> list[int]:
    out = {2**k + 1 for k in range(1, 16)} | {3 * 2**k + 1 for k in range(1, 15)}
    return sorted(n for n in out if n <= 65535)


def tolerance(kind: str, n: int, prec: str) -> float:
    big = 2 * (n - 1) if kind == "dct1" else 2 * (n + 1)
    return EPS[prec] * (3.0 * math.sqrt(max(1.0, math.log2(big))) + 11.0)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    exe, options = argv[1], " ".join(argv[2:])
    cases = []
    for kind in ("dct1", "dst1"):
        for n in lengths():
            for prec in ("f64", "f32"):
                cases.append((kind, n, prec))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cases.txt"
        path.write_text(
            "".join(
                f"id={k}-{n}-{p} kind=r2r dims={n} batch=2 place=out sign=-1 prec={p} r2r={k}\n"
                for k, n, p in cases
            )
        )
        run = subprocess.run(
            [
                exe,
                "run",
                "--cases",
                str(path),
                "--options",
                options,
                "--check-only",
                "--check-budget",
                str(64 * 1024 * 1024),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    results = {}
    describe = ""
    for line in run.stdout.splitlines():
        rec = json.loads(line)
        if rec.get("type") == "describe":
            describe = f"{rec.get('library')} isa {rec.get('isa')}, options '{options}'"
        if rec.get("type") == "case":
            results[rec["id"]] = rec
    print(f"# {describe}")
    print(f"{'case':<18} {'rms':>10} {'tol/2':>10} {'rms/(tol/2)':>12} route")
    worst: dict[int, float] = {}
    for kind, n, prec in cases:
        rec = results[f"{kind}-{n}-{prec}"]
        if rec.get("status") != "ok":
            print(f"{kind}-{n}-{prec}: {rec.get('status')} {rec.get('message', '')}")
            worst[n] = math.inf
            continue
        rms = rec["check"]["err_rms"]
        half = tolerance(kind, n, prec) / 2.0
        folded = "folded" in rec.get("note", "")
        worst[n] = max(worst.get(n, 0.0), rms / half)
        print(
            f"{kind}-{n}-{prec:<6} {rms:>10.3e} {half:>10.3e} {rms / half:>12.3f} "
            f"{'folded' if folded else 'extension'}"
        )
    n_acc = 0
    for n in lengths():
        if worst[n] > 1.0:
            break
        n_acc = n
    print(f"N_acc = {n_acc} (largest swept n with every swept n up to it within half the bound)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

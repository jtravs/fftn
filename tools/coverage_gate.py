#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Line coverage of FFTN's hand-written code, against a threshold (gate stage `coverage`).

    tools/coverage_gate.py <build dir> --min-lines <percent> [--dump <fftn_platform_dump>]

Reads <build dir>/fftn-coverage.json (written by CMake when FFTN_COVERAGE is on), merges
the profiles the tests left behind, and counts lines in include/fftn/ and src/, excluding
the generated codelets in src/gen/ (the generator verifies those). A line is covered if
any test executable or translation unit executed it. With --dump, the vocabulary and region
unit of every target the processor cannot run (above the best target in its family, as the
dump reports it) are left out: their lines cannot execute here, and machines and emulated
runs that have the instruction set cover them. Standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

COUNTED = ("include/fftn/", "src/")
EXCLUDED = ("src/gen/",)
# Each family's targets from lowest to highest; a processor runs its best one and those below.
FAMILIES = (("avx2", "avx512"), ("neon", "neon_fcma"))


def unrunnable(best: str) -> list[str]:
    for family in FAMILIES:
        if best in family:
            return list(family[family.index(best) + 1 :])
    return []


def target_files(targets: list[str]) -> tuple[str, ...]:
    return tuple(f for t in targets for f in (f"src/isa/vocab_{t}.inl", f"src/isa/{t}.cpp"))


def counted(path: str, root: Path) -> str | None:
    try:
        rel = Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return None
    if rel.startswith(COUNTED) and not rel.startswith(EXCLUDED):
        return rel
    return None


def llvm_lines(manifest: dict, build: Path, root: Path) -> dict[str, dict[int, int]]:
    raws = sorted((build / "profiles").glob("*.profraw"))
    if not raws:
        sys.exit(f"coverage: no profiles in {build / 'profiles'}; did the tests run?")
    merged = build / "profiles" / "merged.profdata"
    subprocess.run(
        [manifest["profdata"], "merge", "-sparse", "-o", str(merged), *map(str, raws)], check=True
    )
    exes = [e for e in manifest["executables"] if e]
    args = [manifest["cov"], "export", "-format=lcov", f"-instr-profile={merged}", exes[0]]
    for exe in exes[1:]:
        args += ["-object", exe]
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    lines: dict[str, dict[int, int]] = {}
    per_line: dict[int, int] | None = None
    for record in out.splitlines():
        if record.startswith("SF:"):
            rel = counted(record[3:], root)
            per_line = None if rel is None else lines.setdefault(rel, {})
        elif record.startswith("DA:") and per_line is not None:
            n, count = record[3:].split(",")[:2]
            per_line[int(n)] = max(per_line.get(int(n), 0), int(count))
    return lines


def gcc_lines(manifest: dict, build: Path, root: Path) -> dict[str, dict[int, int]]:
    gcdas = sorted(build.rglob("*.gcda"))
    if not gcdas:
        sys.exit(f"coverage: no .gcda files under {build}; did the tests run?")
    lines: dict[str, dict[int, int]] = {}
    for gcda in gcdas:
        out = subprocess.run(
            [manifest["cov"], "--json-format", "--stdout", str(gcda)],
            check=True,
            capture_output=True,
            cwd=gcda.parent,
        ).stdout
        for chunk in out.splitlines():
            if not chunk.strip():
                continue
            text = gzip.decompress(chunk) if chunk[:2] == b"\x1f\x8b" else chunk
            report = json.loads(text)
            cwd = Path(report.get("current_working_directory", gcda.parent))
            for f in report["files"]:
                rel = counted(str(cwd / f["file"]), root)
                if rel is None:
                    continue
                per_line = lines.setdefault(rel, {})
                for ln in f["lines"]:
                    n = ln["line_number"]
                    per_line[n] = max(per_line.get(n, 0), ln["count"])
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("build", type=Path)
    ap.add_argument("--min-lines", type=float, required=True)
    ap.add_argument("--show", type=int, default=10, help="list this many least-covered files")
    ap.add_argument("--dump", type=Path, help="fftn_platform_dump, to leave out unrunnable targets")
    opts = ap.parse_args()

    build = opts.build.resolve()
    manifest = json.loads((build / "fftn-coverage.json").read_text(encoding="utf-8"))
    root = Path(manifest["source_root"]).resolve()
    if manifest["kind"] == "llvm":
        lines = llvm_lines(manifest, build, root)
    else:
        lines = gcc_lines(manifest, build, root)

    if opts.dump is not None and opts.dump.exists():
        # The dump is instrumented too: its profile goes nowhere rather than to the directory
        # the gate runs in.
        env = {**os.environ, "LLVM_PROFILE_FILE": os.devnull}
        out = subprocess.run(
            [str(opts.dump)], capture_output=True, text=True, check=False, env=env
        ).stdout
        best = next((ln.split()[1] for ln in out.splitlines() if ln.startswith("best_target ")), "")
        skip = target_files(unrunnable(best))
        if skip:
            print(f"   left out (this processor cannot run them): {', '.join(skip)}")
        lines = {rel: v for rel, v in lines.items() if rel not in skip}

    total = sum(len(v) for v in lines.values())
    if total == 0:
        print("   nothing to measure: no instrumented lines in include/fftn/ or src/")
        return 0
    hit = sum(1 for v in lines.values() for c in v.values() if c > 0)
    percent = 100.0 * hit / total
    rows = sorted(
        (100.0 * sum(1 for c in v.values() if c > 0) / len(v), rel, len(v))
        for rel, v in lines.items()
        if v
    )
    below = percent < opts.min_lines
    for pct, rel, n in rows[: opts.show]:
        if pct < 100.0:
            print(f"   {pct:6.1f} %  {n:5d} lines  {rel}")
            if below:
                # Which lines, so that a run on another platform says what to test.
                print(f"             missed: {line_ranges(lines[rel])}")
    print(f"   line coverage {percent:.2f} % ({hit} of {total} lines, {len(lines)} files)")
    if below:
        print(f"coverage: {percent:.2f} % is below the threshold of {opts.min_lines} %")
        return 1
    return 0


def line_ranges(per_line: dict[int, int]) -> str:
    """The lines never executed, as ranges: "12-15, 40"."""
    missed = sorted(n for n, c in per_line.items() if c == 0)
    ranges: list[str] = []
    start = prev = None
    for n in [*missed, None]:
        if n is not None and prev is not None and n == prev + 1:
            prev = n
            continue
        if start is not None:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    return ", ".join(ranges)


if __name__ == "__main__":
    sys.exit(main())

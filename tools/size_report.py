#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The library's text size per instruction-set target, as a Markdown table (informational).

    tools/size_report.py <build dir or library> [--tool <size>] [--summary <file>]

Runs a size tool (`size` or `llvm-size`, Berkeley format) over the static library (libfftn.a,
or fftn.lib on Windows; found under the build directory) and groups its objects by the
instruction-set target whose region they compile (src/isa/CMakeLists.txt): each target's main
unit, its codelet row units and its leaf units (pairs, batch, narrow); everything else is the
common code. The text column is the tool's first: GNU size's `text` (code and read-only data),
llvm-size's (code only on ELF and COFF) or `__TEXT` on Mach-O, so compare numbers from one
platform and tool only. The table goes to stdout and, with --summary, is appended to that file
(CI passes $GITHUB_STEP_SUMMARY). The report never fails a build (DESIGN.md 14.3, decision of
2026-11-17): when no library or no tool is found it says so and exits with 0. Standard library
only.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# In the order of the report; the x86 targets first, then Arm's.
TARGETS = ("scalar", "x86_v2", "avx", "avx2", "avx512", "neon", "neon_fcma")
KINDS = ("main", "rows", "leaves")
# A kernel unit's object: <target>.cpp, rows<k>_<target>.cpp, or a leaf unit
# (pairs_, batch_, batch_large_, narrow_) of the target whose region it compiles. CMake names
# objects <source>.o, <source>.obj on Windows; other tools drop the source suffix.
UNIT = re.compile(
    r"^(?:(?P<rows>rows\d+_)|(?P<leaf>(?:batch_large|batch|pairs|narrow)_))?"
    r"(?P<target>scalar|x86_v2|avx512|avx2|avx|neon_fcma|neon)(?:\.cpp)?\.(?:o|obj)$"
)
# Targets of the other architecture compile to empty units (their bodies are guarded); below
# this many bytes a target is reported as absent.
EMPTY_BYTES = 1024
LIBRARY_NAMES = ("libfftn.a", "fftn.lib")


@dataclass
class Report:
    column: str
    common: int = 0
    targets: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, member: str, text: int) -> None:
        m = UNIT.match(member)
        if not m:
            self.common += text
            return
        kind = "rows" if m["rows"] else "leaves" if m["leaf"] else "main"
        sizes = self.targets.setdefault(m["target"], dict.fromkeys(KINDS, 0))
        sizes[kind] += text

    def total(self) -> int:
        return self.common + sum(sum(s.values()) for s in self.targets.values())


def member_name(name: str) -> str:
    """The object's file name from a size tool's last column, whatever its archive form."""
    name = name.strip()
    ex = re.fullmatch(r"(.+?) \(ex .+\)", name)  # GNU size and llvm-size: "a.o (ex lib.a)"
    if ex:
        name = ex[1]
    else:
        paren = re.fullmatch(r".+\((.+)\)", name)  # Apple's size: "lib.a(a.o)"
        if paren:
            name = paren[1]
    return re.split(r"[\\/]", name)[-1]


def parse(output: str) -> Report:
    """A Berkeley-format listing: a header, then one line of numbers and a name per object."""
    report: Report | None = None
    columns = 0
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if not fields[0].isdigit():
            # The header (repeated by some tools for each file); "filename" names no number.
            if report is None:
                columns = len([f for f in fields if f != "filename"])
                report = Report(column=fields[0])
            continue
        if report is None:
            raise ValueError(f"size output has no header before {line!r}")
        parts = line.split(maxsplit=columns)
        if len(parts) <= columns:
            continue  # a total line without a name
        report.add(member_name(parts[columns]), int(parts[0]))
    if report is None:
        raise ValueError("size output is empty")
    return report


def find_library(path: Path) -> Path | None:
    if path.is_file():
        return path
    for name in LIBRARY_NAMES:
        direct = path / "src" / name
        if direct.is_file():
            return direct
    for name in LIBRARY_NAMES:
        found = sorted(path.rglob(name))
        if found:
            return found[0]
    return None


def find_tool(requested: str | None) -> str | None:
    for candidate in (requested, os.environ.get("FFTN_SIZE_TOOL"), "size", "llvm-size"):
        if candidate and shutil.which(candidate):
            return candidate
    return None


def cache_value(build: Path, key: str) -> str:
    try:
        text = (build / "CMakeCache.txt").read_text(errors="replace")
    except OSError:
        return ""
    m = re.search(rf"^{re.escape(key)}:[A-Z]*=(.*)$", text, re.MULTILINE)
    return m[1].strip() if m else ""


def table(report: Report) -> list[str]:
    total = report.total()

    def share(n: int) -> str:
        return f"{100 * n / total:.1f} %" if total else "-"

    lines = [
        "| target | main | rows | leaves | total | share |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    other = 0
    for target in TARGETS:
        sizes = report.targets.get(target)
        if not sizes:
            continue
        t = sum(sizes.values())
        if t < EMPTY_BYTES:
            other += t
            continue
        lines.append(
            f"| `{target}` | {sizes['main']:,} | {sizes['rows']:,} | {sizes['leaves']:,}"
            f" | {t:,} | {share(t)} |"
        )
    lines.append(f"| common | | | | {report.common:,} | {share(report.common)} |")
    if other:
        lines.append(f"| other architecture (empty units) | | | | {other:,} | {share(other)} |")
    lines.append(f"| **library** | | | | **{total:,}** | 100 % |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help="build directory or the static library itself")
    ap.add_argument("--tool", help="size tool (default: $FFTN_SIZE_TOOL, size, llvm-size)")
    ap.add_argument("--summary", type=Path, help="append the report to this Markdown file")
    args = ap.parse_args()

    library = find_library(args.path)
    tool = find_tool(args.tool)
    if library is None or tool is None:
        what = "no libfftn.a or fftn.lib" if library is None else "no size tool"
        print(f"size report: {what} found under {args.path}; nothing to report")
        return 0
    run = subprocess.run([tool, str(library)], capture_output=True, text=True, check=False)
    if run.returncode != 0:
        print(f"size report: {tool} failed ({run.returncode}): {run.stderr.strip()}")
        return 0
    report = parse(run.stdout)
    build = library.parent.parent if library.parent.name == "src" else library.parent
    compiler = cache_value(build, "FFTN_CXX_COMPILER") or "unknown compiler"
    old_x86 = cache_value(build, "FFTN_OLD_X86_TARGETS")
    config = f", FFTN_OLD_X86_TARGETS={old_x86}" if old_x86 else ""
    text = [
        f"### Library text size ({compiler}{config})",
        "",
        f"`{library.name}`, bytes of `{report.column}` from `{Path(tool).name}`; informational,"
        " never a gate (tools/size_report.py).",
        "",
        *table(report),
        "",
    ]
    out = "\n".join(text)
    print(out)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

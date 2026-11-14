# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Command line (gen/README.md):

uv run python -m gen.fftn_gen                 # write src/gen/
uv run python -m gen.fftn_gen --out DIR       # write elsewhere
uv run python -m gen.fftn_gen --check DIR     # exit 1 unless DIR holds exactly this output
uv run python -m gen.fftn_gen --table         # operation counts against the reference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalogue import all_codelets, generate
from .reference import MissingReference, compare, table

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fftn_gen", description="FFTN's codelet generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--out", type=Path, default=ROOT / "src" / "gen", help="output directory")
    group.add_argument("--check", type=Path, help="compare with this directory, write nothing")
    group.add_argument("--table", action="store_true", help="print operation counts")
    args = parser.parse_args(argv)

    if args.table:
        codelets = all_codelets()
        try:
            print(table(codelets))
        except MissingReference as e:
            print(f"fftn_gen: {e.args[0]}", file=sys.stderr)
            return 1
        problems = [p for c in codelets if (p := compare(c))]
        for p in problems:
            print(p, file=sys.stderr)
        return 1 if problems else 0

    files = generate()  # raises if any codelet fails verification: nothing is written
    if args.check is not None:
        present = (
            {p.name for p in args.check.iterdir() if p.is_file()} if args.check.is_dir() else set()
        )
        bad = sorted(name for name, text in files.items() if not _same(args.check / name, text))
        extra = sorted(present - files.keys())
        for name in bad:
            print(
                f"fftn_gen: {args.check / name} differs from the generator's output",
                file=sys.stderr,
            )
        for name in extra:
            print(f"fftn_gen: {args.check / name} is not written by the generator", file=sys.stderr)
        if bad or extra:
            print(
                "fftn_gen: run `uv run python -m gen.fftn_gen` and commit src/gen/", file=sys.stderr
            )
            return 1
        print(f"fftn_gen: {len(files)} files in {args.check} match the generator")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        if not _same(args.out / name, text):
            (args.out / name).write_text(text, encoding="utf-8", newline="\n")
            print(f"fftn_gen: wrote {args.out / name}")
    return 0


def _same(path: Path, text: str) -> bool:
    return path.is_file() and path.read_bytes() == text.encode("utf-8")


if __name__ == "__main__":
    sys.exit(main())

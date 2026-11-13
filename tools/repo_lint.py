#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Repository hygiene checks for the FFTN gate (scripts/check.sh).

    tools/repo_lint.py spdx    every source file carries the SPDX header
    tools/repo_lint.py docs    relative links in tracked Markdown resolve; PLAN.md has
                               its status section; no file is referenced that is ignored
    tools/repo_lint.py knobs   experimental switches are registered and not expired

Standard library only. Files are taken from git (tracked plus untracked-but-not-ignored),
so a new file is checked before its first commit.
"""

from __future__ import annotations

import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COPYRIGHT = "SPDX-FileCopyrightText:"
LICENSE_ID = "Apache-2.0"

SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".inl", ".c", ".py", ".sh", ".toml", ".cmake", ".sbatch"}
SOURCE_NAMES = {"CMakeLists.txt", ".clang-format", ".clang-tidy"}
# Generated files carry the header too (the generator writes it); nothing is exempt.

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
KNOB_FILE = ROOT / "dev" / "KNOBS.toml"
KNOB_USE = re.compile(r"\bFFTN_X_[A-Z0-9_]+\b")


def repo_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line and (ROOT / line).is_file()]


def is_ignored(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True  # outside the repository
    done = subprocess.run(["git", "check-ignore", "-q", str(rel)], cwd=ROOT, check=False)
    return done.returncode == 0


def check_spdx(files: list[Path]) -> list[str]:
    problems = []
    for f in files:
        if f.suffix not in SOURCE_SUFFIXES and f.name not in SOURCE_NAMES:
            continue
        head = f.read_text(encoding="utf-8", errors="replace")[:1024]
        rel = f.relative_to(ROOT)
        if COPYRIGHT not in head:
            problems.append(f"{rel}: no {COPYRIGHT} header in the first lines")
        elif f"SPDX-License-Identifier: {LICENSE_ID}" not in head:
            problems.append(f"{rel}: no SPDX-License-Identifier: {LICENSE_ID}")
    return problems


def check_docs(files: list[Path]) -> list[str]:
    problems = []
    tracked = set(files)
    for f in files:
        if f.suffix != ".md":
            continue
        rel = f.relative_to(ROOT)
        text = f.read_text(encoding="utf-8")
        in_code = False
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            for target in LINK.findall(line):
                if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                    continue  # URL or in-page anchor
                path = (f.parent / target.split("#", 1)[0]).resolve()
                if not path.exists():
                    problems.append(f"{rel}:{lineno}: broken link {target}")
                elif (path.is_file() and path not in tracked) or is_ignored(path):
                    problems.append(f"{rel}:{lineno}: link to an ignored path {target}")
    plan = ROOT / "dev" / "PLAN.md"
    if plan.exists():
        text = plan.read_text(encoding="utf-8")
        for heading in ("## Status", "## Phases"):
            if heading not in text:
                problems.append(f"dev/PLAN.md: missing section '{heading}'")
    return problems


def check_knobs(files: list[Path]) -> list[str]:
    """Experimental switches (FFTN_X_*) must be registered with an owner and an expiry."""
    used: dict[str, Path] = {}
    for f in files:
        if f.suffix in {".cpp", ".hpp", ".inl", ".h", ".py", ".sh", ".toml"} and f != KNOB_FILE:
            if f == Path(__file__).resolve():
                continue
            for name in KNOB_USE.findall(f.read_text(encoding="utf-8", errors="replace")):
                used.setdefault(name, f)
    registered: dict[str, str] = {}
    if KNOB_FILE.exists():
        current = None
        for line in KNOB_FILE.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\[(FFTN_X_[A-Z0-9_]+)\]", line.strip())
            if m:
                current = m.group(1)
                registered[current] = ""
            m = re.match(r'expires\s*=\s*"?(\d{4}-\d{2}-\d{2})"?', line.strip())
            if m and current:
                registered[current] = m.group(1)
    problems = []
    today = datetime.date.today().isoformat()
    for name, f in sorted(used.items()):
        if name not in registered:
            problems.append(f"{f.relative_to(ROOT)}: switch {name} is not in dev/KNOBS.toml")
    for name, expires in sorted(registered.items()):
        if not expires:
            problems.append(f"dev/KNOBS.toml: {name} has no expires date")
        elif expires < today:
            problems.append(f"dev/KNOBS.toml: {name} expired on {expires}: promote or delete it")
        if name not in used:
            problems.append(f"dev/KNOBS.toml: {name} is registered but no longer used")
    return problems


def main() -> int:
    checks = {"spdx": check_spdx, "docs": check_docs, "knobs": check_knobs}
    wanted = sys.argv[1:] or list(checks)
    files = repo_files()
    failed = False
    for name in wanted:
        if name not in checks:
            print(f"unknown check {name!r}; choose from {', '.join(checks)}")
            return 2
        problems = checks[name](files)
        for p in problems:
            print(p)
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

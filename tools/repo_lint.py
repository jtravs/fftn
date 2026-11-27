#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Repository hygiene checks for the FFTN gate (scripts/check.sh).

    tools/repo_lint.py spdx    every source file carries the SPDX header; shell scripts are
                               executable in git
    tools/repo_lint.py docs    relative links in tracked Markdown resolve; PLAN.md has
                               its status section; no file is referenced that is ignored
    tools/repo_lint.py knobs   experimental switches are registered and not expired
    tools/repo_lint.py markers no file holds a merge's conflict markers
    tools/repo_lint.py snippets  code blocks in docs/ and README.md are quoted from files in
                                 examples/

A snippet is a fenced block in docs/*.md or README.md whose info string names its source:

    ```cpp file=examples/rows.cpp region=plan
    ...
    ```

The block must equal the lines of that file between `// snippet-begin: plan` and
`// snippet-end: plan` (without `region=`, the whole file after its SPDX header), after
removing their common indentation. Every C or C++ block in docs/ and README.md must name a
source, so no untested code reaches a document.

Standard library only. Files are taken from git (tracked plus untracked-but-not-ignored),
so a new file is checked before its first commit.
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COPYRIGHT = "SPDX-FileCopyrightText:"
LICENSE_ID = "Apache-2.0"

SOURCE_SUFFIXES = {
    ".cpp", ".hpp", ".h", ".inl", ".c", ".py", ".sh", ".toml", ".cmake", ".sbatch",
    ".yml", ".yaml", ".ps1",
}  # fmt: skip
SOURCE_NAMES = {"CMakeLists.txt", ".clang-format", ".clang-tidy"}
# Generated files carry the header too (the generator writes it); nothing is exempt.

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
KNOB_FILE = ROOT / "dev" / "KNOBS.toml"
KNOB_USE = re.compile(r"\bFFTN_X_[A-Z0-9_]+\b")
# A merge's conflict markers at the start of a line (the "=======" separator alone is a
# Markdown heading underline, so the outer markers decide).
MARKER = re.compile(r"^(<{7}|>{7}|\|{7})( |$)")


def repo_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line and (ROOT / line).is_file()]


def git_modes() -> dict[str, str]:
    """The mode git records for each tracked file ("100755" when executable)."""
    out = subprocess.run(
        ["git", "ls-files", "--stage"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    modes = {}
    for line in out.splitlines():
        info, _, path = line.partition("\t")
        modes[path] = info.split()[0]
    return modes


def is_ignored(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True  # outside the repository
    done = subprocess.run(["git", "check-ignore", "-q", str(rel)], cwd=ROOT, check=False)
    return done.returncode == 0


def check_spdx(files: list[Path]) -> list[str]:
    problems = []
    modes = git_modes()
    for f in files:
        if f.suffix not in SOURCE_SUFFIXES and f.name not in SOURCE_NAMES:
            continue
        head = f.read_text(encoding="utf-8", errors="replace")[:1024]
        rel = f.relative_to(ROOT)
        if COPYRIGHT not in head:
            problems.append(f"{rel}: no {COPYRIGHT} header in the first lines")
        elif f"SPDX-License-Identifier: {LICENSE_ID}" not in head:
            problems.append(f"{rel}: no SPDX-License-Identifier: {LICENSE_ID}")
        # CI runs scripts from a checkout, which takes the mode from git, not from the disk
        # of whoever committed them. An untracked script is judged by its mode on disk.
        if f.suffix == ".sh":
            mode = modes.get(rel.as_posix())
            executable = mode == "100755" if mode else os.access(f, os.X_OK)
            if not executable:
                problems.append(f"{rel}: not executable (git update-index --chmod=+x {rel})")
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


def check_markers(files: list[Path]) -> list[str]:
    problems = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError, OSError:
            continue  # binary or unreadable: not a merge's text
        for lineno, line in enumerate(text.splitlines(), 1):
            if MARKER.match(line):
                problems.append(f"{f.relative_to(ROOT)}:{lineno}: a merge's conflict marker")
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


FENCE = re.compile(r"^(\s*)```(\S*)(.*)$")
SNIPPET_LANGS = {"c", "cpp", "c++", "cxx", "h", "hpp"}


def dedent(lines: list[str]) -> list[str]:
    lines = [ln.rstrip() for ln in lines]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln]
    cut = min(indents, default=0)
    return [ln[cut:] for ln in lines]


def snippet_source(path: Path, region: str | None) -> list[str] | str:
    if not path.is_file():
        return f"no such file {path.relative_to(ROOT)}"
    lines = path.read_text(encoding="utf-8").splitlines()
    if region is None:
        body = lines
        while body and (body[0].startswith("//") or not body[0].strip()):
            body = body[1:]  # SPDX header and file comment
        return dedent(body)
    begin = [i for i, ln in enumerate(lines) if ln.strip() == f"// snippet-begin: {region}"]
    end = [i for i, ln in enumerate(lines) if ln.strip() == f"// snippet-end: {region}"]
    if len(begin) != 1 or len(end) != 1 or end[0] < begin[0]:
        return f"region {region!r} not found exactly once in {path.relative_to(ROOT)}"
    body = [ln for ln in lines[begin[0] + 1 : end[0]] if "// snippet-" not in ln]
    return dedent(body)


def check_snippets(files: list[Path]) -> list[str]:
    problems = []
    for f in files:
        rel = f.relative_to(ROOT)
        if f.suffix != ".md" or (rel.parts[0] != "docs" and rel != Path("README.md")):
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            m = FENCE.match(lines[i])
            if not m:
                i += 1
                continue
            start = i
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            block = dedent(lines[start + 1 : i])
            i += 1
            lang, info = (
                m.group(2).lower(),
                dict(kv.split("=", 1) for kv in m.group(3).split() if "=" in kv),
            )
            if lang not in SNIPPET_LANGS:
                continue
            if "file" not in info:
                problems.append(f"{rel}:{start + 1}: {lang} block names no source (file=...)")
                continue
            source = snippet_source(ROOT / info["file"], info.get("region"))
            if isinstance(source, str):
                problems.append(f"{rel}:{start + 1}: {source}")
            elif source != block:
                problems.append(f"{rel}:{start + 1}: block differs from {info['file']}")
    return problems


def main() -> int:
    checks = {
        "spdx": check_spdx,
        "docs": check_docs,
        "knobs": check_knobs,
        "markers": check_markers,
        "snippets": check_snippets,
    }
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

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Arms: `label=library[:build][+OPTION=value...]` (DESIGN.md 16.1), and their binaries.

The build is a build directory, or a git revision that is built from `git archive` into its
own directory under bench-out/builds/ (content, not mtimes, decides what is built; a dirty
working tree cannot leak into a commit's arm).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from .store import REPO

ARM = re.compile(
    r"^(?:(?P<label>[A-Za-z0-9_.~-]+)=)?(?P<library>[a-z][a-z0-9_]*)"
    r"(?::(?P<build>[^+]+))?(?P<options>(?:\+[A-Za-z][A-Za-z0-9_]*=[^+\s]*)*)$"
)
# Options the harness handles for every library (test-only; DESIGN.md 16, bench/README.md).
HARNESS_OPTIONS = {"DELAY", "PERTURB"}
# Options the driver handles: FFTW wisdom per campaign.
DRIVER_OPTIONS = {"WISDOM"}
# Executables carry a suffix on Windows only.
EXE_SUFFIX = ".exe" if os.name == "nt" else ""


def executable(build: Path, library: str) -> Path:
    """The path of a library's backend in a build directory."""
    return build / "bench" / f"fbench-{library}{EXE_SUFFIX}"


@dataclass(frozen=True)
class Arm:
    label: str
    library: str
    build: str | None
    options: tuple[tuple[str, str], ...] = ()
    control: bool = False  # the automatic identical-arm copy of the baseline

    @property
    def spec(self) -> str:
        opts = "".join(f"+{k}={v}" for k, v in self.options)
        build = f":{self.build}" if self.build else ""
        return f"{self.label}={self.library}{build}{opts}"

    def option(self, key: str) -> str | None:
        return dict(self.options).get(key)

    def harness_args(self) -> list[str]:
        args: list[str] = []
        for key, value in self.options:
            if key in HARNESS_OPTIONS:
                args += [f"--{key.lower()}", value]
        return args

    def backend_options(self) -> dict[str, str]:
        return {
            k: v for k, v in self.options if k not in HARNESS_OPTIONS and k not in DRIVER_OPTIONS
        }

    def as_control(self) -> Arm:
        return replace(self, label=f"{self.label}~", control=True)


def parse(spec: str) -> Arm:
    m = ARM.match(spec.strip())
    if not m:
        raise ValueError(
            f"bad arm {spec!r}: expected label=library[:build][+OPTION=value...], "
            "e.g. B=fftw+PLANNER=PATIENT"
        )
    options = tuple(
        (k, v) for k, _, v in (o.partition("=") for o in m.group("options").split("+") if o)
    )
    for key, value in options:
        if key in HARNESS_OPTIONS:
            float(value)  # raises ValueError for a non-number
    library = m.group("library")
    return Arm(m.group("label") or library, library, m.group("build"), options)


@dataclass
class Binary:
    """A resolved backend executable and what it says about itself."""

    path: Path
    commit: str | None = None  # set when fbench built the arm from a commit
    describe: dict = field(default_factory=dict)


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_identity(cwd: Path = REPO) -> dict:
    """The commit of the working tree and whether it has uncommitted changes."""
    try:
        commit = git("rev-parse", "HEAD", cwd=cwd)
        dirty = bool(git("status", "--porcelain", "--untracked-files=no", cwd=cwd))
    except OSError, subprocess.CalledProcessError:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


# How each competitor backend is configured (bench/CMakeLists.txt); a backend is built only
# when its variable is set at configure time.
BACKEND_SETTINGS = {
    "fftw": "FFTN_BENCH_FFTW_ROOT",
    "pocketfft": "FFTN_BENCH_POCKETFFT",
    "kfr": "FFTN_BENCH_KFR_ROOT",
    "accelerate": "FFTN_BENCH_ACCELERATE",
    "mkl": "FFTN_BENCH_MKL_ROOT",
    "fftl": "FFTN_BENCH_FFTL_ROOT",
}


def _build_commit(sha: str, library: str, out_dir: Path, log) -> Path:
    cxx = os.environ.get("CXX", "")
    # Every backend setting is part of the key: they choose what the build links.
    settings = sorted((k, v) for k, v in os.environ.items() if k.startswith("FFTN_BENCH_"))
    key = hashlib.sha256(f"{sha}|{cxx}|{settings}".encode()).hexdigest()[:10]
    root = out_dir / "builds" / f"{sha[:12]}-{key}"
    exe = executable(root / "build", library)
    if exe.exists():
        return root / "build"
    src = root / "src"
    if not src.exists():
        log(f"extracting {sha[:12]} with git archive into {src}")
        archive = subprocess.run(
            ["git", "archive", "--format=tar", sha], cwd=REPO, check=True, capture_output=True
        ).stdout
        src.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(src, filter="data")
    if not (src / "bench" / "CMakeLists.txt").exists():
        raise RuntimeError(f"commit {sha[:12]} has no fbench (bench/CMakeLists.txt)")
    build = root / "build"
    log(f"building fbench-{library} at {sha[:12]} in {build}")
    configure = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DFFTN_BUILD_TESTS=OFF",
        "-DFFTN_BUILD_BENCH=ON",
        f"-DFBENCH_SOURCE_COMMIT={sha}",
    ]
    for cmd in (configure, ["cmake", "--build", str(build), "--target", f"fbench-{library}"]):
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{done.stdout[-3000:]}{done.stderr}")
    return build


def resolve(arm: Arm, default_build: Path, out_dir: Path, log=print) -> Binary:
    """The executable of an arm, building it first when the arm names a commit."""
    commit = None
    if arm.build is None:
        build = default_build
    elif Path(arm.build).is_dir():
        build = Path(arm.build)
    else:
        try:
            commit = git("rev-parse", "--verify", f"{arm.build}^{{commit}}")
        except subprocess.CalledProcessError:
            raise ValueError(
                f"arm {arm.label}: {arm.build!r} is neither a build directory nor a commit"
            ) from None
        build = _build_commit(commit, arm.library, out_dir, log)
    exe = executable(build, arm.library)
    if not exe.exists():
        setting = BACKEND_SETTINGS.get(arm.library)
        hint = f" (configure with {setting} set to build it)" if setting else ""
        raise FileNotFoundError(f"arm {arm.label}: no {exe}{hint}")
    return Binary(exe.resolve(), commit)


def describe(binary: Binary, options: dict[str, str]) -> dict:
    """Runs the backend's describe with the arm's backend options (planner, etc.)."""
    done = subprocess.run(
        [str(binary.path), "describe"], capture_output=True, text=True, timeout=60, check=False
    )
    lines = [json.loads(x) for x in done.stdout.splitlines() if x.startswith("{")]
    if done.returncode != 0 or not lines or lines[0].get("type") != "describe":
        raise RuntimeError(f"{binary.path} describe failed: {done.stdout}{done.stderr}")
    info = lines[0]
    info["options"] = options
    return info


def library_problem(info: dict, binary: Path | None = None) -> str | None:
    """Why a backend must not be trusted, or None: it loaded a shared library from outside
    the installation it was built against (DESIGN.md 16.1). A file reached through the
    installation's directory is inside it, even when that directory holds symlinks."""
    loaded, root = info.get("loaded_from"), info.get("expected_root")
    if not loaded or not root:
        return None
    loaded_real = os.path.realpath(loaded)
    if binary is not None and loaded_real == os.path.realpath(binary):
        # A backend bug, not a wrong library: it located its own copy of a symbol.
        return f"describe names the backend executable itself ({loaded}) as the library"
    # Inside the root either as the loader reached it or after resolving symlinks. An
    # installation may link its lib directory into a build tree, so the real file can lie
    # outside the real root although it was loaded through the root; what this check catches
    # is another copy (LD_LIBRARY_PATH, a system install). The lexical test is trusted only
    # when no ".." in the path walked back out through a symlink.
    logical = os.path.abspath(loaded)
    through_root = os.path.realpath(logical) == loaded_real and _under(
        logical, os.path.abspath(root)
    )
    if not (through_root or _under(loaded_real, os.path.realpath(root))):
        return f"loaded {loaded} but was built against {root}"
    return None


def _under(path: str, root: str) -> bool:
    # Lexical and separator-aware: Windows paths use backslashes after abspath/realpath.
    return Path(path) != Path(root) and Path(path).is_relative_to(root)


def timing_problem(info: dict) -> str | None:
    """Why a binary must not be timed, or None: only optimised, uninstrumented builds."""
    harness = info.get("harness", {})
    if harness.get("sanitize"):
        return f"built with sanitizers ({harness['sanitize']})"
    if harness.get("build_type") not in ("Release", "RelWithDebInfo"):
        return f"build type {harness.get('build_type')!r} is not Release"
    return None


def configuration(binary: Binary, arm: Arm) -> str:
    """What an arm measures, for refusing arms that are meant to differ but do not: two
    build directories of the same commit, compiler, flags and library are the same arm.
    The commit is the one recorded at configure time (or the one fbench built)."""
    info = {k: v for k, v in binary.describe.items() if k != "options"}
    return json.dumps(
        {"describe": info, "commit": binary.commit, "options": sorted(arm.options)},
        sort_keys=True,
    )

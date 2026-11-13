#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Record this machine's topology sources as a test fixture for FFTN's platform readers.

    uv run --frozen python tools/capture_platform.py <output>
    uv run --frozen python tools/capture_platform.py --from-tree <dir> <output>

Linux: <output> is a sysfs snapshot file (format in src/platform/file_source.hpp): exactly
the files the sysfs reader reads (src/platform/topology_linux.cpp), for the online CPUs.
/proc/self/status is reduced to its Cpus_allowed_list line, and the cgroup path in
/proc/self/cgroup is renamed level by level (c1/c2/...) so that no job, user or service name
is recorded; the cpu.max (or v1 cpu.cfs_*) files along that path are kept to match.

--from-tree converts a directory tree of such files (an older capture, or a hand-made
synthetic machine) into a snapshot, taking every regular file in it.

macOS: <output> is a file of the `sysctl` keys the macOS readers use (features and
topology), in `sysctl` output form.

The CPU model is printed on stdout, to name the fixture (by model, never by host name).
Run it the way FFTN would run: inside the batch job or container whose limits should be
recorded. Standard library only.
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from pathlib import Path

SNAPSHOT_HEADER = "# fftn-sysfs-snapshot 1"
CPU_FILES = (
    "cpu_capacity",
    "topology/core_cpus_list",
    "topology/thread_siblings_list",
    "topology/physical_package_id",
)
CACHE_FILES = (
    "level",
    "type",
    "size",
    "coherency_line_size",
    "ways_of_associativity",
    "shared_cpu_list",
)
CGROUP_V1_MOUNTS = ("cpu,cpuacct", "cpuacct,cpu", "cpu")
SYSCTL_PREFIXES = (
    "hw.nperflevels",
    "hw.perflevel",
    "hw.pagesize",
    "hw.cachelinesize",
    "hw.logicalcpu",
    "hw.physicalcpu",
    "hw.packages",
    "hw.l1dcachesize",
    "hw.l2cachesize",
    "hw.l3cachesize",
    "hw.cacheconfig",
    "hw.optional.",
    "machdep.cpu.brand_string",
)


def parse_cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for item in text.strip().split(","):
        if not item:
            continue
        first, _, last = item.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def escape(path: str, text: str) -> str:
    """A file's content as one snapshot field: the final newline dropped, then backslash,
    tab and newline escaped. Other control characters are refused."""
    if text.endswith("\n"):
        text = text[:-1]
    bad = sorted({c for c in text if (ord(c) < 32 and c not in "\t\n") or ord(c) == 127})
    if bad:
        sys.exit(f"{path}: control characters {bad!r} cannot be recorded")
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def write_snapshot(files: dict[str, str], out: Path) -> None:
    lines = [SNAPSHOT_HEADER]
    for path in sorted(files):
        parts = path.split("/")
        if "\t" in path or "\\" in path or any(p in ("", ".", "..") for p in parts):
            sys.exit(f"{path!r}: not a path a snapshot can hold")
        lines.append(f"{path}\t{escape(path, files[path])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def read(root: Path, rel: str) -> str | None:
    try:
        return (root / rel).read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return None


def take(root: Path, files: dict[str, str], rel: str) -> bool:
    text = read(root, rel)
    if text is not None:
        files[rel] = text
    return text is not None


def capture_cgroup(root: Path, files: dict[str, str]) -> None:
    kept = []
    for line in (read(root, "proc/self/cgroup") or "").splitlines():
        if line.count(":") < 2:
            continue
        hierarchy, controllers, path = line.split(":", 2)
        parts = [p for p in path.split("/") if p]
        renamed = [f"c{i + 1}" for i in range(len(parts))]
        if hierarchy == "0" and controllers == "":
            mounts = [""]
            names = ("cpu.max",)
        elif "cpu" in controllers.split(","):
            mounts = list(CGROUP_V1_MOUNTS)
            names = ("cpu.cfs_quota_us", "cpu.cfs_period_us")
        else:
            continue
        kept.append(f"{hierarchy}:{controllers}:/" + "/".join(renamed))
        for mount in mounts:
            for depth in range(len(parts) + 1):
                source = "/".join(p for p in ["sys/fs/cgroup", mount, *parts[:depth]] if p)
                target = "/".join(p for p in ["sys/fs/cgroup", mount, *renamed[:depth]] if p)
                for name in names:
                    text = read(root, f"{source}/{name}")
                    if text is not None:
                        files[f"{target}/{name}"] = text
    files["proc/self/cgroup"] = "".join(f"{k}\n" for k in kept)


def capture_linux(root: Path) -> tuple[dict[str, str], str]:
    files: dict[str, str] = {}
    cpu_root = "sys/devices/system/cpu"
    if not take(root, files, f"{cpu_root}/online"):
        sys.exit("no /sys/devices/system/cpu/online: is /sys mounted?")
    for cpu in parse_cpu_list(files[f"{cpu_root}/online"]):
        base = f"{cpu_root}/cpu{cpu}"
        for name in CPU_FILES:
            take(root, files, f"{base}/{name}")
        cache = root / base / "cache"
        for index in sorted(cache.glob("index[0-9]*")) if cache.is_dir() else []:
            for name in CACHE_FILES:
                take(root, files, f"{base}/cache/{index.name}/{name}")
    node_root = root / "sys/devices/system/node"
    for node in sorted(node_root.glob("node[0-9]*")) if node_root.is_dir() else []:
        take(root, files, f"sys/devices/system/node/{node.name}/cpulist")
    for pmu in ("cpu_core", "cpu_atom"):
        take(root, files, f"sys/devices/{pmu}/cpus")
    status = read(root, "proc/self/status")
    if status is not None:
        allowed = [line for line in status.splitlines() if line.startswith("Cpus_allowed_list:")]
        files["proc/self/status"] = "".join(f"{line}\n" for line in allowed)
    if (root / "proc/self/cgroup").is_file():
        capture_cgroup(root, files)
    for line in (read(root, "proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return files, line.split(":", 1)[1].strip()
    return files, ""


def tree_files(tree: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(p for p in tree.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = path.relative_to(tree).as_posix()
        files[rel] = path.read_text(encoding="utf-8")
    if not files:
        sys.exit(f"{tree}: no files")
    return files


def capture_macos(out: Path) -> str:
    text = subprocess.run(
        ["sysctl", "hw", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True
    ).stdout
    kept = sorted(line for line in text.splitlines() if line.startswith(SYSCTL_PREFIXES))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{line}\n" for line in kept))
    model = [line for line in kept if line.startswith("machdep.cpu.brand_string: ")]
    return model[0].split(": ", 1)[1] if model else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="snapshot file (Linux) or sysctl file (macOS)")
    parser.add_argument("--root", type=Path, default=Path("/"), help="Linux: filesystem root")
    parser.add_argument("--from-tree", type=Path, help="convert this directory tree instead")
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists; choose a new name")
    if args.from_tree is not None:
        write_snapshot(tree_files(args.from_tree), args.output)
        return
    if platform.system() == "Darwin":
        model = capture_macos(args.output)
        system = "macos"
    elif platform.system() == "Linux":
        files, model = capture_linux(args.root)
        write_snapshot(files, args.output)
        system = "linux"
    else:
        sys.exit("capture_platform: Linux and macOS only")
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    print(f"model: {model}")
    suffix = "sysctl" if system == "macos" else "sysfs"
    print(f"suggested fixture name: {system}-{slug}.{suffix}")


if __name__ == "__main__":
    main()

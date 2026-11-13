#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Record this machine's topology sources as a test fixture for FFTN's platform readers.

    uv run --frozen python tools/capture_platform.py <output>

Linux: <output> is a directory. Exactly the files the sysfs reader reads
(src/platform/topology_linux.cpp) are copied into it under their own paths, for the online
CPUs. /proc/self/status is reduced to its Cpus_allowed_list line, and the cgroup path in
/proc/self/cgroup is renamed level by level (c1/c2/...) so that no job, user or service name
is recorded; the cpu.max (or v1 cpu.cfs_*) files along that path are copied to match.

macOS: <output> is a file. The `sysctl` keys the macOS readers use (features and topology)
are written in `sysctl` output form.

The CPU model is printed on stdout, to name the fixture (by model, never by host name).
Run it the way FFTN would run: inside the batch job or container whose limits should be
recorded. Standard library only.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

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


def copy(root: Path, out: Path, rel: str) -> bool:
    source = root / rel
    if not source.is_file():
        return False
    try:
        data = source.read_bytes()
    except OSError:
        return False
    target = out / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return True


def capture_cgroup(root: Path, out: Path) -> None:
    lines = (root / "proc/self/cgroup").read_text().splitlines()
    kept = []
    for line in lines:
        hierarchy, controllers, path = line.split(":", 2)
        parts = [p for p in path.split("/") if p]
        renamed = [f"c{i + 1}" for i in range(len(parts))]
        if hierarchy == "0" and controllers == "":
            mounts = [""]
            files = ("cpu.max",)
        elif "cpu" in controllers.split(","):
            mounts = list(CGROUP_V1_MOUNTS)
            files = ("cpu.cfs_quota_us", "cpu.cfs_period_us")
        else:
            continue
        kept.append(f"{hierarchy}:{controllers}:/" + "/".join(renamed))
        for mount in mounts:
            for depth in range(len(parts) + 1):
                source = Path("sys/fs/cgroup", mount, *parts[:depth])
                target = Path("sys/fs/cgroup", mount, *renamed[:depth])
                for name in files:
                    if (root / source / name).is_file():
                        (out / target).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(root / source / name, out / target / name)
    (out / "proc/self").mkdir(parents=True, exist_ok=True)
    (out / "proc/self/cgroup").write_text("".join(f"{k}\n" for k in kept))


def capture_linux(out: Path, root: Path) -> str:
    cpu_root = "sys/devices/system/cpu"
    if not copy(root, out, f"{cpu_root}/online"):
        sys.exit("no /sys/devices/system/cpu/online: is /sys mounted?")
    for cpu in parse_cpu_list((root / cpu_root / "online").read_text()):
        base = f"{cpu_root}/cpu{cpu}"
        for name in CPU_FILES:
            copy(root, out, f"{base}/{name}")
        cache = root / base / "cache"
        for index in sorted(cache.glob("index[0-9]*")) if cache.is_dir() else []:
            for name in CACHE_FILES:
                copy(root, out, f"{base}/cache/{index.name}/{name}")
    node_root = root / "sys/devices/system/node"
    for node in sorted(node_root.glob("node[0-9]*")) if node_root.is_dir() else []:
        copy(root, out, f"sys/devices/system/node/{node.name}/cpulist")
    for pmu in ("cpu_core", "cpu_atom"):
        copy(root, out, f"sys/devices/{pmu}/cpus")
    status = (root / "proc/self/status").read_text().splitlines()
    allowed = [line for line in status if line.startswith("Cpus_allowed_list:")]
    (out / "proc/self").mkdir(parents=True, exist_ok=True)
    (out / "proc/self/status").write_text("".join(f"{line}\n" for line in allowed))
    if (root / "proc/self/cgroup").is_file():
        capture_cgroup(root, out)
    model = ""
    for line in (root / "proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    return model


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
    parser.add_argument("output", type=Path, help="fixture directory (Linux) or file (macOS)")
    parser.add_argument("--root", type=Path, default=Path("/"), help="Linux: filesystem root")
    args = parser.parse_args()
    if platform.system() == "Darwin":
        model = capture_macos(args.output)
    elif platform.system() == "Linux":
        if args.output.exists():
            sys.exit(f"{args.output} exists; choose a new directory")
        model = capture_linux(args.output, args.root)
    else:
        sys.exit("capture_platform: Linux and macOS only")
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    print(f"model: {model}")
    system = "macos" if platform.system() == "Darwin" else "linux"
    print(f"suggested fixture name: {system}-{slug}")


if __name__ == "__main__":
    main()

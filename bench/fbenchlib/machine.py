# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""The machine: host fingerprint, cache size and the quiet-machine check (DESIGN.md 16.3).

The fingerprint names the hardware and OS, never the host: results are stored and may be
shared, and hostnames are private.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except OSError, subprocess.TimeoutExpired:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or platform.machine()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                key, _, value = line.partition(":")
                if key.strip() in ("model name", "Model", "CPU part"):
                    return value.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def performance_cores() -> int | None:
    """Performance cores on Apple silicon (perflevel0); None elsewhere."""
    if platform.system() != "Darwin":
        return None
    value = _sysctl("hw.perflevel0.physicalcpu")
    return int(value) if value.isdigit() else None


def last_level_cache_bytes() -> int:
    """The largest cache the OS reports (0 if none). Apple's system-level cache is not
    reported, which is one reason the cold pool has a generous minimum."""
    sizes: list[int] = []
    if platform.system() == "Darwin":
        for name in ("hw.l3cachesize", "hw.perflevel0.l2cachesize", "hw.l2cachesize"):
            value = _sysctl(name)
            if value.isdigit():
                sizes.append(int(value))
    else:
        for size in Path("/sys/devices/system/cpu/cpu0/cache").glob("index*/size"):
            text = size.read_text().strip()
            mult = {"K": 1024, "M": 1024 * 1024}.get(text[-1:], 1)
            digits = text.rstrip("KM")
            if digits.isdigit():
                sizes.append(int(digits) * mult)
    return max(sizes, default=0)


def fingerprint() -> dict:
    """CPU model, core counts, OS and caches; no hostname."""
    info = {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count() or 0,
        "performance_cores": performance_cores(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "llc_bytes": last_level_cache_bytes(),
    }
    key = f"{info['cpu']}|{info['logical_cpus']}|{info['os']}|{info['machine']}"
    info["host_id"] = hashlib.sha256(key.encode()).hexdigest()[:12]
    return info


def _proc_stat() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(v) for v in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    return sum(values), idle


def busy_cores(exclude: set[int] | None = None) -> float | None:
    """CPU cores' worth of work done by other processes right now, or None if unknown.

    Linux: from /proc/stat over a quarter of a second (the driver sleeps meanwhile).
    macOS: the sum of ps's %cpu (a decaying average) over other processes.
    """
    ncpu = os.cpu_count() or 1
    system = platform.system()
    try:
        if system == "Linux":
            t0, i0 = _proc_stat()
            time.sleep(0.25)
            t1, i1 = _proc_stat()
            total = t1 - t0
            return 0.0 if total <= 0 else ncpu * (total - (i1 - i0)) / total
        if system == "Darwin":
            out = _run(["ps", "-A", "-o", "pid=,%cpu="])
            skip = (exclude or set()) | {os.getpid()}
            busy = 0.0
            for line in out.splitlines():
                pid, _, cpu = line.strip().partition(" ")
                if pid.isdigit() and int(pid) not in skip:
                    busy += float(cpu.strip() or 0.0)
            return busy / 100.0
    except OSError, ValueError:
        return None
    return None


@dataclass(frozen=True)
class QuietLimits:
    max_load: float
    max_busy: float

    @staticmethod
    def defaults() -> QuietLimits:
        # Half the logical CPUs for the load average (which counts our own process and,
        # on macOS, the efficiency cores' work); a tenth of them, at least one core, for
        # the other processes' current use.
        ncpu = os.cpu_count() or 1
        return QuietLimits(max_load=0.5 * ncpu, max_busy=max(1.0, 0.1 * ncpu))


def load_state() -> dict:
    load1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    return {"load1": load1, "busy": busy_cores()}


def is_quiet(state: dict, limits: QuietLimits) -> tuple[bool, str]:
    load1, busy = state.get("load1"), state.get("busy")
    if load1 is not None and load1 > limits.max_load:
        return False, f"load average {load1:.2f} > {limits.max_load:.2f}"
    if busy is not None and busy > limits.max_busy:
        return False, f"other processes use {busy:.2f} cores > {limits.max_busy:.2f}"
    return True, ""


def thermal_note() -> str:
    """One line on frequency or thermal state where the OS exposes it cheaply."""
    if platform.system() == "Darwin":
        lines = _run(["pmset", "-g", "therm"]).splitlines()
        keep = [ln.strip() for ln in lines if "limit" in ln.lower() or "level" in ln.lower()]
        return "; ".join(keep)
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return f"governor {gov.read_text().strip()}"
    except OSError:
        return ""

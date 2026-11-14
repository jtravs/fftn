# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Verdicts from stored records: the same code serves `run` (between rounds) and `show`."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

from . import stats

WRONG = "wrong output"
BASE_WRONG = "baseline wrong"
UNSUPPORTED = "unsupported"
ERROR = "error"
TIMEOUT = "timed out"
# Timing statuses the driver writes for a process that hit --process-timeout: the case it
# was on, and the cases it never reached. Both are left out of the statistics.
TIMED_OUT = "timeout"
NOT_REACHED = "skipped"


@dataclass
class CaseResult:
    case: str
    verdict: str
    interval: stats.Interval | None = None
    base_ns: float | None = None  # median over rounds of the baseline's per-process statistic
    arm_ns: float | None = None
    err_rms: float | None = None  # worst over rounds, this arm
    note: str = ""


@dataclass
class Analysis:
    baseline: str
    control: str | None
    band: float  # log units
    valid_rounds: list[int]
    invalid_rounds: int
    control_results: dict[str, stats.Interval] = field(default_factory=dict)
    results: dict[str, dict[str, CaseResult]] = field(default_factory=dict)
    geomean: dict[str, stats.Interval | None] = field(default_factory=dict)

    def unresolved(self, targets: set[str]) -> list[tuple[str, str]]:
        out = []
        for arm, cases in self.results.items():
            for case, r in cases.items():
                if case not in targets or r.verdict not in (
                    stats.FASTER,
                    stats.SLOWER,
                    stats.NO_EVIDENCE,
                ):
                    continue
                if r.interval is None or not stats.resolved(r.interval, self.band):
                    out.append((arm, case))
        return out


def _seed(text: str, base: int) -> int:
    return base * 1_000_003 + sum(ord(c) * (i + 1) for i, c in enumerate(text))


def analyse(
    records: list[dict],
    stat: str = "median",
    min_band: float = 0.01,
    seed: int = 0,
    resamples: int = 2000,
) -> Analysis:
    run = next(r for r in records if r["type"] == "run")
    arms = [a["label"] for a in run["arms"]]
    baseline = arms[0]
    control = next((a["label"] for a in run["arms"] if a.get("control")), None)
    cases = [c["id"] for c in run["cases"]]
    weights = {c["id"]: c.get("weight", 1.0) for c in run["cases"]}
    valid = sorted(r["round"] for r in records if r["type"] == "round" and r.get("valid"))
    invalid = sum(1 for r in records if r["type"] == "round" and not r.get("valid"))
    valid_set = set(valid)

    key = "min_ns" if stat == "min" else "median_ns"
    values: dict[str, dict[str, dict[int, float]]] = {a: {c: {} for c in cases} for a in arms}
    wrong: dict[str, set[str]] = {a: set() for a in arms}
    status: dict[str, dict[str, str]] = {a: {} for a in arms}
    errs: dict[str, dict[str, float]] = {a: {} for a in arms}
    # Rounds in which a process timeout cost a case its timing, per arm and case: the
    # rounds it timed out in, and the rounds it was not reached in.
    timed_out: dict[str, dict[str, set[int]]] = {a: {} for a in arms}
    not_reached: dict[str, dict[str, set[int]]] = {a: {} for a in arms}
    for r in records:
        if r["type"] != "timing" or r["arm"] not in values or r["case"] not in values[r["arm"]]:
            continue
        arm, case = r["arm"], r["case"]
        if r.get("status") in (TIMED_OUT, NOT_REACHED):
            if r["round"] in valid_set:
                lost = timed_out if r["status"] == TIMED_OUT else not_reached
                lost[arm].setdefault(case, set()).add(r["round"])
            continue
        if r.get("status") != "ok":
            status[arm][case] = r.get("status", ERROR)
            continue
        check = r.get("check", {})
        if not check.get("pass", False):
            wrong[arm].add(case)  # in any round, valid or not: wrong output is never noise
        err = check.get("err_rms")
        if err is not None:
            errs[arm][case] = max(errs[arm].get(case, 0.0), err)
        if r["round"] in valid_set and r.get(key):
            values[arm][case][r["round"]] = r[key]

    result = Analysis(baseline, control, math.log1p(min_band), valid, invalid)
    if control is not None:
        for c in cases:
            ratios = stats.paired_log_ratios(values[baseline][c], values[control][c])
            if ratios:
                rng = random.Random(_seed(c, seed))
                result.control_results[c] = stats.interval(ratios, rng, resamples)
        result.band = stats.noise_band(list(result.control_results.values()), min_band)

    for arm in arms:
        if arm in (baseline, control):
            continue
        per_case: dict[str, CaseResult] = {}
        per_round: dict[str, dict[int, float]] = {}
        for c in cases:
            res = CaseResult(c, stats.NO_EVIDENCE, err_rms=errs[arm].get(c))
            base_vals, arm_vals = values[baseline][c], values[arm][c]
            if base_vals:
                res.base_ns = statistics.median(base_vals.values())
            if arm_vals:
                res.arm_ns = statistics.median(arm_vals.values())
            if c in wrong[arm]:
                res.verdict = WRONG
            elif c in wrong[baseline]:
                res.verdict = BASE_WRONG
            elif status[arm].get(c) or status[baseline].get(c):
                res.verdict = status[arm].get(c) or status[baseline].get(c) or ERROR
                res.note = "arm" if status[arm].get(c) else "baseline"
            else:
                paired = {
                    r: math.log(arm_vals[r] / base_vals[r])
                    for r in sorted(base_vals.keys() & arm_vals.keys())
                }
                lost_in = {
                    who: timed_out[a].get(c, set()) | not_reached[a].get(c, set())
                    for who, a in (("arm", arm), ("baseline", baseline))
                }
                if paired:
                    rng = random.Random(_seed(arm + c, seed))
                    res.interval = stats.interval(list(paired.values()), rng, resamples)
                    res.verdict = stats.verdict(res.interval, result.band)
                    per_round[c] = paired
                    lost = lost_in["arm"] | lost_in["baseline"]
                    if lost:
                        res.note = f"{len(lost)} rounds lost to timeouts"
                elif lost_in["arm"] or lost_in["baseline"]:
                    # Timed out (or never reached) in every round that has the other side.
                    res.verdict = TIMEOUT
                    who = "arm" if lost_in["arm"] else "baseline"
                    a = arm if who == "arm" else baseline
                    res.note = who if timed_out[a].get(c) else f"{who}, never reached"
            per_case[c] = res
        result.results[arm] = per_case
        result.geomean[arm] = stats.weighted_geomean_interval(
            per_round, weights, random.Random(_seed(arm, seed))
        )
    return result

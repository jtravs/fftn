# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Statistics of fbench A/B runs (DESIGN.md 16.3).

The unit of replication is a process round: process-to-process spread dominated the
prototype's noise (DESIGN.md finding E10), so every arm runs once per round, ratios are
paired by round, and intervals come from resampling rounds. Everything is computed on
log ratios, so that "2x slower" and "2x faster" are symmetric.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

FASTER = "faster"
SLOWER = "slower"
NO_EVIDENCE = "no evidence"

# Below this many paired rounds a bootstrap interval of a median means little.
MIN_PAIRS = 3


def williams_orders(k: int) -> list[list[int]]:
    """Arm orders for successive rounds: a Williams design.

    Every arm takes every position equally often, and every arm directly follows every other
    arm equally often (first-order carry-over balance), over len(result) rounds: k for even
    k, 2k for odd k.
    """
    if k < 1:
        raise ValueError("need at least one arm")
    first = [0]
    low, high = 1, k - 1
    while len(first) < k:
        first.append(low)
        low += 1
        if len(first) < k:
            first.append(high)
            high -= 1
    rows = [[(a + i) % k for a in first] for i in range(k)]
    if k % 2 == 1 and k > 1:
        rows += [list(reversed(r)) for r in rows]
    return rows


def paired_log_ratios(base: Mapping[int, float], other: Mapping[int, float]) -> list[float]:
    """log(other / base) for every round present in both, in round order."""
    return [
        math.log(other[r] / base[r])
        for r in sorted(base.keys() & other.keys())
        if base[r] > 0 and other[r] > 0
    ]


def bootstrap_ci(
    values: Sequence[float],
    rng: random.Random,
    stat: Callable[[Sequence[float]], float] = statistics.median,
    resamples: int = 2000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval of stat(values), resampling values with replacement."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    if n == 1:
        return values[0], values[0]
    draws = sorted(stat(rng.choices(values, k=n)) for _ in range(resamples))
    tail = (1.0 - level) / 2.0
    lo = draws[math.floor(tail * resamples)]
    hi = draws[min(resamples - 1, math.ceil((1.0 - tail) * resamples) - 1)]
    return lo, hi


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (statistics.quantiles needs two or more values)."""
    if not values:
        raise ValueError("no values")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    i = math.floor(pos)
    j = min(i + 1, len(xs) - 1)
    return xs[i] + (xs[j] - xs[i]) * (pos - i)


@dataclass(frozen=True)
class Interval:
    """A median log ratio with its bootstrap interval, over `pairs` rounds."""

    median: float
    lo: float
    hi: float
    pairs: int


def interval(log_ratios: Sequence[float], rng: random.Random, resamples: int = 2000) -> Interval:
    lo, hi = bootstrap_ci(log_ratios, rng, resamples=resamples)
    return Interval(statistics.median(log_ratios), lo, hi, len(log_ratios))


def noise_band(control: Sequence[Interval], min_band: float, q: float = 0.95) -> float:
    """The noise band in log units, from the identical-arm (A/A) control.

    The band is the q-quantile over cases of |median log ratio| of A against its identical
    copy, and never less than log(1 + min_band): 95 % of identical-arm comparisons in this
    run land inside it. An effect is claimed only when its whole interval lies beyond the
    band, so a case needs both a resolved interval and an effect larger than identical arms
    show (an order, position or drift effect would show in the control first). Simulated
    with lognormal process noise of 2-3 %, this gave no false verdict in 2400 identical
    cases; a band from the A/A intervals' reach was as safe but needed about twice the
    rounds to detect a 5 % change.
    """
    floor = math.log1p(min_band)
    usable = [abs(c.median) for c in control if c.pairs >= MIN_PAIRS]
    if not usable:
        return floor
    return max(floor, quantile(usable, q))


def verdict(iv: Interval, band: float) -> str:
    """faster, slower or no evidence: whether the interval of log(B/A) clears the band."""
    if iv.pairs < MIN_PAIRS:
        return NO_EVIDENCE
    if iv.lo > band:
        return SLOWER
    if iv.hi < -band:
        return FASTER
    return NO_EVIDENCE


def resolved(iv: Interval, band: float) -> bool:
    """A case is resolved when it has a verdict, or its interval lies inside the band."""
    if iv.pairs < MIN_PAIRS:
        return False
    return verdict(iv, band) != NO_EVIDENCE or (iv.lo >= -band and iv.hi <= band)


def weighted_geomean_interval(
    per_round: Mapping[str, Mapping[int, float]],
    weights: Mapping[str, float],
    rng: random.Random,
    resamples: int = 1000,
) -> Interval | None:
    """Weighted mean over cases of the median log ratio, with a bootstrap over rounds.

    per_round[case][round] is the paired log ratio of that case in that round. Rounds are
    resampled as wholes, which keeps the correlation between cases of the same round.
    """
    cases = [c for c in per_round if per_round[c] and weights.get(c, 0.0) > 0]
    if not cases:
        return None
    rounds = sorted(set().union(*(per_round[c].keys() for c in cases)))
    total = sum(weights[c] for c in cases)

    def mean_of(chosen: Sequence[int]) -> float:
        acc = 0.0
        for c in cases:
            vals = [per_round[c][r] for r in chosen if r in per_round[c]]
            if not vals:
                vals = list(per_round[c].values())
            acc += weights[c] * statistics.median(vals)
        return acc / total

    centre = mean_of(rounds)
    if len(rounds) < 2:
        return Interval(centre, centre, centre, len(rounds))
    draws = sorted(mean_of(rng.choices(rounds, k=len(rounds))) for _ in range(resamples))
    lo = draws[int(0.025 * resamples)]
    hi = draws[min(resamples - 1, math.ceil(0.975 * resamples) - 1)]
    return Interval(centre, lo, hi, len(rounds))

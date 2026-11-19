#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Combines process rounds of t3_kokkos (experiment T3): t3.py ROUND.jsonl...

Each file is one process. For every probe, case and arm it prints the median over the rounds
of the in-process median time, and of the paired ratio to the probe's first arm, with the
range of that ratio over the rounds.
"""

import json
import statistics
import sys
from collections import defaultdict


def main(paths: list[str]) -> None:
    rows: dict[tuple, list[dict]] = defaultdict(list)
    order: list[tuple] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("{"):
                    continue
                row = json.loads(line)
                key = (row["backend"], row["threads"], row["probe"], row["case"], row["arm"])
                if key not in rows:
                    order.append(key)
                rows[key].append(row)
    print(f"{len(paths)} rounds")
    last = None
    for key in order:
        group = key[:4]
        if group != last:
            print(f"\n{key[0]} {key[1]} threads, {key[2]}: {key[3]}")
            last = group
        values = rows[key]
        time_ns = statistics.median(v["median_ns"] for v in values)
        ratios = [v["ratio"] for v in values]
        print(
            f"  {key[4]:24s} {time_ns:12.1f} ns   ratio {statistics.median(ratios):7.4f}"
            f"  [{min(ratios):.4f}, {max(ratios):.4f}]"
        )


if __name__ == "__main__":
    main(sys.argv[1:])

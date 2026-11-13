# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Emission order (DESIGN.md 7.2, step 5): greedy list scheduling for few live values.

At each step the scheduler emits, among the arithmetic operations whose operands are
ready, the one that increases the number of live values least: it counts the value it
creates, the loads it needs first, and the operands it reads for the last time. Ties go
to the operation whose newest operand was computed most recently, which finishes one
sub-transform before starting the next (the depth-first order of Frigo, "A fast Fourier
transform compiler", PLDI 1999). Loads go just before their first use and stores just
after their producer.

Measured over radices 2-16 and all variants, this keeps fewer values live than both the
recursive red/blue partition of that paper and the builders' own (already recursive)
creation order: in total 502 against 598 and 549 (gen/README.md, "Subtleties").
"""

from __future__ import annotations

from .lower import Op

_LOADS = ("load", "load_tw")


def schedule(ops: list[Op]) -> list[Op]:
    n = len(ops)
    producer = {op.dst: i for i, op in enumerate(ops) if op.dst >= 0}
    preds = [sorted({producer[a] for a in op.args}) for op in ops]
    compute = [i for i, op in enumerate(ops) if op.kind not in (*_LOADS, "store")]
    is_load = [op.kind in _LOADS for op in ops]
    # Remaining arithmetic readers of each operation's value (stores are placed afterwards).
    readers = [0] * n
    waiting = [0] * n  # arithmetic operands not yet computed
    users: list[list[int]] = [[] for _ in range(n)]
    for i in compute:
        for p in preds[i]:
            readers[p] += 1
            users[p].append(i)
            if not is_load[p]:
                waiting[i] += 1
    when = [-1] * n  # step at which a value was computed or loaded
    ready = {i for i in compute if waiting[i] == 0}
    seq: list[int] = []

    def score(i: int) -> tuple[int, int, int]:
        change = 1
        for p in preds[i]:
            change += int(is_load[p] and when[p] < 0)
            change -= int(readers[p] == 1)
        newest = max((when[p] for p in preds[i] if not is_load[p]), default=-1)
        return (change, -newest, i)

    while ready:
        i = min(ready, key=score)
        ready.remove(i)
        step = len(seq)
        for p in preds[i]:
            if is_load[p] and when[p] < 0:
                when[p] = step
            readers[p] -= 1
        when[i] = step
        seq.append(i)
        for u in users[i]:
            waiting[u] -= 1
            if waiting[u] == 0:
                ready.add(u)
    assert len(seq) == len(compute), "the operations form a DAG"

    after: dict[int, list[int]] = {}
    for i, op in enumerate(ops):
        if op.kind == "store":
            after.setdefault(preds[i][0], []).append(i)
    out: list[int] = []
    placed: set[int] = set()
    for i in seq:
        for p in preds[i]:
            if is_load[p] and p not in placed:
                placed.add(p)
                out.append(p)
        out.append(i)
        out.extend(after.get(i, []))
    assert sorted(out) == list(range(n)), "every load has a user and every store a producer"
    return [ops[i] for i in out]


def peak_live(ops: list[Op] | tuple[Op, ...]) -> int:
    """Largest number of values (complex and twiddle registers) live at once."""
    last: dict[int, int] = {}
    for i, op in enumerate(ops):
        for a in op.args:
            last[a] = i
    live, peak = 0, 0
    for i, op in enumerate(ops):
        if op.dst >= 0:
            live += 1
            peak = max(peak, live)
        for a in set(op.args):
            if last[a] == i:
                live -= 1
    return peak

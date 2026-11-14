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

With the same load and store placement, this keeps no more values live than the builders'
own, already recursive, creation order (creation_order()) on any codelet, and fewer in
total; a test holds both over every codelet. The recursive red/blue partition of that
paper, implemented first, lost to both: it is in commit aa966ef with the same test, which
measured over radices 2-16 in both directions a total peak of 1004 live values for this
order, 1196 for the partition and 1098 for creation order.
"""

from __future__ import annotations

from .lower import Op

# Loads of the complex-register lowering (lower.py) and of the split one (split.py).
_LOADS = ("load", "load_tw", "load_re", "load_im", "load_twr", "load_twi")


def _graph(ops: list[Op]) -> tuple[list[list[int]], list[int], list[bool]]:
    """Operand positions of each operation, the arithmetic operations, which are loads."""
    producer = {op.dst: i for i, op in enumerate(ops) if op.dst >= 0}
    preds = [sorted({producer[a] for a in op.args}) for op in ops]
    compute = [i for i, op in enumerate(ops) if op.kind not in (*_LOADS, "store")]
    is_load = [op.kind in _LOADS for op in ops]
    return preds, compute, is_load


def place(ops: list[Op], seq: list[int]) -> list[Op]:
    """The arithmetic operations in the order seq (positions in ops), each load just before
    its first use and each store just after its producer (the later of its two, for a split
    store of a real and an imaginary part)."""
    preds, compute, is_load = _graph(ops)
    assert sorted(seq) == compute, "seq orders the arithmetic operations"
    rank = {i: r for r, i in enumerate(seq)}
    after: dict[int, list[int]] = {}
    for i, op in enumerate(ops):
        if op.kind == "store":
            after.setdefault(max(preds[i], key=lambda p: rank[p]), []).append(i)
    out: list[int] = []
    placed: set[int] = set()
    for i in seq:
        for p in preds[i]:
            if is_load[p] and p not in placed:
                placed.add(p)
                out.append(p)
        out.append(i)
        out.extend(after.get(i, []))
    assert sorted(out) == list(range(len(ops))), "every load has a user, every store a producer"
    return [ops[i] for i in out]


def creation_order(ops: list[Op]) -> list[Op]:
    """The builders' own order (already recursive), with the same placement: a baseline."""
    return place(ops, _graph(ops)[1])


def schedule(ops: list[Op]) -> list[Op]:
    """The greedy order, or the creation order where that keeps fewer values live."""
    greedy, created = greedy_order(ops), creation_order(ops)
    return created if peak_live(created) < peak_live(greedy) else greedy


def greedy_order(ops: list[Op]) -> list[Op]:
    n = len(ops)
    preds, compute, is_load = _graph(ops)
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
    return place(ops, seq)


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

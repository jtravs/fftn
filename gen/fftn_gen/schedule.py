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

With the same load and store placement, this keeps no more values live than the recursive
red/blue partition of that paper (partition()) or the builders' own, already recursive,
creation order (creation_order()) on any codelet, and fewer in total; a test measures all
three over every codelet. Over radices 2-16 in both directions the totals are 1004
(greedy), 1196 (partition) and 1098 (creation order).
"""

from __future__ import annotations

from .lower import Op

_LOADS = ("load", "load_tw")


def _graph(ops: list[Op]) -> tuple[list[list[int]], list[int], list[bool]]:
    """Operand positions of each operation, the arithmetic operations, which are loads."""
    producer = {op.dst: i for i, op in enumerate(ops) if op.dst >= 0}
    preds = [sorted({producer[a] for a in op.args}) for op in ops]
    compute = [i for i, op in enumerate(ops) if op.kind not in (*_LOADS, "store")]
    is_load = [op.kind in _LOADS for op in ops]
    return preds, compute, is_load


def place(ops: list[Op], seq: list[int]) -> list[Op]:
    """The arithmetic operations in the order seq (positions in ops), each load just before
    its first use and each store just after its producer."""
    preds, compute, is_load = _graph(ops)
    assert sorted(seq) == compute, "seq orders the arithmetic operations"
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
    assert sorted(out) == list(range(len(ops))), "every load has a user, every store a producer"
    return [ops[i] for i in out]


def creation_order(ops: list[Op]) -> list[Op]:
    """The builders' own order (already recursive), with the same placement: a baseline."""
    return place(ops, _graph(ops)[1])


def schedule(ops: list[Op]) -> list[Op]:
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


def partition(ops: list[Op]) -> list[Op]:
    """The recursive red/blue partition of Frigo (PLDI 1999), with the same placement.

    Kept for one commit as the evidence for choosing schedule() (gen/README.md): the
    operations of a set are split by depth into those nearer its sources (red, closed
    under predecessors) and the rest (blue), each half split into connected components,
    recursively."""
    # Loads and stores take part in the partition (they shape the depths and components);
    # place() then moves them.
    preds, compute, _ = _graph(ops)
    everything = list(range(len(ops)))
    cpreds = {i: preds[i] for i in everything}
    csuccs: dict[int, list[int]] = {i: [] for i in everything}
    for i in everything:
        for p in cpreds[i]:
            csuccs[p].append(i)

    def components(nodes: list[int]) -> list[list[int]]:
        inside, seen, out = set(nodes), set(), []
        for start in nodes:
            if start in seen:
                continue
            comp, stack = [], [start]
            seen.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for w in cpreds[v] + csuccs[v]:
                    if w in inside and w not in seen:
                        seen.add(w)
                        stack.append(w)
            out.append(sorted(comp))
        return sorted(out)

    def order(nodes: list[int]) -> list[int]:
        if len(nodes) <= 2:
            return sorted(nodes)
        comps = components(nodes)
        if len(comps) > 1:
            return [v for c in comps for v in order(c)]
        inside = set(nodes)
        depth_in: dict[int, int] = {}
        for v in nodes:
            depth_in[v] = max((depth_in[p] + 1 for p in cpreds[v] if p in inside), default=0)
        depth_out: dict[int, int] = {}
        for v in reversed(nodes):
            depth_out[v] = max((depth_out[w] + 1 for w in csuccs[v] if w in inside), default=0)
        red = [v for v in nodes if depth_in[v] <= depth_out[v]]
        blue = [v for v in nodes if depth_in[v] > depth_out[v]]
        return order(red) + order(blue)

    arithmetic = set(compute)
    return place(ops, [i for i in order(everything) if i in arithmetic])


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

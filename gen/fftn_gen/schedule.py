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

The fewest live values is the wrong aim once a codelet cannot fit the register file: for
radix 32 and 64 (peaks of 35 and 67 against AVX2's 16 registers) what costs is the
compiler's spill traffic, and GCC on x86 allocates registers in the order it is given (it
schedules only after allocation). So a complex codelet whose order keeps more values live
than the fewest registers of any SIMD target is also ordered by fitted_order(), a list
scheduler that tracks a model register file, and that order is taken when spill_moves(),
a furthest-next-use model of the allocator, says it saves at least a quarter of the moves
and at least MIN_SAVED of them (experiment X54). Only the order changes: every operation
and every rounding is the same, so the results are bit-identical.
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
            # A store of a load (output 0 of a pruned codelet of radix 2, dif_head: its one
            # input) goes just after that load.
            computed = [p for p in preds[i] if p in rank]
            anchor = max(computed, key=lambda p: rank[p]) if computed else max(preds[i])
            after.setdefault(anchor, []).append(i)
    out: list[int] = []
    placed: set[int] = set()
    for i in seq:
        for p in preds[i]:
            if is_load[p] and p not in placed:
                placed.add(p)
                out.append(p)
                out.extend(after.get(p, []))
        out.append(i)
        out.extend(after.get(i, []))
    assert sorted(out) == list(range(len(ops))), "every load has a user, every store a producer"
    return [ops[i] for i in out]


def creation_order(ops: list[Op]) -> list[Op]:
    """The builders' own order (already recursive), with the same placement: a baseline."""
    return place(ops, _graph(ops)[1])


# The fewest registers of a SIMD target that compiles the complex codelets: AVX2's 16 (X54).
REGISTERS = 16

# A fitted order replaces the fewest-live one only if it saves at least this many of
# spill_moves() (and a quarter): radices 14 to 16 would save 2 moves, not worth new text.
MIN_SAVED = 16


def schedule(ops: list[Op], registers: int | None = None) -> list[Op]:
    """The greedy order, or the creation order where that keeps fewer values live; with
    registers, the fitted order where it spills much less in that many registers."""
    greedy, created = greedy_order(ops), creation_order(ops)
    chosen = created if peak_live(created) < peak_live(greedy) else greedy
    if registers is None or peak_live(chosen) <= registers:
        return chosen
    before = spill_moves(chosen, registers)
    # The model register file a little smaller than the real one leaves room for the
    # temporaries the vocabulary needs (a swap for addj, a twiddle's two halves); the best
    # of a few sizes and weights, the first on ties.
    fitted = min(
        (
            fitted_order(ops, size, weight)
            for size in range(registers - 4, registers + 1)
            for weight in (1, 2)
        ),
        key=lambda o: spill_moves(o, registers),
    )
    after = spill_moves(fitted, registers)
    return fitted if before - after >= max(MIN_SAVED, before / 4) else chosen


def fitted_order(ops: list[Op], registers: int, miss_weight: int) -> list[Op]:
    """List scheduling against a model register file: the `registers` values produced or
    read most recently, among those still needed, are in registers. The ready operation
    with the lowest miss_weight * (operands outside the model file) + (change in live
    values) goes next; ties to the one whose newest operand was touched most recently."""
    n = len(ops)
    preds, compute, is_load = _graph(ops)
    readers = [0] * n
    waiting = [0] * n
    users: list[list[int]] = [[] for _ in range(n)]
    for i in compute:
        for p in preds[i]:
            readers[p] += 1
            users[p].append(i)
            if not is_load[p]:
                waiting[i] += 1
    loaded = [False] * n
    touched = [-1] * n
    ready = {i for i in compute if waiting[i] == 0}
    held: list[int] = []  # the model register file, most recently touched last
    seq: list[int] = []

    def score(i: int) -> tuple[int, int, int]:
        misses = 0
        change = 1
        for p in preds[i]:
            if is_load[p] and not loaded[p]:
                change += 1
            elif p not in held:
                misses += 1
            change -= int(readers[p] == 1)
        newest = max((touched[p] for p in preds[i]), default=-1)
        return (miss_weight * misses + change, -newest, i)

    while ready:
        i = min(ready, key=score)
        ready.remove(i)
        step = len(seq)
        for p in preds[i]:
            loaded[p] = True
            readers[p] -= 1
            touched[p] = step
            if p in held:
                held.remove(p)
            if readers[p] > 0:
                held.append(p)
        touched[i] = step
        held.append(i)
        held = held[-registers:]
        seq.append(i)
        for u in users[i]:
            waiting[u] -= 1
            if waiting[u] == 0:
                ready.add(u)
    assert len(seq) == len(compute), "the operations form a DAG"
    return place(ops, seq)


def spill_moves(ops: list[Op] | tuple[Op, ...], registers: int) -> int:
    """Stack stores, reloads and second loads that a register allocator evicting the value
    used furthest ahead (Belady) needs to run ops in `registers` registers. A twiddle takes
    two registers; an evicted input or twiddle is loaded again from its source, a computed
    value is stored once and reloaded at each later use. GCC 16's AVX2 lane passes come
    close: 327 and 101 stack moves a unit for radix 64 and 32 against 292 and 93 here
    (X47, X54)."""
    uses: dict[int, list[int]] = {}
    for i, op in enumerate(ops):
        for a in op.args:
            uses.setdefault(a, []).append(i)
    kind = {op.dst: op.kind for op in ops if op.dst >= 0}
    width = {v: 2 if k == "load_tw" else 1 for v, k in kind.items()}
    nxt = dict.fromkeys(uses, 0)
    never = len(ops)

    def next_use(v: int, now: int) -> int:
        u = uses[v]
        while nxt[v] < len(u) and u[nxt[v]] < now:
            nxt[v] += 1
        return u[nxt[v]] if nxt[v] < len(u) else never

    held: set[int] = set()
    stacked: set[int] = set()
    seen: set[int] = set()
    used = moves = 0

    def make_room(need: int, now: int, keep: set[int]) -> None:
        nonlocal used, moves
        while used + need > registers:
            v = max(sorted(held - keep), key=lambda v: next_use(v, now))
            held.discard(v)
            used -= width[v]
            if next_use(v, now) < never and kind[v] not in _LOADS and v not in stacked:
                stacked.add(v)
                moves += 1

    for i, op in enumerate(ops):
        if op.kind in _LOADS:
            continue  # loaded at first use, below
        keep = set(op.args)
        for a in op.args:
            if a not in held:
                if a in seen:
                    moves += 1  # a reload, or a second load of an input or twiddle
                seen.add(a)
                make_room(width[a], i, keep)
                held.add(a)
                used += width[a]
        for a in keep:
            if next_use(a, i + 1) == never:
                held.discard(a)
                used -= width[a]
        if op.dst >= 0 and op.dst in uses:
            make_room(1, i, set())
            seen.add(op.dst)
            held.add(op.dst)
            used += 1
    return moves


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

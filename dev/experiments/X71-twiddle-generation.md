# X71: planning time, twiddle generation by octant symmetry and shared root tables

- **Status:** running
- **Owner:** x71 session, 2026-11-24
- **Phase / task:** client acceptance, criterion (4): plan construction under 10 ms

## Question

Can plan construction of the client catalogue's long real rows (18432–40960 points, r2c and
c2r, and 32768) come under 10 ms on the Mac, the EPYC 7763 and the Xeon 6240, with the twiddle
tables unchanged bit for bit, so that no transform gets slower or changes its results?

## Hypothesis

The client catalogue check of 2026-11-24 measured plan construction, rules effort, one thread:
worst case medians 10.2 ms (Mac), 20.9 ms (EPYC), 28.6 ms (Xeon); over 10 ms the real rows of
18432–40960 and 32768 on x86-64 (8 shapes), 32768 on the Mac (runs 20261124-102656-dadb Mac,
-102950-017c EPYC, -103457-da59 Xeon). Planning time is linear in the length: almost all of it
is the twiddle tables (DESIGN.md 8.4), each root a double-double Taylor evaluation of sine and
cosine, 360 ns a root on the Mac (`unit_roots<double>(32768)` 6.1 ms, 16385 roots). An r2c or
c2r plan of n builds the root table of n/2 for its half-length chain and of n for its pairing
twiddles (`chain_builder::roots`), and the grid computes its phase-1 and pairing twiddles one
root at a time (`unit_root`): r2c 32768 computes 24578 roots, r2c 40960 on the EPYC's grid
plan about 26000.

Three changes remove most of that work without changing a single bit:

1. **Octant symmetry.** For n divisible by 8 the first half of the table has n/8 + 1 distinct
   reduced angles, each shared by up to four entries (m = i, 2p − i, 2p + i, 4p − i with
   p = n/8). Computing each angle's series once and placing it by the existing octant rule gives
   the table for a quarter of the evaluations, from the same operands, so bit for bit.
2. **Shared root tables.** `root(2m, 2n)` is `root(m, n)` with every operand of every
   double-double operation scaled by two (integers converted exactly, the division's quotient
   unchanged, no underflow), so it is equal bit for bit. The chain's table of n/2 is sliced from
   the table of n when that exists, and the table of n takes its even entries from the table of
   n/2 when that exists: an r2c of n computes the roots of n once.
3. **The grid's roots through a cache** (`root_cache`): its tables' entries one at a time, as
   before, but an angle's series is kept once computed (33 bytes for each of the n/8 + 1 angles),
   so entries sharing an angle cost one evaluation. No table of the n roots is built.

Expected on the Mac: r2c 32768 from about 9 ms to 1.6 ms, r2c 40960 (grid) from 6 ms to 2.5 ms;
the x86-64 machines scale with their root cost (0.8–0.9 µs a root), so every client shape under
10 ms there too.

## Change under test

Arm A: `main` at 1b1d129 (the base of the branch). Arm B: the branch's change commit (named in
Results): `unit_roots` builds tables by octant symmetry where 8 divides n and takes a table of
n/2 to copy its even entries (`src/core/twiddles.cpp`); `chain_builder::roots` slices or extends
its root tables across lengths n 2^k; `grid_builder` takes its roots from a `root_cache` per
length. No planner rule, no plan, no table entry and no kernel changes. No switch.

**Bit identity, the evidence:**
- Tests (in CI on every target): `Twiddles.TablesHoldTheEntriesComputedAlone*` (every entry of
  every table of lengths 1–400 and eight long lengths, both signs, double and float, equals
  `unit_root`'s, the per-entry computation that is unchanged), `Twiddles.TablesFromOtherTables*`
  (tables built from the table of n/2, and entries m 2^k of tables of n 2^k for 2^k = 2, 4, 8,
  equal the table), `Twiddles.RootCacheGivesTheEntries*`; the existing reference checks of every
  table entry against quadruple precision.
- Development (not a test, recorded here): the new `unit_roots` against a copy of the old one,
  every table compared with memcmp, both signs, double and float, whole tables, tables from the
  half table and the half-table slices: every length 1–16384, every 104th multiple of 8 from
  16392 to 131072, f · 2^k for f in 1, 3, 5, 9, 15 up to 4194304, and the client catalogue's
  lengths: 0 mismatches.

So every plan's program and tables are those of arm A, and transform times and results are
unchanged by construction: no transform timing decides.

## Method

- Cases: the client catalogue's tracking tier (52 shapes, `pdev/catalogues/` file, private) and
  the standard catalogue's tracking tier (`--tier tracking`). fbench `run A=fftn:<A> B=fftn:<B>`
  with the A/A control, one thread, rules effort (the default), short timing batches
  (`--batch-time 0.002 --batches 3 --warmup 0.005`): the metric is the plan time fbench records
  per case and round (`plan_s`, the construction of one plan in a fresh process per round).
- Machines: the Mac (M5 Max, Clang 23, `neon_fcma`; on AC power, checked before and after),
  the EPYC 7763 (DMOG, GCC 16, `avx2`, shared 8-core allocation), the Xeon 6240 (GCC 14,
  `avx512`, shared allocation, `--serial`). 8 rounds each (fixed: `--min-rounds 8 --max-rounds 8`).
- Metric: per case and arm, the median plan time over the rounds (the maximum is reported too).

**Decision rule** (written before the judging runs). Adopt B if, on each of the three machines:

1. every case's FFTN note (the plan) is identical between A and B;
2. the sum over the client catalogue's 52 cases of B's median plan times is at most 0.6 of A's;
3. no case of either catalogue has B's median plan time above A's × 1.10 + 0.05 ms (plan times of
   small plans are tens of microseconds, where a process's first allocations dominate);

and the bit-identity tests pass in CI on all four required checks. Otherwise reject.

**Target, reported and not a condition:** criterion (4), every client shape's median plan time
under 10 ms on all three machines (and its maximum over the rounds, as the client check also
reports).

## Results

(to be filled after the judging runs)

## Verdict

(to be filled)

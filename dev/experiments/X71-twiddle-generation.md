# X71: planning time, twiddle generation by octant symmetry and shared root tables

- **Status:** done
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

Arm A 1b1d129, arm B cbfcc29 (the rule, 298fd4c, was committed and pushed before every run).
8 rounds each, one thread, rules effort. Plan times are each case's median over the rounds
(maximum in brackets). The A/A control's sum of medians was within 1.2 % of A's in every run.

| Machine (run, catalogue) | Load | Notes identical | Sum of medians A → B (B ÷ A) | Slower cases (rule 3) | Worst case A → B | Over 10 ms A → B |
|---|---|---|---|---|---|---|
| Mac M5 Max, Clang 23 (20261124-221912-bb81, client) | 6.8–9.1, AC throughout | 52 of 52 | 97.1 → 21.3 ms (0.219) | 0 | r2c-32768 9.33 (9.71) → 1.59 (1.67); worst B r2c-40960 2.65 (3.06) | 0 → 0 |
| Mac (20261124-222350-028f, standard) | 4.3–4.4, AC | 61 of 61 | 1850 → 467 ms (0.252) | 0 | c2c-4194304 804 → 213 | 13 → 6 |
| EPYC 7763, GCC 16 (20261124-222427-ac05, client) | 41–48 (shared node) | 52 of 52 | 224.6 → 49.5 ms (0.221) | 0 | r2c-40960 20.85 (21.27) → 5.13 (5.54) | 8 → 0 |
| EPYC (20261124-222427-e7b3, standard) | 42–48 | 61 of 61 | 3868 → 1022 ms (0.264) | 0 | c2c-4194304 1697 → 475 | 15 → 8 |
| Xeon 6240, GCC 14 (20261124-222921-127d, client) | 2.3–14 | 52 of 52 | 252.2 → 58.1 ms (0.230) | 0 | r2c-40960 26.49 (26.98) → 5.68; worst B c2r-40960 5.65 (6.19) | 8 → 0 |
| Xeon (20261124-224111-b0f2, standard) | 2.1–2.3 | 61 of 61 | 3568 → 1064 ms (0.298) | 0 | c2c-4194304 1873 → 482 | 14 → 8 |

Client catalogue, the shapes that were over 10 ms on x86-64 (medians, ms; A → B):

| Shape | Mac | EPYC | Xeon |
|---|---|---|---|
| r2c / c2r 40960 × 6 | 6.26 → 2.65 / 2.15 → 0.77 | 20.85 → 5.13 / 20.75 → 5.01 | 26.49 → 5.68 / 26.77 → 5.65 |
| r2c / c2r 32768 | 9.33 → 1.59 / 9.19 → 1.60 | 19.49 → 3.73 / 19.47 → 3.53 | 20.79 → 3.99 / 21.20 → 4.19 |
| r2c 20480 × 21 / c2r 20480 × 3 | 5.83 → 1.02 / 5.81 → 1.03 | 12.14 → 2.20 / 12.19 → 2.33 | 13.32 → 2.83 / 13.18 → 2.75 |
| r2c / c2r 18432 | 5.28 → 0.93 / 5.25 → 0.92 | 10.98 → 2.03 / 11.02 → 2.10 | 12.07 → 2.52 / 11.85 → 2.34 |

Over the standard tier the remaining long plans are the long rows themselves (4194304 and
1048576 points, 262144, 100000) and the prime 65537 (Bluestein/Rader: its chirp and kernel roots
are computed alone; 63.7 → 43.3 ms on the Mac): their tables hold n roots, so they stay linear in
n at a quarter of the old cost.

The Mac client run's checkout had an uncommitted documentation edit (fbench marks it dirty);
both arms were built from their commits.

CI on cbfcc29: the four required checks passed (lint, gcc-16, clang-23 ASan, macOS arm64), with
the bit-identity tests.

## Verdict

**Adopted.** Every criterion holds on all three machines: identical plans in every case, sums
of median plan times 0.219–0.298 of A's (rule: at most 0.6), no case slower beyond 10 % +
0.05 ms, and the bit-identity tests green in CI. The target is met: every client shape plans
in under 10 ms on the Mac, the EPYC and the Xeon, by median and by maximum over the rounds
(worst 2.65 / 5.13 / 5.68 ms, maxima 3.06 / 5.54 / 6.19 ms). Since the tables are unchanged bit
for bit, transform times and results are unchanged by construction; no rule reach applies (no
planner rule, every target takes the change). No near-miss re-run or narrowing was used.

Follow-ups, not done: the Bluestein chirp and Rader kernels could take their roots through
`root_cache` or the chain's tables (65537: 43 ms on the Mac); a cheaper exactly rounded
generator (a table of angles and short series) would cut the long rows' tables further, but it
is a different function of (m, n) and would need the bit-identity argument replaced by a
correct-rounding one. DESIGN.md 8.4 and LESSONS.md 4.25 record the method.

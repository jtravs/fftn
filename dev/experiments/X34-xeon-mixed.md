# X34: long rows of mixed length on the Xeon

- **Status:** done
- **Owner:** x34 session, 2026-11-16
- **Phase / task:** 1 (PLAN lead (c); follows X12b, X25, X26, X26b, X29)

## Question

Why is FFTN's c2c row of 100000 points 1.99 times FFTW's time on the Xeon 6240 (46080:
1.29; MKL 1.31 and less), when X29's chains took the same rows to 0.66–0.75 of `main` on
the EPYC and the Mac, and which rule closes it without a vendor switch? The answer sets the
chain-or-grid rule and the grid's block and factors for lengths that are not powers of two
on machines with little L3 per core.

## Hypothesis

X29's chains for such lengths key on 2 MiB of L3 per CPU (EPYC) or a shared L2 (Apple); the
Xeon (1 MiB private L2, 0.69 MiB of L3 per CPU) keeps `main`'s grid. The grids it builds
there are poor for lengths with a large odd part:

- 100000 = 2⁵ · 5⁵ is a three-phase grid 40 × 50 × 50 with blocks of 4 columns. A block is
  a power of two dividing both the first factor and the columns of phase 1, so the five
  factors of 2 allow at most 4: every row a copy moves is one 64-byte register (X25:
  `block_copy` 40 % of the cycles, 8.2M instructions per transform against FFTW's 2.4M).
- 46080 = 2¹⁰ · 45 (720 KiB, beyond L2/2 and within L2) is a two-phase grid 192 × 240,
  whose longest factor (240) limits the block to 4 within the L1 budget.

X29's Xeon screen showed chains losing beyond L2/2 for powers of two (c2c 65536 1.23,
2¹⁹ 1.17) but did not time c2c rows with a large odd part. Expected: either the chain
(fewer passes over L3 than the grid's three copies plus passes) or a grid whose blocks are
wider (a multiple of the register's columns that need not be a power of two, 20 at 100000;
three phases at 46080) takes these rows well below `main`'s time.

## Change under test

Chosen after the screen (step 1), at most three arms, each a commit against `main`:

- (a) a chain-or-grid rule for lengths that are not powers of two on machines with a
  private L2, keyed on the L2 (or on the measured crossover), not on the vendor or L3 per CPU;
- (b) the grid's block a multiple of the register's complex count dividing the first factor
  and the columns of phase 1, not only a power of two (100000: 20 instead of 4);
- (c) whatever the screen and counters point to (factors or phase radices of lengths with a
  large odd part).

Bits: a changed plan changes the rounding of the rows it covers; both routes are tested
against the derived tolerances (the chain and grid suites force each).

## Method

- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512, MKL 2026.1, Slurm
  `-c 8` shared, `--serial`); EPYC 7763 (`avx2`, GCC 16.2, DMOG `-c 8` shared) and the Mac
  M5 Max (`neon_fcma`, Homebrew Clang, NEON FFTW) for collateral checks of any arm that
  changes a plan there. fbench interleaved rounds with the A/A control.
- Step 1, screen (Xeon): c2c rows 2^k × odd from X29's `x29-odd-part` group and the long
  mixed rows (`bench/catalogues/x29.toml`), arms A = `main`, C = `STRATEGY=chain`, F = FFTW
  (`PLAN_NOTE=1`), M = MKL; for 100000 and 46080 also forced grids (`GRID_BLOCK`,
  `GRID_FACTORS`, `GRID_THREE_BYTES`). `run --pmu` counters where the time does not explain
  itself.
- Step 2, each arm against A = `main` on the screened cases it changes and the standard
  tracking tier, on every machine-target whose plans it changes.
- Metric: median paired ratios with 95 % intervals; weighted geomeans over the cases whose
  plan an arm changes; the A/A band.
- Decision rule, written before running: an arm is adopted on a machine-target if over the
  cases whose plan it changes the weighted geomean B ÷ A is below 1 − band with its interval
  below 1, no such case is slower than 1 + max(band, 3 %), and no tracking case is slower than
  1 + max(band, 3 %) (re-run once if a case is just past it on a loaded machine). Release
  toolchains decide (GCC on x86 Linux, Clang on macOS). The EPYC and the Mac keep `main`'s
  plans unless the arm also passes this rule there.

## Results

### Step 1, the screen (Xeon)

Run 20261116-113928-9d84 (Slurm 16011; arms A = `main` 4ebcee5, C = chain forced, F, M;
40 c2c rows of x29's odd-part and long groups; stopped after 6 rounds, load 1.1–1.5 other
cores; the run's A/A spread ±4 %). **Chains lose on the Xeon at every row the rules make a
grid**: C ÷ A 1.05–1.45 (46080 1.33, 61440 1.45, 100000 1.06, 10⁶ 1.05); the rows that
are chains in `main` (≤ 32768 points, and the odd 59049 and 78125) are unchanged. So X29's
chains are not the Xeon's answer, whatever the rule's key. FFTN ÷ FFTW in that run: 100000
2.02 (A 7.6 ns per point, FFTW 3.8), 46080 1.08, 10⁶ 1.04, 2 · 10⁶ 0.97, the other grids
1.0–1.5. The plans: FFTN 100000 = grid 40 × 50 × 50 in blocks of 4 (phases 5 × 8, 5 × 10,
5 × 10); FFTW 25 × 20 × 10 × 20, recursive DIT with no copies; FFTN's chain 4 × 5 × 5 × 10
× 10 × 10.

A forced-grid screen (run 20261116-115712-7427, Slurm 16013) stopped at round 0 on a bad
option value (`GRID_THREE_BYTES=0`); its one round gave blocks of 20 at 100000 0.77 of A
and blocks of 16 at 46080 1.02.

### Counters (Xeon, user mode, `run --pmu`, min of 2–3 processes)

Campaign directory `~/awork/fftn-x34` on the Xeon (Slurm 16026, 16030, 16031; node load
13–30 from other users, so cycles, not times). c2c 100000 per transform:

| plan | cycles | instructions | L2 misses | stall cycles |
|---|---|---|---|---|
| `main`: 40 × 50 × 50, blocks of 4 | 2.73M | 8.05M | 129k | 0.45M |
| 40 × 50 × 50, blocks of 20 (arm b) | 2.20M (0.80) | 5.16M | 124k | 0.47M |
| 200 × 500, blocks of 4 (two phases) | 1.92M (0.70) | 5.84M | 78k | 0.21M |
| 200 × 500, blocks of 20 | 1.88M (0.69) | 4.42M | 83k | 0.36M |
| chain 4 × 5 × 5 × 10 × 10 × 10 | 2.81M (1.03) | 2.20M | 343k | 1.72M |

FFTW (X25): 1.33M cycles, 2.35M instructions. The chain has FFTW's instruction count but
stalls on memory; the three-phase grid's extra pass over memory, not its copy instructions
alone, is the rest. Two phases against three on every mixed row the Xeon makes a
three-phase grid (`GRID_THREE_BYTES=never`, `main`'s blocks), cycles ÷ `main`'s:

| odd part ≥ 15 (the rule's rows) | ratio | odd part 3–9 | ratio |
|---|---|---|---|
| 71680, 86016 | 0.924, 0.915 | 73728, 81920, 98304 | 0.951, 0.967, 0.958 |
| 100000 | 0.730 | 294912 | 1.126 |
| 184320 | 1.014 | 327680 | 1.125 |
| 204800, 221184, 245760, 256000 | 0.898, 0.875, 0.888, 0.923 | 393216 | 1.082 |
| 737280, 819200, 983040 | 0.878, 0.835, 0.765 | | |
| 10⁶, 1024000, 2 · 10⁶, 4 · 10⁶ | 0.698, 0.880, 0.738, 0.895 | | |

L2 misses per point fall from 1.3 to 0.8–1.1 with two phases. Rows nearer a power of two
lose at 294912–393216 (likely L1 set conflicts in the two-phase gathers, whose rows lie
nearly a multiple of 4 KiB apart there), hence the odd-part key.
The same screen on the EPYC 7763 (DMOG Slurm 510630, cycles and instructions): two phases
took **1.13–1.67** of the cycles on every such row below 16 MiB (only 2 · 10⁶ and 4 · 10⁶,
0.89 and 0.76, gained), so the rule stays off where X29's chains apply. Other counter
probes: 46080 and 64000 in three phases 1.24 and 1.09 of `main`'s two (kept), 51200 1.21.

Arm (b) (blocks that are other multiples of the register's columns, commits 8cf4187 and
713b99f) was superseded before its timing run: its one gain, 100000 in three phases (0.80),
is below two phases with `main`'s blocks (0.70), and with two phases no measured row keeps
the narrowest block in three phases. It was removed (a9c4bf1).

### Step 2, arm (c) against `main` (Xeon)

Arm (c) = commit d33ea09 (`grid_two_odd_part` = 15 on the Xeon; later commits only format
and comment it). Run 20261116-125601-0542 (Slurm 16033 build, 16034 timing; 16 rounds, 6
under load; A/A band ±1.95 %), x29's 23 mixed rows from 46080 to 2 · 10⁶, arms A =
`main` 29f9772, B = d33ea09, F = FFTW, M = MKL:

- The 14 rows whose plan changes (odd part ≥ 15, beyond L2): geomean B ÷ A **0.861**, every
  one at or below 1 (largest 184320 0.996; 100000 0.711 [0.696, 0.723], 10⁶ 0.724, 2 · 10⁶
  0.742, 819200 0.814, 983040 0.818, 737280 0.874, 1024000 0.866, 245760 0.891, 204800
  0.892, 221184 0.900, 256000 0.938, 86016 0.968, 71680 0.983). fbench's weighted geomean
  over all 23 rows 0.915 [0.901, 0.934].
- The 9 rows whose plan does not change (46080, 51200, 64000 within L2; odd part 3–9):
  0.989–1.032, all "no evidence" (327680 1.032 [0.971, 1.107], an unchanged plan).
- FFTN ÷ FFTW: 100000 2.03 → **1.44**, 10⁶ 1.01 → 0.73, 2 · 10⁶ 0.99 → 0.73; over the 23
  rows 1.19 → 1.09. FFTN ÷ MKL: 100000 1.34 → **0.95**, 10⁶ 1.27 → 0.92; over the 23 rows
  1.10 → 1.01. 46080 (a two-phase grid within L2, unchanged) stays 1.30 of FFTW and 1.01 of
  MKL.

Tracking tier (only 100000 changes plan there): run 20261116-131703-f74f (Slurm 16035
build, 16036 timing; 59 rounds, 11 under load; band ±1.68 %): c2c 100000 0.706 [0.699,
0.722], every other case "no evidence", the largest r2c 1024² 1.030 [0.999, 1.056] (an
unchanged plan); geomean 0.995. FFTN ÷ FFTW over the tier 0.905 → 0.900, ÷ MKL 1.007 →
1.002; 100000 2.04 → 1.44 of FFTW and 1.33 → 0.94 of MKL.

The EPYC and the Mac: `grid_two_odd_part` is 0 there (X29's large-L3 rule and the shared
L2 respectively; `Schedule.GridSizesFromCaches` checks it), so no plan changes and no run
was needed.

## Verdict

- **Diagnosis.** On the Xeon, 100000 was a three-phase grid in blocks of one register
  (power-of-two part 32), with 3.4 times FFTW's instructions; the third phase's pass over
  memory beyond a small, slow L3 costs more than the cache fit it buys.
- **Arm (a), chains for these lengths on the Xeon: rejected by the screen** (1.05–1.45 of the
  grid at every grid row; the chain at 100000 has FFTW's instruction count but stalls on
  memory). No rule keyed on L2 can make chains pay here.
- **Arm (b), blocks that are other multiples of the register's columns: dropped**
  (100000 0.80 in cycles, superseded by arm c's 0.70; no row left that needs it).
- **Arm (c), two phases for rows with an odd part ≥ 15 where a private L2 has less than 2 MiB
  of L3 per CPU: adopted on the Xeon** by the decision rule (0.861 over the 14 changed rows,
  none slower; tracking tier 0.995, only 100000 changed, 0.706). The EPYC (two phases 1.13–1.67 of the cycles below 16 MiB) and the
  Mac keep `main`'s plans (the field is 0 there). DESIGN 8.5 and the decision log, LESSONS
  3.14.
- **Unverified and open.** Granite Rapids (X31: 100000 2.16, 46080 1.60) in an 8-vCPU VM
  reports 504 MiB of L3 for 8 CPUs, so X29's chains apply there, not this rule: whether it
  wants the Xeon's plan needs a run with `CHAIN_PART_BYTES=0 GRID_TWO_ODD_PART=15` against
  the default; if it does, cache sizes as a VM reports them cannot separate it from Zen, and
  the key needs a core field or a measured choice (lead (d), planner 2.4). 46080 (1.30 of
  FFTW, a two-phase grid within L2: three phases 1.24, blocks of 16 1.02) and 100000's
  remaining 1.44 (FFTW's recursive 25 × 20 × 10 × 20 without copies) are not closed.

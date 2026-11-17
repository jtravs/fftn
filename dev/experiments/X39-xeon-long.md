# X39: the Xeon's long rows and in-place 65536 (exit-check causes 5 and 6)

- **Status:** done (arms A, C and D adopted on the Xeon profile and `avx512` double, arm B rejected)
- **Owner:** x39 session, 2026-11-16
- **Phase / task:** 1 (phase 1 exit check, the tail rule; follows X25, X26, X26b, X32, X34)

## Question

Why are the Xeon 6240's long double rows still 1.2–1.5 times FFTW's time in the exit check
(run 20261116-161123-f848), and which plan rule closes each one on `avx512` without a
vendor switch and without changing the plans of the EPYC, the Mac, Granite Rapids and
Zen 5? The answer sets the Xeon profile's chain and grid choices for 5000–262144 points.

## Hypothesis

The failing rows, FFTN ÷ FFTW (÷ MKL) in the exit run, with `main`'s plans:

| row | FFTN plan (`main`) | ÷ FFTW | ÷ MKL |
|---|---|---|---|
| 5000, 10000 | chains 4 · 5 · 5 · 5 · 10, 4 · 5 · 5 · 10 · 10 | 1.04, 0.98 | 1.18, 1.16 |
| 16384 | chain 16 · 8 · 8 · 16 (split from pass 2) | 1.25 | 1.28 |
| 20000 | chain 4 · 5 · 10 · 10 · 10 | 1.39 | 0.96 |
| 25000 | chain 4 · 5 · 5 · 5 · 5 · 10 (six passes) | 1.50 | 1.13 |
| 32768 | chain 16 · 8 · 16 · 16 (512 KiB, at `grid_min_bytes`) | 1.48 | 1.24 |
| 46080 | grid 192 × 240, blocks of 4 | 1.31 | 0.98 |
| 65536 in place (and × 3) | grid 256 × 256, blocks of 4 | 1.16 | 1.15 |
| 100000 | grid 200 × 500, blocks of 4 (X34) | 1.45 | 0.93 |
| 131072, 262144 | grids 32 × 64 × 64, 64³, blocks of 32 | 1.24, 1.22 | 1.01, 0.96 |

So the rows up to 32768 are chains on the Xeon (its grid starts beyond L2/2, X12b), not
grids. Candidate causes, to be told apart by the screen:

- **Chains of 256–512 KiB on `avx512`** (16384–32768): radix-16 passes read 16 streams a
  power of two apart (X32: on huge pages they share an L2 set) and the chain makes four
  passes over a row that half fills L2; X32's radix-8 chains were 1.06–1.07 of these on the
  Xeon. A grid from L2/4, or radix-32/64 passes (fewer passes; X3 priced them for float
  only), may beat both.
- **Rows with an odd part 5⁴–5⁵** (20000, 25000, 5000, 10000): FFTN has no radix-20 or 25
  codelet, so 5⁵ takes at least five passes; the rules chose six at 25000. Either another
  order of the same passes, fewer passes, or a two-phase grid (whose phases fit L1) wins.
- **Grids in blocks of 4 columns** (46080, 65536, 100000): every gathered row is one 64-byte
  register (X25: `block_copy` 40 % of the cycles); the phase radices are 12–16. Wider
  blocks (X34: 16 at 46080, 1.02 in one round), phase radices of 32 or 64 (fewer passes per
  phase), or other factor orders.
- **Three-phase grids** (131072, 262144): phases of radix 8 × 8 in blocks of 32; single
  radix-64 passes per phase would halve the passes over L1.
- **In place 65536**: the in-place grid route costs more than out of place (1.16 against
  1.09 of FFTW); its copy structure.

## Change under test

Chosen after the screen (step 1), at most three arms, each a commit against `main`, each
changing plans only where the Xeon's profile (a private L2 with less than 2 MiB of L3 per
CPU behind it, `avx512`) or a narrower key says so:

- (a) the chain-or-grid boundary or the chain's radices for 16384–32768 (and 5000–25000);
- (b) the grid's phase radices or blocks on `avx512` (16384–262144, in place 65536);
- (c) whatever the screen points to (factor orders for rows with an odd part 5⁴–5⁵).

Bits: a changed plan changes the rounding of the rows it covers; both routes are tested
against the derived tolerances (the chain and grid suites force each).

## Method

- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512, MKL 2026.1,
  Slurm `-c 8` shared, fbench `--serial`). Collateral: the EPYC 7763, the Mac, Granite
  Rapids and Zen 5 keep `main`'s plans unless an arm also passes the rule there; the
  profile unit tests on each machine's recorded topology check that no plan changes there.
- Step 1, screen (no library change): per row, user-mode counters (`fbench-fftn run
  --pmu`, cycles, instructions, L2 misses; min over processes, robust to the shared node's
  load) of `main`'s plan against forced alternatives (`STRATEGY=grid|chain`, `RADICES`,
  `GRID_FACTORS`, `GRID_BLOCK`, `GRID_PHASE_RADIX=64`, `GRID_THREE_BYTES`), FFTW's plan
  (`PLAN_NOTE=1`) and `PLANNING=measure`'s choice (does an existing candidate already win?).
  An fbench run of the target rows (arms `main`, measure, FFTW, MKL) gives times.
- Step 2, each arm against A = `main`, fbench interleaved rounds with the A/A control, on
  the standard catalogue's target rows (`--tier full`, filter on the rows above) and on the
  standard tracking tier.
- Metric: median paired ratios with 95 % intervals; weighted geomean over the cases whose
  plan an arm changes; FFTN ÷ FFTW and ÷ MKL before and after.
- Decision rule, written before running: an arm is adopted on the Xeon (`avx512`, GCC 14,
  the release toolchain) if over the cases whose plan it changes the weighted geomean
  B ÷ A is below 1 − band with its interval below 1, no such case is slower than
  1 + max(band, 3 %), and no tracking-tier case is slower than 1 + max(band, 3 %) (a case
  just past the limit on a loaded run is re-run once). Other machines keep `main`'s plans
  (checked by the profile tests); an arm that would change them needs the same rule passed
  there. A row that only `planning::measure` wins, with no rule that captures it, is
  reported with its numbers and left to measure. At most three arms.

## Results

### Step 1, the screen (counters)

Campaign directory `~/awork/fftn-x39` on the Xeon; Slurm 16151, 16152, 16153 (`-c 8`
shared, node load 1.3–2.1); `fbench-fftn run --pmu` from the cached builds of 73eb719
(task 2.4's measure, the same plans as `main` on these rows) and 304a8c5 (FFTW, MKL), user
mode, min of 2 processes per variant, cycles per point and the ratio to `main`'s plan in
the same job. Every number below repeated within ±3 % across the three jobs.

FFTW's plans (`PLAN_NOTE=1`) are recursive decimation in time with three to five passes of
radix 20, 25, 32 or 64 (`t3fv_25`, `t3fv_20`, `t1fv_64`, `n1fv_32`/`n1fv_64` leaves):
25000 = 25 · 5 · 10 · 20, 20000 = 20 · 5 · 10 · 20, 100000 = 25 · 20 · 10 · 20, 46080 =
20 · 9 · 32 · 8, 16384 = 8 · 64 · 32, 32768 = 32 · 16 · 64, 131072 = 64 · 4 · 4 · 128 with
a tiled transpose. FFTN has codelets for 2–16, 17, 19, 23, 32 and 64 only: 5⁵ needs five
passes of radix 5 or 10.

| row (`main`'s plan) | cycles per point, FFTN / FFTW / MKL | best alternative (cycles ÷ `main`) | `PLANNING=measure` chose |
|---|---|---|---|
| 5000 (chain 4 · 5 · 5 · 5 · 10) | 7.96 / 7.67 / 6.71 | chain 8 · 5 · 5 · 5 · 5 (0.91–0.92) | 4 · 10 · 5 · 5 · 5 (0.93) |
| 10000 (chain 4 · 5 · 5 · 10 · 10) | 8.19 / 8.76 / 7.23 | 8 · 5 · 10 · 5 · 5 (0.90), 8 · 10 · 5 · 5 · 5 (0.92) | 4 · 10 · 5 · 5 · 10 (0.92) |
| 16384 (chain 16 · 8 · 8 · 16) | 11.1 / 8.6 / 8.1 | 8 · 8 · 16 · 16, 16 · 32 · 32 (0.97); every grid 1.02–1.69 | 8 · 16 · 8 · 16 (0.97) |
| 20000 (chain 4 · 5 · 10 · 10 · 10) | 13.5 / 9.7 / 13.4 | 8 · 5 · 5 · 10 · 10 (0.95–0.97); grids 1.14–1.49 | 4 · 10 · 5 · 10 · 10 (0.94) |
| 25000 (chain 4 · 5⁴ · 10, six passes) | 16.1 / 10.7 / 14.7 | 8 · 5⁵ (0.92–0.98); five-pass 10 · 10 · 10 · 5 · 5 0.98; grids ≈ 1 | 4 · 5 · 10 · 5 · 5 · 5 (0.97) |
| 32768 (chain 16 · 8 · 16 · 16) | 16.4 / 11.5 / 11.6 | grid 128 × 256 in blocks of 8 (0.88–0.89; blocks of 4 0.90–0.92) | grid 128 × 256 (0.94) |
| 46080 (grid 192 × 240, blocks of 4) | 15.9 / 12.0 / 16.2 | 180 × 256 (0.95–0.97), blocks of 8 (0.96–0.97) | `main`'s |
| 65536 (grid 256 × 256, blocks of 4) | 18.7 / 17.3 / 16.6 | nothing below 0.98 | `main`'s |
| 65536 in place | 19.2 / 17.1 / 17.3 | three phases 32 × 32 × 64 with radix-32/64 phases (0.93), blocks of 8 (0.98) | `main`'s |
| 65536 × 3 in place | 56.5 / 52.3 / 51.7 | the same three phases (0.94–0.96) | `main`'s |
| 100000 (grid 200 × 500, blocks of 4) | 19.8 / 13.4 / 21.1 | 100 × 1000 (0.92–0.94: five passes against six) | `main`'s |
| 131072 (grid 32 × 64 × 64, blocks of 32) | 23.1 / 19.7 / 23.5 | radix-64 phases and blocks of 8 or 16 (0.80–0.89); blocks of 16 alone 0.89–0.91; radix 64 alone 0.91–0.92 | `main`'s |
| 262144 (grid 64³, blocks of 32) | 24.4 / 20.6 / 26.1 | blocks of 16 (0.90–0.92), radix 64 (0.91–0.93), both 0.91–0.92 | `main`'s |

The chains of 256–512 KiB miss L2 0.7–1.7 times per point (FFTW: 0.0–0.3) at an IPC near 1:
the breadth-first chain makes four or five passes over a row that, with its scratch and
twiddles, overfills the 1 MiB L2, where FFTW's recursion finishes 2048-point sub-transforms
in L1. No FFTN grid reproduces that at 16384 (the copies cost 30–50 instructions per point
against the chain's 12). Further rows (Slurm 16153): three-phase rows with a 64-long phase
gain from blocks of 16 and radix-64 phases (73728 0.92–0.93, 98304 0.90–0.91 with both);
where the three-phase block is already 16 or 8 (327680, 524288, 2²⁰, 2²²) radix 64 changes
0.99–1.05 and blocks of 8 lose (1.02–1.09); two-phase blocks of 8 for 192–256-long phases
gain 0–4 % (36864 0.96–0.98, 46080 0.97–0.98, 65536 in place 0.97, 49152 and 61440 1.00,
65536 out of place 0.99–1.01); r2c 65536 as a grid 0.88, c2r 65536 1.13 (its separate
pairing pass); float 131072 (two phases) with radix-32 phases 0.87.

So `measure` alone already takes 32768 to the grid (0.94) and 5000–25000 to better chain
orders (0.93–0.97), but never touches the grids: its candidates do not vary the block or
the phase radix.

### The arms

Chosen from the screen, each a profile field set only where a private L2 has less than 2 MiB
of L3 per CPU behind it (the Xeon 6240; the same condition as X34's `grid_two_odd_part`), so
no other recorded machine's plans change (`Schedule.ProfilesOfRecordedMachines`). Arms are
cumulative commits; each is judged by the rule above against the arm before it (A against
`main`):

- A (cc4536c): three-phase blocks of at most 16 columns (`grid_three_block`).
- B (6320ff0): grid phases of radix up to 64 (`grid_phase_radix`), X3's follow-up.
- C (13883a1): two-phase blocks within twice L1 (`grid_two_block_bytes`) and c2c powers of
  two as grids beyond a quarter of L2 (`pow2_grid_bytes`: 32768 points; real rows excluded,
  since c2r 65536 lost as a grid).

Not built: radix-20 and radix-25 codelets (a generator change, the lever for 5000–100000
with an odd part 5⁴–5⁵ and for 46080), a depth-first long chain (16384–32768), and chain
radix orders for 5000–25000 (the cost tables price no first pass of 8 above 1024 points;
`measure` finds them).

### Step 2, the arms against each other (Xeon)

Run 20261116-220608-84fe (Slurm 16154 build, 16155 timing; not `--serial`, the serial
queue held five other jobs; 16 rounds, 11 under load; A/A band ±3.31 %): the 44 standard
rows of 5000 points and up, arms N = `main` f2f59ff, A, B, C, F = FFTW, M = MKL. Per arm,
the cases whose plan differs from the arm before (the plans' notes), the median per-round
ratio and a bootstrap 95 % interval:

| arm (against) | changed cases | geomean | per case |
|---|---|---|---|
| A (N) | 4 | **0.935** | c2c 131072 0.927 [0.912, 0.969], 262144 0.959 [0.918, 1.040], r2c 262144 0.940 [0.905, 0.959], c2r 262144 0.914 [0.849, 0.954] |
| B (A) | 11 | 0.985 | 131072 0.968, 262144 0.972, r2c / c2r 262144 0.946 / 0.976, 524288 0.941 [0.856, 1.082]; 2²⁰ c2c 0.990–0.999, r2c 0.993–1.012, c2r **1.018–1.027** |
| C (B) | 7 | **0.958** | c2c 32768 0.895 [0.872, 0.903], 46080 0.942 [0.918, 0.965], 65536 in place 0.970, × 3 in place 0.964, × 3 0.969, backward 0.971, out of place 1.002 [0.967, 1.043] |

B changes no two-phase row and C no three-phase row, so C's ratios against B are its ratios
against A on the same plans. Tracking tier, `main` against A + B + C (run
20261116-220608-265f, Slurm 16156; 24 rounds, 13 under load; band ±2.23 %): 7 changed
cases, geomean 0.966 (262144 0.895 [0.847, 0.914], 46080 0.961, r2c 2²⁰ 0.936, c2c 2²⁰ 0.976,
65536 in place 0.984, 65536 1.015 [0.994, 1.025], c2r 2²⁰ 1.000); 61 cases, 2 faster, 59 no
evidence, none above 1.028 (c2r 1024², a plane whose plan does not change); weighted
geomean 0.992 [0.982, 1.001].

FFTN ÷ FFTW (÷ MKL) of the exit check's failing rows, `main` → the adopted plans (A for the
three-phase rows, C for the two-phase rows), same run: 32768 1.51 → **1.29** (1.26 → 1.10),
46080 1.33 → 1.26 (1.02 → 0.95), 131072 1.20 → **1.09** (0.98 → 0.90), 262144 1.26 → 1.23
(0.88 → 0.88; 1.18 with B), 65536 in place 1.13 → 1.09 (1.13 → 1.12), × 3 in place 1.13 →
1.10 (1.20 → 1.12); unchanged: 25000 1.45 (1.10), 100000 1.45 (0.93), 20000 1.35 (0.93),
16384 1.23 (1.23), 5000 1.05 (1.17), 10000 0.95 (1.12). Also r2c 262144 1.14 → 1.07, c2r
262144 1.34 → 1.22.

### Confirmation, the adopted branch against `main` (Xeon)

Run 20261116-225143-281a (Slurm 16165 build, 16166 timing; 20 rounds, all under load; band
±3.01 %), arms N = `main` f2f59ff, X = f417f7b (arms A and C, B removed), F, M; the 32
standard rows of 16384–2²⁰ points, 25000 and 100000. The 11 changed cases: geomean
**0.958**, none above 1.009 (c2c 32768 0.913 [0.881, 0.927], 131072 0.914 [0.879, 0.940],
262144 0.926 [0.903, 0.956], r2c / c2r 262144 0.922 / 0.920, 65536 × 3 in place 0.965,
46080 0.979 [0.960, 1.007], × 3 out of place 0.990, 65536 in and out of place 1.002, backward
1.009 [0.984, 1.041]); 21 unchanged cases no evidence. FFTN ÷ FFTW, N → X: 32768 1.37 →
1.29, 131072 1.14 → 1.03, 262144 1.32 → 1.19, 46080 1.16 → 1.14, 65536 × 3 in place 1.12 →
1.09, 65536 in place 1.12 → 1.13; ÷ MKL: 32768 1.21 → 1.10, 131072 0.99 → 0.88, 262144 0.95
→ 0.86, 65536 × 3 in place 1.17 → 1.12. (FFTW's MEASURE plans vary by run, X25: 46080 was
1.33 of FFTW in run -84fe and 1.16 here with the same FFTN plan.)

### Arm D: radix-20 passes (after X38), screen

X38 added a radix-20 pass codelet on every target (its tables price no radix 20; overrides
choose it). Counters as in step 1, from X38's head 94b37e9 (Slurm 16180 build, 16183 and
16184 screens), cycles and ns per point ÷ the rules' chain (min of 2–3 processes):

| length (rules) | best sequence | ratio | other radix-20 orders |
|---|---|---|---|
| 300 (4 · 5 · 15) | 20 · 15 | 0.87–0.89 | 15 · 20 0.90–0.94 |
| 500 (4 · 5 · 5 · 5) | 20 · 5 · 5 | 0.92 | 5 · 5 · 20 1.15–1.22 |
| 1000 (4 · 10 · 5 · 5) | 20 · 10 · 5 | 0.89 | 20 · 5 · 10 0.94–0.95 |
| 2000 (4 · 5 · 10 · 10) | 20 · 10 · 10 | 0.82 | 10 · 10 · 20 1.04 |
| 3000 (4 · 5 · 10 · 15) | 20 · 10 · 15 | 0.86–0.87 | 10 · 15 · 20 1.09 |
| 3600 (4 · 6 · 10 · 15) | 12 · 15 · 20 | 0.85–0.86 | 20 · 15 · 12 0.87 |
| 4000 (4 · 10 · 10 · 10) | 20 · 20 · 10 | 0.84 | 10 · 20 · 20 1.04 |
| 4800 (4 · 10 · 12 · 10) | 20 · 12 · 20 | 0.87–0.88 | 20 · 20 · 12 0.88–0.90 |
| 5000 (4 · 5 · 5 · 5 · 10) | 20 · 10 · 5 · 5 | 0.83–0.84 | 8 · 5⁴ 0.92; 10 · 20 · 5 · 5 0.95–0.96 |
| 6000 (4 · 10 · 10 · 15) | 20 · 15 · 20 | 0.875–0.88 | 20 · 20 · 15 0.89–0.905 |
| 8000 (8 · 10 · 10 · 10) | 20 · 20 · 20 | 0.885–0.90 | |
| 10000 (4 · 5 · 5 · 10 · 10) | 20 · 20 · 5 · 5 | 0.85 | 20 · 10 · 10 · 5 0.855 |
| 20000 (4 · 5 · 10 · 10 · 10) | 20 · 20 · 5 · 10 | 0.915–0.92 | 20 · 10 · 10 · 10 0.86 cycles, 0.94 ns |
| 25000 (4 · 5⁴ · 10) | 20 · 10 · 5 · 5 · 5 | 0.88 ns (0.99 cycles; 17.1 against 23.8 instructions per point) | 20 · 5 · 5 · 5 · 10 0.90 ns |
| 100, 1200, 2400, 7200, 12000 | none below 0.95 | 1.00–1.07 | |
| 40000 (grid 200 × 200) | phases 10 · 20 (`GRID_PHASE_RADIX=20`) | 0.82–0.83 | |
| 100000 (grid 200 × 500) | phase 200 as 10 · 20 | 0.92 | chains with radix 20 1.27–1.35 |
| 46080, 10⁶ (grids) | phase radix 20 changes no phase | 0.99–1.02 | |

A radix-20 pass first and the larger radices next won wherever the length has 2² · 5 to
spare; a radix 20 last lost (1.04–1.22). Arm D: these sequences as `avx512` double overrides
(the 14 lengths above, 1000 replacing its X2 override) and grid phases of radix up to 20 on
the Xeon profile (`grid_phase_radix` = 20, which adds only radix 20: phase lengths are
13-smooth). Radix-25 proxy: 25000's best plan takes five passes at 0.79 ns per point per
pass; FFTW's four (25 · 5 · 10 · 20) at 3.08 ns per point suggest a radix-25 pass
(25 · 10 · 10 · 10) would take 25000 from 1.28 to about 1.05 of FFTW, and 100000's phase of
500 from three passes to two (about 0.85 of arm D's time); not measured.

### Arm D, judged (Xeon, `--serial`)

First form 76019fa (all 14 overrides, 1000 as 20 · 10 · 5) against C = d50585e (A + C with
X36 and X38), run 20261117-011743-1a01 (Slurm 16192 build, 16193 timing; 20 quiet rounds,
band ±1.71 %), the standard rows of the overridden lengths: 28 changed cases, geomean 0.954;
c2c 300 0.856, 500 0.905, 1000 0.904, 2000 0.828, 3000 0.862, 3600 0.854, 4800 0.880, 5000
0.843, 6000 0.864, 10000 0.847, 20000 0.913, 25000 0.899, 100000 0.916, r2c 10000 0.878;
but c2r 1000 and 10000 **1.19–1.25** (the c2r pairing fuses into the first pass, so the rule
moved a radix 5 in front of the radix 20: 5 · 20 · 5, 5 · 20 · 10 · 5) and planes and
batches of 1000 1.02–1.04. It fails the rule. Refined form 3cb6b32: 1000 keeps X2's
override, and a c2r line whose override starts with radix 20 takes the costs' sequence
(`chain_builder::real_radices`; avx2's X38 overrides do not start with 20 where a small
radix could move). Its tracking tier (run 20261117-012733-b1e1, Slurm 16194; 20 rounds, band ±1.45 %) had the
same picture: 1000 0.909, 100000 0.922, r2c 1000 0.949, c2r 1000 **1.235**, nothing else
changed.

The final form 274f870 also takes two of task 2.4's P1 misses on the Xeon
(`P1-rules-vs-measure.md`), which fit the same profile fields; screens Slurm 16208 and
16211 (cycles and ns ÷ the rules' plan):

- r2c rows' half-length powers of two follow `pow2_grid_bytes` (a grid beyond L2/4): r2c
  65536 as a two-phase grid 0.89 out of place, 0.96–0.99 in place. c2r stays a chain (1.13 out
  of place, 0.94–0.96 in place); `plan_grid` now takes the plan's kind and placement.
- c2c powers of two out of place stay in two phases beyond half of L3
  (`grid_two_pow2_bytes`): 2²⁰ 0.89–0.98, 2²¹ 0.91, 2²² 0.88–0.92 (P1: measure 1.14 faster),
  2²³ 0.92–0.94; 2¹⁹ 0.95 cycles / 1.05 ns, 2¹⁸ 1.03–1.21; in place 2²⁰ 1.10–1.22 and c2r 2²⁰
  1.20–1.23, so neither takes it (r2c 2²⁰ 0.97–0.985, left out).

Final form 274f870 against C, run 20261117-033154-673d (Slurm 16213 build, 16214 timing,
`--serial`; 16 quiet rounds, band ±1.70 %): 26 changed cases, geomean **0.905**, none above
1.010 (r2c 1000² 1.002–1.010). c2c 300 0.867, 500 0.914, 2000 0.824, 3000 0.863, 3600 0.854,
4800 0.882, 5000 0.849, 6000 0.866, 10000 0.867, 20000 0.919, 25000 0.903, 100000 0.908;
two phases 2²⁰ 0.874, 2²¹ 0.903, 2²² 0.870, 2²³ 0.923, 2²⁴ 0.952; r2c 65536 0.900 out of
place, 0.968–0.985 in place; r2c 1000 0.940, 10000 0.876. c2r rows and 1000 no longer change.
FFTN ÷ FFTW (÷ MKL), C → D: 5000 1.04 → 0.89 (1.17 → 0.99), 10000 0.96 → 0.84 (1.13 →
0.96), 20000 1.40 → 1.28, 25000 1.50 → 1.34 (1.12 → 1.00), 100000 1.36 → 1.24, 2²² 0.86 →
0.75, r2c 65536 1.42 → 1.27. Tracking tier (run 20261117-042017-32c8, Slurm 16215, `--serial`; 16 rounds, 5 under load, band ±2.75 %): 6 changed cases 0.912 (100000 0.911, 2²⁰ 0.880, 2²² 0.864, r2c 65536 0.902 and 0.977, r2c 1000 0.941), none of 61 above 1.018, weighted geomean 0.979 [0.954, 0.992].

## Verdict

- **Diagnosis.** The failing rows up to 32768 points are chains on the Xeon, not grids.
  Two causes keep them 1.2–1.5 times FFTW's time.
  - The breadth-first chain misses L2 at 256–512 KiB (0.7–1.7 misses per point, IPC near 1),
    where FFTW's recursion finishes sub-transforms in L1 (LESSONS 3.22).
  - Rows with an odd part 5⁴–5⁵ need five or six passes without radix-20 and radix-25
    codelets. FFTW uses three or four (`t3fv_25`, `t3fv_20`).

  The grids (46080–262144) pay for copies in narrow blocks and for blocks wider than their
  phases want.
- **Arm A, three-phase blocks of at most 16 columns: adopted** on the Xeon profile (x86-64,
  a private L2 with less than 2 MiB of L3 per CPU). Over its 4 changed rows it took 0.935 of
  `main`'s time, below 1 − 3.31 %, and none was slower.
- **Arm B, radix-32 and radix-64 grid phases: rejected and removed** (f417f7b).
  - Over its 11 changed rows it took 0.985 of arm A's time, not below the band.
  - It gained 3–5 % at 131072–262144 and lost 1.8–2.7 % on c2r 2²⁰.
  - X3's open follow-up is now measured; a size-limited form would be a fourth arm.
- **Arm C, two-phase blocks within twice L1 and c2c powers of two as grids beyond L2/4:
  adopted**, at 0.958 over its 7 changed rows (32768 0.895, 46080 0.942, at most 1.002).
- **Collateral.** Tracking tier: none slower than 1.028, which was a plan that did not
  change. Other machines are unchanged: the fields are set only on that profile, on x86-64
  only (the Windows arm64 CI runner's caches matched the condition), and the
  `Schedule.ProfilesOfRecordedMachines` test now checks every new field.
- **Confirmation.** The adopted branch (A + C) against `main`: 0.958 over its 11 changed
  cases, none above 1.009 (run 20261116-225143-281a).
- **Arm D, radix-20 passes and two of P1's misses: adopted** in its final form (274f870):
  0.905 over its 26 changed cases, none above 1.010 (run 20261117-033154-673d, `--serial`).
  It has four parts:
  - `avx512` double overrides with a radix-20 first pass for 300–25000;
  - radix-20 grid phases on the Xeon profile;
  - r2c half-lengths as grids beyond L2/4;
  - c2c powers of two out of place in two phases beyond L3/2.

  The first form lost on c2r rows (1.19–1.25) and planes of 1000 (1.02–1.04); both are fixed.
  The overrides apply to every `avx512` machine (Zen 5 and Granite Rapids too, unmeasured).
- **What stays open, with numbers.**
  - After arm D, 25000 (1.34 of FFTW), 20000 (1.28) and 100000 (1.24) still trail: a radix-25
    pass (FFTW's `t3fv_25`) would take 25000 to about 1.05 by the pass-count proxy above;
    46080 (1.26) is unchanged.
  - 100000 also gains 0.92–0.94 as 100 × 1000 (five passes against six), a factor-choice
    rule that X39 did not take.
  - 16384 (1.23 of FFTW and of MKL) needs a depth-first long chain: no grid or radix order
    gained more than 3 %.
  - 5000 and 10000 (1.17 and 1.12 of MKL): chains with a first radix of 8 take 0.90–0.92 of
    the rules' cycles. `PLANNING=measure` already finds such orders (0.92–0.93), as it does
    for 20000 and 25000 (0.94–0.97). The cost tables price no radix-8 first pass above 1024
    points, so these rows are left to measure.
  - In-place 65536: three phases with radix-32/64 phases took 0.93 of the cycles in place
    but 0.99–1.04 out of place. No rule was tried; candidates for measure.

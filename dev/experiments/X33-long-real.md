# X33: long real rows (r2c and c2r of 65536 and 2²⁰)

- **Status:** running
- **Owner:** x33 session, 2026-11-16
- **Phase / task:** 1–2 (PLAN lead (b); follows X6, X25, X29)

## Question

Where does FFTN's time go on long real rows (r2c and c2r of 65536 and 2²⁰, double, one
thread), and does a pairing that costs no sweep of its own over memory close the gap to
FFTW? The answer decides how the real rows' pairing runs beside the grid and the chain.

## Hypothesis

FFTN ÷ FFTW on these rows (runs 20261116-103657-af06, -105105-30b3, -110608-e327): Mac
1.20–1.27, EPYC 1.17–1.44 (r2c 2²⁰ 1.43, c2r 2²⁰ 1.33, c2r 65536 1.29), Xeon 1.08–1.42
(r2c and c2r 65536 1.40–1.43). X29 measured the real overhead of the grid route,
time(real n) ÷ time(c2c n/2) − 1, at 65536–2²⁰: r2c 0.07–0.13 (Mac), 0.10–0.19 (EPYC),
0.14–0.26 (Xeon); c2r out of place 0.16–0.42 (before X29's paired gather on Apple). The
grid runs the half-length transform and then (r2c) or before it (c2r) a separate pairing
pass in place over the h + 1 bins: one more read and write of the whole half spectrum,
from L2 at 65536 and from L3 or memory at 2²⁰. Expected: that sweep is most of the real
overhead on the x86 machines, and fusing the pairing into the grid's last phase (r2c),
where each block's values are in L1 anyway, removes most of it (5–15 % of the row's time),
less on the Mac (shared L2, X6's real overhead beyond L1 2–16 %). The rest of the gap is
the half-length complex transform itself (lead (a), X32).

## Change under test

Step 1 (screen, no library change) decides which arms are built; at most three:

- Arm (b1), the r2c pairing fused into the grid's last phase with **block pairs**: the
  last phase runs over pairs of column sets, one iteration per pair, a low set and the
  set of its mirror columns L − l, both transformed into scratch, paired there (bins K and
  h − K), then scattered; column 0 (bins L k, which pair among themselves, and X_0, X_h)
  in a small stage of its own. No separate pairing pass. The program gets a loop that
  moves a second cursor into the output in the opposite direction (driver change).
- Arm (b2), the Xeon's chain: the r2c pairing fused into the half-length chain's last
  pass beyond L1 (today only within L1, and never after a split run).
- Arm (b3), c2r: the pairing in the grid's first gather where it still is a separate pass
  (in place, three phases, the x86 machines).

Results change bits for the rows whose plan changes (the pairing's arithmetic is the same
per pair but some bins are formed from the mirror's side); the tests' derived tolerances
apply (the grid suites force the route).

## Method

- Cases: `bench/catalogues/x33.toml`: r2c and c2r of 32768, 65536, 131072, 262144 and
  2²⁰, in and out of place, and the c2c rows of half those lengths, out of place; one
  transform, double, hot. Collateral: the standard tracking tier.
- Machines: EPYC 7763 (`avx2`, GCC 16.2, FFTW 3.3.11 with AVX2, DMOG `-c 8` shared), Xeon
  Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512), Apple M5 Max (`neon_fcma`,
  Homebrew Clang, the NEON FFTW, shared). fbench interleaved rounds with the A/A control,
  `--serial` on the Slurm machines; `PLAN_NOTE=1` for FFTW's plans; FFTL.jl on the EPYC
  and the Mac.
- Step 1, screen: arms N = `main`, C = `+STRATEGY=chain`, G = `+STRATEGY=grid`, F =
  FFTW. Per real case: FFTN's real overhead r = time(real n) ÷ time(c2c n/2) − 1 under N
  and G, FFTW's under F (with its c2c n/2), and the share of FFTN's gap to FFTW that the
  half-length transform explains (N(c2c n/2) ÷ F(c2c n/2)) against the share the real
  overhead explains ((1 + r_N) ÷ (1 + r_F)).
- Metric: median paired ratios with 95 % intervals; weighted geomeans over the cases whose
  plan an arm changes; the A/A band.
- Decision rules, written before running:
  1. An arm is built only if the screen shows the real overhead of the route it changes at
     least 0.10 on the target rows of at least one machine-target where FFTN trails FFTW
     there, and FFTW's own real overhead at least 0.05 smaller (so a fused pairing can
     plausibly gain 5 % or more).
  2. An arm is adopted on a machine-target if over the target rows whose plan it changes
     the weighted geomean B ÷ A is below 1 − band with its interval below 1, no such row is
     slower than 1 + max(band, 3 %), and no tracking-tier case is slower than
     1 + max(band, 3 %) (re-run once if a case is just past it on a loaded machine).
     Release toolchains decide (GCC on x86 Linux, Clang on macOS); a win on one target and
     a loss on another is adopted per target.
  3. The plans judged are those `main` picks at the time of the confirmation run (X32 may
     change whether long half-length rows are chains or grids): `origin/main` is merged
     before each timing round.

## Results

Ratios are medians of paired per-round ratios. In the screen tables, times are medians in
µs; r = time(real n) ÷ time(c2c n/2) − 1 per arm; "half" = N ÷ F on c2c n/2 and "real" =
(1 + r_N) ÷ (1 + r_F), so that N ÷ F on the real row = half × real.

### Step 1, the screen (main 3d8b129)

Runs 20261116-114247-1dff (Mac, 24 rounds, load 9.7 median, band ±1.24 %),
20261116-114605-838e (EPYC, 30 rounds, none under load, band ±1.00 %, FFTL as well),
20261116-115726-0dd9 (Xeon, 30 rounds, 15 under load, band ±2.29 %).

| row | Mac N÷F | r_N | r_F | half | real | EPYC N÷F | r_N | r_F | half | real | Xeon N÷F | r_N | r_F | half | real |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2c 65536 out | 1.17 | 0.10 | 0.08 | 1.15 | 1.01 | 1.16 | 0.13 | 0.08 | 1.11 | 1.04 | 1.35 | 0.17 | 0.32 | 1.52 | 0.89 |
| r2c 65536 in | 1.23 | 0.10 | 0.03 | 1.15 | 1.07 | 1.19 | 0.17 | 0.09 | 1.11 | 1.07 | 1.39 | 0.24 | 0.35 | 1.52 | 0.91 |
| r2c 2²⁰ out | 1.38 | 0.09 | −0.14 | 1.08 | 1.28 | 1.34 | 0.21 | −0.24 | 0.84 | 1.59 | 1.46 | 0.31 | −0.24 | 0.86 | 1.71 |
| r2c 2²⁰ in | 1.26 | −0.01 | −0.15 | 1.08 | 1.16 | 1.24 | 0.18 | −0.20 | 0.84 | 1.47 | 1.13 | 0.03 | −0.22 | 0.86 | 1.32 |
| c2r 65536 out | 1.25 | 0.17 | 0.07 | 1.15 | 1.09 | 1.30 | 0.23 | 0.05 | 1.11 | 1.17 | 1.45 | 0.16 | 0.22 | 1.52 | 0.96 |
| c2r 65536 in | 1.21 | 0.12 | 0.06 | 1.15 | 1.05 | 1.23 | 0.17 | 0.06 | 1.11 | 1.10 | 1.23 | 0.12 | 0.38 | 1.52 | 0.81 |
| c2r 2²⁰ out | 1.35 | 0.08 | −0.13 | 1.08 | 1.25 | 1.35 | 0.26 | −0.22 | 0.84 | 1.61 | 1.18 | 0.23 | −0.11 | 0.86 | 1.38 |
| c2r 2²⁰ in | 1.07 | −0.02 | −0.01 | 1.08 | 0.98 | 0.89 | 0.16 | 0.10 | 0.84 | 1.05 | 0.86 | 0.02 | 0.01 | 0.86 | 1.01 |

Where the time goes:

- **65536**: mostly the half-length complex transform. Mac and EPYC run c2c 32768 at
  1.11–1.15 of FFTW, the Xeon at 1.52 (its chain of 32768, 512 KiB, X25's memory-bound
  case; as a grid it is no faster, 166 against 165 µs). FFTN's real overhead is 0.10–0.24,
  FFTW's 0.03–0.38 (the Xeon's FFTW pays more for its real rows than FFTN). On the Xeon the
  grid route is the better r2c 65536 (G ÷ N 0.90 out of place, 0.89 in place: r_G 0.05–0.10)
  although the c2c half is not: the chain's real row is what is slow, not its half (X29 arm
  a measured 0.94–0.98 for this; the arm b2 screen below repeats it with b1).
- **2²⁰**: FFTW's real rows cost **less** than its c2c of half their length (r_F −0.11 to
  −0.24 out of place on all three machines: its r2c plans are rdft2 radix-2–16 DIT/DIF steps
  over sub-transforms of n/4–n/16 points, `PLAN_NOTE=1`), while FFTN's half-length grid of
  2¹⁹ beats FFTW's c2c 2¹⁹ on x86 (0.84–0.86). The gap at 2²⁰ is the real side: FFTN's
  separate pairing pass plus an r2c grid no better than its c2c, against an FFTW real plan
  whose sub-transforms fit the cache better than its own c2c 2¹⁹.
- **The chain on the EPYC** (arm C) took 0.64–0.69 of main's time on every real row at
  2²⁰, 0.70–0.80 at 65536–262144 (except in place at 262144, 1.02–1.05) and c2c 2¹⁹ 0.66: X29's finding (steady beyond 2 MiB,
  bimodal at 1–2 MiB) and FFTL's lead there (FFTL 0.56–0.62 of FFTN at 2²⁰) are the
  half-length chain-or-grid rule, lead (a)'s (X32), not the real rows'.
- In place against out of place at 2²⁰ on the Xeon: 5754 against 7295 µs (r2c), because
  the out-of-place three-phase grid carries the input, the output and the workspace block
  (24 MiB, the whole shared L3).

Decision rule 1: the grid route's r2c overhead is ≥ 0.10 with FFTW's ≥ 0.05 smaller on at
least one machine-target (EPYC 65536–2²⁰, Xeon 2²⁰, Mac 65536 in place and 2²⁰), so arm
b1 was built. The c2r side (arm b3) also qualifies (EPYC 0.16–0.26 against −0.22–0.10).

### Arm b1: the r2c pairing in the grid's last phase

Commits 25f724e (first form), bcf5bf7 (a fused last phase, radices ≤ 8, keeps the pass:
the paired phase must gather and scatter, and lost 3.5 % at the Mac's 262144), 56252ed
(sets at least a cache line wide) and b82b77d (sets of exactly half the block, at least a
register: the final form). Each step came from a measured loss:

- First form, sets of max(C, b/2) columns. Mac (run 20261116-115601-e28a, A = 3d8b129,
  band ±1.37 %): r2c 65536–131072 0.96–0.97, 2²⁰ 0.96–0.97, 262144 1.035–1.037 (its
  last phase 512 = 8 × 8 × 8 is fused in main). EPYC (20261116-122106-d5e7, A = main
  29f9772, 30 rounds all under load, band ±1.00 %): r2c 65536 **1.051–1.073** (blocks of
  4: sets of 2 complex doubles, so each output line is written by two or three iterations
  far apart in time), 131072 0.96–0.99, 262144 0.97–0.99, 2²⁰ 1.003–1.012; c2r and c2c
  unchanged (0.99–1.03, no evidence). Xeon (20261116-123753-9989, A = 29f9772, no round
  under load, band ±1.01 %): r2c 262144 0.941–0.953, 2²⁰ 0.953–0.967, 131072
  **1.080–1.090** (blocks of 4 with C = 4: sets of 4 columns, twice the block's scratch, the
  whole of L1).
- Final form (b82b77d; on the Mac and the EPYC its plans are 56252ed's). Mac, r2c rows
  (20261116-123913-0fa6, 40 rounds, load 17.6 median, band ±1.52 %): 32768 0.977–0.995,
  65536 0.966–0.976, 131072 0.941–0.989, 2²⁰ 0.971–0.972; 262144 unchanged (1.008–1.011);
  geomean of the eight changed rows 0.973. Mac tracking tier (20261116-124942-42d5, band
  ±1.00 %): geomean 0.999, r2c 65536 0.953–0.966, 2²⁰ 0.951; slowest r2c 2048 × 36 1.047
  [1.015, 1.074], whose plan is identical in both arms (as in the first form's tracking run,
  20261116-122100-c121: 1.044); nothing else above 1.02.

### Arm b2: the Xeon's r2c 65536 as a grid (screen only)

Runs 20261116-125342-8af5 (Xeon, band ±1.00 %) and 20261116-124535-cd41 (EPYC), arm G =
bcf5bf7 forced to the grid: Xeon r2c 65536 0.917 out of place, 0.999 in place (with the
first form's over-wide sets); r2c 32768 1.04–1.13. EPYC 32768 1.20, 65536 1.06–1.07. A
rule sending r2c rows to the grid earlier on the Xeon alone would gain one case by 8 % and
needs a profile property that separates it from the EPYC (whose r2c 32768 must stay a
chain); X29's arm (a) found c2r losing there (1.153). Not built.

## Verdict

(pending)

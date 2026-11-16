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

(pending)

## Verdict

(pending)

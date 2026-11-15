# X29: real rows in the chain-or-grid rule, and the grid's pairing

- **Status:** running
- **Owner:** x29 session, 2026-11-15
- **Phase / task:** 1–2 (PLAN lead 5; follows task 2.1, X6, X12b, X25)

## Question

Should a real row (r2c, c2r) switch from the chain to the grid at a different size than the
c2c row of its half length, and is the separate pairing pass after (r2c) or before (c2r) the
grid worth fusing into the grid's last (first) phase? The answer sets the real rows'
chain-or-grid rule (`plan_grid` in `src/core/plan_core.cpp`) and decides whether lead 5
needs grid work.

## Hypothesis

FFTN's real rows lose most at the long lengths (FFTN ÷ FFTW, tracking tier: r2c and c2r of
65536 and 2²⁰ 1.41–1.62 on the EPYC, 1.36–1.46 on the Xeon, 1.30–1.39 on the Mac; runs
20261115-101319-f98f, -114105-ea43, -094953-38c5). X25 found r2c 65536 on the Xeon bound by
memory: its half-length chain of 32768 points (512 KiB, exactly the Xeon's `grid_min_bytes`
of L2/2, so a chain) runs at IPC 0.84 with 3.4 times FFTW's L2 misses. The rule sizes the
half-length complex transform as if it were a c2c row and ignores what the real row adds:
the pairing pass reads both ends of the half spectrum (two streams) after the chain has
left it in L2 or beyond, and the in-place real grid (and every c2r grid, which runs in place
on the output) carries the workspace block as well. Expected: on the x86 machines, whose
private L2 lets a chain stream well up to L2/2 (X12b), real rows near that limit (r2c and
c2r 32768–65536) are faster as grids, i.e. the switch should come at a smaller half length
for real rows; on the Mac (grid beyond L1 already) nothing moves. The pairing pass at
2²⁰ is one more pass over memory on top of the grid's three, so fusing it into the grid's
last phase could save up to a quarter of the time there; X6 measured the Mac's real
overhead beyond L1 at 2–16 %, which bounds what fusion can gain on the Mac.

## Change under test

Arm (a), the rule: a real row is a grid when k × (its half-length complex bytes) exceed
`grid_min_bytes` (main: k = 1), with k from the screen below; nothing else changes (the
grid's split, the pairing's route and fusion limit stay). Arm (b), only if the screen says
it can pay (decision rule 3): the pairing fused into the r2c grid's last phase and the
c2r grid's first phase (blocks of columns l and L − l together), no separate pass.

Results change bits for the real rows whose plan changes (a different factorisation); both
routes are tested against the derived tolerances (the grid and chain suites force each).

## Method

- Cases: `bench/catalogues/x29.toml`: r2c and c2r rows of 4096, 8192, 16384, 32768, 65536,
  131072, 2²⁰, 1000 and 46080 points, in and out of place, and the c2c rows of half those
  lengths (out of place), one transform, double, hot. Collateral: the standard tracking tier.
- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512, Slurm `-c 8`
  shared); EPYC 7763 (`avx2`, GCC 16.2, FFTW 3.3.11 with AVX2, DMOG `-c 8` shared); Apple
  M5 Max (`neon_fcma`, Homebrew Clang, the NEON FFTW, shared). fbench interleaved rounds
  with the A/A control, `--serial` on the Slurm machines.
- Step 1, screen: arms A = `fftn` (main, the rules), C = `fftn+STRATEGY=chain`, G =
  `fftn+STRATEGY=grid`, F = `fftw`. Per real case, C ÷ G says which side of the switch the
  row belongs on; time(real n) ÷ time(c2c n/2) − 1 under G is the real overhead of the
  grid route (the pairing pass and the reals' reading).
- Step 2, arm (a): B = the rule's commit against A = main, x29 catalogue and the tracking
  tier, on each machine.
- Metric: median paired ratios with 95 % intervals; weighted geomeans over the cases whose
  plan an arm changes; the A/A band.
- Decision rules, written before running:
  1. k is the value of {1, 2, 4} (the switch at L2/2, L2/4 and L2/8 of half-length bytes on
     x86) whose implied choice of C or G per real case has the lowest geomean ÷ A in the
     screen, per machine; if the best k differs between machines with a private L2 and the
     Mac, k is set per that profile property (as X12b's thresholds are).
  2. Arm (a) is adopted on a machine-target if over the real cases whose plan it changes the
     weighted geomean B ÷ A is below 1 − band with its interval below 1, no such case is
     slower than 1 + max(band, 3 %), and no tracking case is slower than 1 + max(band, 3 %)
     (re-run once if a case is just past it on a loaded machine). A win on one target and a
     loss on another is adopted per target; release toolchains decide (GCC on x86 Linux,
     Clang on macOS).
  3. Arm (b) is built only if the grid route's real overhead (step 1) is at least 15 % at
     65536–2²⁰ on at least two machines, so that fusion could plausibly gain 7 % or more;
     otherwise the file records why not. If built, it is judged by rule 2 over the grid's
     real rows.

### Arms added after the screen (written before their confirmation runs)

The screen's chain arm beat the grid far outside the real rows' question: on the Mac for
every 46080-point row (0.62–0.67 of the grid's time), and on the EPYC for every long row,
c2c included (0.53–0.74, to 2¹⁹ points). The rule's thresholds were set on powers of two
(X12, X12b) and never tried chains beyond L2 on x86. Two arms follow, each a rule:

- Arm (c), shared L2 (Apple): a row whose length is not a power of two stays a chain when
  the power-of-two part of its length, in bytes, is at most L1/8, or the row is at most
  2 · L1 (`chain_part_bytes`, `chain_mixed_bytes`; fbench `CHAIN_PART_BYTES`,
  `CHAIN_MIXED_BYTES`). From a Mac screen of c2c rows 2^k × odd (3 to 3125) and large rows.
- Arm (d), private L2 (x86): `grid_min_bytes` raised from L2/2 to the value the long-row
  screens (c2c and real rows to 4 Mi points, chain against grid, EPYC and Xeon) support.
- Decision rule for (c) and (d): rule 2 above over the rows whose plan the arm changes, on
  the x29 catalogue and the standard tracking tier, per machine-target.

## Results

(pending)

## Verdict

(pending)

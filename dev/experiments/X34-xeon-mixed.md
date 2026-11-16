# X34: long rows of mixed length on the Xeon

- **Status:** running
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

(pending)

## Verdict

(pending)

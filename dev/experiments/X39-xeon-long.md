# X39: the Xeon's long rows and in-place 65536 (exit-check causes 5 and 6)

- **Status:** running
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

## Verdict

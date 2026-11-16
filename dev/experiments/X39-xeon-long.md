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

## Verdict

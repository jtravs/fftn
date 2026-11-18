# X50: large primes against MKL on the Xeon (Bluestein on the depth-first chain)

- **Status:** running
- **Owner:** x50 session, 2026-11-18
- **Phase / task:** 1 (phase 1 exit check, the tail rule; cause 1, the primes); follows X36,
  X36b, P1b and X46

## Question

Do the Xeon 6240's large primes, 999983 (1.43 of MKL's time, 1.12 of FFTW's) and 131071
(1.33, 1.20), close when a Bluestein stage's convolution runs as two depth-first chains
(X46) instead of two breadth-first chains, and does the EPYC's 65537 (Rader against
Bluestein, bimodal by process in P1b) need a page-robust Rader stage? The answer adds a
form of the Bluestein stage and a rule for when it is taken.

## Hypothesis

- The Bluestein stage's cost is its two transforms of M points (X36: the chirp passes are
  about two pass-equivalents of ten). On the Xeon the convolution is always a breadth-first
  chain (X36b: a grid convolution needs a second movable workspace slot), and a chain of
  2¹⁸ points took 7.4 ns per point against the depth-first chain's 4.4 (X36b's run
  20261117-002122-58ba; X46's 262144 row, 0.685 of its grid). 131071 (M = 2¹⁸, 4 MiB, the
  top of X46's range) should take about 0.65 of its time, under MKL's.
- The depth-first chain fits the program without a new slot: the chirp pass writes scratch
  a; the forward chain's first pass reads a and writes its sub-row blocks into the workspace
  block (movable, so a loop runs the sub-rows); its top pass writes back into a, times the
  kernel spectrum (a factor edge on the top pass); the backward chain does the same from a
  into the block, its top pass in place; the chirp pass out reads the block. Scratch a is
  dead after each first pass, so the sub-rows use it.
- 999983 (M about 2²¹, 32 MiB, beyond X46's range) needs two top factors (sub-rows of about
  8100 points); X46 found two tops no better than one up to 2¹⁸, but there the sub-rows of
  one top fit L2. Unknown until screened.
- EPYC 65537: P1b's three runs disagree (Bluestein 0.57 to 1.49 of Rader's time); Rader's
  gather and scatter visit 256 pages of 4 KiB at random. Left for a counters screen after the
  Xeon (dTLB misses per point in the edge passes).

## Change under test

Arms are commits of `exp/x50-primes` against N = its base (X46's head 80dbcdb, the plans of
`main` after pull request 59):

- **A0 (the stage, forced only).** `chain_builder::bluestein_depth` and the factor edge of
  the depth-first top pass (`depth_top[d][2]`); with `STRATEGY=depth` (and `DEPTH_TOPS`) a
  c2c row of one axis whose chain is one Bluestein stage of one copy runs its convolution as
  two depth-first chains. No plan changes by the rules.
- **A (the rule).** The rules take it where a row of the convolution's length M would be a
  depth-first chain (the profile's depth-first fields on M's bytes, X46's conditions): on
  the Xeon M of 16384–262144 points (8179, 32749, 131071), on the Mac M of 8192–16384 points
  with a power-of-two part above 1024 points; nothing on the EPYC.
- **B (beyond the range; named after the screen, before the judging run).** The screen
  (below) found one top of 16 better than two even at 2²¹ points, so B is: a profile field
  `depth_convolution_max_bytes` extends the range for convolutions only, with the same tops;
  the Xeon profile sets no limit (262139, 524287, 999983 change). Other profiles keep A's
  range.

Bits: a changed plan changes the rounding of its rows (another factorisation of the
convolution), within the derived tolerances; tests hold both precisions to them and the
threaded runs bit for bit to the serial ones.

## Method

- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512, oneMKL 2026.1;
  Slurm `-c 8` shared; judging through `fbench_remote.py submit --serial`), then the EPYC
  7763 (`avx2`, GCC 16, DMOG, collateral only: no plan changes) and the M5 Max
  (`neon_fcma`, Homebrew Clang 23, NEON FFTW) on the cases whose plans change there.
- Screen (no default change): the Xeon's c2c 131071, 999983, 65537 and the convolution
  lengths 262144, 2000000, 2073600, 2097152 under the rules (N), the chain (C), the grid (G),
  depth-first with one top of 16 (D) and two (E), FFTW and MKL.
- Judging: catalogue `x36`, tier `tracking` (the large primes in double and float, and
  controls), plus the standard catalogue's prime rows; then the standard tracking tier.
- Metric: median paired ratio B ÷ A per case with its 95 % interval; weighted geomean over
  the cases whose plan the arm changes; FFTN ÷ FFTW and ÷ MKL before and after.
- Decision rule, written before the judging run: an arm is adopted on a machine's profile
  (the release toolchain: GCC on x86-64, Clang on macOS) if over the cases whose plan it
  changes the weighted geomean B ÷ A is below 1 − band with its interval below 1, no such
  case is slower than 1 + max(band, 3 %), and no tracking-tier case is slower than
  1 + max(band, 3 %) (a case just past the limit on a loaded run is re-run once). A slower
  changed case keeps the old plan by a rule on its shape measured in the same run, or the arm
  is rejected on that profile. At most three arms.

## Results

### Screen (Xeon, no default change)

Run 20261117-235037-3881 (Slurm 16426 build, 16427 timing, not `--serial`; GCC 14.4,
`avx512`, double; 10 rounds, 3 under load, band ±3.33 %): N = 80dbcdb (rules), C = chain,
G = grid, D = depth-first with one top of 16, E = two tops of 16, F = FFTW, M = MKL; ns per
point:

| case | N (rules) | C | G | D | E | FFTW | MKL |
|---|---|---|---|---|---|---|---|
| c2c 131071 | 42.7 (Bluestein, chain M 2¹⁸) | 44.2 | 45.0 | as N | as N | 35.6 | 32.1 |
| c2c 999983 | 81.7 (Bluestein, chain) | 83.9 | 83.5 | as N | as N | 73.0 | 56.4 |
| c2c 65537 | 20.4 (Rader) | 20.6 | 20.3 | as N | as N | 39.0 | 61.4 |
| c2c 262144 | 4.42 (depth-first 16) | 8.16 | 6.46 | 4.50 | 4.95 | 5.95 | 7.44 |
| c2c 2000000 | 12.7 (grid) | 18.3 | 12.6 | 11.9 | 12.9 | 17.3 | 16.3 |
| c2c 2073600 | 11.3 (grid) | 17.0 | 11.8 | **9.50** | 10.5 | 17.6 | 13.5 |
| c2c 2097152 | 13.8 (grid) | 16.8 | 13.8 | **10.4** | 10.8 | 17.6 | 18.5 |

(The primes' plans do not change with D or E in this build: the depth-first strategy does
not fit a prime.) The convolution's chains are the C column: at 2¹⁸ the depth-first chain
takes 0.54 of their time, at 2073600 0.56 and at 2²¹ 0.62, with one top of 16 better than
two. By the stage's arithmetic (two transforms of M and two chirp passes), 131071 should
take about 0.65 of its time (about 0.87 of MKL's) and 999983 about 0.62 (about 0.90 of
MKL's). The rows beyond 4 L2 gain too (2073600 0.85, 2²¹ 0.75 of the grid): a lead for
X51 (rows beyond 4 L2), not changed here.

### Judging

(pending)

## Verdict

(pending)

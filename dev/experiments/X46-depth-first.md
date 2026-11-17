# X46: a depth-first long chain (rows between L2/4 and a few times L2)

- **Status:** running
- **Owner:** x46 session, 2026-11-17
- **Phase / task:** 1 (phase 1 exit check, the tail rule; cause 5); follows X3, X26, X30b,
  X39, X42

## Question

Does a depth-first chain, in which every sub-row runs to completion while it is in L1 or L2,
close the rows between about a quarter of L2 and a few times L2, where FFTN's breadth-first
Stockham chain overfills L2 and the two-phase grid's copies cost too much (Xeon 6240 16384
1.21 of FFTW and 1.16 of MKL, 32768 1.22 / 1.13, 65536 ÷ MKL 1.15–1.23, 20000 1.13, 25000
about 1.18, 100000 1.28; Mac 8192–65536 1.05–1.10)? The answer adds a strategy (a program
of the existing pass kinds plus two new pass loops) and a profile rule for when to take it.

## Hypothesis

LESSONS 3.22: on the Xeon 6240 (1 MiB private L2) the chains of 16384–32768 complex doubles
(256–512 KiB) missed L2 0.7–1.7 times per point at an IPC near 1 with fewer instructions
than FFTW, which plans 16384 as recursive decimation in time (8 · 64 · 32) whose 2048-point
sub-transforms finish in L1 and misses L2 0.0–0.3 times per point. The breadth-first chain
keeps three lines live through four passes (input, output, scratch a: 768 KiB at 16384) plus
the first pass's twiddle table (about n entries); the grid's two phases each gather and
scatter (30–50 instructions per point against the chain's 12).

A two-level decimation in time keeps one line hot after its first pass. With n = R1 · M:

1. *depth-first first pass* (one pass over the row): for every residue r < R1, the first
   pass of the length-M chain of x[R1 j + r], with lanes over C adjacent residues (the input
   is R1 interleaved sub-rows, the layout a Stockham lane pass reads with s = R1 copies) and
   its outputs transposed in registers into each sub-row's own block of the output,
   out[r M + R2 p + k];
2. *sub-rows* (a loop of R1 independent iterations): the rest of each sub-row's chain on its
   block of M (16–32 KiB) and one or two scratch lines of M, ending in its block: every pass
   but the first in L1 or L2;
3. *top pass* (one pass over the row, in place): the radix-R1 decimation-in-time butterflies
   X[k + M t] = Σ_r ω_R1^(r t) w_n^(r k) Y_r[k], lanes over adjacent k, legs M apart, the
   twiddles per lane (the grouped first pass's table for (n, R1)).

So the row is touched in three sweeps (input to output, the sub-rows' blocks once each, the
top pass in place) by two lines instead of three, and every middle pass is cache-resident.
Expected on the Xeon: 16384 and 32768 near FFTW (0.80–0.90 of the current plans), L2 misses
per point down to FFTW's; 65536–100000 maybe, with M in L2; the EPYC (512 KiB L2) and the
Mac (a 16 MiB L2 shared by the cluster) probably differ, so the rule is per profile.

## Change under test

Arms are commits of branch `exp/x46-depth-first` against A = its base (X42's head merged
with `main`, b27970f):

- **A0 (the strategy, no default change).** Two pass loops on every target (the depth-first
  first pass and the top pass, `src/passes/depth.inl`), the strategy as a program of them and
  the chain's lane passes (`src/strategies/depth.*`), `strategy_choice::depth` and its split
  (`build_settings::depth_radix`) for tests and fbench (`STRATEGY=depth`, `DEPTH_RADIX=R1`).
  No plan changes; the strategy is tested in both precisions on every target.
- **A (the rule).** A profile field: c2c rows contiguous out of place whose bytes lie in a
  range keyed on the private L2 take the depth-first chain, R1 and M chosen by a rule fitted
  on the screen (step 1) and written here before the judging run.
- **B (derived twiddles for the top pass)**, only if the counters show the top pass's table
  (about n complex values) is a large part of its L2 traffic: w_n^(r k) formed in registers
  from w_n^k and a few stored powers, measured on the changed rows only (LESSONS 3.10).
- **C (the factor rule X39 left)**: 100000 as 100 × 1000 on the Xeon profile (five passes
  against six, 0.92–0.94 in X39's screen), if the depth-first chain does not take 100000.

Bits: the depth-first chain is another factorisation, so a changed plan changes the
rounding of its rows; tests hold it to the derived tolerances in both precisions and to
bitwise equality between serial and threaded runs.

## Method

- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512, oneMKL 2026.1;
  Slurm `-c 8` shared; timing through `fbench_remote.py submit --serial`), then the EPYC 7763
  (`avx2`, GCC 16, DMOG `-c 8`) and the M5 Max (`neon_fcma`, Homebrew Clang, NEON FFTW).
- Step 1, screen (no default change): on the Xeon, user-mode counters (`fbench-fftn run
  --pmu`: cycles, instructions, `l2_rqsts.miss`; min over processes) and times of the rules'
  plan against forced depth-first splits (`STRATEGY=depth`, `DEPTH_RADIX`), FFTW and MKL, on
  8192, 16384, 20000, 25000, 32768, 46080, 65536, 100000 and 131072. The rule of arm A is
  written from it.
- Step 2, each arm against the one before, fbench interleaved rounds with the A/A control, on
  the standard catalogue's rows the rule changes (`--tier full`), then the standard tracking
  tier.
- Metric: median paired ratios with 95 % intervals; weighted geomean over the cases whose
  plan an arm changes; FFTN ÷ FFTW and ÷ MKL before and after.
- Decision rule, written before running: an arm is adopted on a machine's profile (the
  release toolchain there: GCC on x86-64, Clang on macOS) if over the cases whose plan it
  changes the weighted geomean B ÷ A is below 1 − band with its interval below 1, no such
  case is slower than 1 + max(band, 3 %), and no tracking-tier case is slower than
  1 + max(band, 3 %) (a case just past the limit on a loaded run is re-run once). A profile
  on which the arm is not judged keeps the old plans (the field off; the profile tests check
  the recorded machines). At most three arms.

## Results

(pending)

## Verdict

(pending)

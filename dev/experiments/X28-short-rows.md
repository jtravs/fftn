# X28: single short rows

- **Status:** running
- **Owner:** x28 session, 2026-11-15
- **Phase / task:** 1 (exit criteria); PLAN lead 4; follows X20 and X25

## Question

What is left of the cost of one short row (≤ 120 points, batch 1) after X20, and which of
three changes closes it: a lighter entry, 128-bit leaves on x86, or a flat path for chains
of two passes? Each one that wins becomes the default on the targets where it wins.

## Hypothesis

Baseline (tracking tier, FFTN ÷ FFTW): single rows ≤ 64 points 1.14 on the Mac, 1.38 on
the EPYC, 1.51 on the Xeon; 8 points 1.80–1.83 on the Xeon; 60 points 1.60 on the Xeon and
1.42 on the EPYC; 120 points 1.62 on the Xeon (runs 20261115-094953-38c5,
20261115-101319-f98f, 20261115-114105-ea43). X25 split the gap:

- at 8 points 23 of the 25 cycles of the Xeon's gap (14 of 15 on the EPYC) are fixed cost
  per call. X20's harness puts only about 2 ns between the API and the leaf on the Xeon
  (8 points: `api` 13.0 ns, the leaf alone through its pointer 10.9 ns), so most of that
  fixed cost is in the leaf's own call: its set-up of 16 leg offsets for any stride, six
  saved registers and `vzeroupper`, and on the Xeon a 512-bit leaf that runs at frequency
  licence 1 where FFTW's 128-bit `n2fv_8_sse2` runs at licence 0 (1.62 times the cycles,
  1.84 times the time);
- at 60 points on the EPYC 37 of 79 gap cycles are the two-pass chain's driver
  (`run_flat_one`, `moved_in_nest`): each stage copies its 144-byte `pass_args`, reaches its
  stage through a `std::variant` and its slots through switches;
- at 60 points on the Xeon the partial-unit radix-4 grouped pass takes 57 % of the time
  (left to X23b, which changes those partial units).

Expected: (a) a few per cent at 2–16 points; (b) on the Xeon most of the 8–16-point gap
(licence 0, no broadcasts of 512 bits), little on the EPYC (Zen 3 runs 256-bit operations at
full rate); (c) 5–15 % at 24–120 points on all machines.

## Change under test

Arm A: `main` (3f90275). Then one commit per arm, each on top of the one before:

- (a) the lighter entry: what a measurement of each piece (X20's harness with the new
  `public` and `copy` levels, `--pmu` counts) shows is worth removing between
  `plan::execute` and the kernel of a one-row plan.
- (b) 128-bit leaves on x86: the leaves of n ≤ 16 compiled with a 128-bit vocabulary (one
  complex double per register, as on NEON) inside the `avx2` region, and used by `avx2` and
  `avx512` plans whose program is rows of leaves. The vocabulary repeats `avx2`'s operations
  at half width, so every lane rounds as before: bits unchanged.
- (c) a flat path for chains of two passes: the stages of a flat program prepared so that
  one transform runs its passes without the variant walk and with the least copying the
  calling convention allows.
- (d), added after the harness showed what (c) leaves (Results): every pass kernel takes
  its source and destination as arguments, as a leaf does, and `pass_args` no longer holds
  them, so the driver calls each kernel on the plan's own arguments instead of a fresh copy
  per stage and transform. Judged by the same rule.

No switch in `dev/KNOBS.toml`: each arm is a commit (b 7d0e216, a 4b5c833, c d6a84ca,
d 6f55ab9, in that order on the branch).

## Method

- Cases: catalogue `x28` (`bench/catalogues/x28.toml`): c2c rows 8, 12, 16, 24, 32, 48,
  60, 64, 120 at batch 1 in double and float (targets), the same at batches 64 and 4096 in
  double, r2c and c2r 16, 32, 64 at batch 1. Collateral: the standard catalogue's tracking
  tier, final arm against A.
- Machines: Xeon Gold 6240 (GCC 14, `avx512` and forced `avx2`, FFTW 3.3.11 with AVX-512
  codelets, MKL); EPYC 7763 (DMOG, GCC 16, `avx2`); Apple M5 Max (Clang, `neon_fcma`, NEON
  FFTW), as the brief allows for this lead. fbench interleaved rounds with the A/A control;
  Slurm `-c 8`, shared, `--serial`.
- Diagnosis: X20's harness (`bench/x20/x20_overhead`) per arm, ns per transform at the
  levels `public`, `api`, `driver`, `copy`, `kernel`, `once` (and, before arm d, `prep`:
  every stage's arguments copied before the first kernel runs); `--pmu` counts per call
  (cycles, instructions; on the Xeon also the AVX frequency licences and 128/512-bit FP
  operations).
- Metric: fbench paired ratio arm ÷ A per case; weighted geometric mean over the batch-1
  c2c cases (targets), and FFTN ÷ FFTW.
- Decision rule, written before running, per arm and per machine-target, on the release
  toolchains (GCC on x86-64 Linux, Clang on macOS):
  1. every test passes; arms (a) and (c) keep results bit-identical (same kernels on the
     same arguments), arm (b) too (same operations per lane), held by
     `Contracts.BatchedRowsMatchSingleRows` and the forced-target suites;
  2. adopt where the geometric mean over the target cases is below 0.97 against the arm
     before it, with its interval clear of the A/A band;
  3. and no case of `x28` (batches included) or of the tracking tier is slower than the
     arm before it by more than the band; a case just past the band on a loaded machine is
     re-run before it decides. A win on one target and a loss on another is adopted per
     target.

## Results

Toolchains: GCC 14.4 on the Xeon, GCC 16.2 on the EPYC (DMOG, `-c 8`), Clang 23.1.4 on the
Mac; Release; one thread; out of place; forward. Every machine was loaded throughout (Xeon
load average 30–50, Mac 9–10, all rounds marked loaded); verdicts are judged against each
run's own A/A band.

### Where the time of one short row went (harness, before)

EPYC, `main` plus the harness (4647a31), ns per transform at batch 1, double, median of
four interleaved process rounds (`x20_overhead`, job 509654). `public` is what
`plan::execute` costs, `kernel` the stages' kernels called through their pointers on the
plan's arguments, `once` a leaf alone:

| n | public | api | driver | copy | kernel | once |
|---|---|---|---|---|---|---|
| 8 | 12.05 | 11.40 | 11.40 | 11.07 | 10.74 | 8.79 |
| 16 | 20.07 | 19.84 | 20.69 | 18.61 | 18.71 | 16.67 |
| 24 | 47.03 | 46.52 | 42.74 | 39.01 | 29.67 | |
| 32 | 56.08 | 55.91 | 53.27 | 45.67 | 38.88 | |
| 48 | 71.73 | 71.44 | 66.88 | 64.83 | 54.73 | |
| 60 | 96.66 | 95.36 | 91.52 | 89.79 | 77.86 | |
| 64 | 90.28 | 89.70 | 87.41 | 83.70 | 74.26 | |
| 120 | 170.09 | 169.52 | 167.29 | 164.64 | 151.92 | |

- Leaves (8, 16): 3.3 ns between the public entry and the leaf alone, of which 1.3 ns
  is the entry (two library calls, the records walk) and 2 ns the kernel level's own
  addressing; the leaf itself is most of the time.
- Chains (24–120): the per-stage copy of `pass_args` (copy − kernel) cost 7.4–12.3 ns per
  transform, 3.7–6 ns per stage; the rest of the interpretation (driver − copy) 1–7 ns and
  the entry with its `slot_addresses` round trip (public − driver) 2–5 ns. With every
  stage's copy made before the first kernel (level `prep`, at e6154aa) the copy cost half
  as much (4.1–6.5 ns): much of it is the fresh copy's store-to-load forwarding into the
  kernel, not the copying. On the Mac the copy cost about 1.5–2.7 ns per stage; the Xeon
  (loaded, noisy) about 3.5.

That made the copy, not the walk, the largest fixed cost of a chain on x86, so arm (d) was
added.

### Counters (user mode, per call, median of three rounds; jobs 509659 and 15781)

| case | EPYC cycles A / d | EPYC instructions A / d | Xeon `avx512` cycles A / d | Xeon instructions A / d | Xeon licence cycles (1 + 2) A / d |
|---|---|---|---|---|---|
| c2c-8 | 44 / 35 | 195 / 173 | 65 / 47 | 200 / 179 | 65 / 0 |
| c2c-16 | 71 / 59 | 315 / 295 | 110 / 85 | 321 / 300 | 110 / 0 |
| c2c-24 | 141 / 100 | 624 / 502 | 267 / 229 | 674 / 544 | 267 / 226 |
| c2c-60 | 308 / 251 | 1266 / 1137 | 401 / 356 | 1012 / 886 | 401 / 353 |
| c2c-64 | 280 / 237 | 1151 / 1022 | 278 / 229 | 852 / 725 | 280 / 226 |
| c2c-120 | 532 / 484 | 2120 / 1993 | 686 / 642 | 2038 / 1867 | 686 / 638 |

(A = 4647a31, d = 4ab1bd0, the final branch.) FFTW for comparison (X25): EPYC 29 / 50 /
221 / 211 / 413 cycles at 8 / 16 / 60 / 64 / 120; Xeon 40 / 78 / 258 / 231 / 437. The
128-bit leaves run at licence 0 (no 512-bit FP operation left at 8 and 16 points: 31 and
91 128-bit ones instead); under forced `avx2` the 16-point leaf had spent its 100 cycles at
licence 1 and now spends none there. At 64 points the Xeon's cycles match FFTW's.

### fbench, Mac

Run 20261115-193513-a1ac (arms A, a, c; 10 rounds, band ±2.48 %) and run
20261115-194803-d367 (A, c, d; 10 rounds, band ±3.27 %). Geometric means of paired ratios,
arm ÷ the arm before it:

| arm | single f64 | single f32 | batches 64, 4096 | real rows | worst case |
|---|---|---|---|---|---|
| a ÷ A | 0.986 | 0.992 | 1.010 | 0.990 | 60 × 4096 1.050 (inside the band: verdict no evidence) |
| c ÷ a | 0.976 | 0.955 | 1.001 | 0.914 | 48 × 4096 1.029 (no evidence) |
| d ÷ c | 0.921 | 0.877 | 0.961 | 0.830 | 12 × 4096 1.025 (no evidence) |
| d ÷ A | 0.878 | 0.822 | 0.949 | 0.747 | 32 × 64 1.020 (no evidence) |

d ÷ A per case: 24 points 0.685 (f32 0.740), 48 0.830, 60 0.833, 64 0.842, 120 0.878; leaves
8–16 0.94–0.98; r2c and c2r 16 0.60–0.62, 32 0.73–0.81, 64 0.88–0.89; batches of 24–120
0.87–0.96. FFTN ÷ FFTW (NEON), single c2c rows: 1.144 → 1.005; real rows 1.161 → 0.862.

Tracking tier, run 20261115-200944-b5dc (A, d; 10 rounds, band ±3.56 %): d ÷ A 0.953, 14
cases faster (17 points 0.79, 64 × 64 0.81, 60 0.84, 64 0.84, 128 0.89, the 2048 × 36
planes 0.87–0.91, r2c 64³ 0.88), none slower; FFTN ÷ FFTW 1.067 → 1.017. (A first tracking
run, 20261115-195155-e57c, overlapped a GCC build of mine: band ±8 %, not used.)

### fbench, EPYC

Run 20261115-214625-9bf0 (Slurm 509647; arms A, b, a, c, d and FFTW; GCC 16, `avx2`; 10
rounds, every one overloaded: other processes used 93 of 128 CPUs; band ±1.38 %). Arm ÷
the arm before it:

| arm | single f64 | single f32 | batches 64, 4096 | real rows | slower than the band |
|---|---|---|---|---|---|
| b ÷ A | 0.987 | 0.985 | 0.989 | 1.004 | none |
| a ÷ b | 0.964 | 0.971 | 1.002 | 1.001 | 120 points f32 1.066 |
| c ÷ a | 0.962 | 0.955 | 1.001 | 1.056 | r2c 16 1.18, r2c 32 1.10, c2r 16 1.13, c2r 32 1.04 (against A) |
| d ÷ c | 0.896 | 0.896 | 0.988 | 0.746 | none |
| d ÷ A | 0.819 | 0.822 | 0.976 | 0.788 | none |

- b touches only the leaf lengths: 8 0.958, 12 0.975, 16 0.964 (f32 8 0.958, 16 0.937);
  every other case within ±1.3 %. Its geometric mean over all nine lengths misses 0.97.
- a: leaves 8–16 0.89–0.94 in double, 8 points f32 0.83; one slower case (120 f32, three
  passes, which a does not change: code placement), 0.898 of A once d is in.
- c alone slowed the single real rows of 16–32 points by 4–18 % (their two stages from
  the contiguous body, with a copy of a copy); d, which removes that copy, takes them to
  0.71–0.81 of A.
- d ÷ A per case, double: 8 0.854, 12 0.857, 16 0.867, 24 0.716, 32 0.754, 48 0.789,
  60 0.829, 64 0.817, 120 0.907; float 0.70–0.90; batches 0.90–1.01; r2c and c2r 16 0.71,
  32 0.80–0.81, 64 0.84–0.87.
- FFTN ÷ FFTW, single c2c rows in double: 1.366 → 1.122 (8 points 1.36 → 1.17, 60 1.42 →
  1.18, 64 1.32 → 1.09, 24 1.43 → 1.03); real rows 1.098 → 0.862.

Tracking tier, run 20261115-215643-eaa9 (Slurm 509649; A, d; 8 rounds, all overloaded,
band ±2.75 %): d ÷ A 0.964, 12 cases faster (17 points 0.81, 64 0.83, 60 0.84, 8 0.85,
16 0.86, r2c 64 × 64 0.90, 120 0.92, c2c 64 × 64 0.92, r2c 64³ 0.92), none slower (worst
16384 1.03 and r2c 2²⁰ 1.03, inside the band); FFTN ÷ FFTW 1.321 → 1.273.

### fbench, Xeon

(pending: jobs 15769 and 15771)

## Verdict

(pending)

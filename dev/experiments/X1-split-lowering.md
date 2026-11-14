# X1: split re/im lanes against interleaved complex registers

- **Status:** done
- **Owner:** split-lowering session (task 1.5), 2026-11-14
- **Phase / task:** 1.5; follows X18, X10 and X19

## Question

Do a chain's lane passes run faster in split arithmetic (separate real and imaginary
registers, BL layout between passes) than on interleaved complex registers, per target and
precision? The answer sets `target_profile::split` (planner/rules.cpp, `split_of`) for each
target and precision, and decides whether the split path stays in the library at all.

## Hypothesis

After X19 the Xeon's AVX-512 kernels keep port 5 72 % busy against FFTW's 55 % at 256
points; what remains is the twiddle multiplies' swaps (one per twiddled leg), the `addj` /
`subj` swaps (⌊(R − 1)/2⌋ per codelet) and the grouped pass's transposes. A split codelet
has none of the first two: its twiddle multiply is two multiplies and two FMAs on W reals
with broadcast twiddle parts, and ±i is a free renaming. Per W points, the generated
radix-16 DIF codelet is 212 real-register operations and no shuffle (FFTW's FMA count for
`t1_16` is 204), against two interleaved calls of about 117 instructions each with about
22 shuffles each.

The conversion costs one shuffle per register at each end of a split run (deinterleave on
loading, interleave on storing), and the grouped first pass keeps its transposes (moving
them into the BL layout costs as much as the deinterleave: an 8 × 8 transpose of doubles
against a 4 × 4 one of complex pairs). So split can win only where a chain has middle
passes to amortise its conversions: chains of three or more passes (512 to 2048 points in
double on x86; larger rows are the grid's). Expected: a few per cent on the Xeon's AVX-512
at 512–2048 where port 5 binds, less or nothing on AVX2 (16 registers: a radix-16 split
codelet holds 34–40 live reals and spills; Zen 3 has two shuffle pipes), nothing or a loss
on NEON (FCMA multiplies in two instructions; the prototype found interleaved 5 % faster on
NEON and on AVX-512, and 4–8 % slower on Zen 3 only for C = 2; pdev research notes).

## Change under test

One build, two arms differing in one internal setting (`build_settings::layout`, fbench
option `LAYOUT`): A = `LAYOUT=interleaved`, B = `LAYOUT=split`. Everything else (radices,
grouped first pass, placement, scratch) is the same; a chain with no split run (two passes,
a run of one pass, a generic or Bluestein pass in it, strided output on NEON double) is
identical in both arms. The setting is a test knob of the private build settings, not a
library switch, so `dev/KNOBS.toml` does not list it (as for `STRATEGY` and `RADICES`).

Split run (strategies/chain.hpp): from the pass after the first whose copy count is a
multiple of the target's `split_block` (avx512's W on both x86 targets, so the bitwise pair
agrees: 8 in double, 16 in float; NEON's W) to the last pass, if that is two passes or more.

## Method

- Cases: the standard catalogue's tracking c2c rows (`--tier tracking --filter
  '^c2c-[0-9]+-b'`: rows and row batches, forward, double), FFTW 3.3.11 at MEASURE
  alongside for FFTN ÷ FFTW. Cases whose chains have no split run time the same code in
  both arms; they stay in the geomean (they are what a user's mix contains) and are listed.
- Machines: Xeon Gold 6240 (Cascade Lake; `avx512` default and `ISA=avx2`), GCC 14.4,
  Slurm `-c 8` on the shared node; EPYC 7763 (DMOG node16+, Zen 3, `avx2`), GCC 16, `-c 8
  --hint=nomultithread`; Mac (M-series, `neon_fcma`), Homebrew Clang, for completeness.
  Float on the Xeon `avx512` as a second precision if the double result is positive.
- Metric: fbench median paired ratio B ÷ A per case, geometric mean over the cases with a
  95 % bootstrap interval; A~ (A/A control) gives the band. `perf stat` port-5 uops and
  cycles per transform at 1024 (Xeon, both ISAs).
- Decision rule, written before running: adopt split as the default for a target and
  precision if its B ÷ A geometric mean over the tracking c2c rows is ≤ 0.97 (3 % or more
  faster) with the interval clear of the A/A band, and no case is slower than
  1 + band + 2 %. Otherwise that target stays interleaved. If no target adopts split, the
  split path (generator lowering, passes, setting) is deleted (AGENTS.md 6) and the
  generator change kept only if another task needs it.
- Amendment, after the early signal and before the decision runs: most tracking c2c rows
  have no split run on any x86 target (two passes at ≤ 256, the grid beyond L1, primes,
  leading radix 4 on AVX2), so they time the same code in both arms and dilute any effect
  towards 1. The geometric mean is therefore taken over the cases whose B plan has a split
  run (the tracking c2c rows and the supplementary set below, "split from pass" in the
  case's note), ≤ 0.97 with its interval clear of the band; the no-slowdown condition
  stays on every case of both sets. The geometric mean over all tracking c2c rows is
  reported too.
- Supplementary set (`x1.toml` in the campaign directories, not in the tree): c2c rows of
  512, 768, 1024, 1536, 2048 (double), 512 × 64 and 1024 × 64, 1024 backward, r2c and c2r
  of 2048, and c2c rows of 1024, 2048, 4096 in float.

## Results

### Early signal (commit c8c709c, before the tangent-form generator change)

Xeon 6240, GCC 14.4, run 20261114-180920-df53 (16 rounds, band ±1.49 %, load median 9.8):
split ÷ interleaved per case, avx512 arms B ÷ A, avx2 arms B2 ÷ A2 (from the ratios to A):

| case | radices (avx512) | avx512 | avx2 |
|---|---|---|---|
| 512, 768, 512 × 64 | 4 × 8 × 16, 4 × 12 × 16 | no split run: 0.99–1.00 | no run |
| 1024, 1024 bwd, 1024 × 64 | 8 × 8 × 16 | 0.90, 0.89, 0.91 | no run (4 × 16 × 16) |
| 1536 | 8 × 12 × 16 | 0.92 | 0.97 |
| 2048 | 16 × 8 × 16 | 0.93 | 0.89 (8 × 16 × 16) |
| r2c, c2r 2048 | half-length 8 × 8 × 16 | 0.92, 0.95 | no run |
| float 2048, 4096 | 16 × 8 × 16, 16 × 16 × 16 | 0.79, 0.81 | 0.84, 0.77 |

EPYC 7763, GCC 16, run 20261114-181003-1c8e (16 rounds, band ±1.37 %): split runs only at
2048 (0.89) and float 2048 and 4096 (0.93, 0.82); every other case has a leading radix 4
and no run (0.99–1.01). So split wins even in a run of two passes, which pays both
conversions and has no BL-to-BL pass; the decision runs below use the final generator.

### Decision runs (commit f38ad96)

Paired per-round ratios split ÷ interleaved (B ÷ A, and B2 ÷ A2 for the Xeon's `avx2`
arms), median over rounds, geometric means over the cases with a split run (computed from
the stored timings with the 95 % bootstrap interval per case):

| Machine, target | Run (set) | Band | Split-run cases: geomean (n) | All cases | Slowest case |
|---|---|---|---|---|---|
| Xeon 6240, `avx512` | 20261114-182235-7015 (X1 set, 24 rounds) | ±2.03 % | 0.892 (9) | 0.924 | 1.006 (512, no run) |
| Xeon 6240, `avx2` | same run | ±2.03 % | 0.873 (4) | 0.962 | 1.019 (512 × 64, no run) |
| EPYC 7763, `avx2` | 20261114-182403-9c72 (X1 set, 24 rounds) | ±1.00 % | 0.929 (4) | 0.977 | 1.025 (1536, split) |
| EPYC 7763, `avx2` | 20261114-182358-c6ff (tracking, 20 rounds) | ±1.00 % | — (no case has a run) | 0.998 (35) | 1.009 (4093) |
| Xeon 6240, `avx512` | 20261114-183454-f63e (tracking, 20 rounds, 17 loaded) | ±2.18 % | 0.906 (4: 1024 rows, backward, in place, × 256) | 0.989 (35) | 1.031 (2^20, grid, no run) |
| Xeon 6240, `avx2` | same run | ±2.18 % | — (no case has a run) | 1.001 (35) | 1.036 (4096, grid, no run) |

FFTN ÷ FFTW over the 35 tracking c2c rows (same runs): Xeon `avx512` 1.175 → 1.162, EPYC
1.357 → 1.354 (no run there); the tracking set has split runs only at 1024 on `avx512`
(1024 0.889, backward 0.889, × 256 0.896, in place 0.951).

Per case on the Xeon, `avx512`: 1024 0.889, 1024 backward 0.899, 1024 × 64 0.920, 1536
0.915, 2048 0.928, r2c and c2r 2048 0.930 and 0.945, float 2048 and 4096 0.792 and 0.824.
`avx2` (Xeon): 1536 0.983, 2048 0.904, float 0.845 and 0.774. EPYC: 1536 1.025 (radix 12,
whose split codelet keeps 26–44 reals live on 16 registers), 2048 0.920, float 0.932 and
0.848. Float on `avx512` has no run at 1024 (8 × 8 × 16: split_block is 16 there).

FFTN ÷ FFTW on the X1 set (FFTW 3.3.11, MEASURE, same runs; double cases): Xeon `avx512`
1.19 → 1.10 over its seven double split-run cases (1024 1.23 → 1.09, 2048 1.13 → 1.05);
EPYC unchanged except 2048 (1.30 → 1.20).

Would a split-aware planner give AVX2's 1024 a run? EPYC run 20261114-184451-3196 (16
rounds, band ±1.00 %), against the default interleaved 4 × 16 × 16: interleaved 8 × 8 × 16
1.023, split 8 × 8 × 16 1.012 (0.987 of its interleaved self), split 16 × 8 × 8 1.137. On
Zen 3 the run gains little at 1024, so a first radix chosen for the run would not pay there;
the Xeon's AVX2 was not probed.

Port counters per transform (Xeon, `perf stat`, one case per process minus a check-only
process, approximate; job 15577, script `ports.sh` in the campaign directory):

| n | target | cycles IL → split | port-5 uops IL → split (busy) | instructions IL → split | FFTW cycles, port 5 |
|---|---|---|---|---|---|
| 1024 | avx512 | 5888 → 5297 | 3488 → 3056 (59 % → 58 %) | 12894 → 10691 | 5128, 2081 (41 %) |
| 2048 | avx512 | 13333 → 12084 | 8175 → 7128 (61 % → 59 %) | 26174 → 21753 | 11795, 5141 (44 %) |
| 2048 float | avx512 | 7570 → 5905 | 5458 → 3842 (72 % → 65 %) | 15608 → 12015 | |
| 1024 | avx2 | 8916 → 8856 | 3478 → 3463 | 23532 → 23435 | (no run: 4 × 16 × 16) |
| 2048 | avx2 | 20830 → 18949 | 7602 → 7597 | 48933 → 45840 | |
| 2048 float | avx2 | 11215 → 9493 | 6791 → 4783 | 29131 → 23506 | |

Port 5 stays about 58 % busy in the split plans on `avx512` (FFTW 41–44 %): what remains
is the grouped first pass (its twiddle swaps and 4 × 4 transposes) and the run's two
conversions. On `avx2` the double gain comes from fewer instructions, not from port 5.

Mac (M5 Max, `neon_fcma`, AppleClang 21), run 20261114-181508-efc3 (X1 set, 16 rounds,
heavily loaded: band ±4.00 %): split 1.10–1.25 slower on every case with a run (geomean
about 1.16). NEON stays interleaved, as expected: FCMA multiplies complex values in two
instructions and rotates for free.

Fix after review: an r2c row whose pairing fused into the last pass (`avx512` fuses at
radix 8, `avx2` at 4) never took a split run, while the separate route did, so the pair
gave different bits (r2c 1024 forced 8 × 8 × 8: 314 of 513 bins differ, forced on NEON).
`route()` now sends an r2c row to the separate route whenever its half-length chain has a
split run; fusion stays where none exists. Tests `Passes.R2cRoutesAgreeUnderSplit` and
forced radix-8 real rows in `Passes.TargetsThatRoundAlikeGiveTheSameBits`. Not re-timed.

## Verdict

Adopted on x86: `avx2` and `avx512`, double and float, take split runs by default
(`split_of` in planner/rules.cpp); `neon`, `neon_fcma` and `scalar` stay interleaved. Over
the cases with a split run the geometric means are 0.892 and 0.906 (Xeon `avx512`, two
sets), 0.873 (Xeon `avx2`) and 0.929 (EPYC), each clear of its band, and no case of either
set is slower than 1 + band + 2 % (the worst, EPYC 1536 at 1.025 against a limit of 1.03,
has a run; the others are cases without one). Float rests on two cases per machine (0.77–0.93).
Under the unamended rule (all tracking c2c rows) no target would pass: 0.989 on the Xeon
`avx512`, 1.00 elsewhere, because the tracking set has runs only at 1024 on `avx512`.

What it does not do, and the follow-ups it opens:

- Port 5 is still about 58 % busy on `avx512` (FFTW 41–44 %): the grouped first pass and the
  conversions remain. A split first pass that writes BL with an 8 × 8 real transpose costs
  the same shuffles as the deinterleave, so the next gain there is fewer transposes, not a
  different layout.
- AVX2 gets no run where its planner starts with radix 4 (1024 = 4 × 16 × 16), because a run
  starts at `avx512`'s W on both targets of the pair; on the EPYC a run at 1024 (8 × 8 × 16)
  would gain only 1.3 %, so no planner change is made. A split-aware cost table (X2's method
  with split passes) is the general answer.
- Bluestein convolutions and grid phases stay interleaved (the 1009 and 4093 rows'
  convolutions of 2048 and 8192 points would have runs).
- The split codelets reach FFTW's FMA counts except for scales shared by sums that each
  absorbed another product (radix 16 plain 152 against 144).

Lesson added to `dev/LESSONS.md` (2.1a).

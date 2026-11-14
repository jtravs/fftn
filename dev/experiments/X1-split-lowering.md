# X1: split re/im lanes against interleaved complex registers

- **Status:** running
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

## Results

(pending)

## Verdict

(pending)

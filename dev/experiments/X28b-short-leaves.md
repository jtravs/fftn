# X28b: what is left at single short rows: leaves in registers

- **Status:** running
- **Owner:** X28b agent, 2026-11-16
- **Phase / task:** 1 (exit criteria); PLAN lead 4; follows X28 and X27

## Question

After X28, which of three leaf changes closes what is left of the single short rows on
x86 (24, 48, 60, 120 points; leaves of 8 to 23 points): a whole two-pass transform held in
registers, a leaf with constant offsets for rows of stride 1, or 128-bit leaves for the
prime codelets? Each one that wins becomes the default on the targets where it wins.

## Hypothesis

After X28 (FFTN ÷ FFTW, single c2c rows in double; runs 20261116-000445-a095 Xeon,
20261115-214625-9bf0 EPYC): Xeon `avx512` 24 points 2.02, 60 1.52, 120 1.49, 32 1.15, 48
1.09; Xeon forced `avx2` 24 1.25, 60 1.39, 120 1.60; EPYC 60 1.18, 120 1.11, 48 1.11; leaves
of 8 to 16 points 1.08–1.19 on both machines. FFTN's plans on the Xeon (`avx512`, double):
24 = 4 × 6, 48 = 4 × 12, 60 = 4 × 15, 120 = 4 × 3 × 10.

- **Short composites.** On `avx512` in double (C = 4) the first pass of 24 = 4 × 6 has six
  butterflies, one whole unit of four and a partial one; 60 = 4 × 15 has fifteen (X25: the
  radix-4 pass with its partial unit was 57 % of the 60-point time, with 1.7 times FFTW's
  stores and its store-forwarding blocks). Every two-pass factorisation of 60 has a pass
  whose vector dimension C does not divide, because 16 does not divide 60. FFTW's plan for
  60 is 15 × 4, the first pass a 512-bit codelet on four butterflies and the second a
  128-bit one. A transform of n = R1 M points with M a multiple of C fits in n / C
  registers (60: 15 of AVX-512's 32): the first pass as one grouped pass of M butterflies
  (M / C whole units), its outputs transposed in C × C blocks in registers, which is
  exactly the second pass's input (lanes over its R1 copies, padded to a multiple of C with
  zeros), and the second pass's outputs stored as runs of C, the last one masked. No
  partial unit in memory, no round trip between the passes, one call. Expected: 24 points
  on the Xeon from 2.0 to near 1.0 of FFTW, 60 from 1.5 to below 1.2; 20–64 points
  0.6–0.9 of `main` on `avx512`; less on `avx2` in double (C = 2, 16 registers: only n up
  to about 32 fits) and more in float (C = 4 and 8).
- **Unit stride.** A leaf works for any stride: it computes 2 R leg offsets from the
  strides and holds them in registers. For rows of stride 1 the offsets are constants in
  the addressing; X28 put the remaining 8–16-point gap (6–9 cycles per call on both
  machines) on this set-up. Expected 3–8 % at 8–16 points.
- **Prime leaves.** X27's codelets of 17, 19 and 23 run single rows as 256- or 512-bit
  leaves, which idle 1 to 7 lanes and on the Xeon run at a lower frequency licence (X25,
  X28). Expected: 17–23-point rows 0.8–0.9 of `main` on the Xeon (as X28's 8–16), less on
  the EPYC.

Against: a two-pass leaf of R1 legs per unit has no benefit where the chain has no partial
unit and the memory round trip is cheap (EPYC, double 60 = 6 × 10); it adds code (B3:
X27 already added 25 %), which is why the lengths are chosen per target and precision.

## Change under test

Arm A: the branch point (`origin/lead/merge-x27` 63943cc: `main` with X27). One commit per
arm, each on top of the one before:

1. **Pair leaves** (`src/passes/pair_leaf.inl`, units `src/isa/pairs_avx2.cpp` and
   `pairs_avx512.cpp`): the two-pass transform R1 × M of the paragraph above as one
   `leaf_fn`, from the same codelets and the same twiddle table as the chain R1 × M, so it
   gives that chain's bits. Used for c2c rows of stride 1 on both sides whose length has
   an entry in the target's table for the precision, at any batch (a batch of rows is one
   call, as X20's leaf rows). Lengths and factors per target and precision, chosen by n / C
   against the register file.
2. **Unit-stride leaves**: every target's leaves and the narrow leaves get a variant with
   constant leg offsets, used when both strides are 1. Bits unchanged.
3. **Narrow prime leaves**: the 128-bit leaves of X28 extended to 17, 19 and 23 on x86.
   Bits unchanged (same operations per lane).

No switch in `dev/KNOBS.toml`; the pair leaves have a forcing parameter for the tests
(`build_settings::pair_leaves`).

## Method

- Cases: catalogue `x28b` (`bench/catalogues/x28b.toml`): c2c single rows of 8, 12, 16,
  17, 19, 20, 23, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 120, 128 points in double
  and float (targets: double first); c2c batches of 64 and 4096 rows at 8, 16, 17, 24, 48,
  60, 120 (collateral); r2c and c2r single rows of 16, 32, 34, 48, 64, 120. Collateral at
  the end: the standard catalogue's tracking tier, final arm against A.
- Machines: Xeon Gold 6240 (GCC 14, `avx512` and forced `avx2`, FFTW 3.3.11 with AVX-512
  codelets, `PLAN_NOTE=1`), EPYC 7763 (DMOG, GCC 16, `avx2`), Apple M5 Max (Clang 23,
  `neon_fcma`, NEON FFTW; arm 2 only, the others do not change NEON). fbench interleaved
  rounds with the A/A control, Slurm `-c 8` shared, `--serial`.
- Metric: fbench paired ratio arm ÷ the arm before it per case; geometric mean over each
  arm's target cases (the cases whose plan or kernel the arm changes); FFTN ÷ FFTW.
- Library size: text of `libfftn.a` (`size`) per arm on x86-64 (GCC, the remote build
  cache) and arm64.
- Decision rule, written before running, per arm and per machine-target on the release
  toolchains (GCC on x86-64 Linux, Clang on macOS), double and float judged separately:
  1. every test passes; arms 2 and 3 keep results bit-identical; arm 1 gives the bits of
     the chain R1 × M forced through `build_settings::radices` (a test holds it), so its
     results change only where the planner's chain was another factorisation, within the
     derived tolerances;
  2. adopt where the geometric mean over the arm's target cases is at most 0.97 of the arm
     before it, with its interval clear of the A/A band;
  3. and no case of `x28b` (batches included) or, for the final arm, of the tracking tier is
     slower than the arm before it beyond the band. For arm 1 a length whose pair leaf is
     slower beyond the band (single row or batch) leaves that target's table, rather than
     rejecting the arm; where only batches lose, the pair leaf is limited to single rows.
     A case just past the band on a loaded machine is re-run before it decides. A win on
     one target and a loss on another is adopted per target.

## Results

Toolchains: GCC 16.2 on the EPYC (DMOG, `-c 8`, shared), GCC 14.4 on the Xeon, Clang 23.1.4
on the Mac; Release; one thread; out of place; forward. Arms: A 63943cc, P f687614 (arm 1),
U 5443035 (arm 2), N 031160b (arm 3). Ratios are of each arm's median time against the arm
before it in the same run (fbench's paired ratios against A for the verdicts); "targets"
are the cases whose plan or kernel the arm changes.

Correctness: the whole suite passes natively on the Xeon (GCC 14 `-Werror`, `avx512` and
`avx2`, 1777 tests, Slurm job 15964, at 1b64c9e). The native run of arm 1 alone (job 15950)
found two contracts the pair leaves had broken, fixed before any final measurement: the
lanes strategy (float batches on x86) must run the row's own chain, now the pair's
(16c2774, 1b64c9e); and a serial execution of a plan of leaf rows (X28's fast path) did
not reset the workspace's team state, which `TeamExecute.ResetAfterASkippedExecution`
caught once 64-point rows became one leaf under AVX-512 (1b64c9e; a latent X28 bug).

Library size (text of `libfftn.a`, x86-64, GCC 16, the EPYC build cache): A 47.18 MB; arm 1
+0.35 MB (+0.75 %); arm 2 +1.65 MB (+3.5 %), of which the x86 targets' own unit-stride
leaves, which no plan selects (they take the narrow ones), were removed afterwards
(631a3a2), leaving the scalar target's and the narrow ones; arm 3 +0.50 MB (+1.1 %).

### EPYC 7763 (`avx2`)

Run 20261116-055020-465f (Slurm 510170; arms A, P, U, N and FFTW with `PLAN_NOTE=1`; 10
rounds, every one overloaded: other processes on 64 of 128 CPUs; band ±1.14 %).

| arm | targets f64 | targets f32 | batches 64, 4096 | real rows | slower beyond the band |
|---|---|---|---|---|---|
| P ÷ A (pair leaves) | 0.631 (20–48; worst 36 0.760) | 0.541 (20–64; worst 64 0.778) | 0.862 (24 0.51, 48 0.67–0.70, others 0.99–1.01) | 0.939 (r2c/c2r 48 0.76–0.87, 64 0.85–0.88) | none |
| U ÷ P (unit stride) | 0.804 (8–23; worst 17 0.920) | 0.721 | 1.012 (17 × 64, 4096: 1.10–1.12, the 256-bit leaf of 17, which arm 3 replaces) | 1.011 (r2c 120 1.139, plan unchanged) | 17 batches, r2c 120 |
| N ÷ U (prime leaves) | 0.906 (17 0.853, 19 0.930, 23 0.938) | 0.964 | 17 batches 0.865–0.867 | 1.0 | c2c 128 f32 1.139 (plan unchanged) |
| N ÷ A | 0.841 over all 32 single f64 cases | 0.703 | 0.857 | | r2c 120 1.128, 128 f32 1.124 |

FFTN ÷ FFTW (double, FFTW's own plans): single c2c rows 0.926 → 0.724; real rows 0.815 →
0.773; batches 0.859 → 0.736. Per case after: 20 points 1.28 → 0.65, 24 0.99 → 0.53, 48
0.99 → 0.69, 8 1.24 → 0.90, 16 1.22 → 0.97, 17 0.42 → 0.32. Left above 1: 60 (1.09, plan
6 × 10 against FFTW's 10 × 6), 120 (1.05), 80 (1.02), batches of 16 (1.16–1.24), r2c 120
(1.19). The two cases slower than A, r2c 120 and float 128, keep their plans and kernels
(the plan notes are identical in all arms); they are re-run against the final branch
below.

### Apple M5 Max (`neon_fcma`)

Run 20261116-051638-105c (A, U, FFTW; 10 rounds, 8 loaded; band ±2.13 %). Only arm 2
changes NEON plans. U ÷ A: single rows 8–23 points 0.836 in double (worst 19 0.866, best 23
0.800), 0.867 in float; the 32-point leaf (NEON double) 0.892; batches 1.007, real rows
1.015, the other single rows 1.014, all "no evidence" by fbench's paired verdicts (14
faster, 52 no evidence, none slower). FFTN ÷ FFTW, single c2c rows in double: 0.800 →
0.764 (8–23 points 0.581 → 0.486). Tracking tier, final branch (1b64c9e) against `main`
(622bd95, the same library), run 20261116-055731-b624 (10 rounds, none loaded, band
±1.00 %): 0.990, 5 faster (17 points 0.81, 8 and 16 0.86, r2c 64² 0.94, r2c 64³ 0.96),
none slower; FFTN ÷ FFTW 0.940 → 0.930 (geometric means over the cases FFTW
runs).

## Verdict

(to come)

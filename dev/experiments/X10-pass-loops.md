# X10: pass loops on x86: addressing, register budget, shuffles

- **Status:** running
- **Owner:** x86-perf session, 2026-11-14
- **Phase / task:** 1 (exit criteria); follows X18

## Question

The plan's X10 asks whether generated whole passes beat templated pass loops. X18 found
what such loops must fix on x86: for the same 16 × 16 plan, FFTN's AVX2 loops execute 1.2–1.3
times FFTW's instructions per butterfly pair (5386 against 2340 per 256-point transform with
the default plan), from per-unit address arithmetic, spills on 16 registers, and shuffles.
Can the templated loops reach FFTW's instruction counts, or do they need generating? The
answer decides whether X10 becomes generator work (whole passes) or stays in the templates.

## Hypothesis

From X18's hot-loop listings (Xeon, `ISA=avx2`, 16 × 16; FFTW `n2fv_16` 176 and `t2fv_16`
222 instructions per two butterflies):

1. **Per-unit address arithmetic.** The plain radix-16 lane loop spends 60 (Clang) to 72
   (GCC) of its 204–211 instructions on scalar work: a multiply per unit for each base
   pointer, the tail test, the group and block counters, and the 2R leg offsets, which do not
   fit 16 registers (GCC rebuilds them every unit, and on AVX-512 parks them in vector
   registers with 23 `vmovq` on port 0). Fixed pointer steps per unit, the partial unit out of
   the loop, and the offsets of radix ≥ 8 read from a table at each access (FFTW's scheme)
   should bring it near FFTW's 21–34.
2. **Spills.** The grouped radix-16 pass keeps all 16 outputs live until its transposes
   (GCC: 51 stores where 16 are outputs). Storing each C × C block as soon as its legs exist
   should remove most of them.
3. **Shuffles.** The radix-16 codelet uses 19 shuffles where FFTW's core uses about 7, and
   the grouped pass adds 15 twiddle swaps and 16–32 transpose shuffles. Fewer rotations by
   ±i in the generator's output, and the transpose stores split from the twiddle multiplies,
   reduce port-5 pressure on Intel.

Each is a separate change and arm; 1 is measured first.

## Change under test

Arms are commits of branch `exp/x10-pass-loops` from main 8affea0:

- A: 8affea0 (main).
- L: bdd3946, unit loops with fixed pointer steps, partial units after the loop (leg tables
  present but off).
- T: 6ba3297, L plus leg offsets from a table for radix ≥ 8 on `avx2` and `avx512`.
- Later arms (block-wise transposed stores, generator shuffles) are added here before they
  are measured.

No switch: arms are commits; `dev/KNOBS.toml` is untouched. None changes the arithmetic or
its order (only addressing, loop shape and store order), so outputs are bitwise identical to
A; the tests' cross-target comparisons and fbench's check confirm it.

## Method

- Cases: `x18.toml` (X18; c2c double rows 64–16384, 60, 120, 360, 1000; batches 128 × 4096
  and 1024 × 256), on the Xeon with `ISA=avx512` (default) and `ISA=avx2`, on the EPYC with
  `avx2`. NEON: the standard catalogue's tracking c2c rows on the Mac.
- Compilers: GCC 14 and Clang 23 on x86; Homebrew Clang 23 on the Mac.
- Metrics: (a) instructions and cycles per transform from `perf stat` on the Xeon at 256
  (16 × 16 and default), 1024 and 4096, both ISAs; (b) fbench median paired ratio arm ÷ A per
  case and geomean, against the A/A band, and FFTN ÷ FFTW before and after.
- Decision rule, written before running: adopt an arm if its geomean ÷ A is ≤ 0.98 on at
  least one x86 machine and ISA, with its interval clear of the A/A band, no case slower than
  1 + band + 2 % on any machine, ISA or compiler, and on the Mac the tracking c2c rows' geomean
  ≤ 1.01 with no case slower than 1.03. An arm that only removes instructions without a time
  gain is rejected (deleted).

## Results

Arms on branch `exp/x10-pass-loops` (from 8affea0), each bit-identical to main by
construction (addressing, loop shape and store order only): L fixed-step pointers with tails
after the loop; T = L plus a leg-offset table from radix 8 on x86; B = T plus each
grouped-pass block stored right after its last leg; T2 tables for every radix; U main's
per-unit loops plus tables from radix 8 plus block stores; V = U without tables. Paired arm ÷
main, geometric mean over the X18 cases (GCC 14 / Clang 23):

| Arm | Xeon avx512 | EPYC avx2 | Xeon avx2 | Mac neon_fcma |
|---|---|---|---|---|
| T | 1.013 / 1.013 | 0.988 / 1.029 | 0.969 / ≈ 1.00 | — |
| B | 1.000 / 1.018 | 0.992 / 1.031 | 0.969 / ≈ 1.00 | 1.035 |
| U | 0.971 / 1.027 | 1.002 / 1.033 | 0.974 / 1.006 | 1.001 |
| V | 1.004 / 1.015 | 0.988 / 1.007 | 0.985 / 0.985 | — |

Runs: Xeon 20261114-145403-ff20 (band 1.19 %), 20261114-145726-c67a (avx2, 6 rounds, band
3.5 %), 20261114-150747-e7dc and 20261114-150419-9843 (12 and 24 rounds); EPYC
20261114-145419-0d55, 20261114-150419-126f (band 2.07 %); Mac 20261114-145423-acda,
20261114-150105-8f44. Per case the arms trade gains (256 at 0.89–0.95, 128 × 4096 at
0.91–0.94, EPYC 1000 at 0.92–0.95) against losses (120 at 1.13–1.26 and 360 at 1.12 with
the fixed-step loops, which pay a per-group setup when a group has 1–3 blocks; tables cost
Clang 3–5 % on the EPYC; Mac 64 and 128 at 1.07–1.11 under B, 2187 at 1.12 under U). The
tables cut instructions per transform by 5–15 % (perf, Xeon avx512 at 256: 2825 → 2293) but
time barely moves. Leg tables help GCC by about 3 % and cost Clang about 3 %; block stores
alone are neutral. FFTN ÷ FFTW is unchanged by every arm: Xeon AVX-512 1.29–1.30, Xeon AVX2
1.75, EPYC 1.28–1.33.

What the measurements showed instead: on AVX-512 FFTN's pass loops already execute about as
many instructions per butterfly as FFTW's (496 against 557 per four radix-16 butterflies),
and the time goes to port 5 (shuffles), 73 % busy against FFTW's 55 %. The generated
radix-16 codelet needs 19 distinct multiplications by ±i (each a shuffle in interleaved
form) against FFTW's ~7; radix 8 needs 6. A tie-break in the lowering changes nothing (the
stores' rotations decide every choice); an algebraic rewrite in the IR does (for the w8
pair, combining before scaling drops 4 swaps to 1 at the same operation count, bit-exact).
FFTN's AVX2 grouped pass is also 1.7 times slower than its AVX-512 one on the Xeon, so
mixed-width plans (FFTW runs untwiddled passes on ymm, twiddled ones on zmm) wait for
better AVX2 code.

## Verdict

Rejected: no arm reaches the rule (≤ 0.98 geometric mean on one x86 machine with no case
worse than band + 2 % and the Mac ≤ 1.01); the arms stay on the branch, unmerged. The x86
work continues with the generator's ±i rotations (X19) and then split lowering (X1).

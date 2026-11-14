# X19: multiplications by i in the generated codelets

- **Status:** running
- **Owner:** gen-rotations session, 2026-11-14
- **Phase / task:** 1 (exit criteria); follows X10 and X18

## Question

Does rewriting the generator's DAG so that codelets multiply by ±i fewer times (each a
register swap, a port-5 shuffle on Intel) make FFTN faster on x86 without slowing NEON? The
answer decides whether the real-coefficient form becomes the generator's only form.

## Hypothesis

X10 found that on the Xeon (AVX-512) FFTN's pass loops execute about FFTW's instructions
per butterfly (496 against 557 per four radix-16 butterflies) and still take about 1.28
times FFTW's time, with port 5 73 % busy against 55 %. In interleaved form `addj` and
`subj` swap their second operand (`vpermilpd`, port 5), and the radix-16 codelet swaps 19
distinct values where FFTW's `n1fv_16` swaps 7. Writing every node as R + i·I (real and
imaginary parts of its coefficients) leaves the factors of i to the outputs, one swap per
output pair X_k, X_{n−k}: ⌊(R − 1)/2⌋ per codelet. Expected: a few per cent on x86 where
radix 8 and 16 passes dominate, nothing on NEON with FCMA (`FCADD` rotates for free), but
the form also has fewer vector operations, so NEON should not lose.

## Change under test

Arm A: main 00a5455. Arm B: branch `exp/gen-rotations`, the generator's real-coefficient
form (`gen/fftn_gen/realify.py`) applied to every codelet, `src/gen/` regenerated. The
pass loops, planner and tables are unchanged. No switch.

Swaps per codelet (distinct second operands of `addj`/`subj`, forward plain and twiddle
variants alike; the twiddle multiplies' own swaps, R − 1 per twiddle codelet, are
unchanged) and vector operations (plain, forward):

| Radix | swaps A | swaps B | FFTW `n1fv` | vector ops A | vector ops B |
|---|---|---|---|---|---|
| 2 | 0 | 0 | | 2 | 2 |
| 3 | 1 | 1 | | 7 | 7 |
| 4 | 1 | 1 | 1 | 8 | 8 |
| 5 | 2 | 2 | | 19 | 19 |
| 6 | 2 | 2 | | 20 | 20 |
| 7 | 3 | 3 | | 33 | 33 |
| 8 | 6 | 3 | 3 | 28 | 28 |
| 9 | 11 | 4 | | 54 | 48 |
| 10 | 4 | 4 | | 48 | 48 |
| 11 | 5 | 5 | | 75 | 75 |
| 12 | 7 | 5 | | 52 | 52 |
| 13 | 6 | 6 | | 102 | 102 |
| 14 | 6 | 6 | | 80 | 80 |
| 15 | 11 | 7 | | 92 | 92 |
| 16 | 19 | 7 | 7 | 84 | 80 |
| 32 (not generated) | 54 | 15 | 15 | 228 | 212 |
| 64 (not generated) | 139 | 31 | 31 | 580 | 528 |

`addj`/`subj` operations at radix 16 fall from 26 to 14. Unfused real additions are equal
for every codelet; unfused multiplies are equal or fewer (radix 9: 36 against 40, below
FFTW's published 40). The two forms compute the same values exactly (the generator verifies
every codelet in 256-bit arithmetic); they round differently, which DESIGN.md 15.5 allows
between library versions. Every target runs the same text, so the bitwise pairs (`avx2`
and `avx512`, `neon` and `neon_fcma`) still agree. Peak live values are unchanged except
where noted in `--table`.

## Method

- Cases: the standard catalogue's tracking c2c rows (`--tier tracking --filter
  '^c2c-[0-9]+-b'`: rows and row batches, forward, double).
- Machines: Xeon Gold 6240 (`ISA=avx512`, the default, and `ISA=avx2`), GCC 14.4, Slurm
  `-c 8` on the shared node; EPYC 7763 (DMOG, `avx2`), GCC 16; Mac (M-series,
  `neon_fcma`), Homebrew Clang. FFTW 3.3.11 at MEASURE alongside for FFTN ÷ FFTW.
- Metric: fbench median paired ratio B ÷ A per case and geometric mean, with A~ (A/A
  control) for the band; `perf stat` port-5 uops per transform at 256 on the Xeon if
  available.
- Decision rule, written before running: adopt B if the x86 geometric mean of B ÷ A
  (Xeon avx512, Xeon avx2, EPYC avx2 together) is ≤ 0.98 with the paired interval clear of
  the A/A band, and no target's geometric mean is slower than 1 + band + 1 %.

## Results

To be filled in.

## Verdict

To be filled in.

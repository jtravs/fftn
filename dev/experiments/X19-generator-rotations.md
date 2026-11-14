# X19: multiplications by i in the generated codelets

- **Status:** done
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

Arms A = main 00a5455, B = 70f4a74 (this branch's generator change). 24 valid rounds each:

| Machine, target | Run | Band | B ÷ A geomean [95 %] | faster / slower | FFTN ÷ FFTW A → B |
|---|---|---|---|---|---|
| Xeon 6240, `avx512` (GCC 14) | 20261114-152415-8058 | ±1.11 % | 0.980 [0.977, 0.984] | 10 / 3 | 1.19 → 1.17 |
| Xeon 6240, `avx2` (GCC 14) | 20261114-154151-9eeb | ±1.67 % | 0.976 [0.970, 0.982] | 9 / 0 | — |
| EPYC 7763, `avx2` (GCC 16) | 20261114-152530-765f | ±1.06 % | 0.971 [0.969, 0.975] | 15 / 3 | 1.36 → 1.32 |
| Mac M5 Max, `neon_fcma` (Clang 23) | 20261114-152124-de44 | ±1.00 % | 0.991 [0.989, 0.994] | 4 / 0 | 1.06 → 1.05 |

FFTN ÷ FFTW is the reciprocal of fbench's W ÷ A geomean (FFTW 3.3.11 at MEASURE, same run)
times B ÷ A. x86 geometric mean of the three: 0.976, each interval clear of its band.

Largest gains are where radix-16 and radix-8 passes dominate: 1024 (Xeon avx2 0.887, EPYC
0.888, Xeon avx512 0.966), 1024 backward and in place (0.87–0.90 on avx2), 128 × 4096
(0.81 Xeon avx512, 0.93–0.94 avx2), 2187 (radix 9: 0.89–0.92 everywhere), 256 (0.92
Xeon avx512, 0.95 EPYC). Slower beyond the band: Xeon avx512 60 (1.027), 1009 (1.058) and
4093 (1.063, Bluestein lengths; on the EPYC 1009 is 0.936 and 4093 0.976, on Xeon avx2 4093
is 0.975); EPYC 60 (1.038), 120 (1.029), 257 (1.027). 60 and 120 are mostly radices 3–5,
whose codelets are unchanged; the Bluestein lengths run power-of-two convolutions, whose
own rows improved on the same machines, and the losses change sign between machines and
targets. Unexplained (code placement is the likely cause); none moves a target's geomean
past the rule, but 1009 and 4093 on the Xeon are worth a recheck in the next x86 run.

Port counters per transform on the Xeon (`perf stat`, `uops_dispatched_port.port_5`,
GCC 14, single case, approximate; script `ports.sh` in the campaign directory):

| n | target | cycles A → B | port 5 A → B | FFTW cycles, port 5 |
|---|---|---|---|---|
| 256 | avx512 | 1067 → 998 | 798 → 723 (75 % → 72 % busy) | 850, 468 (55 %) |
| 256 | avx2 | 1882 → 1841 | 815 → 697 | |
| 1024 | avx512 | 5842 → 5630 | 3507 → 3251 | 4912, 2194 |
| 1024 | avx2 | 8909 → 8080 | 3474 → 2923 | |
| 4096 | avx512 | 41764 → 40360 | 22055 → 21184 | 31864, 18269 |

Port-5 uops fall 4–16 %; FFTN still issues about 1.5 times FFTW's port-5 uops at 256 and
1024: the twiddle multiplies' swaps (R − 1 per twiddle codelet) and the grouped pass's
transposes are untouched by this change and are what remains (split lowering, X1).

Gates: Mac gate (spdx gen build sanitize bench) passed; GCC 16 `-Werror` build on the Mac
and its ISA tests passed; DMOG (EPYC, GCC 16) gen and build stages passed (1087 tests).

## Verdict

Adopted: the x86 geometric mean is 0.976 (rule ≤ 0.98) with every interval clear of the
band, and no target is slower (Mac 0.991). The real-coefficient form is the generator's only
form (`gen/fftn_gen/realify.py`, DESIGN.md 7.2 step 3, gen/README.md step 4); the direct
lowering stays reachable (`build_ops(..., real=False)`) only for the test that compares the
two. The remaining port-5 gap to FFTW is the twiddle swaps and transposes: next is split
lowering for x86 (X1).

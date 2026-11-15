# X22: Intel's icpx: does FFTN build, pass and run fast with it?

- **Status:** done
- **Owner:** icpx session, 2026-11-15
- **Phase / task:** 1 (toolchains, DESIGN.md 14.3)

## Question

Is Intel's oneAPI C++ compiler (icpx, LLVM-based) a supported compiler for FFTN, what must
its build pass to keep the numerical contract (DESIGN.md 15.5), and is it worth
recommending on Intel hardware?

## Hypothesis

icpx is Clang with Intel's middle-end and back-end tuning. FFTN's kernels are written in
intrinsics-style vocabulary code inside ISA regions, where X18 found Clang 23 and GCC 14
within 1 % on x86 (geomean 0.992 on the Xeon), so icpx should land within the noise band
of Clang. The risk is its default floating-point model, which is fast-math-like: FFTN's
`-ffp-contract=off` alone may not give IEEE arithmetic.

## Change under test

Arm A: FFTN built with GCC 14.4 (the usual x86 compiler). Arm B: the same commit built with
Clang 23.1.2. Arm C: the same commit built with icpx 2026.1.1. All Release, no `-march`,
from commit b1969dc (this branch's CMake changes, which add `-fno-fast-math`: a no-op on
GCC and Clang).

## Method

- Compatibility: `scripts/check.sh` stage `build` (Release, warnings as errors, tests on;
  the suites force `avx2` and `avx512` in turn, and `isa.asm_check` disassembles the units)
  with icpx, on main 6844ad0 (no icpx-specific flags) and on this branch. The default
  floating-point model probed with `-###` and `-dM -E`, and MXCSR read at the start of an
  optimised `main()`.
- Speed: fbench, interleaved process rounds, A/A control (A~), defaults for rounds and
  budget, `--max-load 80 --max-busy 60`. Run 1: the standard catalogue's tracking c2c rows
  (`--tier tracking --filter '^c2c-[0-9]+-b'`, double, `avx512` by default). Run 2: X2's
  catalogue, every 7-smooth length 64–1024, batch 1, double and float. Run 3: run 1 with
  every arm forced to `ISA=avx2`.
- Machine: Xeon Gold 6240 (Cascade Lake, AVX-512), Slurm `-c 8` on the shared node,
  loaded by other jobs; one thread.
- Metric: per case, fbench's median paired ratio C ÷ A and B ÷ A; per run, the
  catalogue-weighted geometric mean with its bootstrap interval; C ÷ B from the stored
  per-round timings.
- Decision rule, written before running:
  - icpx is documented as supported if the gate's `build` stage passes with it, with
    whatever floating-point flags FFTN's CMake sets for it (no flags asked of users to
    build FFTN itself).
  - icpx is recommended over GCC and Clang only if its geometric mean is at least 3 %
    faster than both (C ÷ A and C ÷ B ≤ 0.97 with the interval clear of the A/A band) on
    at least one run, and no case is slower than 1 + band + 2 %.
  - Any case where icpx beats both by more than the band is profiled (`perf`, VTune) to
    say whether the difference could be had portably (in the generator, the vocabulary or
    GCC and Clang flags).

## Results

### 1. Compatibility and the floating-point model

icpx 2026.1.1 (Clang 22 based) builds FFTN with no new warning under `-Werror`: the
gate's `build` stage on main 6844ad0 and on this branch compiled every FFTN target
cleanly. GoogleTest's own `-Winline` draws icpx's "unsupported option" note (#10430), a
third-party flag, not ours. `isa.asm_check` passes: icpx inlines everything the
disassembly check requires and puts no instruction of a wider ISA outside its region.
Two changes were needed:

- **The FFTW test oracle** found no C compiler: the driver beside `icpx` is `icx`
  (`cmake/FftnTestDeps.cmake` now maps the name).
- **The floating-point model.** icpx's default (`-fp-model=fast`) passes
  `-funsafe-math-optimizations -mreassociate -fno-signed-zeros -freciprocal-math
  -fapprox-func -fdenormal-fp-math=preserve-sign` to its front end, yet defines neither
  `__FAST_MATH__` nor a non-zero `__FINITE_MATH_ONLY__`, so FFTN's header guard
  (`config.hpp`) does not see it. FFTN's `-ffp-contract=off` turns off contraction only.
  An optimised `main()` compiled this way also sets FTZ and DAZ in MXCSR at start-up (read
  back: `0x9fc0`; `0x1f80` with `-O0`, `-fno-fast-math` or `-fp-model=precise`). With main
  6844ad0's flags, 55 of 1608 tests fail: the vocabulary's reference checks (38, products
  differ in the last bit), Parseval at n = 351 and 375 (RMS 4.27e-15 against a tolerance of
  4.10e-15), twiddle factors 1 ulp from the correctly rounded value (the library's own
  table code, reassociated), the exact-arithmetic and seeded-fault checks, and fbench's
  self-test (`unit_root(1, 6) is inaccurate`).
- **The fix**, portable and warning-free: FFTN's targets get `-fno-fast-math
  -ffp-contract=off` with every GCC-style compiler (a no-op for GCC and Clang), and an
  icpx build adds `-fno-fast-math` to every target, third-party ones included (the tests'
  `main()` is GoogleTest's, and FTZ/DAZ would otherwise be on in every test process) and
  to the FFTW oracle's C flags. `-fp-model=precise -ffp-contract=off` would also give
  IEEE arithmetic, but icpx warns that the second overrides the first
  (`-Woverriding-option`), which `-Werror` turns into an error; `-fp-model=strict` adds
  `-frounding-math` and strict exception semantics, which cost speed and are not needed.
  With the fix all 1608 tests pass on the Xeon (job 15619; `avx2`, `avx512` and scalar
  forced in turn). Users who compile their own `main()` with icpx's defaults still get
  FTZ/DAZ in their process, which no library flag can undo: `docs/conventions.md`
  ("Accuracy") says so.

### 2. Speed

Xeon, Slurm `-c 8` on the shared node, load average 8–17 (other jobs 7–18 cores), no
round under the overload threshold. Ratios are medians of paired per-round ratios; C ÷ B
from the stored timings (`pair.py` in the campaign directory, 400-sample bootstrap).

| Run (ID) | cases | rounds | band | B ÷ A (Clang ÷ GCC) | C ÷ A (icpx ÷ GCC) | C ÷ B |
|---|---|---|---|---|---|---|
| 1: tracking c2c, `avx512`, double (20261115-004902-a9d2) | 35 | 80 | ±1.00 % | 0.975 [0.972, 0.979] | 1.008 [1.004, 1.012] | 1.033 |
| 2: X2 7-smooth 64–1024, b1, `avx512`, double (20261115-012528-0c01) | 105 | 76 | ±1.00 % | 0.895 | 0.896 | 1.001 |
| 2: same, float | 105 | 76 | ±1.00 % | 0.841 | 0.668 | 0.795 |
| 3: tracking c2c, `ISA=avx2`, double (20261115-022627-7d3b) | 35 | 80 | ±1.00 % | 0.998 [0.996, 1.001] | 1.051 [1.048, 1.054] | 1.053 |

- **Double** (runs 1 and 2): icpx is Clang's equal on the X2 lengths (C ÷ B 1.001; 24
  cases below 0.97, 26 above 1.03) and 3 % behind it on the tracking rows, where it loses
  most on primes and Bluestein (17: 1.064 of Clang, 1009: 1.059), 46080 (1.079), the
  batch 8 × 4096 (1.085) and 4096 (1.037; 1.20 of GCC). It beats both GCC and Clang by
  more than the band at two tracking rows only: 128 (0.938 of GCC, 0.974 of Clang) and
  2^22 (0.935, 0.990).
- **AVX2** (run 3, double): icpx is 5 % behind both (1.053 of Clang; 23 of 35 rows slower
  than GCC beyond the band, worst 17: 1.21, 46080: 1.16, 100000: 1.15) and beats neither
  anywhere.
- **Float** (run 2): icpx is much faster: 0.58–0.71 of GCC's time by length band and
  0.77–0.81 of Clang's; 88 of 105 lengths more than 3 % faster than Clang, none slower.
- **Clang against GCC**, a by-product: X18 found them within 1 % on the tracking rows,
  and run 1 agrees (0.975), but on the X2 lengths Clang is 0.86–0.92 of GCC in double and
  0.75–0.88 in float. The X2 lengths are mostly not powers of two, so their passes have
  partial runs (below) and radices 3, 5, 7 whose codelets GCC compiles worse.

### 3. Why icpx wins in float, and whether it can be had portably

`perf stat`, `perf record` (`perf.sbatch`, `hot.py`) and uiCA (Cascade Lake) on single
cases:

- **Partial runs** (the tails of lane and grouped passes, where fewer than C copies
  remain; `passes/access.inl`, `load_part` and `store_part`). n = 105 float is 7 × 15:
  every unit of both passes is partial (s = 7 and 15 against C = 8). Cycles per transform:
  GCC 1369, Clang 1198, icpx 763. The element-by-element tile loop is what differs: icpx
  turns it into masked AVX-512 loads and stores (`vmovups ... {%k1}{z}`; the radix-15 lane
  pass is 827 instructions with 99 stack references), Clang partly (2480 instructions, 247
  stack references), GCC not at all (2821 instructions, 671 stack references: scalar
  stores into a stack tile then a 512-bit load of it, a store-forwarding stall per leg).
  In double C = 4 and the tails are short, which is why double does not show it. **This
  can be had portably**: write partial runs in the vocabulary with explicit masks
  (`_mm512_maskz_loadu_ps` / `_mm512_mask_storeu_ps` from `valid` when the run is
  contiguous, AVX2's `vmaskmov` likewise), keeping the tile loop for strided runs. It is a
  vocabulary change and needs its own experiment (follow-up below); icpx's numbers bound
  what it could give: 0.5–0.7 of GCC's time for float rows whose passes are mostly partial.
- **Register allocation in big codelets.** n = 512 float (32 × 16, full runs only): GCC
  1193 cycles, Clang 1336, icpx 1077. The radix-32 grouped pass's hot loop has the same
  arithmetic in all three (140 add, 77 FMA, 57 mul) and differs in spills: 169 stack
  references (GCC), 215 (Clang), 139 (icpx). At n = 128 double the radix-16 lane loop is
  212 / 184 / 181 instructions (uiCA: 63.8 / 48.0 / 46.5 cycles per iteration); GCC
  spends 49 more integer instructions on addresses and keeps pointers in vector registers
  (23 `vmovq`), Clang and icpx spill to the stack (52 and 35 references). This is the
  compiler's register allocator on a codelet that does not fit in the registers; the
  portable lever is the generator's schedule (fewer live values, X3's open spill
  question), not a compiler flag.
- icpx also replaces `memcpy` with its own `__intel_avx_rep_memcpy` (1.7–2.7 % of samples,
  against glibc's `__memmove_evex_unaligned_erms` for Clang): neutral.
- Code size: icpx's library is 0.84 of GCC's (`libfftn.a` text 25.4 against 30.1 MB); its
  AVX-512 float codelets are half the size, mostly from the masked tails.

## Verdict

**icpx is supported; it is not recommended as the compiler for FFTN; its float advantage
is a portable vocabulary change to make.**

- Supported: with this branch's CMake changes, icpx 2026.1 builds FFTN under `-Werror`
  and passes every test, the disassembly check included, with no flag asked of the user
  for building FFTN itself. The user-side caveat (FTZ/DAZ from an icpx-compiled `main()`)
  is documented in `docs/conventions.md`. DESIGN.md 14.2 and 14.3 updated.
- Not recommended: by the rule, icpx beats GCC and Clang by far more than 3 % on run 2's
  float half, but it is 3 % behind Clang on the tracking rows with `avx512` and 5 % behind with
  `avx2`, and slower than GCC by more than band + 2 % on 13 `avx512` tracking rows (4096:
  1.20). The users' compiler choice stays theirs.
- Follow-up (not started here, for the plan): masked partial runs in the `avx512` and
  `avx2` vocabularies, measured on the X2 float lengths against GCC and Clang; expected to
  close most of icpx's float lead with any compiler. Second: spill-aware ordering in the
  generator for radix 16–32 codelets on x86.
- Lesson: [LESSONS.md](../LESSONS.md) 2.5 (corrected: the compiler matters for tails and
  big codelets) and 6.13 (icpx's floating-point default).

Campaign directory on the Xeon: `~/awork/fftn-icpx` (scripts `gate.sbatch`,
`build.sbatch`, `fb.sbatch`, `perf.sbatch`, `pair.py`, `hot.py`, `mix.sh`).

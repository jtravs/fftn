# X42: a radix-25 codelet and register-fitting radix-32/64 codelets (phase 1 tail)

- **Status:** running
- **Owner:** x42 session, 2026-11-17
- **Phase / task:** 1 (phase 1 exit check, the tail rule; causes 4 and 5); follows X3, X8,
  X27, X38, X39

## Question

Do (1) a radix-25 pass codelet and (2) radix-25, 32 and 64 codelets that keep their live
values within 16 registers make FFTW's plan shapes pay in FFTN: 20000, 25000, 100000, 5000,
15625 on `avx512` (Xeon) and `avx2` (EPYC), and two-pass 512–2048 on `avx2`? The answers
change the generator's catalogue, the `avx2` and `avx512` double radix tables in
`src/planner/rules.cpp`, and which codelet text the x86 targets compile.

## Hypothesis

- **Radix 25.** FFTW plans 25000 = 25 · 5 · 10 · 20, 20000 = 20 · 5 · 10 · 20 and
  100000 = 25 · 20 · 10 · 20 on the Xeon (`t3fv_25`; X39); FFTN, with radix 20 but not 25,
  needs five passes at 25000 (20 · 10 · 5 · 5 · 5) and X39's pass-count proxy puts
  25 · 10 · 10 · 10 at about 1.05 of FFTW against 1.34. The flat radix-25 codelet (Cooley–Tukey
  5 × 5: 352 additions, 184 multiplications, the derived reference; 222 vector operations
  plain, 246 DIF) peaks at 29 live values: it fits AVX-512's and NEON's 32 registers, not
  AVX2's 16.
- **Register fitting.** FFTN's radix-32 and 64 codelets peak at 35 and 67 live values; on
  AVX2's 16 registers the compiler spills them where its allocator chooses (X3 counted 67–681
  stack stores per pass), and X38 measured FFTW's two-pass plans 1.01–1.29 times FFTN's
  three radix-16 passes at 640–2048 on the EPYC, while FFTW itself is 1.13–1.22 times faster
  than FFTN there. A two-level form, R = n1 × n2 with the n1 n2 values staged once through a
  local array behind a compiler fence (each level ordered on its own: radix 64 as 8 × 8 peaks
  at 13 live values, 32 as 8 × 4 at 13, 25 as 5 × 5 at 15), costs 4–12 % more vector
  operations (DIF: 25 275 against 246, 32 253 against 243, 64 657 against 591) and exactly
  R stores and R loads, but no unplanned spill. Expected: radix-32 and 64 passes on `avx2`
  become cheaper than the two radix-4/8/16 passes they replace, so two-pass 512–2048 (FFTW's
  shape) win, by up to the 13–22 % FFTW leads.

## Change under test

Arms are commits of branch `exp/x42-codelets` against A = `origin/main` (4d00f0a):

- **B (radix-25 codelet).** The generator gains radix 25, flat, on every target, a pass radix
  only (never a leaf, as radix 20 in X38): no plan changes by itself (bits identical).
- **B2 (radix-25 sequences).** `avx512` and `avx2` double overrides with a radix-25 pass,
  selected by X2's search driver (`--radices 2-16,20,25,32,64 --require 25`) on the Xeon
  (`ISA=avx512`) and the EPYC (`ISA=avx2`) for the lengths with 5³ or more: 625, 1250, 2500,
  3125, 5000, 6250, 10000, 12500, 15625, 20000, 25000, 50000 (100000 is a grid on both
  machines; its phase of 500 is a candidate only if a phase radix of 25 is screened first).
  A length gets an override when its double winner is the same sequence at batch 1 and 64
  and at most 0.97 of the planner's at both in the confirmation (X38's selection rule).
- **C (two-level codelets on x86).** The generator writes a two-level form of the plain and
  DIF codelets of 25 (5 × 5), 32 (8 × 4) and 64 (8 × 8), which targets whose
  `codelet_registers` is 16 compile instead of the flat form. `avx2` and `avx512` round
  alike and must give the same bits with the same passes (DESIGN 6.2), so both take it
  (`avx512` takes `avx2`'s 16, as it takes its `split_block`); NEON and scalar keep the flat
  forms. Plans do not change; plans with a 25, 32 or 64 pass on x86 change bits (within the
  tests' tolerances).
- **D (re-fitted `avx2` double sequences, 512–4096).** X2's search driver on the EPYC with
  C's build, candidates with radix 32 or 64 (`--radices 2-16,20,25,32,64 --require 32,64`),
  lengths 512, 640, 768, 1024, 1536, 2048, 3072, 4096; overrides by B2's selection rule.

Order of the commits (amended before any timing): B, C, then B2 and D, whose sequences are
searched on C's build, since that is the codelet text both x86 targets then run; C is
judged against B on plans that already take a radix-32 pass on `avx2` (256, 384, 512, 640)
and on forced radix-25/32/64 sequences, B2 and D against C.

No switch: arms are commits, `dev/KNOBS.toml` is untouched. Library text size is recorded
per arm (Mac static library, Clang; x86-64 Linux, GCC 16).

## Method

- Screens (not deciding): `fbench-fftn run --pmu` and fbench runs with forced `RADICES` of
  the two-level and flat builds, the search driver's screening and refinement stages.
- Judging runs: `fbench_remote.py submit --serial`, one thread, interleaved process rounds
  with the A/A control:
  - Xeon Gold 6240 (GCC 14.4, `avx512`, FFTW 3.3.11 AVX-512, MKL): B2 and C against the arm
    before, the exit check's cases (standard catalogue, filter `^c2c-[0-9]+-b`, tier `full`,
    restricted to the affected lengths), then the tracking tier.
  - EPYC 7763 (DMOG, GCC 16.2, `avx2`): B2, C and D against the arm before, the same.
  - Apple M5 Max (Clang, `neon_fcma`): B (no plan change; collateral only) and, if a NEON
    override is tried, the same rule.
- Metric: fbench's median paired ratio arm ÷ arm before per case with its interval; the
  geometric mean over the cases whose plan (or, for C, whose codelet text) changes; FFTN ÷
  FFTW and ÷ MKL before and after.
- **Decision rule, written before any arm is timed.** Judged on the release toolchains (GCC
  on x86-64 Linux, Clang on macOS), per target, in double:
  - B2 and D (overrides), on the target's deciding machine (`avx512`: Xeon; `avx2`: EPYC):
    (a) the geometric mean over the changed target cases is at most 0.97 with its interval
    clear of the band; (b) no case of the judged set or of the tracking tier is slower than
    1 + band + 2 % (a case past it on a loaded machine is re-run once, `--serial`, and the
    re-run decides); (c) a length whose cases are slower than 1 + band at any batch or
    direction loses its override before adoption.
  - C (codelet text, both x86 targets together): adopted if on the EPYC (`avx2`) the
    geometric mean of C ÷ B2 over the judged cases whose plans contain a radix-25, 32 or 64
    pass is at most 1 − band with its interval below 1, and on the Xeon (`avx512`) that
    geometric mean is at most 1 + band with no such case slower than 1 + band + 2 %;
    collateral as (b) on both. If C wins on `avx2` and loses on `avx512` past that limit,
    the pair contract decides: C is rejected as written and the per-target form (the
    contract changed for codelets above 23) is put to the lead, not adopted here.
  - An arm that fails is reported with its numbers and removed; at most three arms per part.

## Results

(pending)

## Verdict

(pending)

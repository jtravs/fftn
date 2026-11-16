# X14: the old targets (`x86_v2`, `avx`, plain `neon`) and operation order without FMA

- **Status:** running
- **Owner:** task-1.3 session, 2026-11-16
- **Phase / task:** 1.3 and the phase 1 exit ("the `x86_v2`, `avx` and plain `neon` targets,
  forced on the same machines, at or below FFTW restricted to the same instruction sets")

## Question

Three decisions:

1. Is the separate `avx` target (256 bit, no AVX2, no FMA) worth keeping over `x86_v2` (128
   bit, SSE4.2) for Sandy Bridge-class processors (DESIGN.md 6.1: "kept if experiment X14
   shows a gain over `x86_v2`")?
2. Do the old targets meet DESIGN.md 1.3's "Older x86-64" row (≤ 0.95 of FFTW built with its
   SSE2 and AVX codelets, best effort) and the phase 1 exit (at or below that FFTW), and
   plain `neon` the exit against FFTW's NEON build?
3. Operation order without FMA: do the targets without FMA need anything beyond the
   generator's complex-register text, whose lowering is already in the non-FMA order, or
   would an FMA-shaped text de-fused ("naive") serve as well?

## Hypothesis

1. On the machines available (Zen 3, Cascade Lake), 256-bit arithmetic runs at full width,
   so `avx` should beat `x86_v2` by roughly the gain of `avx2` over a 128-bit target minus
   FMA: 15–35 % on rows that fit in L1/L2, less on long rows. On a real Sandy Bridge the
   gain would be smaller (256-bit loads take two cycles on its two 128-bit load ports, so
   memory-bound passes gain nothing), which these machines cannot show: no old hardware is
   available (DESIGN.md 1.3).
2. FFTW's SSE2 and AVX codelets keep one complex double per 128-bit lane pair, with about
   three shuffles per twiddle multiply (finding E4), and its non-FMA codelets are its own
   generator's; FFTN's interleaved passes with pre-arranged twiddles need one shuffle per
   multiply. Expected: both old x86 targets at 0.80–0.95 of the restricted FFTW, the tails
   (primes, short rows) as against the full FFTW in the exit check (PLAN Status). Plain
   `neon` takes the FCMLA order with FMLA and is expected near `neon_fcma`'s 0.78 of FFTW.
3. The complex-register lowering never fuses a scale used twice (gen/README.md,
   "Subtleties"), so on an unfused target its real operation count equals the algorithm's
   (a test holds it): it is the non-FMA order. The split lowering is FMA-shaped (tangent
   forms, a product fused into each of its users), so de-fused it costs more multiplies.
   Expected: the split text de-fused needs more real operations than the complex text, and
   split runs, which `avx2` and `avx512` take by default (X1), lose on the targets without
   FMA or gain less.

## Change under test

No library default changes before the verdict: the two new targets are added (task 1.3)
with interleaved passes (no split runs, `split_of`), `avx2`'s pass-cost tables of the
nearest complex count (the fallback in `rules.cpp`, as for any target without its own), no
128-bit leaves (those are `avx2` code) and no pair leaves. The arms differ by the forced
instruction set and, for question 3, by the forced layout:

- `V`: `fftn+ISA=x86_v2`; `A`: `fftn+ISA=avx`; `VS`, `AS`: the same with `LAYOUT=split`
  (split runs where the chain allows them, as on `avx2`);
- `F`: `fftw`, built from FFTW 3.3.11 in the campaign directory with only `--enable-sse2`
  (run 1) or `--enable-sse2 --enable-avx` (run 2), default CFLAGS (no `-march`, so its generic
  code is baseline x86-64), double, MEASURE as fbench runs it;
- on the Mac: `N`: `fftn+ISA=neon` against `F` = FFTW's NEON build.

No switch is added (`ISA` and `LAYOUT` are fbench options of the FFTN backend).

## Method

- Cases: the phase 1 exit set, catalogue `standard`, `--tier full --filter '^c2c-[0-9]+-b'`
  (201 c2c cases, one thread), as the exit runs 20261116-161226-416e (EPYC) and
  20261116-161123-f848 (Xeon); `--min-rounds 6 --max-rounds 12`.
- Machines: EPYC 7763 (DMOG, GCC 16), Xeon Gold 6240 (GCC 14.4), shared nodes, `-c 8`; the
  M5 Max (Clang, NEON FFTW). Run 1 per x86 machine: `V A VS F` with the SSE2 FFTW; run 2:
  `V A AS F` with the SSE2 + AVX FFTW.
- Static count (question 3): real operations per codelet on an unfused target, complex text
  against split text, from the generated metadata (`src/gen/codelets.hpp`).
- Metric: geometric mean over the cases of the per-case median time ratio, and the share of
  cases above 1.05 (DESIGN.md 1.3's tail rule), with fbench's A/A band.
- Decision rules, written before running:
  1. Keep `avx` if `A ÷ V` is ≤ 0.95 (geometric mean) on both machines in both runs;
     otherwise delete it and let Sandy Bridge-class processors take `x86_v2`.
  2. Reported against DESIGN.md 1.3: met if `V ÷ F` (run 1) and the kept target's `÷ F`
     (run 2) are ≤ 0.95 with at most 5 % of cases above 1.05; the exit's own bar is ≤ 1.00.
     Not met is recorded with its causes; it does not block the targets, which are best
     effort.
  3. Split runs become the default on a target without FMA only if `AS ÷ A` (or `VS ÷ V`) is
     ≤ 0.97 on both machines with no more than 5 % of cases above 1.05; otherwise the
     interleaved complex text stays the only text these targets run, and the generator gets
     no unfused split lowering.

## Results

(pending)

## Verdict

(pending)

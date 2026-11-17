# X14: the old targets (`x86_v2`, `avx`, plain `neon`) and operation order without FMA

- **Status:** done (`avx` kept; both old x86 targets at or below the restricted FFTW, `avx` ≤ 0.95, the tail rule failed everywhere; no split runs without FMA; `x86_v2` takes a leaf of 32 in double; see Verdict)
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

### Amendment: three arms for the old targets' weak cases

Written after the Xeon's run 1 showed `x86_v2` at 0.98 of FFTW's SSE2 build with its worst
cases at 32 points (1.56–1.75) and the short rows, and after the first rounds of the EPYC's
run 1 (so not blind to those rounds; the rules are the protocol's usual ones):

- `VL`: `x86_v2` with `LEAF_LIMIT=32` (a leaf of up to 32 points in double, as plain `neon`
  takes from X8; `x86_v2` has C = 1 in double like it). Adopt if the rows of 17–32 points
  gain and the geometric mean over all cases is ≤ 1.00 on both machines with no case
  beyond the A/A band by more than 5 %.
- `VN`: `x86_v2` with plain `neon`'s pass-cost tables (commit d393f1a on
  `exp/x14-v2-neon-tables`) instead of `avx2`'s nearest (DESIGN.md 6.1: the two share a
  shape). Adopt if the geometric mean is ≤ 0.98 on both machines, same collateral rule.
- `AN`: `avx` with 128-bit leaves compiled from `x86_v2`'s vocabulary in the `avx` region
  (commit 984c26b on `exp/x14-avx-narrow`), as `avx2` has its narrow leaves (X28). Adopt if
  the rows up to 23 points gain and the geometric mean is ≤ 1.00 on both machines, same
  collateral rule.

Each is timed in the EPYC's run 1 (FFTW SSE2) and a third Xeon run (arms `A AN VN V VL`).

## Results

### Question 3, static count

Real arithmetic per codelet on a target without FMA (forward, radices 2–16, plain and DIF),
from `src/gen/codelets.hpp`: the complex-register text costs adds + fmas + 2 cmuls additions
and muls + fmas + 4 cmuls multiplications (each `fmadd_r` a multiply and an add); the split
text de-fused costs adds + fmas additions, muls + fmas multiplications and its negations.
The additions are equal in every codelet; the split text needs 0–17 % more operations in
all (radix 9 plain 136 against 116, radix 3 plain 18 against 16, radix 16 plain 176 against
168; radices 2, 4, 7, 11, 13, 14 equal), 3.1 % over the 30 codelets, all of it in
multiplications: the tangent forms c·(u + t·v) and the products fused into several sums
each cost a multiply that the complex text shares. So the complex-register text is the
non-FMA order the design asks for, and a de-fused FMA text ("naive") costs more; whether
split runs still pay for that through their saved shuffles is the timing below.

### Plain `neon` on the Mac

Run 20261116-232302-42c6 (M5 Max, Clang 23, `ISA=neon` forced, FFTW 3.3.11 NEON build, 12
rounds, 8 under load, A/A band ±1.2 %, `N~ ÷ N` 1.004): FFTN ÷ FFTW 0.886 over 193 cases,
57 (30 %) above 1.05. The worst are the exit check's causes (PLAN Status): primes outside
Rader's range (131071 2.49, 999983 2.25, 4093 1.43), short single rows (20 points 1.71, 32
1.36, 2 × 3 in place 1.47), the long powers of two (65536 1.33–1.37); 17- and 289-point
rows 0.19–0.24. For comparison `neon_fcma` was 0.78 of the same FFTW in the exit run.

### The x86 runs

Library commit ac8c5ed (the arms VN and AN their own commits), `-c 8` on shared nodes, 6–12
rounds. Geometric means of the per-case median paired ratio over the 201 cases (193 where
FFTW is compared: FFTW refuses the float rows of the catalogue's double build), and the
share of cases above 1.05:

| run | machine, FFTW | A/A band | `V ÷ F` | `A ÷ F` | `A ÷ V` | split `÷` interleaved |
|---|---|---|---|---|---|---|
| 20261117-004757-3a1d | EPYC 7763, GCC 16, SSE2 | ±1.1 % | 0.957 (33 %) | 0.651 (3 %) | 0.681 (7 %) | `VS ÷ V` 0.976 (0 %) |
| 20261116-234204-8175 | EPYC 7763, GCC 16, SSE2 + AVX | ±1.5 % | 1.322 (70 %) | 0.897 (29 %) | 0.680 (8 %) | `AS ÷ A` 0.977 (0 %) |
| 20261116-232600-139a | Xeon 6240, GCC 14.4, SSE2 | ±3.2 % | 0.984 (46 %) | 0.720 (13 %) | 0.735 (11 %) | `VS ÷ V` 0.991 (1 %) |
| 20261117-002151-a927 | Xeon 6240, GCC 14.4, SSE2 + AVX | ±3.5 % | 1.296 (74 %) | 0.943 (44 %) | 0.731 (10 %) | `AS ÷ A` 0.995 (0 %) |

Every EPYC round ran under load (a shared node; the band stayed small); the Xeon had 3–5 of
12 rounds under load. fbench's weighted geometric means agree to ±0.01.

- **`avx` against `x86_v2`:** 0.68 on the EPYC and 0.73 on the Xeon, in both runs; slower
  only on short rows (7–16 points, 1.09–1.16; float 8 × 2 on the Xeon 1.52), where a leaf
  broadcasts one line into a 256-bit register, and nowhere by more than 1.16 in double.
- **`x86_v2` against FFTW's SSE2 build:** 0.957 (EPYC) and 0.984 (Xeon): at or below FFTW,
  not ≤ 0.95, and a third to a half of the cases above 1.05: 32-point rows and batches of
  them (1.56–1.75; FFTN runs 2 × 16 where FFTW has one 32-point codelet), 20 points (1.59–
  1.69), the primes outside Rader's range (65537, 131071: 2.2 on the EPYC), 1018 (1.63),
  in-place 65536 (1.42–1.55), 100000 on the Xeon (1.70).
- **`avx` against FFTW's SSE2 + AVX build:** 0.897 (EPYC) and 0.943 (Xeon), ≤ 0.95 on both,
  with 29 % and 44 % of the cases above 1.05: batches of 16 and 32 points (1.34–1.66),
  13 and 20 points, 240 (1.42), the primes outside Rader's range, the Xeon's long rows
  (100000 1.83, 131072–262144 1.32–1.35), as against FFTW's full build in the exit check.
- **Split runs without FMA:** 0.976–0.977 on the EPYC, 0.991–0.995 on the Xeon; nothing
  above 1.05, the gains in long in-place rows (0.61–0.73).

The amendment's arms (EPYC run 20261117-004757-3a1d; Xeon run 20261117-014841-2862, band ±2.4 %, 3 of 12 rounds under load, `A ÷ V` 0.739 again):

| arm | EPYC | Xeon |
|---|---|---|
| `VL ÷ V` (leaf of 32) | 0.991, 32-point rows 0.62–0.64, worst 1.05 (65536 backward) | 0.988, 32-point rows 0.58–0.64, nothing above 1.05 |
| `VN ÷ V` (neon's tables) | 1.013, 26 cases above 1.05 (999983 2.05, 10403 1.37) | 1.018, 29 above 1.05 (999983 1.36, 8192 1.29) |
| `AN ÷ A` (128-bit leaves) | 0.986, rows of 11–23 points 0.76–0.82; 61, 100, 120 1.08–1.14 | 0.977, 8–23 points 0.64–0.83, worst 1.06 (24) |

### Library size and build time

GCC 16.2, Release, EPYC node (`size` of the kernel objects, the gate build of ac8c5ed):
`x86_v2` 9.9 MB of text (main and row units), `avx` 15.5 MB, against `scalar` 14.5, `avx2`
14.4, `avx512` 18.1 (`avx` carries masked partial runs with GCC, `x86_v2` none); the x86
library's text grew from 49.3 MB to 74.7 MB (+52 %). Kernel compile time, summed over units
(16 parallel jobs): `x86_v2` 379 s, `avx` 649 s, against 2040 s for `scalar`, `avx2` and
`avx512` together (+50 %); the slowest unit 165 s (`avx` radix 64), below `avx512`'s 208 s.
Each new unit compiles the same rows as an `avx2` unit, so the peak memory per unit, which
the row split keeps within the CI runners' 16 GB, does not rise; the total work does.

The `AN` losses on the EPYC are two-pass rows whose last pass is a leaf over several lines
(100 = 10 × 10, 120 = 10 × 12, Rader's 60 = 6 × 10): `avx`'s own leaf computes two lines per
register there, the 128-bit one a line at a time. `avx2` has the same narrow leaves but runs
such rows as pair leaves (X28b), which `avx` lacks.

## Verdict

1. **`avx` is kept** (rule 1): 0.68 (EPYC) and 0.73–0.74 (Xeon) of `x86_v2`'s time in all
   three runs per machine; Sandy Bridge-class processors get it (dispatch already prefers it).
2. **Against the restricted FFTW** (rule 2): `x86_v2` 0.957 (EPYC) and 0.984 (Xeon) of
   FFTW's SSE2 build, `avx` 0.897 and 0.943 of FFTW's SSE2 + AVX build: both at or below
   FFTW, as the phase 1 exit asks; `avx` meets DESIGN.md 1.3's 0.95 on both machines,
   `x86_v2` does not; the tail rule fails for both (29–46 % of cases above 1.05), with the
   causes of the exit check (primes outside Rader's range, short rows and batches of 16–32
   points, the Xeon's long rows) plus 32-point rows for `x86_v2`, which the leaf of 32 now
   halves. Plain `neon` 0.886 of FFTW's NEON build meets the exit's bar and the 0.90 mean
   DESIGN.md sets for Neoverse (timed on the M5, not a Neoverse core), not its tail rule (30 %
   above 1.05). Best effort: recorded, not blocking.
3. **No split runs without FMA** (rule 3): 0.976–0.995, short of 0.97 on the Xeon; the
   complex-register text stays the only one the unfused targets run, and the generator gets
   no unfused split lowering. The static count agrees: the de-fused FMA text costs 3 % more.
4. Amendment: **`VL` adopted** (`leaf_limit_of`: 32 for `x86_v2` in double, as plain
   NEON; 32-point rows 0.58–0.64, all cases 0.99, nothing above 1.05 beyond the band);
   **`VN` rejected** (1.013–1.018; `avx2`'s tables stay, 16 registers matter more than C);
   **`AN` rejected** by its collateral rule on the EPYC (1.08–1.14 on two-pass rows); a
   follow-up could give `avx` the narrow leaves for whole-row leaves only, or pair leaves.
   The arm branches `exp/x14-v2-neon-tables` and `exp/x14-avx-narrow` stay unmerged (local
   only).

Not measured: real Sandy Bridge or Nehalem hardware (none available: emulation shows
correctness and dispatch, not speed); float (the exit set is double); the old targets'
`rader_below` (NEON's 2¹⁶) and grid-copy prefetch (on, as `avx512`), which follow the
defaults of targets other than `avx2`. LESSONS.md 2.10.

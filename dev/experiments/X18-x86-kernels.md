# X18: why FFTN trails FFTW on x86, and what closes the gap

- **Status:** done (diagnosis; three fixes committed; the larger changes are planned work)
- **Owner:** x86-perf session, 2026-11-14
- **Phase / task:** 1 (exit criteria), 1.5, 1.6, X2, X5, X10

## Question

FFTN beats NEON FFTW on the Mac for smooth c2c rows, but a first one-round fbench run on
the Xeon Gold 6240 (GCC 14) gave FFTN ÷ FFTW (AVX-512, MEASURE) of 2.15 at 64, 1.61 at
256, 1.28 at 1024, 1.10 at 4096, 1.09 at 1000, and 1.5–2.7 with `avx2` forced. Where does
the time go, and what does it take to reach the phase 1 exit (≤ 0.90 of FFTW on Zen 3,
≤ 1.10 of MKL on the Xeon)? The answer orders tasks 1.5 (split lowering, X1), 1.6 (radix 32
and 64, X3), X2's long lengths, X5 (twiddle modes) and X10 (generated whole passes).

## Hypothesis

Candidates, to be ranked by measurement: fixed per-call and per-pass overhead in the
driver; more passes than FFTW (radix ≤ 16); per-pass kernel cost (address arithmetic,
shuffles, tails, spills); memory footprint; code generation (GCC against Clang); the
planner's choice (AVX2 tables fitted on the EPYC up to 1024; AVX-512 on the nearest C).

## Change under test

A diagnosis, with each fix found on the way a separate commit and A/B:

- a344bff `core: no integer division on the serial execute path`
- 9d9d455 `strategies: chains of three passes or more use the destination as scratch`
- 94ef09f `planner: AVX2 double overrides for 4096 and 16384`

## Method

- Cases: `x18.toml` (c2c, double, forward, out of place): rows 64, 128, 256, 512, 1024,
  2048, 4096, 16384, 60, 120, 360, 1000; batches 128 × 4096 and 1024 × 256. The file
  lives in the campaign directories (`~/awork/fftn-x86perf` on the Xeon,
  `/mnt/data/shared/jt52/fftn-x86perf` on DMOG), not in the tree.
- Machines: Xeon Gold 6240 (Cascade Lake, AVX-512, 32 KiB L1D, 1 MiB L2; Slurm `-c 8`,
  shared, loaded by other users outside Slurm); EPYC 7763 (Zen 3, AVX2, 32 KiB L1D,
  512 KiB L2; DMOG node21, `-c 8 --hint=nomultithread`, shared node).
- Compilers: GCC 14.4, GCC 16.2, Clang 23.1.2; Release, no `-march`. Competitors: FFTW
  3.3.11 (Xeon: SSE2 to AVX-512 codelets; EPYC: AVX2) at MEASURE with fresh wisdom; oneMKL
  2026.1 on the Xeon.
- fbench A/B with interleaved process rounds and the A/A band; FFTN arms compared with
  each other from the stored timings (paired per-round ratios, bootstrap interval).
- `perf record`/`annotate` and `perf stat` (instructions, cycles, uops per port, L1 misses)
  on the harness running one case, on the Xeon (`perf_event_paranoid` = 1; DMOG has no
  `perf`). Instruction mix of each hot loop: the innermost backward-branch range around the
  hottest instruction, classified by mnemonic.

## Results

### 1. Where FFTN stands (FFTN ÷ FFTW, median over rounds)

Xeon run 20261114-123623-22f3 (30 rounds, band ±1.85 %) and the three-line run
20261114-124839-874c (24 rounds, load 15–24, band ±1.55 %); EPYC run
20261114-124018-d017 (30 rounds, band ±1.00 %) and 20261114-124830-bc8e (band ±1.10 %).
"Before" is main 9af469e with GCC 14; "after" is 9d9d455 (both fixes) with GCC 14 and
Clang 23.

| case | Xeon before | Xeon after GCC / Clang | EPYC before | EPYC after GCC / Clang |
|---|---|---|---|---|
| 64 | 2.22 | 1.30 / 1.40 | 1.54 | 1.34 / 1.34 |
| 128 | 1.76 | 1.33 / 1.25 | 1.47 | 1.38 / 1.32 |
| 256 | 1.56 | 1.31 / 1.18 | 1.52 | 1.43 / 1.40 |
| 512 | 1.62 | 1.29 / 1.39 | 1.55 | 1.47 / 1.41 |
| 1024 | 1.32 | 1.17 / 1.17 | 1.38 | 1.35 / 1.42 |
| 2048 | 1.19 | 1.19 / 1.17 | 1.28 | 1.28 / 1.31 |
| 4096 | 1.08 | 1.03 / 1.07 | 1.52 | 1.49 / 1.43 |
| 16384 | 1.65 | 1.35 / 1.39 | 1.62 | 1.57 / 1.65 |
| 60 | 2.19 | 1.60 / 1.43 | 1.54 | 1.33 / 1.23 |
| 120 | 2.25 | 1.63 / 1.57 | 1.35 | 1.24 / 1.18 |
| 360 | 1.11 | 0.88 / 0.89 | 1.04 | 0.96 / 0.98 |
| 1000 | 1.12 | 1.04 / 1.06 | 1.04 | 0.99 / 0.95 |
| 128 × 4096 | 1.58 | 1.70 / 1.45 (noisy) | 1.28 | 1.24 / 1.15 |
| 1024 × 256 | 1.13 | 1.04 / 1.03 | 1.20 | 1.19 / 1.26 |
| geomean | 1.50 | 1.25 / 1.23 | 1.37 | 1.29 / 1.28 |

MKL ÷ FFTW on the Xeon is 0.95 (geomean; 0.80 at 64, 0.87 at 4096), so the MKL exit
(≤ 1.10 of MKL) means about ≤ 1.05 of FFTW there. Forced `avx2` on the Xeon was 2.00 (GCC
14, before) and 1.70 (Clang, after the first fix). The exits are 18 % (Xeon) and 30 %
(EPYC) away.

### 2. Ranked diagnosis

1. **Integer division in the serial driver (fixed, a344bff).** `run_stage_chunks` found each
   chunk's ends with `chunk_start` (four 64-bit `div` per chunk), the batch loop divided once
   per transform, and lane passes divided their first unit. A 64-bit `div` costs 40–90
   cycles on Cascade Lake: at 64 points `perf annotate` put 31.5 % of the time in
   `run_stage_chunks`, almost all on three `div` instructions, plus 7.6 % in
   `run_transforms` (one `divq`) and 12.6 % of the radix-16 lane pass's samples on its `div`.
   After the fix (paired, same run): Xeon GCC 14 0.855 geomean (64: 0.589, 120: 0.722,
   256: 0.826, 128 × 4096: 0.902), Clang 0.903; EPYC GCC 0.955, Clang 0.982 (Zen 3 divides
   fast). Bits unchanged: chunks are adjacent ranges of independent units.
2. **The AVX2 kernels execute too many instructions.** *Correction (X10): FFTW's plans on
   the Xeon mix widths: its untwiddled, transposing passes (`n2fv_16`, `n2fv_32`, `n2fv_64`)
   run the AVX2 codelets on `ymm`, its twiddled passes (`t2fv_16`, `t3fv_32`, `t2fv_64`) the
   AVX-512 ones on `zmm`. The first version of this item read FFTW's other copies of the
   same symbol names and called the comparison same-width; FFTW's 2340 below includes a
   `zmm` pass. The per-iteration rows for FFTW are its AVX2 (non-FMA for `n2fv_16`) copies:
   valid as a static comparison of AVX2 loops, not as what FFTW ran. What FFTW ran at 256:
   `n2fv_16` on `ymm`, 175 instructions per two butterflies, and `t2fv_16` on `zmm`, 207 per
   four (X10 has the per-transform counts).* At 256 FFTN with `ISA=avx2` executes 5386
   instructions per transform (default plan 2.8.16) against FFTW's 2340, and takes 1789
   cycles against 849 at about 3.0 instructions per cycle, near the 4-wide rename limit;
   no port is saturated (p0 942, p1 987, p5 701 uops). With the same plan, 16 × 16, per loop
   iteration of two radix-16 butterflies:

   | loop (ymm, 2 butterflies) | instr. | add/sub | FMA | mul | shuffle | loads (+ mem operands) | stores | scalar |
   |---|---|---|---|---|---|---|---|---|
   | FFTW `n2fv_16` (no twiddles, transposing store) | 176 | 72 | 0 | 12 | 23 | 20 (+28) | 20 | 21 |
   | FFTW `t2fv_16` (twiddles, in place) | 222 | 60 | 27 | 19 | 22 | 22 (+35) | 25 | 34 |
   | FFTN grouped 16 (twiddles + transposes), GCC 14 | 319 | 46 | 30 | 27 | 50 | 51 (+42) | 51 | 55 |
   | FFTN grouped 16, Clang 23 | 259 | 46 | 30 | 27 | 55 | 28 (+35) | 27 | 39 |
   | FFTN last 16 (plain), GCC 14 | 211 | 46 | 15 | 12 | 19 | 19 (+12) | 19 | 72 |
   | FFTN last 16, Clang 23 | 204 | 46 | 15 | 12 | 23 | 20 (+4) | 20 | 60 |

   The arithmetic is the same (FFTN's `addj`/`subj` are the FMA column); the excess is
   (a) address arithmetic, 99–127 scalar instructions per pair of passes against 55:
   the 32 leg offsets of a radix-16 unit do not fit 16 GPRs, so GCC recomputes and spills
   them each unit (and on AVX-512 parks them in `xmm` registers, 23 `vmovq` per unit on
   port 0) and Clang reloads them from the stack; (b) spills of vector values on 16
   registers (radix 16 has 18 live values): GCC stores 51 times where 16 are outputs;
   (c) shuffles, 69–78 against 45: the grouped pass does the twiddle multiply and the
   output transposes together (FFTW's DIT plan puts the transposes in the untwiddled
   pass), and the plain radix-16 codelet uses 19 shuffles where FFTW's core needs about 7.
3. **Plan choice on AVX2 above 1024 (two lengths fixed, 94ef09f).** X2's table was fitted up
   to 1024. On the EPYC (Clang build, forced sequences against the default): 4096 8.8.8.8
   0.753 (default 4.8.8.16), 16384 8.8.16.16 0.843 (default 4.16.16.16), 1024 8.8.16 0.903
   (default 4.16.16, X2's measured winner with GCC 16 and batches 1 and 64), 256 16.16 0.941
   (default 2.8.16); runs 20261114-125134-ac87, -125200-05ea, -125215-cc40, -125232-d808.
   The pattern is the register pressure of radix 16 on 16 registers. With the two overrides
   (94ef09f; run 20261114-125833-44b1, 6 rounds, band ±1.00 %), paired against 9d9d455:
   4096 0.751 (GCC 14) and 0.723 (Clang), 16384 0.895 and 0.840; FFTN ÷ FFTW at 4096 is
   then 1.14 (GCC) and 1.07 (Clang), at 16384 1.39 and 1.41. On AVX-512 the planner
   is not the problem: no forced sequence beat the default by more than 4 % at 64, 128, 256,
   512, 1024, 16384, 60 or 120 (runs 20261114-124216-c5cc to -124341-0867), and X16's search
   (campaign xeon-avx512-smooth) puts planner v1 within 0.6–2.1 % of the best sequence for
   7-smooth n ≤ 1024. On AVX2 the misses are above X2's range (1024 apart).
4. **Fewer, larger passes in FFTW.** FFTW's plans (Xeon, `perf report`): 64 = 32 × 2,
   256 = 16 × 16, 1024 = 32 × 32 (`n2fv_32` + `t3fv_32`), 4096 = 64 × 64, 1000 = 20 × 10 × 5.
   FFTN's radix ≤ 16 needs 3 passes at 512–4096 and 4 at 16384. At 4096 on the Xeon the per
   pass, per point cost is lower in FFTN (about 0.65 ns against 0.85 for FFTW's radix-64
   passes, from `perf` shares) but per bit of the transform higher (about 0.16 against 0.14 ns
   per point per log₂ R).
5. **Memory footprint.** A chain touched four lines (source, a, b, destination) and its first
   pass reads a twiddle table of (R − 1) n / R entries stored as 4 reals each: about twice
   the data. At 1024 that is 64 KiB of lines and 28 KiB of twiddles against FFTW's in-place
   second pass with derived twiddles (`t3fv`); FFTN took about three times FFTW's L1 load
   misses per transform (`perf stat`). Using the destination as the second scratch line
   (9d9d455) gave, paired: Xeon 512 0.87, 1024 0.90–0.94, 16384 0.85–0.87, 1024 × 256
   0.92–0.96, 360 0.95, others within the band (geomean 0.972 GCC, 0.957 Clang); EPYC 512
   0.96, 16384 0.96–0.97, 360 0.95–0.96, geomean 0.987.
6. **Port 5 on AVX-512.** At 256 the AVX-512 grouped pass keeps port 5 74–75 % busy (803
   of 1092 cycles, GCC; 723 of 958, Clang): its 66–70 shuffles per unit (32 for the 4 × 4
   transposes of 128-bit units, 15 twiddle swaps, 19 in the codelet) compete with half of
   the 512-bit FP throughput. FFTW's grouped work at 256 is 77 ns (`t2fv_16`) + 111 ns
   (`n2fv_16`); FFTN's is 175 ns (grouped) + 98 ns (plain lane pass).
7. **Compiler: not the cause.** After the division fix Clang 23 ÷ GCC 14 is 0.992 on the
   Xeon and 0.989 on the EPYC (geomean), Clang ahead by 5–12 % at ≤ 256 and on the short
   batch, GCC ahead by 3–7 % at 512–4096. Before the fix Clang was 0.937 on the Xeon: most
   of that difference was code around the divisions. GCC 16 ÷ GCC 14: 1.000 (Xeon), 1.012
   (EPYC). On the Mac (lead's run) GCC 16 is 1.07 behind Clang, with outliers at 4096 and
   1009; neither shows on x86 (4096: Clang ÷ GCC 1.016 Xeon, 0.956 EPYC).

### 3. Short rows: fixed cost per call, per transform, and in the kernel

Runs 20261114-130648-8888 (Xeon, band ±1.26 %) and 20261114-130701-40c6 (EPYC, band
±1.15 %), commit 9d9d455 (EPYC: 94ef09f), GCC 14; ns per transform, batch 1 / batch 64:

| n | Xeon FFTN | Xeon FFTW | EPYC FFTN | EPYC FFTW |
|---|---|---|---|---|
| 8 (leaf) | 26.5 / 22.2 | 10.1 / 4.6 | 21.8 / 17.6 | 9.6 / 4.0 |
| 16 (leaf) | 43.2 / 38.7 | 21.6 / 12.0 | 32.7 / 28.9 | 16.2 / 9.7 |
| 32 (2 passes) | 56.4 / 54.3 | 45.3 / 29.5 | 58.6 / 53.1 | 39.0 / 25.2 |
| 64 (2 passes) | 83.5 / 83.9 | 64.9 / 75.8 | 93.1 / 91.8 | 70.0 / 60.2 |

- **Fixed cost per call** (plan entry to the first stage: batch 1 minus batch 64 per
  transform) is about 4 ns on both machines: small.
- **Driver cost per transform** (`run_transforms`, `moved_in_nest`, `run_records`,
  `run_stage_chunks` with its copy of `pass_args` and one indirect call per stage) is paid
  for every transform of a batch: 16 % of a 64-point transform at batch 64 on the Xeon
  (`run_stage_chunks` 7.7 %, `run_transforms` 4.4 %, `moved_in_nest` 3.9 %; about 13 ns),
  23 % of a 16-point leaf at batch 1 (about 9 ns).
- **The leaf kernel** broadcasts each input into every lane, computes the same transform C
  times and stores lane 0 through a stack tile: 65 % of the 16-point time. FFTW's `n1fv`
  vectorises across the transforms of a batch (4.0–4.6 ns per 8-point transform at batch
  64, a quarter of FFTN's) and is 2.2–2.4 times faster at batch 1 too.
- At 32 and 64 the two passes are 78–82 % of the time; their per-point cost is item 2 above
  (the AVX2 loops) and the radix-4 grouped pass at 64.

So at n ≤ 64 the work is, in order: leaves that use the lanes (C transforms per register in a
batch, task 2.2; a SIMD layout inside one transform at batch 1, X8), a per-transform driver
that does not re-walk the program (a precompiled stage list per line, `pass_args` by
pointer), then the pass loops of item 2.

## Verdict

Diagnosis done. Committed: the division fix (adopted: 0.855–0.903 on the Xeon, 0.955–0.982
on the EPYC, no case slower), the three-line chain (adopted: up to 0.85 at 512–16384,
nothing slower than the band), and two AVX2 overrides (adopted: 0.72–0.75 at 4096,
0.84–0.90 at 16384 on the EPYC; they change only those two lengths on targets that use the
AVX2 double table). With the two kernel-independent fixes FFTN ÷ FFTW is 1.23–1.25 on the
Xeon and 1.28–1.29 on the EPYC (geomean of the 14 cases; about 1.25 on the EPYC with the
overrides).

Recommendation, in order (expected gains from the measurements above):

1. **Lane-pass addressing and register budget (X10, generated whole passes; days).** Give
   every pass loop a fixed address scheme: a per-pass table of leg offsets read as memory
   operands (FFTW's scheme) or base pointers bumped per unit, never 2R live offsets; and a
   register budget per target in the generator's schedule (spill-aware ordering for 16
   registers, radix-16 codelets split into two radix-4 halves on AVX2). Target: the
   per-iteration instruction count of FFTW's loops (−25 to −35 % on AVX2 loops, where the
   machine is front-end bound). Expected: EPYC 1.28 → about 1.0, Xeon `avx2` similar.
2. **Radix 32 and 64 on AVX-512 (task 1.6, X3).** 1024 and 2048 in two passes, 16384 in
   three. Expected on the Xeon: 512–2048 and 16384 from 1.17–1.39 to about 1.0, i.e. the
   MKL exit there.
3. **Split lowering for x86 (task 1.5, X1), after 1 and 2.** It removes the twiddle swaps
   and the `addj`/`subj` shuffles (34 of the 66–70 port-5 uops per AVX-512 grouped unit),
   and in a four-step layout moves the transposes to one pass. It does not fix 2 (a)–(b),
   which is why it comes second; the prototype's split four-step reached 0.93–1.03 of FFTW
   on these machines. Plan: the generator's split lowering for `avx2` and `avx512` with the
   `BL` layout of DESIGN.md, a four-step driver for the middle passes only (first and last
   pass convert), measured per pass position as X1 specifies.
4. **Twiddle footprint (X5).** Compact (2-real) or derived twiddles for the first pass's
   table; a check against item 1 on Intel, where the extra shuffles go to port 5.
5. **X2 over the lengths above 1024 on both AVX2 and AVX-512 machines**, which replaces the
   two overrides; it should include 1024 (8.8.16 measured 0.90 of X2's winner with Clang).
6. **Whole-transform leaves (X8)** for n ≤ 128, where the driver and the second pass are
   most of the gap.

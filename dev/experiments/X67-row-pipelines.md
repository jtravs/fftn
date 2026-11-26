# X67: row convolution pipelines against three separate calls and FFTW with a loop

- **Status:** running. The decision rules were drafted with 5.2's design (2026-11-25), before
  anything was built, and adjusted before the first judging run (2026-11-26, "Amendments before
  the first judging run" below, each with its reason).
- **Owner:** task AF (5.2g), 2026-11-26 (overnight)
- **Phase / task:** 5.2g, on 5.2a–5.2f as built

## Question

Does a row pipeline (DESIGN 12.2) meet step 5.2's acceptance, at or below 0.8 of three separate
calls for L2-resident sizes? Which pair should the pipeline rules choose at each length and
machine (chain, depth-first or lanes), and which fused middle variants pay?

## Hypothesis

- **Saved traffic.** Three separate calls write the spectrum to memory and read it back.
  - At 2m = 18432 (`<T, C, T>`, double), the spectrum is 147 KB per row. Inside L2 its round
    trip costs about two sweeps of the line, against the eight to ten passes of the two
    transforms' chains. The user's multiply loop adds a third sweep.
  - The pipeline saves those sweeps and fuses the multiply into a pass. It also saves the
    caller's zero-padding and cropping loops where the separate calls lack runs.
  - The expected result is 0.70–0.85 of S's time inside L2, and more where the spectrum spills
    (beyond L2 each saved sweep is a DRAM sweep).
- **Fan-out.** It saves F − 1 forward transforms against S only if S also reuses its spectrum,
  which S does (see the arms). The gain per copy is then the spectrum's round trip and the
  multiply loop, as above.
- **FFTW.** FFTW with a loop pays the same sweeps as S, so the pipeline should beat it wherever
  FFTN's rows match FFTW's (the client catalogue: 0.85–1.00 of FFTW at one thread on 2026-11-24, PLAN "Status").
- **Depth-first pair.** From X50 it should win over the chain pair where a row of 2m is
  depth-first by X50's rules: the Xeon 6240's long convolutions, and Apple's beyond 2 L1.

## Change under test

There is no single library change: X67 judges the built pipelines against user code, and
compares the strategies inside FFTN by forcing them (`build_settings`). Arms, in one fbench
session per machine:

- **P:** the pipeline as the rules (or X67's candidate rules) build it, one execution per
  batch.
- **S:** three separate calls, the code a user writes with FFTN.
  - A forward plan with the input run (5.1d) writes the spectrum to a buffer.
  - The user's multiply loop runs over the spectrum, written in the style of X65's U arm:
    plain C++, vectorised by the compiler, cloned for AVX2 and AVX-512 on x86-64 Linux.
  - An inverse plan with the output run follows.
  - For fan-out F: one forward, then F times the loop (into a second spectrum buffer) and the
    inverse.
- **S_e** (informative): S with the multiply as the inverse plan's input edge (plans with
  edges, 5.1).
- **W:** FFTW 3.3.10, built as its best users build it (SIMD and FMA options; the NEON build on
  the Mac), at MEASURE, with fresh wisdom per campaign. It uses r2c and c2r (or dft) many-plans
  over the rows, the user's zero-padding, multiply and cropping loops, and the same fan-out
  scheme as S. PATIENT is informative.
- **Inside FFTN, forced:** the chain pair (C), the depth-first pair (D) and, on strided lines,
  the lanes pair (L). The middle of `<T, C, T>` runs as an edge stage (M0) or fused into the
  pairing passes (M1, if built).

## Method

- **Cases:** a catalogue `x67` (`bench/catalogues/x67.toml`, written with the fbench pipeline
  options before any run):
  - kinds `<T, C, T>` (primary) and `<C, C, C>`;
  - 2m = 18432, 24576, 32768, 49152, 65536, 98304 and 131072;
  - valid halves: the input run is m points from logical 0, and the output run is m points from
    logical m/2 (the "same" half for a centred kernel of m + 1 taps). The full line without
    runs is informative;
  - fan-out 1, 4 and 16, with a kernel per copy;
  - 1 and 32 rows, with the kernel shared by the rows;
  - double and float.
- **Collateral** (that batched forms hold, X65): `<C, C, C>` and `<T, C, T>` rows of 64, 256,
  1024 and 4096 in batches of 16 to 1024 rows. Also `<C, C, C>` on strided lines (the lines of
  a 64 × 512 and a 512 × 512 plane) for the lanes pair.
- **L2-resident**, fixed per machine in this file before the judging run: a case whose S
  working set per row (stored input, spectrum and stored output, in bytes) is at most the L2
  of one core in the machine's profile. On the Mac that is the cluster's L2 divided by its
  cores.
- **Machines:**
  - the Mac M5 Max: on AC, one fbench at a time, `pmset -g batt` before and after each run;
  - the Xeon 6240 (`--serial`, its load recorded);
  - the EPYC 7763 (DMOG, a small shared allocation, never `--exclusive`).

  GCC 16 on Linux and Clang on macOS (the release toolchains). One thread, and the machine's
  full width (the client's widths where the client catalogue lists them).
- **Metric:** fbench's median paired ratios P/S, P/W, D/C, L/C and M1/M0 per case, with 95 %
  intervals and an A/A control for the band.
- **Decision rules,** written before running:
  1. **Acceptance (step 5.2).** At one thread, on each machine and in each precision, the
     geometric mean of P/S over the L2-resident cases is at most 0.80. No L2-resident case's
     ratio lies above 0.90 beyond the band.
  2. **No loss.** No case, at either thread count and in the collateral set included, has P/S
     above 1 beyond the band. A case that loses is fixed within 5.2g, or, with the
     maintainer's agreement, recorded as a lead together with its cause.
  3. **Depth-first pair.** Per machine and precision, the rules take the depth-first pair from
     the smallest measured length at which D/C is below 1 beyond the band at that length and
     every longer one. The limit becomes a profile field and reaches other machines by the
     cache-key rule (AGENTS.md section 6).
  4. **Lanes pair on strided lines.** The rules take the lanes pair where L/C is below 1 beyond
     the band, up to the longest measured length at which it still is.
  5. **Fused middle (M1)**, if built. Adopt it per target when the geometric mean of M1/M0
     over the `<T, C, T>` cases is at most 0.95 beyond the band and no case is slower beyond
     the band. This is X65's P2 rule.
  6. **FFTW.** Any case with P/W above 1.05 × (1 + band) (DESIGN 1.3's tail rule) is recorded
     as a lead for the performance phase. It does not block 5.2's acceptance: features come
     first (the maintainer, 2026-11-25).
  7. **Fan-out** (a check). If the geometric mean of P/S at F = 4 or F = 16 is above that at
     F = 1 beyond the band, the copies' layout becomes a lead before adoption.
  8. **Regression.** Plans without edges and without pipelines are unchanged: the `describe()`
     and `workspace_bytes()` probes are identical, and the plain driver functions disassemble
     identically.

  Near misses are re-run (AGENTS.md section 6).

## Amendments before the first judging run (task AF, 2026-11-26)

Written and committed before any judging run. Each changes the drafted method or rules above;
where the two disagree, these hold.

1. **Machines.** The Xeon 6240 (`--serial`, its load recorded) and the EPYC 7763 (DMOG, a
   small shared allocation, never `--exclusive`) tonight; the Mac leg is a daytime follow-up
   (the maintainer's quiet-Mac rule for the night). Why: the brief for the night. A rule measured
   on one vendor is keyed to that vendor (the Xeon's class and the EPYC's class), and reaches
   other machines only by AGENTS.md section 6's cache-key rule.
2. **FFTW** is the build each machine has: 3.3.11 (AVX-512 build on the Xeon, GCC 16 build
   with AVX2 on the EPYC), at MEASURE with fresh wisdom per run. PATIENT is dropped. Why: these
   are the installations every FFTN comparison on these machines uses; PATIENT is informative
   only and would double FFTW's planning time at the long lengths.
3. **S_e is dropped.** Why: it is informative only, and the inverse plan's input edge is
   already judged against the user's loop by X65; P/S decides the acceptance.
4. **The arms as fbench arms** (one build of FFTN, the options of bench/README.md, "FFTN
   options"):
   - P: `fftn+PIPELINE=fused` (the rules); C, D, L: the same with `STRATEGY=chain`, `depth` or
     `lanes` (forced where the pair can be built; each case's note names the pair built);
   - S: `fftn+PIPELINE=separate`; W: `fftw+PIPELINE=separate`;
   - the valid halves with `PIPE_RUNS=valid` on every arm, fan-out with `PIPE_FANOUT=F`.

   With runs or fan-out, S and W keep a spectrum buffer of their own (the forward plan with
   the input run into it, the multiply loop in place, or into a second buffer per copy for
   fan-out, then the inverse plan with the output run); W, which has no runs, zero-pads each
   line into a buffer of 2m before its forward plan and crops the inverse's line into the output
   (the user's loops). Without runs and fan-out, S and W are 5.5's: no buffer of their own.
5. **Fan-out cases** count the copies in the case's batch: a case of batch R F is R lines of
   F copies each, line r read from the case's input row r F and copy k written to output row
   r F + k (`fanout_out_dist` one output row, the lines F rows apart), each copy with its own
   kernel array. fbench checks every copy against its line delayed by one element.
6. **The catalogue `x67`** (bench/catalogues/x67.toml), trimmed from the drafted product (336
   cases, among them 16 copies of 32 lines of 131072 points, 1 GiB of output per case):
   - primary: `<T, C, T>` (r2c cases) and `<C, C, C>` (c2c forward cases), 2m = 18432, 24576,
     32768, 49152, 65536, 98304 and 131072, valid halves, fan-out 1, 1 and 32 lines, double and
     float (56 cases);
   - fan-out: F = 4 and 16 at one line, double, 2m = 18432, 32768, 65536 and 131072 (16);
   - full lines without runs (informative): one line, double, the same four lengths (8);
   - collateral: c2c and r2c rows of 64, 256, 1024 and 4096 in batches of 16 and 1024, double
     (16); `<C, C, C>` on the lines of a 64 x 512 and a 512 x 512 plane (`layout =
     "interleaved"`: lines of 64 and 512 points, 512 adjacent lines), double and float (4).

   Why: fan-out is a check (rule 7), and full lines are informative; the trimmed set keeps every
   length and both precisions for the rules that decide.
7. **L2-resident cases**, fixed here per machine: the valid-half cases of fan-out 1 whose S
   working set per line (stored input m, spectrum and stored output m: 64m bytes for `<C, C, C>`
   double, 32m float; 32m + 16 for `<T, C, T>` double, 16m + 8 float) is at most one core's L2.
   - Xeon 6240 (1 MiB): `<C, C, C>` double 2m ≤ 32768, float ≤ 65536; `<T, C, T>` double
     ≤ 49152, float ≤ 98304.
   - EPYC 7763 (512 KiB): `<C, C, C>` double none, float ≤ 32768; `<T, C, T>` double ≤ 24576,
     float ≤ 49152.

   Rule 1 judges each machine and precision over these cases of both kinds and both line
   counts; a precision with no such case of a kind judges the other kind alone.
8. **Runs and threads.**
   - R1, the pairs (one thread, the code as merged, before any rule): A = C, then D, L, S
     and W. Every pipeline by the merged rules takes the chain pair, so C is today's P, and R1
     gives D/C, L/C (rules 3, 4) and today's P/S and P/W.
   - R2, deciding (one thread, the final code with the adopted rules): A = P, then S, W and C.
     It decides rules 1, 2, 6 and 7 and shows every case the rules do not move on its chain pair.
   - R3 (eight threads, the final code): A = P, then S, on the cases of 32 lines and the fan-out
     cases, for rule 2 at full width (the client's width on both machines, PLAN "Status").
     fbench's ratios are arm ÷ A; P/S is the inverse of S/P, its interval inverted.
9. **Rule 3 (depth-first pair)** is judged on the `<C, C, C>` valid-half cases of fan-out 1,
   both line counts: D/C below 1 beyond the band at a length means for both. The limit is a
   machine-profile field in bytes of the complex line (n times the complex value's size), one
   per machine class; where the two precisions' limits differ in bytes, the field takes the
   larger, and R2 must show no case it moves slower than C beyond the band (else it is narrowed
   per precision, AGENTS.md section 6). `<T, C, T>` has no depth-first pair (5.2d), so its cases
   stay on the real chain pair whatever rule 3 says.
10. **Rule 4 (lanes pair) and bit agreement** (5.2f's review, DESIGN 6.2): a lanes pair taken
    by a threshold that differs between the `avx2` and `avx512` targets (their profiles differ)
    is taken only for pipelines whose middle the chain pair also runs as a separate pass (no
    complex factor of stride 1 along the line, no middle callback with `first` backward), where
    the two pairs give the same bits; pipelines whose kernel the chain pair fuses keep the chain
    pair, unless a fused tile pass is built and adopted (not built: rule 5's fused middles are
    not built either). fbench's kernel is complex of stride 1, so the collateral cases measure
    L/C with that kernel; the rule then extends to the separate-pass middles only if L/C wins
    on those cases, the tile's pass being the same pass with either kernel stride.
11. **Rule 5 (fused middles)**: neither M1 nor a fused tile pass is built, so rule 5 does not
    apply; each is a lead for the performance phase with R1's evidence.
12. **Measured planning and wisdom** (5.2g's row): the candidates (the rules' pair, the chain,
    depth-first and lanes pairs where each builds) are timed on scratch data and scratch kernel
    arrays, the way plans with edges are, and recorded in wisdom keyed by the description. They
    change no pipeline built by the rules (rule 8's probes), so they take no judging run of
    their own; a test shows the candidates and the wisdom round trip.

## Results

Not run.

## Verdict

Not run.

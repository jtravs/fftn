# X67: row convolution pipelines against three separate calls and FFTW with a loop

- **Status:** planned. The decision rules were drafted with 5.2's design (2026-11-25), before
  anything was built. The agent who runs X67 claims it here and may adjust the rules only
  before the first judging run, stating each change and why.
- **Owner:** unclaimed (task 5.2g)
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

## Results

Not run.

## Verdict

Not run.

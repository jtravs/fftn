# X67: row convolution pipelines against three separate calls and FFTW with a loop

- **Status:** done (2026-11-27). Step 5.2's acceptance is not met on either machine. Adopted: the
  depth-first pair on the Xeon 6240's class and the lanes pair for a plane's strided lines on Zen
  3's and the Xeon 6240's classes, each narrowed to what was measured. Rule 2's losses are leads
  for the maintainer. The rules were drafted with 5.2's design (2026-11-25) and amended before the
  first judging run ("Amendments before the first judging run" below, each with its reason).
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
   - The runs and fan-out are arm options, so each of R1–R3 is one fbench run per option set,
     on the catalogue's groups as its header names them (valid halves; fan-out 4; fan-out 16;
     whole lines with the collateral cases), one after another in the same Slurm chain.
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

## R2 declared deciding (2026-11-27, before the Xeon's R2 started)

R2 judges the rules as adopted from R1, narrowed (AGENTS.md section 6) to what R1 measured: the
depth-first pair (Xeon 6240's class, avx512, 288 KiB to 2 MiB, a kernel of stride 1 or none, at
most one run per side, no prologue or epilogue factor, accumulation or callback) and the lanes
pair for a plane's strided lines of 64 to 512 points (Zen 3's class and the Xeon 6240's). It is the
deciding run of amendment 8 and of the narrowing: every case the rules exclude must show its
chain pair, and no case they move may be slower than C beyond the band. The Xeon's R2 runs
commit a2378229 (the final rules). The EPYC's R2 started at d8fc1250, whose programs for every
x67 case are the final code's: the later commits only exclude adjacent lines under 64 points
(none in the catalogue), add the Xeon's lanes field and pin a test.

## Results

Ratios are arm ÷ the run's first arm (fbench's medians, 95 % intervals in brackets); P/S is the
inverse of fbench's S/P (or S/C in R1, where C is today's P). Every band below is the A/A control's,
1.0 % (its floor) in every R1 run. GCC 16.2 (EPYC) and GCC 14.4 (Xeon), Release, one thread; FFTW
3.3.11 at MEASURE with fresh wisdom per run.

### R1, the pairs (commit f246ac76, the code as merged before any rule)

**EPYC 7763** (DMOG node20, shared Slurm, `-c 8`, never `--exclusive`; avx2; no round loaded). A
first attempt on node16 (load 91 from other users' builds, band ±23.5 %) was cancelled after 5
rounds and is not used. Runs: valid halves 20261127-001033-14f4, fan-out 4 20261127-005940-2ae5,
fan-out 16 20261127-010514-2eee, whole lines and collateral 20261127-011234-9a31 (80 rounds each).

- **D/C** (rule 3): faster beyond the band only at 98304 (0.84–0.87 on two of its four cases), with
  131072 slower again (1.45–3.14); elsewhere no evidence or slower (valid halves up to 3.14, fan-out
  1.04–3.06, whole lines 1.30–2.58). No length qualifies at it and every longer one: no rule.
- **L/C** (rule 4): a plane's strided lines 0.46–0.96 (64 × 512: 0.959 double, 0.605 float; 512 ×
  512: 0.903, 0.457); batches of short rows 1.31–1.79 (slower).
- **P/S with P the chain pair** (rule 1 informative, before the rules): over the L2-resident cases
  1.000 in double (4 `<T, C, T>` cases; no `<C, C, C>` double case is L2-resident on 512 KiB) and
  0.987 in float (14); 0.959 over all 56 valid-half cases. Above 1 beyond the band (rule 2): `<T, C,
  T>` 18432 (1.021–1.027 double, 1.086–1.101 float), 24576 float (1.033–1.048), 32768 float one line
  (1.018). Fan-out: P/S 1.00–1.05 (`<C, C, C>`) and 1.07–1.21 (`<T, C, T>`) at F = 4, 0.93–1.10 and
  1.07–1.17 at F = 16. Whole lines: 1.02–1.16 (`<C, C, C>`), 1.12–1.14 (`<T, C, T>`); batches of
  short rows 1.02–1.25 but `<C, C, C>` 4096 × 1024 (0.95); a plane's strided lines 0.48–0.57 (the
  pipeline faster).
- **P/W** above 1.05 (1 + band) (rule 6): `<T, C, T>` valid halves at 18432–49152, 98304 and 131072
  (1.08–1.29), `<C, C, C>` 98304 (1.19); FFTW's real rows on Zen 3 are the client catalogue's known
  gap (PLAN "Status").

**Xeon 6240** (`--serial`, one at a time behind task AC's X66; Slurm binds no CPUs; 17–36 of 80
rounds marked loaded by other users' jobs, 1 started overloaded). Runs: valid halves
20261127-003028-89a2, fan-out 4 20261127-013554-ff89, fan-out 16 20261127-014130-bbd2, whole lines
and collateral 20261127-015052-528a.

- **D/C** (rule 3): `<C, C, C>` double 0.63–0.81 at every length from 18432 (the shortest measured)
  to 131072, one and 32 lines; float 0.91–0.92 at 32768 and 0.64–0.75 from 49152, no evidence at
  18432–24576 (0.98–1.00) but one line of 18432 slower (1.03). Fan-out 4 and 16: 0.62–0.83. Whole
  lines: 0.72–0.81.
- **L/C** (rule 4): a plane's strided lines 0.58–0.72; batches of short rows 1.37–2.08.
- **P/S with P the chain pair**: L2-resident geomean 0.909 in double (14 cases) and 0.892 in float
  (22); 0.889 over all 56; no valid-half case above 1 beyond the band. Fan-out: `<C, C, C>`
  0.88–1.22 (131072 loses: 1.11 at F = 4, 1.22 at F = 16), `<T, C, T>` 0.97–1.02. Whole lines:
  1.37–1.71 (`<C, C, C>`) and 1.24–1.43 (`<T, C, T>`): the separate plans take the Xeon's
  depth-first chains (X46, X75), the pipeline's chain pair does not. Short rows: `<C, C, C>`
  0.72–0.90 except 64 × 16 (1.34), `<T, C, T>` 1.04–1.44 except 4096 × 1024 (0.75). Strided lines
  0.54–0.62, except 64 × 512 in float (2.73: the separate plans' lanes columns and loop beat the
  chain pair's lines there).
- **P/W** above 1.05 (1 + band): `<C, C, C>` one line from 24576 (1.14–1.29) and `<T, C, T>` one
  line at every length (1.07–1.38).

**Diagnosis of the `<T, C, T>` losses** (EPYC, one thread, a timing loop outside fbench, job 520839;
not a judging run): at 18432 points in float the pipeline took 38.8 µs, the same pipeline without a
middle factor 32.9 µs, the r2c and c2r plans with runs 17.1 and 15.8 µs. The middle pass over the
n/2 + 1 bins costs 6 µs: it is the baseline edge pass (vectorised for SSE2 only, outside every ISA
region), where the separate arm's loop is cloned for AVX2 (about 2.5 µs by the difference of
fbench's medians). A unit-stride form of that pass, which GCC vectorised the same, changed nothing
(job 520840) and was dropped. Nothing else in the real pair is fused: the input run's read and the
output run's store are separate passes, as in the plans with runs.

### R2, deciding (one thread, the rules as adopted), and R3 (eight threads)

**EPYC 7763** (node20, as R1; commit d8fc1250, whose programs for the x67 cases are the final
code's). Runs: valid halves 20261127-014336-157d, fan-out 4 20261127-023040-0fa5, fan-out 16
20261127-023553-c727, whole lines and collateral 20261127-024257-fb89 (80 rounds each, band 1.0 %,
no round loaded); R3 20261127-025706-ea00 (32 lines, 80 rounds, band 1.31 %) and
20261127-030907-41ac (fan-out 4, 6 rounds, band 14.6 %: every case resolved at once).

- **The rules' moves:** only the strided lines move (to the lanes pair): C/P 1.03–2.16, so P is
  faster than the chain pair on all four. Every other case is on its chain pair: C/P no evidence in
  all 56 valid-half, 16 fan-out and 24 other cases.
- **Rule 1 (acceptance):** P/S over the L2-resident cases 1.009 in double (4) and 1.001 in float
  (14); not met. 0.97 over all 56 valid-half cases.
- **Rule 2:** above 1 beyond the band at one thread: `<T, C, T>` valid halves 18432 (1.03–1.10),
  24576 and 32768 in float (1.02–1.06), 65536 and 131072 one line in double (1.08–1.11, wide
  intervals); fan-out 1.02–1.29 on most cases; whole lines 1.01–1.16; short-row batches 1.02–1.24
  except `<C, C, C>` 4096 × 1024 (0.95). At eight threads (R3): the 32-line cases mostly gain (P/S
  0.51–0.96) but `<T, C, T>` 65536 in double (1.12) and 131072 in float (1.09) lose, and one line of
  four copies loses heavily (P/S 1.56–3.56): its copies run one after another in a team (they work
  in the workspace block's second region, task 5.2e), where the separate calls share each transform.
- **Rule 6 (P/W):** as R1 (`<T, C, T>` 1.10–1.25, `<C, C, C>` 49152 and 98304 1.20–1.37).
- **Rule 7 (fan-out):** over the lengths fan-out was measured at, P/S's geometric mean is 1.03 at F
  = 1, 1.09 at F = 4 and 1.08 at F = 16: the copies' layout is a lead.
- Strided lines on the lanes pair: P/S 0.25–0.51.

**Xeon 6240** (`--serial`, commit a2378229; Slurm binds no CPUs; 39 of 80 rounds of the valid halves
marked loaded by other users' jobs). Runs: valid halves 20261127-025103-3880, fan-out 4
20261127-035459-926a, fan-out 16 20261127-040044-68e1, whole lines and collateral
20261127-041011-15cb; R3 20261127-042650-bbce (32 lines, 6 rounds, band 5.3 %) and
20261127-042803-2184 (fan-out 4, 9 rounds, band 4.8 %).

- **The rules' moves:** the `<C, C, C>` lines from 18432 points in double and from 49152 in float
  take the depth-first pair: C/P 1.24–1.58 (P faster on every moved valid-half case); the lines the
  rule leaves (float up to 32768, every `<T, C, T>`) stay on the chain pair (C/P no evidence). With
  fan-out the moved lines take 0.62–0.85 of C's time (C/P 1.18–1.64), whole lines 0.71–0.80 (C/P
  1.25–1.41); the strided lines take the lanes pair (C/P 1.39–1.75). No moved case is slower than C;
  every case the rules leave is on its chain pair (C/P no evidence, and one unmoved case, `<C, C,
  C>` 1024 × 16, with C 1.016 of P's time).
- **Rule 1 (acceptance):** P/S over the L2-resident cases 0.765 in double (14), with `<T, C, T>`
  cases at 0.94–1.01 (above 0.90 beyond the band), and 0.840 in float (22); not met. 0.769 over all
  56 valid-half cases.
- **Rule 2:** no valid-half case above 1 beyond the band at one thread. Fan-out: `<C, C, C>`
  0.69–0.77, `<T, C, T>` 0.97–1.02, where 131072 × 16 copies lies above 1 beyond the band (1.02) and
  65536 × 4 just inside it (1.02; both on the chain pair). Whole lines: `<C, C, C>` 1.09–1.23 even
  on the depth-first pair (the separate plans' X46 chains with their measured tops are faster
  still), `<T, C, T>` 1.24–1.42; short rows `<C, C, C>` 0.71–0.89 except 64 × 16 (1.33), `<T, C, T>`
  1.03–1.42 except 4096 × 1024 (0.75); strided lines 0.31–0.44 except 64 × 512 in float (1.96).
- **Rule 6 (P/W):** `<T, C, T>` one line at every length (1.07–1.39); `<C, C, C>` no longer (the
  depth-first pair). Fan-out `<T, C, T>` 1.15–1.36 and whole `<T, C, T>` lines 1.21–1.38; no
  short-row batch (P/W 0.54–1.01).
- **R3 (eight threads):** 32 lines 0.47–0.80 of the separate calls' time, every case; one line of
  four copies 0.67–0.74 at 18432 but 1.15–2.24 from 32768 (the copies one after another in a team,
  as on the EPYC).
- **Rule 7 (fan-out):** on the same lengths, P/S's mean is 0.77 at F = 1, 0.85 at F = 4 and 0.86 at
  F = 16 (EPYC: 1.03, 1.09 and 1.08): above F = 1's on both machines, so the copies' layout is a
  lead.

## Verdict

**Step 5.2's acceptance is not met on either machine.** The rules as adopted give P/S over the
L2-resident cases of 1.009 (double) and 1.001 (float) on the EPYC 7763, and 0.765 and 0.840 on the
Xeon 6240, against 0.80 with no case above 0.90 beyond the band. The fused pipeline saves the
spectrum's round trip, which inside L2 is worth a few per cent; it loses more where it gives up what
the separate calls have (leads 1 and 4–6 below).

**Adopted (rules 3 and 4, narrowed to what R1 measured; R2 deciding):**
- the depth-first pair for contiguous `<C, C, C>` lines from 288 KiB to 2 MiB of the line in double
  (float to the same 131072 points) on the Xeon 6240's class, avx512
  (`machine_profile::pipeline_depth_min_kib`, `_max_kib`, `_isa`), for a kernel of stride 1 or none,
  at most one run per side, no prologue or epilogue factor, accumulation, callback or fan-out. R2:
  C/P 1.24–1.58 on the moved valid halves, 1.25–1.41 on whole lines (and 1.18–1.64 with fan-out,
  since excluded). The EPYC keeps the chain pair (D/C up to 3.4).
- the lanes pair for a plane's strided `<C, C, C>` lines of 64 to 512 points on Zen 3's class (AMD,
  best target avx2) and the Xeon 6240's (best target avx512) (`machine_profile::pipeline_lanes_adjacent_max`), for a kernel of stride 1 or
  none and none of what the lanes pair refuses: C/P 1.03–2.16 on the EPYC (P/S 0.25–0.51), C/P
  1.39–1.75 on the Xeon (P/S 0.31–0.44, 64 × 512 in float excepted). Batches of short rows keep the
  chain pair (L/C 1.3–2.1).
- Bit agreement (DESIGN 6.2, 5.2f's review): the lanes rule is keyed by the machine (its cache
  class and its processor's best target), so a machine's `avx2` and `avx512` targets take it alike
  and never split where the chain pair fuses the kernel. The depth-first rule is keyed by the plan's
  target (avx512): on the Xeon a forced `avx2` plan keeps the chain pair, so its two targets split
  on that rule, as they already do on its depth-first rows (X46) and Bluestein convolutions (X50);
  DESIGN 6.2 lets `avx2` and `avx512` plans choose different programs.
- Every case the rules exclude is on its chain pair in R2 (C/P no evidence), plans keep their
  programs and bits (2,244 of 2,256 probes against `main` unchanged on the EPYC; the 12 that move
  are adjacent lines of 64–512 points), and the plain driver's object disassembles the same
  (rule 8).

**Narrowed after R2** (the second reader of PR 179, 2026-11-27): the depth-first rule leaves
fan-out (R2 timed the copies at one thread only, and the pair runs them one after another in a
team, where the chain pair shares them), and caps float lines at 131072 points, the longest
measured (the byte limit had reached 262144); the lanes rule is keyed by the processor's best
target as well as its cache class. The excluded cases are back on their chain pairs (exclusions
need no timing: those cases build main's programs, probes), and R2's evidence stands for every
case the rules still move.

**The baselines' multiply loop** (the second reader of PR 172): as merged it lacked `restrict` on
its two arrays while S and W call it in place, which a compiler may answer with a scalar path
after a failed overlap check. Timed in place on the EPYC (9217 complex values, the AVX2 clone, a
loop outside fbench), it took the same time as the restrict-qualified form with GCC 16.2 (4839
against 4839 ns in double, 2283 against 2280 in float) and with GCC 14.4, the Xeon's compiler (4885
against 4821, 2279 against 2278): S and W were not slowed and no P/S above is flattered. The loop
is restrict-qualified again for other compilers (Clang, the Mac leg).

**Not adopted:** no fused middle (rule 5): neither M1 nor a fused tile pass is built. The near
misses of rule 2 (the Xeon's `<T, C, T>` 65536 × 4 copies) were not re-run: clear losses on
both machines decide rule 2 whatever their re-run shows. Rule 2's
losses are not fixed in 5.2g; they are recorded as leads below, which need the maintainer's
agreement (rule 2).

**Leads for the performance phase** (each with its evidence above):
1. **The real pair's middle** is the baseline edge pass (SSE2 vectorisation) where the user's loop
   is AVX2: 6 µs of a 39 µs line at 18432 floats on the EPYC, the whole of `<T, C, T>`'s loss
   against S (1.02–1.10). M1 (the middle fused into the c2r pairing pass) or a vocabulary pass in
   the ISA regions would remove it.
2. **The depth-first pair with fan-out** is excluded until timed with threads (above).
3. **`<T, C, T>` at eight threads:** 32 lines on the EPYC lose at 65536 in double (1.12) and 131072
   in float (1.09) in R3.
4. **Fan-out's copies** lose on the EPYC (P/S 1.08 to 1.29 for `<T, C, T>`, rule 7: the mean at
   F = 4 and 16 is above F = 1's; on the Xeon too, though there P is faster than S) and at eight threads
   (one line of four copies 1.56–3.56 on the EPYC, 1.15–2.24 from 32768 on the Xeon), where copies
   in the block's second region run one after another.
5. **Whole lines and short-row batches against the plans' own strategies:** the separate plans keep
   batch leaves, real rows' depth-first and fused routes (X72, X75, X78) and the Xeon's depth-first
   chains with their measured tops (X46); the pipeline's pairs do not (EPYC whole lines 1.01–1.16,
   short rows up to 1.24; Xeon `<C, C, C>` 1.09–1.23 on the depth-first pair, `<T, C, T>`
   1.24–1.42).
6. **The lanes pair on 64 × 512 float lines on the Xeon** stays slower than the separate plans (P/S
   1.96 in R2).
7. **Against FFTW with a loop** (rule 6, not blocking): `<T, C, T>` lines on both machines
   (1.07–1.39, fan-out included) and the EPYC's `<C, C, C>` 49152 and 98304 (1.20–1.37). The Xeon's
   long `<C, C, C>` lines, 1.14–1.29 of FFTW's time on the chain pair in R1, beat FFTW on the
   depth-first pair (P/W 0.80–0.94 whole, 0.84–0.94 with fan-out).

**Mac leg:** a daytime follow-up (amendment 1).

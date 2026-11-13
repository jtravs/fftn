# X15: what a recent compiler gets from the scalar pass loops

- **Status:** planned
- **Owner:** lead, 2026-11-13
- **Phase / task:** 0 (input to 1.2 and 1.3)

## Question

How much speed does a current compiler find in FFTN's scalar pass loops when it is
allowed to vectorise them (`#pragma omp simd` under `-fopenmp-simd`, no OpenMP runtime),
and how close does that come to code written against a hand vocabulary? The answer
decides two things for phase 1: whether the generated pass loops of the `scalar` target
carry `omp simd`, and whether the old-ISA targets (`x86_v2`, `avx`, plain `neon`, DESIGN.md
6.1) can start as the scalar kernel text compiled inside their region instead of needing
a hand vocabulary first.

## Hypothesis

The walking skeleton's Stockham loops (`src/passes/chain.inl`) have an innermost loop over
the copies `q` with unit stride on both sides whenever the line is contiguous (`s > 1`
stages), and no reductions, so a compiler can vectorise them with the operation order
unchanged. Expected: under `omp simd` a measurable gain on the `s > 1` stages and none on
the first stage (`s = 1`) or the generic prime pass (a pairwise sum over `j`, which `omp
simd` on `q` does not touch). Compiled inside the `avx2` region the same text gets 256-bit
registers but no complex-aware shuffles, so it should land well short of the hand `avx2`
vocabulary (FFTW's genfft and the prototype's evidence: complex arithmetic in a vector
needs explicit swaps that auto-vectorisers produce poorly). The open question is how far
short.

## Change under test

Four arms, each a commit on `exp/x15-omp-simd`; results of every arm must be **bitwise
equal** to arm A (no reductions are reordered, the build keeps `-ffp-contract=off`), and a
test asserts it for every arm the build contains.

- **A: plain scalar.** The skeleton as merged (`scalar` region, no pragma).
- **B: scalar + `omp simd`.** `FFTN_X_OMP_SIMD` (registered in `dev/KNOBS.toml`, owner
  lead, expiry 2026-12-13) builds the library with `-fopenmp-simd` and makes a macro emit
  `_Pragma("omp simd")` on the innermost unit loops of the chain passes. When the switch is
  off the macro expands to nothing, so no `-Wunknown-pragmas` arises anywhere.
- **C: scalar text in the `avx2` region.** On x86-64 only: the chain passes compiled with
  the scalar vocabulary inside the `avx2` region (its target attribute, its namespace),
  with and without the B pragma. A forceable variant that is never the default: the plan
  takes it only when the experiment's switch is on.
- **E: scalar text, whole library at `-march=native`** (`-mcpu=native` on AArch64), with
  and without the B pragma. A diagnostic ceiling, never a candidate: AGENTS.md section 6
  forbids `-march=native` in anything distributable, and a global ISA flag breaks the
  region rules (section 4), so this arm is a throwaway build option of the experiment
  branch, built on the machine that runs it (on DMOG, inside the compute job: the login
  node is a VM). It shows how much of the gap between A and D the compiler closes when it
  may use the whole instruction set everywhere (AVX-512 on the Xeon, Zen 3's AVX2 on the
  EPYC, the M5's full Armv9 set on the Mac). It is excluded from the bitwise-equality test
  only if the compiler contracts nothing and reorders nothing it would not elsewhere;
  otherwise its outputs are checked against tolerances and the difference is recorded.
- **D: hand `avx2` vocabulary.** The pass that exists on every target in phase 0,
  `radix4_pass` of `src/isa/kernels.inl`, compared at pass level across A–D (scalar text
  in the scalar region, with `omp simd`, scalar text in the `avx2` region, hand `avx2`),
  since the chain passes have no hand `avx2` version yet.

## Method

- Cases: whole transforms through the public API, fbench `standard` catalogue, c2c and
  r2c rows, double and float, one thread; target set = smooth lengths 64–16384 with
  batches (where the `s > 1` stages dominate). Pass level (D): `radix4_pass` for
  m = 16 … 4096, in L1 and in L2, from a small harness under `bench/` (no new
  dependency), reporting ns per butterfly, medians over process rounds.
- Machines: EPYC 7763 (DMOG, Zen 3, AVX2; a block of whole cores in one NUMA domain,
  non-exclusive; the A/A band decides whether that is quiet enough, otherwise an exclusive
  node), Xeon Gold 6240 (AVX-512 hardware, `avx2` target; exclusive Slurm job, run when no
  gate is queued), Apple M5 Max (arms A, B and E; only if fbench's quiet check passes).
- Compilers: GCC 14 and 15 on Linux, Clang 23 on both Linux and macOS (Apple Clang
  noted if run). Build: Release, the build a user gets (no `-march=native`).
- Metric: per-case median paired ratio arm ÷ A with its 95 % interval; weighted geometric
  mean over the target set; the A/A noise band of the same session next to every claim.
- Decision rule, written before running:
  - **B is adopted** (generated scalar pass loops carry `omp simd`; the switch becomes the
    default and is removed) if its weighted ratio to A is ≤ 0.95 on at least two of the
    three machines with intervals clear of the A/A band, and no catalogue case outside the
    target set is slower than A by more than 2 % beyond the band. Otherwise B is deleted.
  - **C is recorded as a finding for 1.3**, not adopted as code: if the pass-level time
    of C is within 1.25 × of D on both x86 machines, phase 1 may bring up `x86_v2` and
    `avx` as scalar text compiled in their regions before their hand vocabularies exist
    (with X14 still deciding whether `avx` stays). If not, old targets need hand
    vocabularies from the start. Either way the C code is deleted after the run.
  - **E is never adopted.** It is reported as the share of the A → D gap (pass level) and
    the A → E ratio (whole transforms) per machine. If E comes within 1.10 × of D, the
    finding goes to 1.3 as "the generator's job is the operation order and the shuffles;
    the compiler handles the rest", and C (the distributable form of the same idea) is
    re-examined against it. Its run IDs are stored like any other arm's, labelled
    non-distributable.
  - Any arm whose output is not bitwise equal to A is a bug in the arm, not a result
    (E excepted as stated above).

## Results

(To be filled with run IDs, host, compiler, instruction set, thread mode and load.)

## Verdict

(To be filled.)

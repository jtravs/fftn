# X49: the fixed cost of tiny calls

- **Status:** running
- **Owner:** X49 agent, 2026-11-18
- **Phase / task:** 1 (exit criteria: the tail rule of DESIGN 1.3); exit-check cause 2

## Question

Why do single rows of 2–8 points, the backward pairs 64 × 2 and 15 × 2, and 2 × 3 / 3 × 3
in place still take 1.1–1.5 of FFTW's time on the Mac, the EPYC, the Xeon, Zen 5 and Axion
(X45; PLAN "Phase 1 exit check again"), and which change removes it without slowing anything
else?

## Hypothesis

X20, X28 and X37 took the execute path down to one library call and one leaf call. What
is left, from counters and timings taken before any arm (`main` a0dffe1; FFTN with rules,
FFTW at MEASURE with its plan printed; user-mode counters per call, median of three
processes):

- **Single rows of 2–6 points on x86 are mostly fbench's own cost.** EPYC 7763 (Slurm
  512517): FFTN 20–26 cycles against FFTW's 22–28, already faster, though it retires 6–15
  more instructions per call. Xeon 6240 (Slurm 16430): FFTN 24.5–29 cycles and 93–103
  instructions against FFTW's 21–25 and 72–86 at 2–4 points (1.08–1.17 of FFTW's cycles).
  fbench's FFTN backend reaches `execute_all` through `state()` (a function-local static: a
  guard check and a call) and saves three registers around it, about 16 instructions per
  call; FFTW's backend is five instructions and a jump. A caller of `plan::execute` pays
  none of this. Expected: correcting the backend takes about 16 instructions and 3–5
  cycles off every FFTN case on every machine, most visibly at 2–8 points.
- **2 × 3 in place on the Mac is store-to-load forwarding.** In place with one buffer,
  each execution loads what the previous one stored a few nanoseconds before. On the M5
  Max a hand-written 2-point loop on the same addresses runs 0.75 ns per row with single
  128-bit loads and stores (`ldr`/`str q`) and 3.3–3.8 ns when either the loads or the
  stores are pairs (`ldp`/`stp q`), at any alignment tried (the screen `x49/ip.cpp`, below).
  Clang pairs adjacent accesses in every NEON kernel, so FFTN's 2 × 3 in place runs at 4.1 ns
  against FFTW's 2.8–3.1. The same stall hits codelets that spill (a spill stored as a pair
  and reloaded soon): a whole-library build with LLVM's load/store optimiser off ran 2 × 3 in
  place 0.78, 32 points 0.94 (the Mac's 32 points was 1.20 of FFTW in the exit check), 64
  points 0.94, 1024 0.98 of `main`'s time, and nothing tried slower beyond 1.04 (one
  process round each; a screen, not a verdict).
- **The backward pairs on x86 are kernels, not fixed cost.** EPYC 15 × 2 and 64 × 2: FFTN
  94 and 443 cycles (376 and 1946 instructions, a radix-15 leaf twice and a 4 × 16 chain)
  against FFTW's 87 and 354 (321 and 1374: `n1bv_15_avx2` and `n1bv_64_avx2`, both rows in
  the two halves of one `ymm`). FFTN's lanes strategy forced on them is 1.6–10 times slower
  (built for long batches). On the Xeon 64 × 2 is a pair leaf at 0.52 of FFTW; 15 × 2 is
  1.16 (139 against 120 cycles). Not targeted here: it needs a leaf that runs C rows in the
  lanes of one register, recorded as a follow-up.
- Zen 5's and Axion's 2–8 points (1.14–1.49 of FFTW in X45) could not be measured here (no
  cloud in this round); the backend correction applies to them as it does to every machine.

## Change under test

Arm A: `main` a0dffe1 (before X46's merge; the arms are rebased on `main` before the judging
runs if it has moved). One commit per arm, each on top of the one before:

0. **Benchmark correction** (no library change): fbench's FFTN backend keeps the session's
   execution mode in each plan record, so that a serial execute is a load, a test and a
   tail call of `execute_all`.
1. **U, unpaired NEON** (AArch64 with Clang, which pairs; GCC builds unchanged): the NEON
   units (`neon`, `neon_fcma`: main, row and pair-leaf units) are compiled with the target
   features `+disable-ldp` and `+disable-stp`, where the compiler accepts them (a CMake
   check), so that their loads and stores of vectors and spills are single accesses. Same
   instructions otherwise: bit-identical.

No switch in `dev/KNOBS.toml`: each arm is a commit.

## Method

- Cases: catalogue `x37` (tiny rows, tiny batches in and out of place, backward pairs,
  short batches; double), the standard catalogue's tracking tier for collateral, and the
  exit check's catalogue and filter (`standard`, `^c2c-[0-9]+-b`, tier `full`) on the final
  arm to count the exit cases it moves.
- Machines: Apple M5 Max (Clang 23, `neon_fcma`, NEON FFTW; arms 0 and U), EPYC 7763 (DMOG,
  GCC 16, `avx2`; arm 0), Xeon Gold 6240 (GCC 14, `avx512`, FFTW and oneMKL; arm 0). fbench
  interleaved rounds with the A/A control, `--serial` on the Slurm machines (`-c 8`,
  shared), one run at a time on the Mac. Counters: fbench's harness `--pmu` (cycles,
  instructions, retired branches; on the Xeon also `LD_BLOCKS.STORE_FORWARD`).
- Metric: fbench paired ratio of each arm against the arm before it; geometric mean over
  each arm's target cases; FFTN ÷ FFTW (÷ MKL on the Xeon) per case.
- Decision rule, written before running, per arm and machine-target, on the release
  toolchains (GCC on x86-64 Linux, Clang on macOS), double:
  1. every test passes; arm U is bit-identical (the suites hold results against FFTW and
     the reference; nothing arithmetic changes);
  2. arm 0 is kept whatever it measures if rule 1 holds: it makes fbench time what a caller
     of `plan::execute` runs (as X37's arm 0). Its effect is reported;
  3. arm U is adopted on Clang AArch64 if the geometric mean over the `x37` cases is at
     most 0.99 of arm 0 with its interval clear of the A/A band, 2 × 3 in place is at most
     0.85 of arm 0, and no case of `x37` or of the tracking tier is slower than arm 0 beyond
     the band (a case just past it on a loaded machine is re-run before it decides).
  4. Added after arm U's run (below) missed rule 3, before any run of its own: **arm U2**
     is arm U's code (1caa4f9) judged by fresh runs under a rule aimed at the cases the
     change can reach, which rule 3 diluted with 60 cases it cannot touch. Targets: the
     exit catalogue's in-place batches of three of 2–16 points (2, 3, 5, 7, 8, 12, 16) and
     its single rows of 32 and 64 points (codelets that spill). U2 is adopted on Clang
     AArch64 if the geometric mean over these targets is at most 0.97 of arm 0 with its
     interval clear of the A/A band, and no case of the exit catalogue (`standard`,
     `^c2c-[0-9]+-b`, `full`) or of the tracking tier is slower than arm 0 beyond the band
     (a case just past it is re-run before it decides).

## Results

Toolchains: Clang 23.1.4 on the Mac (`neon_fcma`), GCC 16.2 on the EPYC (DMOG, `-c 8`),
GCC 14.4 on the Xeon (`avx512`); Release, one thread, double. Arms: A a0dffe1 (`main`),
Z 94bf6b7 (arm 0 alone: 88890ae plus its second part, the threaded executes out of line,
cherry-picked onto it for the Mac's comparison), U 1caa4f9 (the branch: arm 0 then arm U;
on x86 its library is A's).

### Counters after arm 0 (Slurm 512540 EPYC, 16442 Xeon)

Cycles and instructions per call, `main` → arm 0 (88890ae, before its second part) against
FFTW. EPYC: 2 points 20.0 / 86 → 15.0 / 66 (FFTW 22.0 / 71), 4 points 23.0 / 97 → 18.1 / 77
(26.0 / 85), 2 × 3 in place 26.8 / 106 → 20.4 / 86 (27.4 / 104): every tiny case now
0.68–0.86 of FFTW's cycles. Xeon: 2 points 24.5 / 93 → 22.0 / 84 (21.0 / 72), 4 points
28.8 / 103 → 24.0 / 94 (24.9 / 86), 2 × 3 in place 29.3 / 112 → 27.4 / 103 (28.7 / 93),
3 × 3 in place 43.6 / 139 → 40.5 / 130 (37.8 / 134): only 9 instructions went, because GCC
14 still saved three registers before testing the mode, hence the second part (Clang 23 and
GCC 16 now emit a load, a test and a tail call). No store-forwarding blocks
(`LD_BLOCKS.STORE_FORWARD`) anywhere on the Xeon.

### Mac, `x37` (run 20261118-002900-6819; 30 rounds, all loaded: load 11.7, other processes 8.1 cores; band ±2.39 %)

- Arm 0 against A: 0.961 over the 69 cases; single rows of 2–12 points 0.79–0.88, tiny
  batches out of place 0.82–0.93, in place 0.95–0.98; nothing slower. FFTN ÷ FFTW, single
  rows of 2–8 points 0.92–1.01 → 0.77–0.81.
- Arm U against arm 0 (paired, the same run): 1.002 over the 69 cases (rule 3 asks 0.99), 2 ×
  3 in place 0.857 (rule 3 asks 0.85), 3 × 3 in place 0.909, 5 × 3 in place 0.961; nothing
  slower beyond the band (worst 8 × 3 in place 1.046 [1.019, 1.059], 16 × 256 1.046 [1.017,
  1.088], 26 points 1.030). **Arm U is not adopted under rule 3.** FFTN ÷ FFTW, 2 × 3 in
  place 1.317 (A) → 1.283 (arm 0) → 1.093 (U); 3 × 3 in place 1.043 → 1.003 → 0.910. The
  catalogue holds no single row of 32 or 64 points, the spilling codelets of the screen.

### Mac, exit catalogue (run 20261118-004545-cfca; Z, U, FFTW; 24 rounds, all loaded; band ±3.36 %)

- Arm U2 against arm 0, its nine targets: in place 2 × 3 0.849, 3 × 3 0.895, 5 × 3 0.986,
  7 × 3 1.002, 8 × 3 1.041, 12 × 3 0.994, 16 × 3 1.012; single rows of 32 and 64 points
  1.008 and 0.999 (the screen's 0.94 had LLVM's whole load/store optimiser off in every
  unit, not only the pairs in the NEON units). Geometric mean **0.974** against the rule's
  0.97. Over the 201 cases 0.999; slower beyond the band: float 8 × 2 in place 1.086 [1.046,
  1.138]; faster: 1024 × 1024 0.891 besides the two targets. **Arm U2 is rejected** (both
  halves of rule 4) and reverted (ad7ac2d).
- Arm 0's FFTN ÷ FFTW over the exit catalogue's 193 rows with a reference (`main` a0dffe1's
  library, so before X46): 0.733, 7 above 1.05: 2 × 3 in place 1.32, 2²³ 1.13, 8192 1.10,
  65536 1.08, 32 × 4096, 64 × 4096 and 32 × 256 1.05–1.07. No single row of 2–8 points
  is among them; the Mac's 32 points (1.20 in the exit check of 509d123) is at 0.98.

## Verdict

(pending)

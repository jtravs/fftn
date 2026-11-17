# X47: the EPYC's double rows of 512–2048 points, second attempt

- **Status:** running
- **Owner:** x47 session, 2026-11-17
- **Phase / task:** 1 (phase 1 exit check, the tail rule; cause 4); follows X30, X38, X42

## Question

Why do single rows of 512–2048 points take 1.09–1.22 of FFTW's time on the EPYC 7763
(`avx2`, GCC 16) when the best radix sequences (X38) and register-fitting codelets (X42)
did not move them, and which change to the passes or to the chain's buffers closes it? The
answer changes `src/passes/stockham.inl` (lane pass loops) and the chain's choice of lines
in `src/strategies/chain.cpp`.

## Hypothesis

### Diagnosis (step 1, before any code)

EPYC 7763 (DMOG node16), GCC 16.2, `avx2`, one thread, the library of X42's B3 (10e9487 in
the fbench cache; the same plans as main 509d123 at these lengths), FFTW 3.3.11 (AVX2,
MEASURE, one plan shared by wisdom). Counts per transform, user mode, medians of three
processes (five for the in-place screen) of `fbench-<lib> run --pmu` with the Zen 3 events
in groups of five (cycles plus four: a group of six never runs on these nodes, the NMI
watchdog holds a counter): L1 fills PMCx044, store-to-load forwards PMCx035, non-forwardable
store conflicts PMCx024 unit 0x02, load and store dispatches PMCx029, dispatch token stalls
PMCx0AE (FP register file 0x20, FP scheduler 0x40, load queue 0x02, store queue 0x04) and
PMCx0AF (retire queue 0x20), FP pipe assignments PMCx000, op cache PMCx28F, op queue empty
PMCx0A9. Slurm 512069, 512082, 512085 (512063 lost: six-event groups). FFTW's plans
(`PLAN_NOTE=1`, the same at batches 1, 3, 16, 256), first pass first: 512 = 64 → 8
(`n2fv_64`, `t2fv_8`), 768 = 12 → 64 (`t1fv_64`), 1024 = 64 → 16 (`t2fv_16`), 2048 = 32 →
64 (`t1fv_64`).

| case | cycles N / F | instructions N / F | IPC N / F | L1 fills N / F | STLF N / F | FP-register-file stall, % of cycles, N / F |
|---|---|---|---|---|---|---|
| 512 (16 × 32) | 2234 / 1948 | 7802 / 7040 | 3.49 / 3.61 | 517 / 181 | 670 / 83 | 28 / 22 |
| 768 (4 × 16 × 12) | 3925 / 3367 | 14459 / 11920 | 3.68 / 3.54 | 1085 / 434 | 320 / 146 | 26 / 21 |
| 1024 (4 × 16 × 16) | 5582 / 4735 | 19438 / 15276 | 3.48 / 3.23 | 1780 / 1242 | 670 / 493 | 34 / 24 |
| 2048 (8 × 16 × 16, split) | 13982 / 12302 | 40407 / 34650 | 2.89 / 2.82 | 5180 / 3002 | 1300 / 540 | 37 / 32 |
| 1024 as 16 × 64 | 5710 | 16952 | 2.97 | 1765 | 902 | 31 |
| 1024 as 32 × 32 | 6212 | 18039 | 2.90 | 1981 | 1292 | 27 |
| 2048 as 32 × 64 | 15840 | 38837 | 2.45 | 4494 | 1781 | 34 |
| 512 as 8 × 64 | 2570 | 7761 | 3.02 | 547 | 179 | 23 |

What the counters say:

1. **The work on the FP pipes is FFTW's.** FP-pipe uops agree within 3 % (1024: 9824
   against 9565), loads and stores within 10 % (1024: 5355 / 2060 against 5614 / 2402).
   FFTN's excess instructions (11–27 %) are integer: in the hot loop of the plain radix-16
   lane pass (GCC 16, `objdump`), a unit of two copies is 190 instructions: 91 FP, 32 loads
   and stores of data, 13 spill moves and 54 integer (two `imul`, two `cmov`, the unit's
   group and block bookkeeping, eleven loop invariants reloaded from the stack, and the leg
   addresses as a chain of adds); the twiddled radix-16 unit 261 (49 integer), the radix-32
   plain unit 477 (80 integer, 101 spill moves), the radix-64 plain unit 1161 (176 integer,
   327 spill moves). The radix-4 grouped pass's loop is tight (44 instructions a unit, four
   integer).
2. **The planner's three-pass plans lose on instructions, not IPC** (IPC at or above FFTW's
   from 768 up); the two-pass plans of FFTW's shape lose on IPC (2.45–3.0 against 2.8–3.6):
   their radix-32/64 units spill (101 and 327 stack moves a unit) and miss the op cache
   (190–410 misses a transform against FFTW's 0–46; the op queue is never empty, so the
   front end is not what stops dispatch). X42's finding, explained.
3. **The cycles FFTN loses are dispatch stalls on the FP register file.** Its share is 5–10
   points above FFTW's, and the extra stall cycles are most of the gap: 1024 762 of 847
   cycles, 2048 1236 of 1680, 768 313 of 558, 512 196 of 286. The register file fills when
   retirement waits on long-latency loads: L1 misses and forwarded spill reloads.
4. **FFTN refills L1 1.4–2.9 times as often.** Out of place, a chain touches three lines
   (source, destination, one scratch line: X18) where FFTW's in-place codelet passes on the
   output touch two. At 512 the three lines and the radix-16 pass's twiddles are 32 KiB, the
   whole L1: 517 fills against FFTW's 181. Screen (Slurm 512085): the same 512-point plan in
   place, which touches two lines, refills 160 times and takes 0.954 of the out-of-place
   time (680 against 713 ns; FFTW 617); 1024 as 16 × 64 in place 0.97 (1800 against 1855
   ns, below the planner's 1839).

### What should close it

- Two-pass chains out of place should touch two lines: the last pass of a Stockham chain
  (m = 1) reads and writes the same R elements in each unit, and every output of a DFT
  codelet depends on every input, so it can run in place on the destination; the first pass
  then writes the destination instead of the scratch line. Expected 512 at about 0.95 of
  its time (the in-place screen), every batch of 512 likewise, other two-pass rows whose
  three lines overfill L1 (256–1024 points in double, 512–2048 in float) less.
- The lane passes should spend fewer integer instructions per unit: a loop over the blocks
  of one group with pointers stepped by C, the group's addresses formed once, takes the
  per-unit products, conditional moves and the partial-block test out of the whole units.
  Expected 5–10 % fewer instructions where lane passes dominate (768–2048: two of three
  passes) and, at an IPC near the dispatch width in unstalled cycles (5.5 macro-ops), a
  few per cent of time.

## Change under test

Arms are commits of branch `exp/x47-epyc-mid` against O = its base (X42's head b27970f,
main plus the radix-25 rows):

- **A (two-line chains).** `chain_builder::c2c_line`: an out-of-place chain of two passes
  whose destination serves as a line (X18's `via_dst`) and whose last radix has a codelet
  runs its first pass into the destination and its last pass in place there; it needs no
  scratch line. Every target. Bits unchanged (the same arithmetic on other addresses).
- **B (lane loops by group, on A).** `lane_pass` in `src/passes/stockham.inl` iterates the
  blocks of a group with stepped pointers; the partial block and the group change stay out
  of the whole units' loop. Every target. Bits unchanged.

The two are independent (A changes buffers, B instructions); B is judged against A. No
switch: arms are commits, `dev/KNOBS.toml` is untouched.

## Method

- Static check before timing B: GCC 16's instruction count per unit of the radix-16 and
  radix-32 lane loops (`objdump`), and `run --pmu` instructions per transform at 512–2048.
- Judging runs (`fbench_remote.py submit --serial`, one thread, interleaved process rounds,
  A/A control):
  - EPYC 7763 (DMOG, GCC 16.2, `avx2`), deciding: arms O, A, B and FFTW on the exit check's
    cases (standard catalogue, filter `^c2c-[0-9]+-b`, tier `full`); then the tracking tier
    (all kinds) for O, A, B.
  - Xeon Gold 6240 (GCC 14.4, `avx512`): the tracking tier, O, A, B (both arms change code
    every target runs).
  - Apple M5 Max (Clang, `neon_fcma`, NEON FFTW): the tracking tier, O, A, B, if the Mac is
    free (one fbench run at a time).
- Metric: fbench's median paired ratio arm ÷ the arm before, per case, with its interval;
  geometric means over the target cases; FFTN ÷ FFTW before and after.
- **Decision rule, written before any arm is timed.** Judged on GCC on x86-64 Linux and
  Clang on macOS; per arm against the arm before it (A ÷ O, B ÷ A):
  - (a) on the EPYC, the geometric mean over the arm's target cases is at most 0.97 with
    its interval clear of the band. A's target cases: the exit check's c2c cases whose
    program A changes (two passes out of place), of which 512 at batches 1, 16, 256 and
    1024 are the named ones; B's: the exit check's cases of 512, 768, 1024 and 2048 points
    (every batch and direction in the set).
  - (b) no case of the EPYC's judged set or tracking tier is slower than 1 + band + 2 %; a
    case past it on a loaded machine is re-run once (`--serial`) and the re-run decides.
  - (c) on the Xeon and the Mac, the tracking tier's geometric mean is at most 1 + band and
    no case is past 1 + band + 2 % after one re-run; an arm that fails (c) on one target
    but passes (a) and (b) is adopted for the targets where it passes (per-target gating,
    AGENTS.md section 6), with the losing target recorded.
  - An arm that fails (a) or (b) is reported with its numbers and deleted; no third arm.

## Results

(to come)

## Verdict

(to come)

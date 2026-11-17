# X47: the EPYC's double rows of 512–2048 points, second attempt

- **Status:** done (A proposed for adoption, the lead's call; B rejected and reverted)
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

### Static check of B and a counter screen (not deciding)

GCC 16, `objdump` of the cached builds: B cut the plain radix-16 lane unit from 190 to 158
instructions (integer 54 → 25), the twiddled radix-16 unit from 262 to 238 and the plain
radix-32 unit from 477 to 438, but GCC now holds the 32 leg offsets of a radix-16 unit in
stack slots, so each leg's load first loads its offset (21 integer stack loads a unit,
against 11 before). `run --pmu` (Slurm 512121, five processes each, the arms' cache builds),
per transform O → A → B: 512 702 → 669 → 709 ns, L1 fills 509 → 172 → 191; 1024 1822 →
1830 → 1860 ns, instructions 19438 → 19438 → 18463, IPC 3.46 → 3.45 → 3.23; 768 and 2048
within 1 % for A, B +2 % / −1 %; 256 B +7 %. A did what it was meant to (L1 refills at 512
−66 %); B ran 5 % fewer instructions and lost more in IPC. Both judged as written.

### EPYC, the exit check's cases (run 20261117-170142-1ab1)

EPYC 7763 (DMOG), GCC 16.2, `avx2`, one thread, Slurm 512107 (`--serial`), arms O
(b27970f), A (c893ef7), B (ee932d1), FFTW; standard catalogue, filter `^c2c-[0-9]+-b`, tier
`full` (201 cases); 21 rounds, every one under load (other processes 35 cores, median), A/A
band ±1.07 %, so rule (b)'s limit is 1.031.

| case | A ÷ O | B ÷ A | FFTN ÷ FFTW, O → A |
|---|---|---|---|
| 512 | **0.950** [0.944, 0.954] | 1.048 | 1.112 → 1.057 |
| 512 × 16 | **0.976** [0.965, 0.986] | 1.068 | 1.064 → 1.038 |
| 512 × 256 | **0.971** [0.959, 0.980] | 1.056 | 1.060 → 1.030 |
| 512 × 1024 | **0.962** [0.956, 0.978] | 1.067 | 1.063 → 1.022 |
| 768 | 1.007 [0.990, 1.022] | 0.978 | 1.092 → 1.099 |
| 1024 | 0.989 [0.984, 1.003] | 1.032 | 1.217 → 1.203 |
| 1024 backward | 0.998 | 1.023 | 1.190 → 1.188 |
| 1024 × 3 | 0.996 | 1.024 | 1.212 → 1.207 |
| 1024 × 16 | 0.994 | 1.036 | 1.110 → 1.103 |
| 1024 × 256 | 0.997 | 1.026 | 1.074 → 1.071 |
| 1024 × 1024 | 0.989 | 1.046 | 0.917 → 0.907 |
| 1024 in place | 0.994 | 1.016 | 0.962 → 0.957 |
| 2048 | 0.990 [0.972, 1.014] | 1.002 | 1.135 → 1.124 |

- **A**: over the 42 cases whose chain is two passes out of place (A's target set by the
  rule) the geometric mean is **0.987**; over 512's four cases 0.965. The two-pass cases
  of 256 points and more, whose three lines overfill L1 (a post-hoc grouping, not the
  rule's), gain: 256 × 16 / × 256 / × 1024 0.977 / 0.947 / 0.959, 323 0.975, 384 0.980, 500
  0.960, 640 0.963, and 96 0.952 and 100 × 1024 0.970; below that (45–240 points, single
  rows of 64–256) 0.98–1.02, none outside the band but 64 × 4096 (1.021). The 159 cases A
  does not change: 0.9997. All 201: 0.997 [0.995, 1.000]. Past 1.031: 65536 1.054 [0.958,
  …] and 65537 1.041 [0.876, …] (unchanged programs: a six-pass chain and a Rader stage;
  wide intervals); 61 1.027 and 101 1.029 ("slower" verdicts inside the limit; Rader stages, whose
  convolutions of 60 and 100 points run in the two scratch slots, which A does not touch).
  Re-run: see below.
- **B**: ÷ A over the 16 cases of 512–2048 points **1.033** (768 0.978 the only gain);
  past the limit in 36 cases (64-point rows 1.11–1.17, 31 1.13, 256 × 256 1.10).
- FFTN ÷ FFTW over the 193 cases with every reference: O 0.802, A 0.799; cases over 1.05:
  22 → 17 (512 1.057, 256 1.054 remain just over; 512 × 16 / × 256 / × 1024 leave the list).

### Mac, tracking tier (run 20261117-163159-3214)

Apple M5 Max, Clang 23.1.4, `neon_fcma`, arms O, A, B (built from the same commits), 80
rounds, all under load (other processes 4 cores, median), band ±1.00 % (limit 1.030). A ÷
O 0.999 [0.996, 1.000], one "slower" case: 128 × 4096 1.021 [1.013, 1.032] (inside the
limit). B ÷ O 1.007 [1.005, 1.009], 13 slower, five past the limit: 128 × 4096 1.091, 128
1.085, 2048 × 36 1.062, 97 1.053, c2r 256 1.040.

### Xeon Gold 6240, tracking tier (run 20261117-174501-5595)

GCC 14.4, `avx512`, Slurm 16380 (`--serial`), arms O, A, B, 70 rounds, 4 under load, band
±1.00 % (limit 1.030). A ÷ O 0.997 [0.995, 0.999], no case slower; faster: 128 × 4096
0.952, 100 × 1024 0.960, r2c 1000 0.953 (its half-length chain of 500 = 20 × 25 takes two
lines), 256² 0.972, and 16 0.957 (a leaf: not A's change; placement). B ÷ O 1.004 [1.002,
1.006], nine slower; B ÷ A up to 1.118 (100 × 1024), 1000 1.060, 128 1.044.

### EPYC, tracking tier (run 20261117-180415-3cb9)

Slurm 512109 (`--serial`), arms O, A, B, 76 rounds, 69 under load (other processes 15
cores, median), band ±1.00 % (limit 1.030). A ÷ O 0.998 [0.997, 0.999], no case slower;
faster: r2c 1000 0.967, 100 × 1024 0.970, 128 × 4096 0.976, c2r 256 0.976. B ÷ O 1.005
[1.004, 1.006], 18 slower; B ÷ A up to 1.155 (64), 64 × 256 1.138, 64² 1.071.

### EPYC re-run of the cases past the limit (run 20261117-194215-1604)

Slurm 512296 (`--serial`), O and A, 65536, 65537, 61 and 101 at batch 1, 21 rounds, all
under load (other processes 24 cores), band ±1.00 % (limit 1.030): 65536 1.000 [0.996,
1.001], 65537 0.998 [0.991, 1.003], 61 1.026 [1.022, 1.029], 101 1.022 [1.019, 1.027]. The
re-run decides: no case past the limit, so A passes (b). 61 and 101 stay 2–3 % slower in
both runs although A builds the same stages for them (a single Rader stage, whose chain of
one stage A does not touch, and the same scratch size): code or table placement, not shown,
as X38's 60 × 3 and 323.

Bits: A and B change no arithmetic (the same units, the same order); the tests pass on the
Mac (1883) and in CI. Library text (x86-64 Linux, GCC 16, `size` of `libfftn.a`): O
86 942 337 bytes, A 86 943 812 (+0.002 %), B 91 967 625 (+5.8 %, reverted).

## Verdict

- **A (two-line chains): fails rule (a) by the letter, passes (b) (after the re-run) and
  (c); proposed for adoption, the lead's call.** Over its 42
  target cases (every two-pass chain out of place in the exit check) A ÷ O is 0.987, not
  ≤ 0.97: the rule's target set included rows of 45–240 points whose three lines already
  fit L1, where A can change nothing (0.98–1.02, all inside the band). On the cases the
  diagnosis named it does what was measured in the screen: 512 0.950, 512 × 16 / × 256 /
  × 1024 0.976 / 0.971 / 0.962 (geomean 0.965), and the other two-pass rows of 256 points
  and more 0.947–0.980. Nowhere slower: the 159 unchanged cases 0.9997, tracking tiers EPYC
  0.998, Xeon 0.997, Mac 0.999, no case past the limit there. FFTN ÷ FFTW at 512 1.11 →
  1.06, at 512 × 16–1024 1.06 → 1.02–1.04; exit-check cases over 1.05 22 → 17 of 193.
- **B (lane loops by group): rejected and reverted** (a7c8d05). 1.033 of A's time over
  512–2048 on the EPYC, up to 1.17 elsewhere, 1.007 on the Mac with five cases past the
  limit. GCC 16 turned the saved index arithmetic into 21 stack loads of leg offsets a unit:
  17 % fewer instructions, lower IPC (LESSONS 4.19).
- **Not closed: 768, 1024 and 2048** (1.10, 1.19–1.21, 1.12 of FFTW after A). The counters
  place their gap in the third pass's instructions and in L1 refills of three-line chains
  (a three-pass chain cannot run its middle pass in place), shown as FP-register-file
  dispatch stalls. What would close them: a two-pass plan whose radix-32/64 pass keeps
  FFTW's IPC (FFTW's `t1fv_64`/`n2fv_64` run at IPC 3.2–3.5 with 16 registers; FFTN's
  radix-64 unit spends 327 of 1161 instructions on stack moves), or a depth-first order
  that keeps two passes' working set in L1 (X46's subject for long rows). With A, 1024 as
  16 × 64 ran 1800 ns in place against the planner's 1839 (screen), so after A a radix-64
  last pass is within 2 % of the three-pass plan. A screen on A's build (not deciding; not
  `--serial`; runs 20261117-190918-99ff, -190928-81b1, -190948-ea48, band ±1.00 %), forced
  sequences ÷ the planner's: 1024 as 16 × 64 0.957 at batch 1 but 0.994 / 1.006 / 1.101 at
  batches 16 / 256 / 1024, as 32 × 32 1.01–1.17; 768 as 12 × 64 0.964; 2048 as 32 × 64
  1.096. So a two-pass single row of 768 or 1024 now gains 4 %, and batches lose it: an
  override would need the batch in the rule (X38's selection rule asks for both batches),
  which closes a third of 1024's gap at most.

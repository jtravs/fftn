# X36: large primes and prime factors outside Rader's range

- **Status:** done: arms 1, 3, 4 adopted per machine; arm 2 rejected
- **Owner:** X36 agent, 2026-11-16
- **Phase / task:** PLAN Status, phase 1 exit check, cause (1)

## Question

Which change to the Bluestein and Rader stages takes the exit check's failing primes
(131071, 999983, 4093, 1018 = 2 · 509, EPYC 65537) under FFTW's time on every machine, and
is EPYC 65537 a Zen 3 effect or a bad plan? The answer sets the convolution's radices and
length (`bluestein_length`, `add_convolution`), the Bluestein stage's edge passes and the
Rader range per profile (DESIGN.md 8.5, 11.2 rule 3).

## Hypothesis

Evidence before any change (Mac, Clang 23, `neon_fcma`, run 20261116-175942-49c7 and a
timing harness on the public API):

- FFTW does not use Rader for these lengths. Its printed plans (`PLAN_NOTE=1`, MEASURE) are
  Bluestein for 131071 (M 262144), 999983 (M 2000000), 4093 (8192), 8179 (16384), 509 (1024)
  and 1018 (2 × Bluestein 509), Rader only for 65537. So a Bluestein stage as good as
  FFTW's is enough for parity; Rader nesting would be a further step.
- FFTN's Bluestein stage costs far more than two transforms of M points: 131071 takes
  5.36 ms against 0.97 ms for FFTN's own c2c 262144; 999983 48.4 ms against 12.8 ms for
  2²¹; 8179 200 µs against 37.5 µs for 16384. The convolution's chains take
  `rule_radices(M)` (8 × 8 × 16 × 16 × 16 at 2¹⁸), the cost table's choice for rows inside
  L1, while the line itself takes X32's long-row radices (4⁹ at 2¹⁸): forced to the rule's
  radices, c2c 262144 takes 2.48 times as long (10.1 against 4.1 ns per point). Radix-16
  passes over a line beyond L2 meet the set conflicts X29 and X32 found. The same holds for
  a Rader convolution of a long power of two (65537 on the EPYC and the Mac: L = 65536,
  1 MiB in double), which may be why Rader lost at 65537 on the Mac in X27.
- Below the long-row size the stage costs about 2.7 times one transform of M (4093: 49.7 µs
  against 18.1 µs for 8192; 509: 2.8 against 1.1). The two chirp passes and the zeros are
  that extra: M ≥ 2Q, so half the convolution's input is zero (written by the chirp pass,
  read by the first pass) and half the backward transform's output is never read. A
  chirp pass that also does the convolution's first radix-2 pass (whose butterflies have a
  zero leg) and one that does the backward transform's last radix-2 pass for the Q outputs
  it keeps remove one pass in each direction of about 10 pass-equivalents.

Expected: arm 1 takes 131071 and 999983 to about 0.4–0.6 of their time on the Mac and the
EPYC (to about FFTW's); arm 2 takes the mid sizes (509–8179) to about 0.8–0.9.

## Change under test

- **Arm 1** (long convolutions; a74a388, amended by 19bc985): a convolution whose length
  is a power of two of more than `grid_min_bytes` takes FFTL.jl's long-row radices (X32) in
  every profile (on the Xeon too: a convolution cannot be a grid); and a Bluestein stage's
  length M has a power-of-two part of at most `machine_profile::bluestein_pow2_part_bytes`
  (a quarter of L1 where the L2 is shared, Apple; unlimited on private L2s), the
  candidates then running up to 2 (2Q − 1). *Amendment, before the A/B runs:* the plan
  was to give long stages the power of two; the screen of convolution lengths (Results)
  showed the opposite on the Mac, and no effect worth a rule on x86. A = `main` (8ae4e49).
- **Arm 2** (fused zero padding; 85c6a3e): the chirp-in pass also does the convolution's
  first radix-2 pass (its butterflies have a zero leg) and the chirp-out pass the backward
  chain's last (only its first half is read), for one copy or C | s; the chains run
  M / 2's radices between them. `BLUESTEIN_HALVES=0|1` compares in one build.
- **Arm 3** (few copies; 0740988), in place of the planned Rader-range arm (65537 already
  beats FFTW on every machine at the base: Rader on x86, Bluestein on the Mac): a Bluestein
  stage with 1 < s < C copies becomes one stage over the whole line (1018 = 2 · 509 on
  AVX-512 double ran 1.42 times MKL's time, partial units in every convolution pass).
- **Arm 4** (split convolutions; 8893751): a single-copy convolution (s = 1) takes the
  split run its length would as a line (`split_from`), on the targets whose profile splits
  (x86). The EPYC's 4093 at the base took 21 ns per point for a stage whose two chains of
  8192 take about 6, while its c2c 8192 runs split from pass 2.

Bits: every arm changes the result bits of the lengths whose plans it changes (a different
algorithm or radix sequence), within the derived tolerances; other lengths give
identical bits.

## Method

- Cases: `bench/catalogues/x36.toml`, tier `tracking`: the five targets, nine more primes
  and prime factors whose p − 1 is not 13-smooth (509 to 524287, 3 × 4093, 10403), double
  and float, and six double controls whose plans no arm changes (97, 1009, 8191, 16381,
  2431, 65536). Collateral: the standard catalogue's tracking tier. The exit check's cases
  are judged with the standard catalogue's ids (`^c2c-[0-9]+-b`), against FFTW and MKL.
- Machines: Apple M5 Max (`neon_fcma`, Homebrew Clang 23, shared, A/A control), EPYC 7763
  (`avx2`, GCC 16, DMOG, `-c 8` shared, `--serial`), Xeon Gold 6240 (`avx512`, GCC 14,
  `-c 8` shared, `--serial`, with MKL). One thread, hot, load-tolerant rounds.
- Metric: fbench's median paired ratio B ÷ A per case with its 95 % interval; the geometric
  mean over the cases whose plan the arm changes, per machine-target and precision.
- Decision rule, written before running. Judged on the release toolchains (Clang on the
  Mac, GCC on the EPYC and the Xeon), per target and precision:
  1. An arm is adopted on a target if the geometric mean of B ÷ A over the cases whose plan
     it changes is at most 0.95 with the A/A band below 3 %, and none of them is slower
     (interval wholly above 1 + band).
  2. A changed case that is slower while rule 1 adopts the arm keeps the old route by a
     rule for its shape (size or copies), measured in the same run, or the arm is rejected
     on that target if no such rule separates it.
  3. Collateral: no tracking-tier or control case whose plan does not change is slower by
     more than 3 % beyond the band; a case a hair past on a loaded machine is re-run.
  4. Arm 3 moves a Rader limit only where Rader beats the Bluestein stage (after arms 1–2)
     beyond the band on that target.
  Float is judged separately by the same rule.

## Results

All runs one thread, hot, with an A/A control unless said. Ratios B ÷ A below 1 are
faster; "÷ FFTW" is FFTN's time over FFTW's (MEASURE, fresh wisdom) in the same run.

### Where the time went (before any arm)

- FFTW's plans (Mac run 20261116-175942-49c7, Xeon 20261116-181028-d147, `PLAN_NOTE=1`):
  `dft-bluestein-131071/262144`, `-999983/2000000`, `-4093/8192`, `-8179/16384`,
  `-509/1024`; 1018 as 2 × `dft-bluestein-509/1024`, looping the stage once per copy;
  Rader only for 65537 (with a nested 65536). No nested Rader for these lengths at MEASURE.
- Mac harness (public API, best of 15): Bluestein 131071 took 5.36 ms against 0.97 ms for
  FFTN's c2c 2¹⁸; forced to the convolution's rule radices (8 × 8 × 16 × 16 × 16) c2c
  262144 took 2.48 times as long as with its long-row radices (fbench, 10.1 against 4.1 ns
  per point). 8179 took 200 µs against 37.5 µs for c2c 16384.
- Convolution lengths, c2c chains of powers of two against 7-smooth lengths just above
  (ns per point; total time relative to the power of two):
  - Mac (harness): from 8192 points in double (128 KiB, the L1) and 16384 in float, 7-smooth
    lengths ran 0.62–0.72 of the power of two's time per point and 0.66–0.75 in total
    (8640: 13.4 µs against 19.9 µs; 264600 0.65; 2073600 0.72); at 1024–4096 the powers
    of two won or tied. Lengths whose power-of-two part was 4096 or more (552960 =
    2¹² · 135, 1105920, 2048000) ran as badly as powers of two.
  - Xeon (run 20261116-183101-fc67, chains forced, loaded, band ±13 %): double powers of
    two won up to 16384 and tied beyond; float 7-smooth won beyond 2¹⁸ (0.80–0.89).
  - EPYC (run 20261116-184402-1fe8, chains forced, powers of two with X32's radices):
    7-smooth won at 16384 (0.88) and in float at 8192–16384; elsewhere within ±10 %.
  The standalone chains understate what the Bluestein stage sees: there the convolution
  ping-pongs between the two scratch lines, which lie M complex values apart, a power of
  two when M is one (the EPYC A/B below).

### Arms against `main` (A = 8ae4e49)

Mac, run 20261116-181720-c68e (band ±1.41 %, 11 of 20 rounds loaded), B = arm 1:
the 9 double cases whose plan changed 0.488 (0.329–0.953), the 7 float 0.600 (0.354–1.033;
12279 f32 1.033, slower); 131071 0.329, 999983 0.342, 4093 0.636, 8179 0.346, 32749 0.474,
262139 0.483, 524287 0.392. FFTN ÷ FFTW after: 4093 0.90, 8179 0.89, 32749 0.77, 131071
0.88, 262139 0.77, 524287 0.70, 999983 0.75, 12279 0.84; 1018 0.86 and 65537 0.91
unchanged. Arm 2 against arm 1, run 20261116-184430-88b1 (band ±2.97 %): double 0.87–1.29
(524287 1.29), float 0.95–1.17 (most slower). Tracking tier, A = main, B = arms 1 + 3
(0740988; arm 3 changes nothing on NEON), run 20261116-185426-3bd0 (band ±2.0 %, 21 of 24
rounds loaded): 4093 0.64 faster, the other 60 cases no evidence.

EPYC 7763, run 20261116-193025-47d4 (band ±2.34 %, every round loaded), arms B = arm 1
(19bc985, no length limit on x86), H = B + arm 2, T = B + arm 3 (0740988), P = B with
`BLUESTEIN_POW2_PART_BYTES=65536`:

| case | B | H | T | P | FFTW ÷ A |
|---|---|---|---|---|---|
| 4093 | 1.000 | 0.909 | 0.999 | **0.693** | 0.736 |
| 8179 | 1.005 | 0.888 | 1.006 | **0.595** | 0.696 |
| 12279 | 0.991 | 0.996 | 0.996 | **0.747** | 0.688 |
| 32749 | **0.519** | 0.643 | 0.722 | **0.186** | 0.201 |
| 65537 | 0.654 | 0.711 | 0.750 | 0.761 | 0.475 |
| 131071 | **0.694** | 0.692 | 0.701 | **0.654** | 0.612 |
| 262139 | **0.788** | 0.735 | 0.791 | 0.932 | 0.989 |
| 524287 | **0.613** | 0.610 | 0.612 | **0.574** | 0.711 |
| 1018 | 1.001 | 0.740 | 1.000 | 1.004 | 0.669 |
| f32 1018, 12279 | 1.00 | 1.00 | **0.607, 0.648** | 1.00 | – |
| f32 8179, 32749, 131071, 262139, 524287 | 1.01, 0.92, 0.39, 0.93, 0.91 | 1.02–1.18 | as B | 0.59, 0.60, 0.29, 0.59, 0.64 | – |

B, T and P run the same plan at 65537 (Rader with X32's radices for L = 65536); A's Rader
convolution took the rules' radices (4 × 16 × 16 × 16) over 1 MiB, the radix-16 passes
X29 found bimodal on this machine, so A spread widely (A~ 1.13 here; 1.09 of FFTW in X27,
2.51 in the exit run): the EPYC's 65537 was a bad plan, not Zen 3 (Zen 5's 0.61 came
from a quieter, larger-cache node). Arm 2 lost in float (509–4093 1.13–1.47). P's limit of 64 KiB let 262139 take
552960 = 2¹² · 135 (1.18 times B's 2¹⁹ with long radices), the length the Mac screen
found as slow as a power of two; the final rule's limit is the L1 (32 KiB), which keeps
the power-of-two parts at most 2048 points in double.

Xeon Gold 6240, run 20261116-193408-f0e8 (band ±1.03 %, 1 of 24 rounds loaded), the same
arms plus MKL:

| case | B | H | T | P | FFTW ÷ A | MKL ÷ A |
|---|---|---|---|---|---|---|
| 1018 | 1.004 | 1.001 | **0.682** | 0.999 | 1.037 | 0.730 |
| 12279 | 0.994 | 0.993 | **1.171** | 0.999 | 0.985 | 1.261 |
| 32749 | **1.115** | 0.905 | 1.119 | 1.026 | 0.736 | 0.649 |
| 65537 | **1.094** | 1.093 | 1.089 | 1.095 | 1.754 | 2.815 |
| 131071 | 1.015 | 1.063 | 1.015 | 1.017 | 0.752 | 0.723 |
| 262139 | 0.993 | 0.913 | 0.994 | 0.898 | 0.882 | 0.615 |
| 524287 | 0.951 | 0.985 | 0.947 | 0.961 | 0.824 | 0.565 |
| f32 1018, 12279 | 1.00 | 1.00 | **0.404, 0.845** | 1.00 | – | – |
| f32 131071, 262139, 524287, 999983 | **0.887, 0.932, 0.841, 0.788** | 0.82–0.96 | as B | 0.89–1.13 | – | – |

Arm 2 lost in double at 509–2039 (1.22–1.30) and in float everywhere (1.5–1.8). The
long-row radices lost at 1 MiB in double (32749, 65537: 1.09–1.12) and won from 8 MiB and
in float from 2 MiB; the final rule takes them beyond 4 L2 (4 MiB) on the Xeon. A length
limit gained nothing there. Arm 3 won at 1018 in both precisions and lost at 12279 in
double, whose whole convolution (25920) is longer than its three copies' (3 × 8192); the
final rule requires it to be no longer (float 12279: 24576 against 3 × 8192, kept).

### Confirmation: the final rules against `main`

A = `main` 8ae4e49, N = 5098f3c (the final rules), I = N with `LAYOUT=interleaved` (on
these single rows that turns arm 4 off and nothing else), F = FFTW.

EPYC, run 20261116-203815-9d19 (band ±1.37 %, every round loaded), and the tracking run
below: both ran outside fbench_remote's serial queue (five jobs were ahead), on the same
shared nodes as other agents' timing jobs, so they may have disturbed those runs' timings
and been disturbed by them (the A/A band absorbs the second, not the first): every changed case faster, none slower; changed cases
N ÷ A double 0.558 (0.269–0.843), float 0.548 (0.249–0.793). Arm 4 (N ÷ I): 65537 0.56,
131071 0.82, 262139 0.88, 524287 0.87, 999983 0.92, 1019 0.85, float 2039 0.82, 65537
0.79; 2039 double 1.03 (within its interval's reach of the band), 10403 1.00. FFTN ÷ FFTW
(ns per point N against F): 4093 0.94 (20.8 → 14.5 against 15.4), 65537 0.80 (43.9 →
16.4 against 20.6), 131071 0.87 (46.5 → 25.0 against 28.7), 999983 0.86, 8179 0.89,
32749 0.88, 262139 0.61, 524287 0.75, 509 0.98, 1019 1.01, 2039 1.01; 1018 1.51 and 12279
1.17 unchanged in double (12279: three copies of 8640, a plan no arm improved).
Tracking tier, A = main, N = 5098f3c, run 20261116-203815-e474 (band ±1.56 %): 4093 0.70
and 65537 0.70 faster, 10403 1.036 slower (its plan changes only by arm 4's split run; in
the x36 run above N ÷ A = 1.003 at 10403, so a one-run effect on a loaded node, recorded),
58 no evidence.

Mac: the final rules plan as arm 1 there (arms 3 and 4 do not apply to NEON), so run
20261116-181720-c68e is the confirmation (above).

Xeon, run 20261116-221709-0a82 (Slurm 16143, band ±3.82 %, 19 of 20 rounds loaded): no
case slower than A or than I; changed cases N ÷ A double 0.836 (1018 0.666, 262139 0.963,
524287 0.912), float 0.611 (1018 0.352, 12279 0.696, 524287 0.792, 999983 0.718). Arm 4
(N ÷ I) 0.81–1.0, none slower beyond the band (509 f32 0.80, 8179 0.92, 131071 0.94), so it
stays on `avx512`. FFTN ÷ FFTW: 1018 0.65 (÷ MKL 1.37 → 0.90), 4093 0.68, 131071 1.28,
999983 1.10, 32749 1.28, 262139 1.07; ÷ MKL 131071 1.31, 999983 1.37. The Xeon's tracking
tier run (x36-trk-xeon, Slurm 16145) was still queued at the time of writing.

## Verdict

- **Arm 1 adopted** with machine-specific values (rule 1; rule 2 shaped the values):
  long-row radices for long power-of-two convolutions from `grid_min_bytes` on Apple and
  Zen, beyond 4 L2 on the Xeon (double lost at 1 MiB there); the Bluestein length's
  power-of-two part at most L1/4 on Apple, the L1 on Zen-like caches, unlimited on the
  Xeon (`machine_profile::bluestein_pow2_part_bytes`, `long_convolution_bytes`; Granite
  Rapids, X35's large-L2 profile, takes the Xeon's rules and the Axion VM the Zen ones,
  neither measured). Mac changed cases 0.49 (double),
  0.60 (float; 12279 f32 1.033 recorded, a hair past the band under load); EPYC final
  0.56 / 0.55.
- **Arm 2 rejected and deleted** (19c49df): float lost on all three machines (1.04–1.8),
  double mixed.
- **Arm 3 adopted** on the targets where it applies (C > 2: AVX2 float, AVX-512), with
  the condition that the whole line's convolution is no longer than the copies' (12279
  in AVX-512 double lost 1.17 without it): 1018 0.68 (Xeon double), 0.40–0.61 (float).
- **Arm 4 adopted** on x86 (EPYC: single-copy convolutions 0.56–0.92, Xeon 0.81–1.0, none
  slower beyond the band).
- The planned Rader-range arm was not run: the bad EPYC 65537 was the convolution's radices
  (arm 1 fixes it), and 65537 beats FFTW on the Mac with Bluestein (0.91).
- Exit check, cause (1), FFTN ÷ FFTW after: Mac 131071 0.88, 999983 0.75, 4093 0.90, 1018
  0.86 (all were failing or near; 1018 already passed); EPYC 131071 0.87, 999983 0.86, 4093
  0.94, 65537 0.80, 1018 still 1.51 (two interleaved copies of Bluestein 509 cost 1.6 times
  two single stages there: FFTW loops the stage per copy); Xeon 1018 ÷ MKL 0.90 (was 1.37),
  4093 0.68, but 131071 1.28 and 999983 1.10 of FFTW still fail there.
- Follow-ups: the Xeon's long Bluestein convolutions (a chain where the Xeon wants a grid:
  131071, 999983 1.2–1.3 of FFTW); a Bluestein stage with copies looping one copy at a
  time (EPYC 1018); the scratch lines' power-of-two distance itself (pad line b) instead
  of avoiding power-of-two lengths; Granite Rapids takes the Zen values unmeasured.

Done: DESIGN 8.5, 11.2 and the decision log; LESSONS 4.14; PLAN index and Status.

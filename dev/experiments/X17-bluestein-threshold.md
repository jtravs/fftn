# X17: generic pass or Bluestein for prime factors above 16

- **Status:** done
- **Owner:** task 2.3 agent, 2026-11-14
- **Phase / task:** 2.3

## Question

Up to which prime factor does the generic pass (direct evaluation, O(q²) per butterfly) beat
a Bluestein stage (a chirp-z transform through a smooth-length convolution, DESIGN.md 8.5)?
The answer is the planner profile's Bluestein threshold (DESIGN.md 11.2).

## Hypothesis

The generic pass costs about q² complex multiply-adds per transform of q points and a
Bluestein stage two chains of M ≥ 2q − 1 points plus two elementwise passes, so the
generic pass should win only for the smallest primes, somewhere between 17 and 50, and lose
by a growing factor beyond. With many copies the generic pass vectorises across them, so
the crossover moves up for composites.

## Change under test

The Bluestein threshold, through the test knob `build_settings::bluestein_above` (an
internal construction setting, not a library switch, so not in `dev/KNOBS.toml`): arm G
keeps every prime factor in generic passes (threshold `SIZE_MAX`), arm B sends every prime
factor above 16 through Bluestein (threshold 16). Same build, same plan otherwise.

## Method

- Cases: c2c forward, one row, out of place, primes 17 to 257 (17, 19, 23, 29, 31, 37, 41,
  43, 47, 53, 59, 61, 67, 71, 79, 97, 127, 257) and composites with a prime factor above 16
  and many copies (34, 68, 272, 1088, 2431 = 11 × 13 × 17, 4352, 323, 4097, 3757, 1564,
  5440, 6137).
- Machine: Apple M5 Max (shared, under the usual load), Homebrew Clang 23, Release,
  `neon_fcma`, one thread, double and float.
- Harness: a short timing program against the library's internal plan constructor (both
  arms built from one source tree, uncommitted): each arm warmed, then the best of 7
  batches of about 2 M point-operations, three interleaved repetitions per arm, best
  taken. A quick screen, as the brief allowed (under 15 minutes in total); no fbench A/B,
  because the effect is far beyond any noise band.
- Decision rule, written before running: the threshold is the largest prime at which arm G
  is faster than arm B in both precisions; 16 if there is none.

## Results

Time of B ÷ time of G (below 1: Bluestein faster):

| n | double | float |
|---|---|---|
| 17 | 0.40 | 0.14 |
| 19 | 0.35 | 0.12 |
| 23 | 0.29 | 0.089 |
| 31 | 0.20 | 0.059 |
| 61 | 0.088 | 0.027 |
| 97 | 0.056 | 0.017 |
| 257 | 0.023 | 0.0074 |
| 34 = 2 × 17 | 0.35 | 0.50 |
| 1088 = 64 × 17 | 0.33 | 0.40 |
| 2431 = 11 × 13 × 17 | 0.34 | 0.43 |
| 5440 = 320 × 17 | 0.36 | 0.49 |
| 4097 = 17 × 241 | 0.033 | 0.031 |

Every other case lies between its neighbours in the table. Bluestein wins everywhere, by
2.5 times at 17 in double (where the generic pass vectorises best: C = 1) and by 2 to 3
times at composites with many copies.

The acceptance run of task 2.3 (fbench, A = NEON FFTW at MEASURE, B = FFTN, threshold 16)
is cited in the task's merge commit.

## Verdict

Adopted: the threshold is 16 on every profile (`default_bluestein_above` in
`src/planner/rules.cpp`), so every prime factor above the codelets goes through Bluestein.
Measured on the Mac only; the other targets take the same value until measured. The generic
pass stays for radices a test forces (and for a profile that raises the threshold); in the
direct form it has, it lost at every length measured, so its symmetric form (DESIGN.md
9.2), or its deletion, is the maintainer's call.

## x86 (2026-11-15)

- **Question.** Does Bluestein beat the generic pass from 17 up on x86 too, so that the
  threshold of 16 holds for the `avx2` and `avx512` profiles, in double and float? The
  generic pass vectorises across copies in C lanes, and C is 2 to 8 on x86 (1 to 2 on
  NEON), so a single prime row (s = 1) idles more lanes there and composites with many
  copies use more; if the generic pass wins anywhere, it is most likely at composites
  with many copies of 17 or 19 on AVX-512.
- **Arms.** The same knob as the Mac run, now an fbench option (`BLUESTEIN_ABOVE`, sets
  `build_settings::bluestein_above`): A = `fftn+ISA=<t>+BLUESTEIN_ABOVE=never` (every prime
  factor in generic passes; arm G above), B = `fftn+ISA=<t>` (the profile's threshold, 16;
  arm B above). One commit, one build; the ratio is B ÷ A, as in the table above (below 1:
  Bluestein faster). An A/A control gives the noise band.
- **Cases.** `bench/catalogues/x17.toml`, tier `tracking`: c2c forward, one row, out of
  place, primes 17 to 257 and composites 34, 68, 272, 1088, 2431, 4352, 5440, 4097, double
  and float.
- **Machines.** The Xeon (Intel Xeon Gold 6240, Cascade Lake, GCC 14, `-c 8` shared Slurm
  allocation), `avx512` and `avx2` forced, one job each; the EPYC (AMD EPYC 7763, Zen 3,
  DMOG, GCC 16, `-c 8` shared), `avx2`. One thread, hot, fbench's load-tolerant rounds.
- **Decision rule, written before running.** Per target and precision, the threshold is
  the largest prime p at which B is `slower` (generic pass faster beyond the noise band) at
  the prime row p, provided every prime row below p is too; 16 if B is not slower at 17.
  Since a profile is per target and complex count, a threshold may differ between double and
  float of one target. Composites do not set the threshold (the rules apply one threshold
  to every factor); if the generic pass wins at composites while losing at the primes, that
  is recorded here, not acted on.

## Follow-up: plan memory of the Bluestein tables (task 2.3b)

Not a threshold change, recorded here because it changes the stage measured above
(DESIGN.md 8.5). The chirp and the kernel spectrum were stored once per element of the
line (s Q and s M entries for s copies); they are now stored once (Q broadcast entries
when C divides s, otherwise s Q; M kernel entries always) and shared by the axes of a plan.

- Plan memory (internal plan core, double, Mac): n = 17 · 2²⁰ (s = 2²⁰, M = 36): tables
  1174 MB before, 285 MB after, of which the Bluestein stage's were 889 MB and are now
  under 1 KB; the rest is the head chain's grouped-pass twiddles (about n entries, as any
  chain of that length has). Workspace unchanged: 1208 MB per team member (two scratch
  lines of s M); running the copies in blocks would cut it but needs moving scratch slots
  in the program (DESIGN.md 8.5, "Not done").
- Where the shared kernel applies: as the forward chain's last output factor when C divides
  s (lanes share one entry), else as the backward chain's first input factor. The input
  form everywhere cost about 5 % at 4096 × 17 and 3 × 11 × 13 × 17 against main (fbench
  20261114-132546-1598, A/A band ±1.0 %).
- Speed, final code against main, Apple M5 Max, Homebrew Clang 23, Release, `neon_fcma`,
  one thread, double, shared machine: fbench 20261114-135103-5f6e (A = NEON FFTW at
  MEASURE, B = main a3903ed, C = f4c9065, which is c541c20 rebased, A/A ±1.0 %) over the tracking tier's prime and
  awkward rows, rows with copies and planes with prime axes (14 cases): C ÷ B from 0.84
  (4352 = 256 × 17) to 1.05, geometric mean 0.99; the cases above 1.02 (4093, 69632) were
  not slower in the repeat 20261114-140239-9661 (69632: 0.97, no evidence) or swing both
  ways between runs, as s = 1 primes whose code did not change do (4093: 0.96 in
  20261114-131632-4a44, 1.04 in 20261114-135103-5f6e). Verdict: adopted; no regression
  beyond the run-to-run spread, faster with many copies.

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

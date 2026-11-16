# X36: large primes and prime factors outside Rader's range

- **Status:** running
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

- **Arm 1** (long convolutions): a convolution whose length is a power of two longer than
  the profile's grid threshold takes FFTL.jl's long-row radices (X32), in every profile
  (on the Xeon too: a convolution cannot be a grid, and the rule's radices are worse than
  either); and a Bluestein stage whose power of two above 2Q − 1 is that long takes that
  power of two as M. A = `main`, B = arm 1.
- **Arm 2** (fused zero padding): the chirp-in pass also does the convolution's first
  radix-2 pass, and the chirp-out pass the backward transform's last radix-2 pass, for a
  single row (s = 1); the convolution chain is M / 2's radices between them. A = arm 1,
  B = arm 2.
- **Arm 3** (Rader range): with arm 1, the Rader range per profile revisited where
  arm 1 changes the Rader convolution (65537 on NEON) and where X27 lost with C ≥ 4.

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

(running)

## Verdict

(running)

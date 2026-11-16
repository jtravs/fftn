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

(running)

## Verdict

(running)

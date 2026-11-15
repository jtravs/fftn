# X27: generated codelets for primes above 16; Rader for primes with smooth p − 1

- **Status:** running (arms 1 and 2 built; x86 timing queued)
- **Owner:** X27 agent, 2026-11-15
- **Phase / task:** PLAN Status lead 2 (one-dimensional double gap); B3 (build size)

## Question

Do generated codelets of radix 17, 19 and 23, used as the leaf of n = p and as the pass of
each such factor in a composite, beat the current route (a Bluestein stage for every prime
factor above 16, X17) per target and precision, and at what cost in library size? Second:
does Rader's algorithm beat Bluestein for primes whose p − 1 is smooth (97, 257, 65537)?
The answers set which radices the generator emits (`gen/fftn_gen/catalogue.py`) and the
planner's route for prime factors above 16 (DESIGN.md 11.2).

## Hypothesis

- **Codelets.** A Bluestein stage for q points costs two chains of M ≥ 2q − 1 points (36 or
  48 for 17) plus two chirp passes: for one row of 17 that is four or more passes over 36+
  points against one codelet call. The generator already builds odd primes in the
  symmetric-pair form (gen/README.md): 17 costs 320 real additions and 242 multiplications
  (168 vector operations, 113 of them fused), 19 costs 396 and 308 (207), 23 costs 572
  and 464 (297), against Bluestein's roughly 2 × 48 × (4 + log₂ 48) complex operations.
  Baselines (PLAN Status): c2c 17 takes 1.57 (Mac), 1.95 (EPYC), 1.79 (Xeon) of FFTW's
  time; 2431 = 11 · 13 · 17 is 2.57 of FFTL's time on the Mac and 1.25 of FFTW's on the
  EPYC. FFTW runs these lengths with its own codelets or direct passes, not Bluestein. So
  the codelet should win by about 2× at single rows of 17–23 and by 1.3–2× at composites,
  most where the factor has many copies (the pass runs across copies in all C lanes).
  Against: peak live values are 20 (17), 22 (19) and 26 (23) complex registers, so the
  codelets spill on AVX2's 16 registers (radix 16, at 18, already does a little), and
  each radix adds about 50 pass instantiations per precision and target (B3).
- **Rader.** For p = 257 the cyclic convolution has length 256 (a power of two) against
  Bluestein's M ≥ 513 (7-smooth, about 525–576); for 65537 it is 65536 against M ≥ 131073.
  Half the convolution length should be worth up to about 2× on those primes if the
  generator-order permutation (a gather by powers of a primitive root) is cheap enough;
  4093 does not qualify (4092 = 4 · 3 · 11 · 31). Arm 2 runs only if time allows.

## Change under test

- **Arm 1** (2e5e4a8): radices 17, 19 and 23 added to the generator's catalogue
  (complex lowering only; the split lowering stays at 2–16), a row unit per prime per
  target, and the planner unchanged in form: those factors were already listed last,
  after the cost-priced radices up to 16, and a factor with a codelet is now a codelet pass
  instead of a Bluestein stage (`chain_builder::radices`); n = p is a leaf, and an r2c or
  c2r of 2p a real leaf. The grid's phase passes and the fused real pairing never take
  these radices, so those entries stay null. A = `main` (3f90275), B = arm 1.
- **Arm 2** (f3b4722, then dd4adf2, a8b0384, cddfe86): a Rader stage in place of the
  Bluestein stage for a single prime q whose q − 1 is smooth, sharing the Bluestein
  stage's convolution. As first written (f3b4722) it stepped g^m mod q with a subtraction
  loop and ran a single row one broadcast register at a time: slower than Bluestein at
  65537 and in float (Mac screen, not kept); dd4adf2 reads g^m from a table and runs one
  copy in scalar arithmetic; a8b0384 adds the size limit q < 2^16; cddfe86 lets q − 1 have
  the factors 11 and 13. A = arm 1, B = the arm-2 commit.

Bits: arm 1 changes the result bits of every length with a factor 17, 19 or 23 (a
different algorithm: within the derived tolerances, which the oracle and FFTW tests
check); every other length runs the same plan and gives identical bits.

## Method

- Cases: `bench/catalogues/x27.toml` tier `tracking`: c2c rows 17, 19, 23, 34, 51, 68,
  289, 2431, 4913, 17408 (= 17 · 1024), 10403 (= 101 · 103, control: no factor changes),
  97, 257, 4093, 65537; batches 17 × 256; r2c and c2r 34 and 4862 (= 2 · 2431); double and
  float. Plus the standard catalogue's tracking tier for collateral damage.
- Machines: Apple M5 Max (`neon_fcma`, Homebrew Clang 23, shared, A/A control), EPYC 7763
  (`avx2`, GCC 16, DMOG, `-c 8` shared, `--serial`), Xeon Gold 6240 (`avx512`, GCC 14,
  `-c 8` shared, `--serial`). One thread, hot, fbench's load-tolerant rounds.
- Metric: fbench's median paired ratio B ÷ A per case with its 95 % interval; the
  geometric mean over the target cases per machine-target and precision; the A/A band.
- Build size: the text size of each target's row units and of `libfftn.a`, before and
  after, on arm64 (Clang 23, Release) and on x86-64 (the same compiler cross-compiling the
  `avx2` and `avx512` units); per radix where the arm keeps fewer than three.
- Decision rule, written before running. Judged on the release toolchains (AGENTS.md
  section 6: Clang on the Mac, GCC on the EPYC and the Xeon), per target and precision:
  1. Arm 1 is adopted on a target if the geometric mean of B ÷ A over the target cases
     (every case with a factor 17, 19 or 23) is at most 0.90 with the A/A band below 3 %,
     and no target case is slower (interval wholly above 1 + band).
  2. A radix (19 or 23) stays only if its own prime row is faster beyond the band on every
     machine measured; otherwise it is dropped from the catalogue (B3: no code for nothing).
  3. Collateral: no tracking case without such a factor slower by more than 3 % beyond the
     band; a case a hair past on a loaded machine is re-run before it decides.
  4. A target case that is slower while the rule adopts the arm (for example a composite
     with many copies, where Bluestein's convolution ran in all lanes) keeps Bluestein by
     a planner rule for that shape, measured in a follow-up arm, not by rejecting the arm.
  5. Arm 2 (Rader) is adopted for the primes where it is faster than Bluestein beyond the
     band on every machine measured; the rule is then "Rader when p − 1 is 7-smooth".
  Float is judged separately by the same rule.

## Results

(pending)

## Verdict

(pending)

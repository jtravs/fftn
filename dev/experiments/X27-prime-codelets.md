# X27: generated codelets for primes above 16; Rader for primes with smooth p − 1

- **Status:** done: arm 1 adopted everywhere; arm 2 (Rader) adopted per target and precision
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

All runs one thread, hot, with an A/A control; the Mac (Apple M5 Max, Homebrew Clang 23,
`neon_fcma`) shared with other agents' runs, the EPYC 7763 (DMOG, GCC 16, `avx2`, `-c 8`
shared, most rounds under load) and the Xeon Gold 6240 (GCC 14, `avx512`, `-c 8` shared)
through `fbench_remote.py --serial`. Ratios are B ÷ A (below 1: B faster); "targets" are
the cases with a factor 17, 19 or 23 (15–16 per precision), "controls" 10403, 97, 257,
4093, 65537, whose plans arm 1 does not change.

### Arm 1: codelets of 17, 19, 23 (A = main 3f90275, B = 2e5e4a8)

| machine, run, band | targets f64: geomean (worst) | targets f32 | controls |
|---|---|---|---|
| Mac, 20261115-193346-8c5f, ±1.3 % | 0.258 (r2c 4862: 0.415) | 0.290 (23: 0.386) | 0.977–1.021 |
| EPYC, 20261115-210826-412f, ±1.0 % | 0.306 (r2c 4862: 0.438) | 0.306 (289: 0.652) | 0.969–1.007 |
| Xeon, 20261115-225300-75d0, ±1.15 % | 0.282 (23: 0.497) | 0.236 (289: 0.476) | 0.994–1.010 |

Every target case is faster beyond the band on every machine in both precisions; no
control moved. Per radix, double, Mac / EPYC / Xeon: 17 0.18 / 0.22 / 0.27, 19 0.22 /
0.28 / 0.37, 23 0.28 / 0.35 / 0.50. FFTN ÷ FFTW after arm 1, double: c2c 17 0.28 (Mac,
was 1.57), 0.42 (EPYC, was 1.95), 0.47 (Xeon); 2431 0.37 / 0.50 / 0.27; 17 × 256 0.19 /
0.31 / 0.33. On the Mac 2431 takes 0.37 of its old time, which puts FFTN at about FFTL's
(it was 2.57 times FFTL's, PLAN Status).

Collateral, tracking tier: Mac A = main, B = a8b0384 (arm 1 + Rader below 2^16), run
20261115-201024-7a7a (80 rounds, all under load, band ±1.0 %): 5 faster (17, 97, 257,
1009, 2431), 55 no evidence, 1 slower: r2c 2048 × 36 at 1.045, whose plan is unchanged;
re-run 20261115-203756-b0b0 (band ±2.2 %): 1.038, no evidence (code-layout noise). EPYC
A = main, B = cddfe86, run 20261116-002503-0a91 (band ±1.0 %): 5 faster, 56 no evidence.
Xeon A = main, B = arm 1, run 20261115-231022-8115 (band ±1.5 %): 2 faster, 59 no evidence.

Build size (B3), text of the kernel units, Clang 23 Release (x86-64 cross-compiled):
neon_fcma 4.18 → 5.16 MB (+23.5 %), avx2 9.90 → 12.49 MB (+26.2 %), avx512 14.65 → 17.74
MB (+21.1 %), scalar +26–29 %; `libfftn.a` +25 % on both. Per prime, of a target: 17
+6 %, 19 +7 %, 23 +10 %. The grid's phase passes and the fused real pairing, which never
take these radices, are not instantiated for them (otherwise +35 %; leaving them out also
avoided a Clang 23 crash in the SLP vectorizer on `c2r_first<float, 17>` for x86-64
scalar). Peak live values: 20, 22, 26 complex registers (spills on AVX2's 16).

### Arm 2: Rader against Bluestein (A = arm 1)

As first written (f3b4722: g^m mod q stepped with a subtraction loop, a single row one
broadcast register at a time), Mac screen 20261115-195223-f669 (stopped after 10 rounds,
band 13 %): 65537 1.5–2.0 and float slower almost everywhere; not kept. With the index table
and scalar single rows (dd4adf2):

| q | Mac f64 / f32 (20261115-200226-b1eb, ±1.2 %) | EPYC f64 / f32 (20261115-233441-5bdc, ±1.0 %) | Xeon f64 / f32 (20261116-002300-c123, ±2.4 %) |
|---|---|---|---|
| 29 | 0.69 / 0.85 | 0.72 / **1.27** | **1.11** / **1.78** |
| 61 | 0.61 / 0.75 | 0.74 / **1.22** | 0.97 / **1.93** |
| 97 | 0.52 / 0.64 | 0.62 / 0.77 | 0.65 / **1.33** |
| 257 | 0.48 / 0.56 | 0.59 / 0.91 | 0.64 / 0.70 |
| 1009 | 0.54 / 0.65 | 0.49 / 0.80 | 0.63 / 0.91 |
| 2017 | 0.46 / 0.61 | 0.44 / 0.47 | 0.62 / 0.77 |
| 4097 = 17 · 241 | 0.57 / 0.61 | 0.72 / 0.79 | 0.72 / 0.80 |
| 7681 | 0.17 / 0.56 | 0.60 / 0.82 | 0.59 / 0.67 |
| 12289 | 0.57 / 0.53 | 0.79 / 0.83 | 0.55 / 0.73 |
| 40961 | 0.82 / 0.62 | 0.55 / 0.96 | 0.49 / 0.57 |
| 65537 | **1.21** / **1.20** | 0.62 / 0.83 | 0.62 / 0.56 |
| 8191, 16381 (13-smooth q − 1; cddfe86) | 0.21 / 0.45, 0.23 / 0.24 (20261115-203357-1ff2) | 0.37 / 0.53, 0.32 / 0.40 | 0.65 / **1.08**, 0.45 / 0.91 |
| 4093 (control) | 1.01 / 1.00 | 1.01 / 1.00 | 1.00 / 1.00 |

Bold: slower beyond the band. Rader loses where a single small prime meets wide vectors
(C ≥ 4: AVX2 float, AVX-512): its gather and scatter of one row are scalar while the
Bluestein stage's chirp passes fill C lanes; and at 65537 on the Mac only (a megabyte
gathered and scattered in random order, and a 2^16-point chain). FFTN ÷ FFTW with Rader at
65537: EPYC 1.09 (was 1.78), Xeon 0.47.

### Confirmation against the new main (d8ca0f1, after X26, X23b, X28)

(pending: A = main d8ca0f1, B = a312699, x27 catalogue; Mac, EPYC job 510124, Xeon job
15889.)

## Verdict

**Arm 1 adopted on every target, both precisions** (rule 1: target geomeans 0.24–0.31,
nothing slower; rule 2: 19 and 23 each faster on every machine, so both stay; rule 3: no
collateral). The library grows by 25 % (B3, recorded for the maintainer's build-size
decision; 23 alone is 10 % of a target for 0.28–0.50 of its old time).

**Arm 2 adopted per target and precision** (rule 5, applied per target as AGENTS.md
section 6 allows; the rule's form changed from "q − 1 7-smooth" to the profile's range
with q − 1 13-smooth, both measured): `target_profile::rader_from` = 16 C for C ≥ 4, else 0
(AVX2 float and AVX-512 double from 64, AVX-512 float from 128), `rader_below` = 2^16 on
NEON and scalar, 2^24 on x86. One measured exception kept by the rule: 8191 on AVX-512
float (1.08); 16381 there is 0.91. The final rule (02fd25c) was not timed as a whole on x86
before the confirmation run; each of its choices is a measured winner above.

Done: generator catalogue, row units, planner, DESIGN (8.5, 9.2, 11.2, 15.2, 18.2,
decision log), LESSONS 4.10 and 4.11. Follow-ups: 128-bit (X28-style) leaves for 17–23 on
x86, where a single row now idles 1–7 lanes; a vectorised gather/scatter for single-row
Rader (would extend it down to small primes on AVX-512 and up past 2^16 on NEON); Rader
inside the generator for 13 (DESIGN 18.2); radices 17–23 inside the grid (lengths with a
factor 17 still run as chains: 17408 is 0.37–0.58 of FFTW's time already).

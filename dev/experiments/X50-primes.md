# X50: large primes against MKL on the Xeon (Bluestein on the depth-first chain)

- **Status:** done: A0, A, B and C adopted (Xeon and Apple profiles); EPYC unchanged; the Rader arm not built
- **Owner:** x50 session, 2026-11-18
- **Phase / task:** 1 (phase 1 exit check, the tail rule; cause 1, the primes); follows X36,
  X36b, P1b and X46

## Question

Do the Xeon 6240's large primes, 999983 (1.43 of MKL's time, 1.12 of FFTW's) and 131071
(1.33, 1.20), close when a Bluestein stage's convolution runs as two depth-first chains
(X46) instead of two breadth-first chains, and does the EPYC's 65537 (Rader against
Bluestein, bimodal by process in P1b) need a page-robust Rader stage? The answer adds a
form of the Bluestein stage and a rule for when it is taken.

## Hypothesis

- The Bluestein stage's cost is its two transforms of M points (X36: the chirp passes are
  about two pass-equivalents of ten). On the Xeon the convolution is always a breadth-first
  chain (X36b: a grid convolution needs a second movable workspace slot), and a chain of
  2¹⁸ points took 7.4 ns per point against the depth-first chain's 4.4 (X36b's run
  20261117-002122-58ba; X46's 262144 row, 0.685 of its grid). 131071 (M = 2¹⁸, 4 MiB, the
  top of X46's range) should take about 0.65 of its time, under MKL's.
- The depth-first chain fits the program without a new slot: the chirp pass writes scratch
  a; the forward chain's first pass reads a and writes its sub-row blocks into the workspace
  block (movable, so a loop runs the sub-rows); its top pass writes back into a, times the
  kernel spectrum (a factor edge on the top pass); the backward chain does the same from a
  into the block, its top pass in place; the chirp pass out reads the block. Scratch a is
  dead after each first pass, so the sub-rows use it.
- 999983 (M about 2²¹, 32 MiB, beyond X46's range) needs two top factors (sub-rows of about
  8100 points); X46 found two tops no better than one up to 2¹⁸, but there the sub-rows of
  one top fit L2. Unknown until screened.
- EPYC 65537: P1b's three runs disagree (Bluestein 0.57 to 1.49 of Rader's time); Rader's
  gather and scatter visit 256 pages of 4 KiB at random. Left for a counters screen after the
  Xeon (dTLB misses per point in the edge passes).

## Change under test

Arms are commits of `exp/x50-primes` against N = its base (X46's head 80dbcdb, the plans of
`main` after pull request 59):

- **A0 (the stage, forced only).** `chain_builder::bluestein_depth` and the factor edge of
  the depth-first top pass (`depth_top[d][2]`); with `STRATEGY=depth` (and `DEPTH_TOPS`) a
  c2c row of one axis whose chain is one Bluestein stage of one copy runs its convolution as
  two depth-first chains. No plan changes by the rules.
- **A (the rule).** The rules take it where a row of the convolution's length M would be a
  depth-first chain (the profile's depth-first fields on M's bytes, X46's conditions): on
  the Xeon M of 16384–262144 points (8179, 32749, 131071), on the Mac M of 8192–16384 points
  with a power-of-two part above 1024 points; nothing on the EPYC.
- **B (beyond the range; named after the screen, before the judging run).** The screen
  (below) found one top of 16 better than two even at 2²¹ points, so B is: a profile field
  `depth_convolution_max_bytes` extends the range for convolutions only, with the same tops;
  the Xeon profile sets no limit (262139, 524287, 999983 change). Other profiles keep A's
  range.
- **C (fused chirp passes; named after B's partial judging results, before its own run).**
  B left 999983 at about 1.1 of MKL's time. The stage still sweeps M elements twice more
  than its chains need: the chirp pass writes the zero-padded line into scratch a, which the
  forward first pass reads back, and the backward top pass writes M elements to the block,
  of which the chirp pass out reads Q. C fuses the chirp into the forward first pass (it
  reads the input times the chirp, zero from Q on: `depth_first_pass<..., Chirp>`) and into
  the backward chain's last top pass (it stores only the first Q outputs, times the chirp,
  to the output: `depth_top_pass<..., Chirp>`). Same plans otherwise, so judged against B on
  the cases whose convolutions B runs depth-first, by the decision rule below, on the Xeon
  and on the Mac (its primes 4813–5119).

Bits: a changed plan changes the rounding of its rows (another factorisation of the
convolution), within the derived tolerances; tests hold both precisions to them and the
threaded runs bit for bit to the serial ones.

## Method

- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512, oneMKL 2026.1;
  Slurm `-c 8` shared; judging through `fbench_remote.py submit --serial`), then the EPYC
  7763 (`avx2`, GCC 16, DMOG, collateral only: no plan changes) and the M5 Max
  (`neon_fcma`, Homebrew Clang 23, NEON FFTW) on the cases whose plans change there.
- Screen (no default change): the Xeon's c2c 131071, 999983, 65537 and the convolution
  lengths 262144, 2000000, 2073600, 2097152 under the rules (N), the chain (C), the grid (G),
  depth-first with one top of 16 (D) and two (E), FFTW and MKL.
- Judging: catalogue `x36`, tier `tracking` (the large primes in double and float, and
  controls), plus the standard catalogue's prime rows; then the standard tracking tier.
- Metric: median paired ratio B ÷ A per case with its 95 % interval; weighted geomean over
  the cases whose plan the arm changes; FFTN ÷ FFTW and ÷ MKL before and after.
- Decision rule, written before the judging run: an arm is adopted on a machine's profile
  (the release toolchain: GCC on x86-64, Clang on macOS) if over the cases whose plan it
  changes the weighted geomean B ÷ A is below 1 − band with its interval below 1, no such
  case is slower than 1 + max(band, 3 %), and no tracking-tier case is slower than
  1 + max(band, 3 %) (a case just past the limit on a loaded run is re-run once). A slower
  changed case keeps the old plan by a rule on its shape measured in the same run, or the arm
  is rejected on that profile. At most three arms.

## Results

### Screen (Xeon, no default change)

Run 20261117-235037-3881 (Slurm 16426 build, 16427 timing, not `--serial`; GCC 14.4,
`avx512`, double; 10 rounds, 3 under load, band ±3.33 %): N = 80dbcdb (rules), C = chain,
G = grid, D = depth-first with one top of 16, E = two tops of 16, F = FFTW, M = MKL; ns per
point:

| case | N (rules) | C | G | D | E | FFTW | MKL |
|---|---|---|---|---|---|---|---|
| c2c 131071 | 42.7 (Bluestein, chain M 2¹⁸) | 44.2 | 45.0 | as N | as N | 35.6 | 32.1 |
| c2c 999983 | 81.7 (Bluestein, chain) | 83.9 | 83.5 | as N | as N | 73.0 | 56.4 |
| c2c 65537 | 20.4 (Rader) | 20.6 | 20.3 | as N | as N | 39.0 | 61.4 |
| c2c 262144 | 4.42 (depth-first 16) | 8.16 | 6.46 | 4.50 | 4.95 | 5.95 | 7.44 |
| c2c 2000000 | 12.7 (grid) | 18.3 | 12.6 | 11.9 | 12.9 | 17.3 | 16.3 |
| c2c 2073600 | 11.3 (grid) | 17.0 | 11.8 | **9.50** | 10.5 | 17.6 | 13.5 |
| c2c 2097152 | 13.8 (grid) | 16.8 | 13.8 | **10.4** | 10.8 | 17.6 | 18.5 |

(The primes' plans do not change with D or E in this build: the depth-first strategy does
not fit a prime.) The convolution's chains are the C column: at 2¹⁸ the depth-first chain
takes 0.54 of their time, at 2073600 0.56 and at 2²¹ 0.62, with one top of 16 better than
two. By the stage's arithmetic (two transforms of M and two chirp passes), 131071 should
take about 0.65 of its time (about 0.87 of MKL's) and 999983 about 0.62 (about 0.90 of
MKL's). The rows beyond 4 L2 gain too (2073600 0.85, 2²¹ 0.75 of the grid): a lead for
X51 (rows beyond 4 L2), not changed here.

### Judging A and B (Xeon)

Run 20261118-001558-967f (Slurm 16433 build, 16434 timing, `--serial`; GCC 14.4, `avx512`;
24 rounds, 11 under load, band ±1.67 %), catalogue `x36` tier `tracking` (46 cases), arms
N = 80dbcdb, A = e024a70, B = ebda2a7, F = FFTW, M = MKL. No case of A or B was slower than
N beyond the band; every case whose plan did not change (float, the controls, 4093, 65537,
1018 and the other copies) was within 0.98–1.02.

| case (double) | A ÷ N | B ÷ N | FFTN ÷ FFTW, N → B | FFTN ÷ MKL, N → B |
|---|---|---|---|---|
| 8179 (M 16384) | 0.796 | 0.801 | 0.94 → 0.76 | 1.11 → 0.89 |
| 8186 = 2 × 4093 (one stage) | 0.808 | 0.799 | – | – |
| 10403 = 101 × 103 (one stage) | 0.788 | 0.776 | – | – |
| 32749 | 0.658 | 0.657 | 1.30 → 0.85 | 1.46 → 0.96 |
| 131071 (M 2¹⁸) | 0.721 | 0.704 | 1.25 → 0.88 | 1.34 → 0.94 |
| 262139 | 1.019 | 0.742 | 1.11 → 0.82 | 1.63 → 1.21 |
| 262142 = 2 × 131071 (one stage) | 1.008 | 0.740 | 1.54 → 1.14 | 1.62 → 1.20 |
| 524287 | 1.001 | 0.727 | 1.13 → 0.82 | 1.64 → 1.19 |
| 999983 (M 2073600) | 1.000 | 0.786 | 1.15 → 0.90 | 1.44 → 1.13 |

A over its 5 changed cases: geometric mean 0.752 (every interval below 0.84). B against A
over the 4 cases B adds: 0.748 (at most 0.786). Both pass the rule on the Xeon profile.

### Judging C (Xeon and Mac)

Xeon, run 20261118-031525-5c5c (Slurm 16447 build, 16448 timing, `--serial`; 20 rounds, 5
under load, band ±1.15 %), B = ebda2a7 against C = 1f7fc8d, F = FFTW, M = MKL, the primes
whose convolutions B runs depth-first and two controls:

| case (double) | C ÷ B | FFTN ÷ FFTW, B → C | FFTN ÷ MKL, B → C | C ÷ N (main), from both runs |
|---|---|---|---|---|
| 8179 | 0.908 | 0.72 → 0.65 | 0.87 → 0.79 | 0.73 |
| 32749 | 0.857 | 0.81 → 0.70 | 0.96 → 0.82 | 0.56 |
| 131071 | 0.869 | 0.92 → 0.80 | 0.96 → 0.84 | 0.61 |
| 262139 | 0.925 | 0.79 → 0.73 | 1.19 → 1.10 | 0.69 |
| 524287 | 0.898 | 0.80 → 0.72 | 1.22 → 1.09 | 0.65 |
| 999983 | 0.908 | 0.91 → 0.83 | 1.16 → 1.05 | 0.71 |
| 4093, 65537 (unchanged) | 1.006, 1.009 | – | – | – |

C over its 6 changed cases: geometric mean 0.894 (at most 0.925, every interval below
0.96). Mac, run 20261118-005441-4014 (10 rounds, all under load, band ±1.72 %): C ÷ B 4813
0.914, 4999 0.919, 5119 0.891. C passes on both profiles.

### A on the Mac

Run 20261118-001926-1f9e (Homebrew Clang 23, `neon_fcma`; 18 rounds, 17 under load, band
±1.00 %), catalogue `x50`, the primes whose convolution is 10240 points: A ÷ N 4813 0.949,
4999 0.949, 5119 0.951 (all faster beyond the band), float unchanged (1.001–1.004); FFTN ÷
FFTW (NEON) 0.83 → 0.79. A passes on the Apple profile. (B changes nothing on the Mac.)

### Collateral

- Mac, standard tracking tier, N against B (run 20261118-002010-e039; 24 rounds, all under
  load, band ±2.22 %): 61 no evidence, weighted geometric mean 0.995; the largest median,
  r2c 2048 × 36 1.046 (a plan B does not change), re-run once: 0.970 (run
  20261118-003128-7bf4, band ±2.26 %).
- EPYC (no plan changes), standard tracking tier, N against B (run 20261118-001628-7178,
  Slurm 512530/512531, `--serial`; 24 rounds, all under load, band ±3.38 %): 61 no evidence,
  weighted geometric mean 1.009; medians above 1.034 on unchanged power-of-two rows (c2c
  65536 1.109, c2r 65536 1.107, r2c 65536 1.064, c2c 262144 1.058, 16384 1.043, c2r 1024²
  1.044): re-run with 30 rounds (run 20261118-031743-095b, Slurm 512554, band ±2.20 %):
  0.979–1.032, all no evidence (c2r 2²⁰ 1.032 [1.003, 1.049], 1.029 in the first run: a plan
  and code path the arms do not change; recorded).
- Xeon, standard tracking tier, N against B and C (run 20261118-032601-14d8, Slurm 16449,
  `--serial`; 20 rounds, 3 under load, band ±3.62 %): C 1 faster (10403 0.711, a whole-line
  Bluestein stage it changes), 60 no evidence, weighted geometric mean 0.997; B 1 faster,
  60 no evidence, 1.005. Medians past 1 + max(band, 3 %) on plans the arms do not change:
  C c2c 1009 1.063 [0.995, 1.092] (a Rader prime; 0.985 and 1.005 in the x36 run above), B
  r2c 4096 1.049, r2c 1024² 1.048, r2c 256² 1.038: re-run once (below).

### EPYC 65537 (P1b's flip)

DMOG, Slurm 512533 and 512537 (`-c 8`, nodes 17 and 18, both loaded): the benchmark binary
of N run directly, 12 processes each of the rules (Rader) and `RADER=0` (Bluestein),
alternating, with Zen 3's L1 data-TLB events (PMCx045: all reloads, 4 KiB reloads, L2 TLB
misses). Rader 15.6–16.0 ns per point in every process, Bluestein 23.0–24.2 (float 9.7–9.9
against 11.2–11.5); Rader missed the L1 DTLB 1.66 times per point (all on 4 KiB pages,
nearly all hitting the L2 TLB: 0.002–0.015 walks per point), Bluestein 0.06–0.15. The
1 MiB buffers were never huge-page backed (THP "always"), so the flip P1 saw is not
reproduced by the caller's rows; it belongs to measured planning's own scratch (LESSONS
3.20, corrected). Rader's TLB reloads cost at most about 11 of its 47.5 cycles per point
(7 cycles each), so a blocked permutation could gain at most about 0.2 of 65537's time,
which already runs at 0.80 of FFTW's: not built.

## Verdict

- **A0 (the stage, forced only): built**, every target, both precisions, in and out of
  place, tested against the reference and bit for bit across team sizes.
- **A (the rule within the depth-first range): adopted** on the Xeon profile (0.752 over 5
  changed cases) and the Apple profile (0.95 over 3); the EPYC has no depth-first range and
  keeps its plans.
- **B (beyond the range, the Xeon profile): adopted** (0.748 over the 4 cases it adds:
  262139, 262142, 524287, 999983).
- **C (fused chirp passes): adopted** (Xeon 0.894 over 6 changed cases, Mac 0.89–0.92).
- Together, against `main`: Xeon 8179–999983 0.56–0.73 of their time (131071 0.61, 999983
  0.71); FFTN ÷ MKL 131071 1.34 → 0.84, 999983 1.44 → 1.05, 262139 and 524287 1.63 → 1.10,
  1.64 → 1.09; FFTN ÷ FFTW 0.65–0.83 on the changed primes; Mac 4813–5119 about 0.86. The
  exit check's 999983 ÷ MKL stays just past 1.05 in this run.
- **EPYC 65537: no change.** P1b's flip was not reproduced (24 processes, Rader 1.5 times
  faster every time; 1.66 L1 DTLB misses per point, all 4 KiB pages); a page-robust Rader
  stage could gain at most about 0.2 and 65537 already runs at 0.80 of FFTW's time there.
  Not built. EPYC 1018 (1.10 of FFTW, s = 2 copies of 509, no depth-first range on Zen) and
  Axion's 65537 (a Bluestein stage on NEON, no depth-first range on that profile) are
  untouched by this experiment.
- **Library size:** text +6.2 % on the Mac (Homebrew Clang 23, `size` of `libfftn.a`,
  21.76 → 23.10 MB, 80dbcdb → 1f7fc8d), +3.5 % on x86-64 (GCC 14.4, the Xeon's build
  cache, 88.72 → 91.83 MB): the depth-first top pass's factor and chirp variants and the
  chirp first pass, per radix, direction and precision.
- **Left open:** float (the depth-first rules are double only); the EPYC and Zen 5 (no
  depth-first range); Granite Rapids (its large-L2 profile takes neither range, unmeasured);
  Bluestein stages of several copies (the block holds one copy's convolution); 999983 ÷ MKL
  (1.05: MKL's 2²¹-point-class convolution is faster than FFTN's depth-first 2073600); rows
  beyond 4 L2 also gained from one top of 16 in the screen (2073600 0.85, 2²¹ 0.75 of the
  grid), a lead for X51.

Done: DESIGN 8.5 and 11.2 and the decision log; LESSONS 3.20 (corrected) and 4.20; PLAN
index and Status.

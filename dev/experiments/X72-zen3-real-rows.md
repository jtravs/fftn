# X72: mid-length real rows on Zen 3

- **Status:** done (arms A and B adopted)
- **Owner:** x72 session, 2026-11-24 (overnight)
- **Phase / task:** 1 (the client catalogue check's largest gap, PLAN "Status")

## Question

The client catalogue check (2026-11-24, `main` d4b169f, run 20261124-102950-017c) found every
real row of 6144–20480 points out of place on the EPYC 7763 at one thread at 1.05–1.14 of
FFTW's time (12 shapes, 40 % of the catalogue's weight), which holds that machine's one-thread
geometric mean at 1.00 against a target of 0.95. Where does the time go, and which change keyed
to Zen 3's class wins it back?

## Diagnosis

### Plans and per-stage cost (run D1)

Run 20261124-220705-a157 (Slurm 518033, node18, `main` 1b1d129 plus this branch's catalogue and
fbench options, f39e659; GCC 16, `avx2`, double, one thread; every round overloaded: load
about 900 on 128 CPUs, band ±1.00 %), catalogue `x72`: the client catalogue's 17 real shapes of
6144–20480 points (lengths and batches as there) and the half-length c2c rows, both
directions. Arms: FFTN's rules (N), FFTW 3.3.11 MEASURE with `PLAN_NOTE=1` (F), FFTN
`PLANNING=measure` (M). Times per row in µs, FFTN ÷ FFTW, and each library's plan:

| case | N µs/row | N ÷ F | FFTN's plan | FFTW's plan |
|---|---|---|---|---|
| r2c 6144 × 48, × 66 | 9.87 | 1.04–1.05 | chain 12 × 16 × 16, pairing pass | rdft2-ct-dit/6 (hc2c 6) → 3 → 64 → 16 |
| c2r 6144 × 48, × 66 | 9.62 | 1.06–1.07 | pairing pass, chain 12 × 16 × 16 | rdft2-ct-dif/2 → 3 → 64 → 16 |
| r2c 6400 | 10.00 | 1.15 | chain 2 × 10 × 10 × 16, pairing pass | rdft2-ct-dit/20 → 10 → 20 → 16 |
| c2r 6400 | 8.77 | 0.96 | chain fused (2 first) | rdft2-ct-dif/2 → 10 → 20 → 16 |
| r2c 9216 | 14.32 | 0.99 | 4 × 8 × 12 × 12, pairing pass | /2 → 12 → 32 → 12 |
| c2r 9216 | 13.23 | 0.97 | fused | /2 → 12 → 32 → 12 |
| r2c 10240, × 28 | 17.14–18.45 | 1.09–1.11 | 4 × 8 × 10 × 16, pairing pass | /16 → 8 → 20 → 32 |
| c2r 10240, × 4 | 16.6–16.7 | 1.05 | fused | /2 → 8 → 20 → 32 |
| r2c 18432 | 37.00 | 1.16 | 4 × 12 × 12 × 16, pairing pass | /6 → 3 → 6 → 32 → 16 |
| c2r 18432 | 35.55 | 1.10 | fused | /2 → 16 → 3 → 32 → 6 |
| r2c 20480 × 21 | 45.9 | 1.16 | 4 × 10 × 16 × 16, pairing pass | /32 → 16 → 64 → 10 |
| c2r 20480 × 3 | 41.6 | 1.13 | fused | /2 → 8 → 20 → 64 |
| c2c 3072 | 7.41 | 1.06–1.07 | 12 × 16 × 16 | 3 → 64 → 16 |
| c2c 3200 | 7.88 | 1.10–1.11 | 2 × 10 × 10 × 16 | 10 → 20 → 16 (bwd 20 → 8 → 20) |
| c2c 4608 | 11.22 | 0.99–1.00 | 4 × 8 × 12 × 12 | 12 → 32 → 12 |
| c2c 5120 | 13.65 | 1.07–1.10 | 4 × 8 × 10 × 16 | 8 → 20 → 32 (bwd 16 → 20 → 16) |
| c2c 9216 | 26.35 | 0.96–0.98 | depth-first 4 × 2304 | 6 → 6 → 8 → 32 |
| c2c 10240 | 30.29 | 0.99 | depth-first 4 × 2560 | 16 → 64 → 10 |

- **18432 and 20480 are the chain's cost.** Their c2c half lengths 9216 and 10240 are
  depth-first chains (X51) and tie FFTW, but a real row's half-length transform never takes the
  depth-first chain: it is the breadth-first chain, then the pairing. r2c 18432 takes 37.0 µs
  where the half length takes 26.4 as a c2c row; r2c 20480 takes 45.9 µs per row where 10240
  takes 30.3.
- **6144–12800 trail where their half lengths trail.** c2c 3072, 3200 and 5120 lose to FFTW by
  6–11 % (the client check's gap 5 is the same rows); 4608 ties, and real rows of 9216 tie.
  The real overhead (time of the real row over its half length's c2c, minus 1) is FFTW's or
  close to it: r2c 6144 0.33 (FFTW 0.36), r2c 10240 0.25 (FFTW 0.24), r2c 6400 0.27 (FFTW
  0.23). FFTW pairs inside a radix pass (hc2c codelets); FFTN's r2c rows of these lengths pair
  in a separate pass (their half spectra exceed L1, and their chains end in a split run), c2r
  in the first pass where a radix of 4 or less can lead.
- **FFTN's own measurement finds little** (arm M): r2c 20480 × 21 0.90 (8 × 8 × 10 × 16), c2r
  6144 0.92 (2 × 8 × 12 × 16 fused), c2r 10240 0.95; the rest 0.96–1.02.

### Screens (not deciding)

EPYC 7763 nodes 17, 19 and 20 (Slurm, `-c 8`, shared; node16 and node18, at load 900–1250,
excluded), GCC 16, one thread, builds 7ca7722 (the depth-first half-length route of commit
7ca7722, off by default, so `main`'s plans) and 4847f67 (the same library). Ratios are forced
arm ÷ the rules' plan; every run's band ±1.00 % unless given.

- **Depth-first half lengths** (run 20261124-222341-7696, Slurm 518047, 14 rounds;
  `STRATEGY=depth` with `DEPTH_TOPS=4` or 8; it forces c2c rows too): real rows of 18432 0.87
  (r2c) and 0.91 (c2r), of 20480 0.86 (r2c × 21) and 0.93 (c2r × 3) with a top of 4; with 8,
  18432 0.82–0.85 but 20480 0.90–0.94 and 10240 1.19–1.23. Below the rules' range (half
  lengths of 3072–5120, under L2/4) a top of 4 gains nothing (0.97–1.14). Reach screen (run
  20261124-222340-a9cf, Slurm 518052, 12 rounds, band ±2.99 %: the rules' grids of these rows
  are bimodal between processes, as X29's were): real rows of 24576–245760 out of place, every
  one whose half length the c2c rule would make a depth-first chain, 0.53–0.84 with a top of 4
  (r2c 40960 × 6 0.72, c2r 0.66); FFTW took 0.65–0.87 of the rules' time there, so the
  depth-first rows run 0.86–1.00 of FFTW's.
- **Radices of the half lengths** (`RADICES`, now taken by real rows for their half length;
  runs 20261124-222341-7292, -222346-0b55, -222346-e2eb, -222346-a7c3, Slurm 518048–518051,
  12 rounds; refinement 20261124-222539-b79a, -222556-de95, -222556-6b3a, -222626-e5be, Slurm
  518089–518092, 10–12 rounds; 15 sequences per length). The best three-pass sequence with a
  radix of 20, 32 or 64 against the rules' four passes of at most 16:

  | half length | rules | best | c2c | r2c | c2r |
  |---|---|---|---|---|---|
  | 3072 | 12 × 16 × 16 | 6 × 16 × 32 | 0.84 | 0.92 | 0.90 |
  | 3200 | 2 × 10 × 10 × 16 | 10 × 20 × 16 | 0.80–0.82 | 0.85–0.88 | 0.95 |
  | 4608 | 4 × 8 × 12 × 12 | 6 × 12 × 64 | 0.91–0.93 | 0.93 | 0.99 |
  | 5120 | 4 × 8 × 10 × 16 | 10 × 32 × 16 | 0.91 | 0.93 | 0.94 (× 4: 1.02) |

  Sequences that start with a radix of 3–5 lost (3 × 64 × 16 1.15, 4 × 20 × 64 1.17, 5 × 64 ×
  16 1.05), as did 20 × 16 × 16 for 5120 (1.19); the next best of each length were within 0.02
  of the best (6 × 32 × 16 and 12 × 32 × 8; 8 × 20 × 20 and 20 × 8 × 20; 12 × 12 × 32; 10 × 16
  × 32). The c2r rows gain less because the rules' sequences lead with a radix of 4 or less
  (3200, 4608, 5120), into which the pairing fuses, where the new ones pair in a separate pass.
- **The pairing fused beyond L1 for r2c** was not screened: at these lengths FFTN's real
  overhead already matches FFTW's (above).

## Hypothesis

- A real row's half-length transform costs what a c2c row of that length costs, so it should
  take the c2c row's depth-first chain where that chain wins (X51 measured 0.63–0.82 of the
  chain's time for c2c rows of 12288–163840 on Zen 3), with the pairing as a separate pass:
  r2c 18432 and 20480 then come to about FFTW's time, and the long real rows of 24576–245760
  (the client check's gap 2 includes 40960 × 6) gain more, because there the rules' plan is a
  grid.
- The cost tables price radix-16 passes too low and radix-32/64 passes not at all (X3, X54) on
  rows that exceed the 32 KiB L1 but fit L2: three passes beat four there, as in FFTW's plans
  (3 → 64 → 16, 10 → 20 → 16, 12 → 32 → 12, 8 → 20 → 32).

## Change under test

Two arms, each a commit on `exp/x72-zen3-real-rows`, cumulative:

- **Arm A** (0a41f58): `machine_profile::depth_real_max_bytes` = `depth_max_bytes` (4 L2 =
  2 MiB) in `set_grid_sizes`' Zen 3 branch (x86-64, private L2 under 1 MiB, at least 2 MiB of L3
  per CPU: X51's key). An even real row of one axis, out of place, contiguous on both sides, in
  double, whose half length h the c2c depth-first rule takes (h of 128 KiB–2 MiB, not a power
  of two, power-of-two part above 512 points: h = 8192–131072, top 4), runs h as that chain:
  r2c with the pairing pass after it in place in the output, c2r with the pairing from the
  input into the workspace block (h complex values), from which the chain's first pass reads
  (`real_depth_tops_of`, commit 7ca7722, which also lets `STRATEGY=depth` force it). Before:
  the breadth-first chain (h below 16384) or the half-length grid (from L2/2), with its pairing.
- **Arm B** (d24225a, on A): `avx2` double radix overrides on AMD's cores (other vendors keep
  the costs' choice): 3072 = 6 × 16 × 32, 3200 = 10 × 20 × 16, 4608 = 6 × 12 × 64, 5120 = 10 ×
  32 × 16 (`avx2_double_overrides`, `avx2_double_amd_only`).

Reach (AGENTS section 6): arm A's key is X51's, `zen3_like` with its depth-first fields: Zen 3,
and Zen 2 (512 KiB L2, 16 MiB per CCX), unmeasured; Zen 4 and Zen 5 (1 MiB, `avx512`), Ice
Lake-SP, the Xeon 6240, Granite Rapids and Apple keep their plans (profile test). Arm B is keyed
as X42's, X54's and X62's `avx2` rules: the `avx2` target on AMD's processors (Zen 2 and Zen 3
run `avx2` plans; Zen 4 and 5 only when forced to `avx2`), unmeasured on Zen 2.

What each arm can move, by its mechanism:

- **A**: real rows as above, any batch, both kinds. In place, strided, single precision, rows
  whose half length is a power of two or outside the range, and every c2c, plane and volume
  plan keep their plans.
- **B**: every `avx2` double chain of one of the four lengths: c2c rows of them (any batch,
  direction, placement), the half lengths of real rows of 6144, 6400, 9216 and 10240, the
  sub-rows of depth-first chains of four times them (c2c 12288, 12800, 18432, 20480; under A
  real rows of 24576, 25600, 36864, 40960), planes' axes of them, and a Bluestein convolution
  whose length is one of them. Grid phases (their own radix rule), lanes and float keep theirs.

## Method

- Machine: EPYC 7763 (DMOG, Slurm `-c 8`, shared nodes, node16 and node18 excluded while at load
  900–1250, `--hint=nomultithread`), GCC 16 Release, `avx2`, double, one thread (unpinned, as
  the client application runs), hot cases, statistic median.
- **Run J** (judging), catalogue `x72` (88 cases: the client catalogue's 17 real shapes of
  6144–20480 points, the half-length c2c rows, the real rows arm A reaches with 40960 × 6, and
  arm B's reach above: c2c rows of the four lengths at 1, 4 and 64 rows both directions and in
  place at 1 and 16, the four depth-first lengths, real 25600 and 36864, five primes of
  1543–2551 and two planes), `--tier full`, arms **O** = 7ca7722 (`main`'s plans; library as
  `main` plus the route off), **A** = 0a41f58, **B** = d24225a, **F** = FFTW 3.3.11 MEASURE,
  with the A/A control O~; 12–20 rounds, `--serial`.
- **Run P** (plan check, before J, one round, no timing claims): arms O, A, B on catalogue
  `x72` and on the standard catalogue's tracking tier. **A case moves under an arm when its
  plan note differs from the arm before** (A against O, B against A); a case whose notes agree
  runs the same plan and does not decide. Done before these rules were committed: runs
  20261124-223706-f8c6 (`x72`, Slurm 518103) and 20261124-223708-de23 (tracking tier, Slurm
  518104), every output checked against the reference and correct. On `x72` A moves 26 cases,
  exactly arm A's family: r2c and c2r of 18432, 20480 × 21 (r2c) and × 3 (c2r), 24576, 36864,
  40960 (× 1 and × 6), 49152, 61440, 81920, 98304, 122880, 163840, 245760 (not 25600, whose
  half length's power-of-two part is 512 points). B moves 58: every c2c row of 3072, 3200,
  4608, 5120 (both directions, 1–64 rows, in place), the real rows of 6144, 6400, 9216 and
  10240, the depth-first c2c rows 12288, 18432, 20480 (not 12800, which is no depth-first
  chain), the real rows 24576, 36864 and 40960 (sub-rows under A) and both planes; none of the
  primes. On the tracking tier neither arm moves a case.
- **Run K** (informative, decides nothing): the client catalogue's tracking tier (52 shapes),
  arms `main`'s plans (7ca7722), B and FFTW, one thread, 12–16 rounds, scored as the client
  check scored its one-thread mode (weighted geometric mean of FFTN ÷ FFTW).
- Metric: arm ÷ the arm before, per moved case the median over valid rounds with its 95 %
  interval; over the moved cases the geometric mean with its interval (fbench's analysis). The
  band is run J's A/A control (O against O~).

### Decision rules (written before run J)

- **Arm A is adopted** on Zen 3's class if, over run J's cases whose notes differ between O and
  A, the geometric mean of A ÷ O is below 1 − band with its interval's upper end below 1, and
  no such case is slower than 1 + max(band, 3 %) (its whole interval above).
- **Arm B is adopted** for `avx2` double on AMD if, over run J's cases whose notes differ
  between A and B, the geometric mean of B ÷ A is below 1 − band with its interval's upper end
  below 1, and no such case is slower than 1 + max(band, 3 %) (whole interval above). B is
  judged against A, so it is adopted only with A; if A is rejected, B is re-judged against O in
  a declared run on the code without A.
- **Moves nothing else**: run P shows no note differing between O and A outside arm A's family,
  nor between A and B outside arm B's (on the tracking tier: none expected). A failure is a bug
  to fix before adoption, not a verdict.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one `--serial` re-run of run J, which decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, arm A may be narrowed by
  `depth_real_max_bytes` (to the largest half length at which every moved case passes, or by
  kind), and arm B by dropping lengths from its table, without a new experiment; a run on the
  final code is then declared here before it starts, and must show the excluded cases on their
  old plans and no remaining moved case slower than 1 + band.
- Run K and FFTW's column in run J (FFTN ÷ FFTW before and after on the client's shapes) are
  reported and decide nothing.

### Not measured

Threaded plans take the same routes (a team shares the depth-first chain's stages as for c2c
rows; tests hold bitwise equality with the serial plan). Zen 2 is reached by both keys
unmeasured (the next cloud screen's n2d family). Float keeps its plans.

## Results

**Run J** 20261124-223906-cac2 (Slurm 518107, node21, `--serial`; 20 rounds, 5 under load, load
average median 9.3, other processes 9.1 cores; band ±1.00 %).

- **Arm A ÷ O** over the 26 cases whose notes differ (exactly run P's list): geometric mean
  **0.761** [0.755, 0.766], none slower than 1.03. Per case 0.646–0.943: r2c 18432 0.874,
  c2r 18432 0.937, r2c 20480 × 21 0.852, c2r 20480 × 3 0.943, r2c 40960 × 6 0.724, c2r 0.705,
  24576 0.771 / 0.870, the longest rows 0.65–0.79 (122880 and 163840 0.65–0.67, 245760 0.79).
- **Arm B ÷ A** over the 58 cases whose notes differ (run P's list): geometric mean **0.906**
  [0.904, 0.909], none slower than 1.03. c2c rows of 3072 0.84–0.88, 3200 0.81–0.85, 4608
  0.90–0.95, 5120 0.91–0.94 (the gain shrinks a little at 64 rows); real rows 0.85–0.99 (r2c
  6400 0.850, c2r 6144 0.89, c2r 9216 0.990, c2r 10240 × 4 0.981 [0.970, 1.003]); the
  depth-first sub-rows 0.94–0.97 (c2c 12288, 18432, 20480; real 24576, 36864, 40960); the planes
  0.887 and 0.914.

FFTN ÷ FFTW (MEASURE) in run J on the client catalogue's shapes, `main`'s plans (O) → A and B:

| case | O ÷ F | B ÷ F |
|---|---|---|
| r2c 6144 × 48, × 66 | 1.021, 1.023 | 0.943, 0.943 |
| c2r 6144 × 48, × 66 | 1.056, 1.060 | 0.946, 0.944 |
| r2c 6400, × 7 | 1.085, 1.031 | 0.928, 0.900 |
| c2r 6400 | 0.976 | 0.925 |
| r2c 9216, c2r 9216 | 1.003, 0.927 | 0.931, 0.917 |
| r2c 10240, × 28 | 1.105, 1.077 | **1.034**, 1.000 |
| c2r 10240, × 4 | 1.066, 1.058 | 0.997, **1.043** |
| r2c 18432, c2r 18432 | 1.150, 1.091 | 1.004, 1.019 |
| r2c 20480 × 21, c2r 20480 × 3 | 1.176, 1.111 | 0.993, **1.046** |
| r2c 40960 × 6, c2r 40960 × 6 | 1.271, 1.435 | 0.883, 0.987 |
| c2c 3072 fwd, bwd | 1.033, 1.047 | 0.888, 0.885 |
| c2c 3200 fwd, bwd | 1.113, 1.112 | 0.901, 0.904 |
| c2c 5120 fwd, bwd | 1.090, 1.051 | 0.984, 0.960 |
| c2c 4608 | 0.975 | 0.892 |

Three of the client's shapes stay above 1.02 in this run: r2c 10240 (1.034), c2r 10240 × 4
(1.043), c2r 20480 × 3 (1.046).

**Informative screen, a depth-first top of 8 on B** (run 20261124-224617-abf6, Slurm 518111, 12
rounds; `DEPTH_TOPS=8`): no clean key. r2c and c2r 18432 0.94, c2c 9216 0.92, some long rows
0.89–0.93, but 24576, 36864, 122880 and c2c 12288–18432 1.05–1.13, 20480 1.00–1.04. Left as a
lead.

**Run K** (informative) 20261124-223708-ded8 (Slurm 518105, node21, 16 rounds, 3 under load,
band ±1.26 %): the client catalogue's tracking tier (52 shapes) at one thread, scored as the
client check scored it. Weighted geometric mean of FFTN ÷ FFTW **0.990** [0.986, 0.996] with
`main`'s plans → **0.933** [0.928, 0.938] with arms A and B (unweighted 0.989 → 0.935), against
the target of 0.95. Shapes above 1.02: 24 → 7, of which three are this experiment's (r2c 10240
1.04, c2r 10240 × 4 1.05, c2r 20480 × 3 1.04) and four the out-of-place c2c batches of 384 ×
7372–7373 (1.03–1.06, not moved here).

Planning time rises on the rows arm A moves (the depth-first chain's tables): c2r and r2c
40960 × 6 20.8 → 26.2 ms (median over rounds), r2c 18432 11.0 → 11.8 ms, c2r 20480 × 3 12.2 →
13.1 ms; these rows were already over criterion (4)'s 10 ms (X71's subject).

**Planning time after X71** (informative, decides nothing). Those numbers were measured before X71
(PR 105) reached `main`. Re-measured by X71's method (fbench `plan_s`, one thread, rules
effort, short timing batches, 8 fixed rounds, with the A/A control) on the EPYC 7763 (Slurm
518295 and 518297, node21, shared 8-core allocation, GCC 16, `avx2`; load 55–67 on a shared
node). Arm A is `main` just before this experiment's merge, 7d8ae4b (X71 in). Arm B is `main`
87c52eb (this experiment plus X73, which changes only the threading engine). Client catalogue,
tracking tier, run 20261125-025002-b543: the plans differ in 26 of 52 shapes, exactly this
experiment's rows. The sum of median plan times goes from 49.1 to 47.2 ms (B ÷ A **0.960**;
the control's sum is 0.998 of A's). No shape's median rose by more than 10 % + 0.05 ms. The moved
rows plan in 0.87–0.98 of A's time. r2c and c2r 40960 × 6 go from 5.07 / 4.98 to 4.39 / 4.32 ms:
on A their half length was a grid whose twiddles come one entry at a time from `root_cache`; on
B the depth-first chain's table of 20480 is built by X71's octant rule, and the pairing's table of
40960 is built from it. r2c 18432 goes 2.04 → 2.00 ms and c2r 20480 × 3 2.39 → 2.22 ms. The
worst shape on B is r2c 40960 × 6, median 4.39 ms (maximum over the rounds 5.52 ms).
Criterion (4) holds: every client shape plans in under 10 ms on the EPYC, by median and by
maximum. The standard catalogue's tracking tier (run 20261125-025002-047d) has the same plans in
all 61 cases, sum 0.996 of A's, and no case rose beyond the margin.

## Verdict

**Both arms adopted** by their rules, no clause used (no near miss, no narrowing):

- Arm A on Zen 3's class (`set_grid_sizes`, `depth_real_max_bytes = depth_max_bytes`): the
  real rows' half-length depth-first chain, 0.761 over the 26 rows it moves, which include the
  client's 18432, 20480 and 40960 × 6.
- Arm B for `avx2` double on AMD: 3072 = 6 × 16 × 32, 3200 = 10 × 20 × 16, 4608 = 6 × 12 × 64,
  5120 = 10 × 32 × 16, 0.906 over the 58 cases it moves.

With both, the EPYC's one-thread client score (run K) goes from 0.990 to 0.933 of FFTW, past
the 0.95 target. Before X71 the cost was 5–26 % more planning time on the rows arm A moves.
With X71 in, those rows plan in 0.87–0.98 of the time they took before, and all are under 10 ms
("Planning time after X71").

Recorded in DESIGN 8.5 (the depth-first chain's EPYC bullet), 9.1 (the *Depth-first* real
route), 13.5 (the vendor-keyed and Zen 3-keyed lists) and the decision log; lesson in
LESSONS 4.21 and 4.26. Reach: Zen 2 by both keys, unmeasured (for the next cloud screen).
Follow-ups: the three shapes still above 1.02 (r2c 10240, c2r 10240 × 4, c2r 20480 × 3; the
c2r rows pair in a separate pass where `main` fused the pairing into a leading radix-4 pass,
so a c2r pairing fused into the depth-first chain's first pass, or a three-pass sequence that
leads with 4, is the next lead); tops of 8 for some lengths (above); the same overrides on
other lengths between L1 and L2/4 (X38 gave 3000 and 3600 radix-20 passes the same way; other
lengths of 3000–6000 points may gain alike, which a search with radix 20, 32 and 64 candidates
would show).

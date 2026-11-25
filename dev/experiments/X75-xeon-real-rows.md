# X75: long real rows on the depth-first chain on the Xeon 6240's class

- **Status:** done (arm A adopted; arm B adopted narrowed)
- **Owner:** x75 session, 2026-11-25 (overnight)
- **Phase / task:** 1 (the client catalogue check's second gap, PLAN "Status")

## Question

The client catalogue check (2026-11-24, `main` d4b169f, run 20261124-103457-da59) found the real
rows of 32768 and 40960 points out of place losing at one thread on every machine, at 1.19–1.44
of FFTW's time; on the Xeon 6240 they are 2.8 % of excess, the largest gap left there (c2r 32768
1.44, c2r 40960 × 6 1.38, r2c 32768 1.28, r2c 40960 × 6 1.20, with c2r 20480 × 3 1.14). X72 gave
real rows' half-length transform the c2c depth-first chain on Zen 3's class only
(`machine_profile::depth_real_max_bytes`). The Xeon 6240's class has had c2c depth-first chains
from L2/4 to 64 L2 since X46 and X51. Does the same route win there, and over which range?

## Diagnosis

### The route forced on and off (run D1)

Run 20261125-004940-024f (Slurm 17015, the Xeon 6240, `--serial`, one thread under `numactl`
on four idle whole cores of one socket with its memory; GCC 14.4 Release, `avx512`; 14 rounds,
none under load, band ±1.33 %), build cc38175 (the X72 branch with this experiment's catalogue
and the fbench option `DEPTH_REAL_MAX_BYTES`), catalogue `x75` without its rows beyond 2^20
points (54 cases). Arms: the rules (N, real rows off the depth-first chain as on `main`); D,
`DEPTH_REAL_MAX_BYTES=never`, real rows' half lengths on the c2c rows' depth-first rule (L2/4 to
64 L2, tops 16, 20, 10); DL, D with `DEPTH_MIN_BYTES=131072` (from L2/8); FFTW 3.3.11 MEASURE
with `PLAN_NOTE=1` (F). Ratios against N:

- **D moves every real row whose half length is from L2/4 (16384 complex doubles), and
  each one gains:** 0.59–0.87, with a geometric mean of 0.791 over all 54 cases (the 12
  below the range unchanged, 0.98–1.02 with the same plans). r2c 32768 0.617, c2r 32768
  0.722, r2c 40960 × 6 0.771, c2r 40960 × 6 0.830; 36864–61440 0.59–0.60 (r2c) and 0.70–0.82
  (c2r); 65536 0.687 / 0.864; 262144 0.805 / 0.779; 2^20 0.736 / 0.875 (there against the
  half-length grid of three phases with the pairing in its last phase, X33); single precision
  65536–2^20 0.67–0.80. The plans are those of the c2c rows of the half length: 32768 as
  16 × 1024 (sub-rows 8 × 8 × 16), 40960 as 16 × 1280, 2^20 as 16 × 32768.
- **From L2/8 (DL) the shorter rows gain too:** r2c and c2r 18432 0.918 / 0.923 (16 × 576),
  r2c 20480 × 21 0.916 and c2r 20480 × 3 0.840, 20480 0.889 / 0.826, 24576 0.759 / 0.727,
  single precision 40960 (half length 160 KiB) 0.951 / 0.645. 16384 (half length 128 KiB,
  L2/8 itself, 16 × 512) ties: 0.986 / 0.977, intervals reaching 0.998 and 1.021. X46 found
  c2c rows of 8192 points (L2/8) gaining nothing; a real row's half length gains where the
  c2c row of that length does not, because the breadth-first chain it replaces is not the
  c2c row's (a split run with a separate or fused pairing).
- FFTN ÷ FFTW before → with DL, on the client's rows: r2c 32768 1.34 → 0.83, c2r 32768 1.46
  → 1.04, r2c 40960 × 6 1.28 → 0.99, c2r 40960 × 6 1.42 → 1.18, c2r 20480 × 3 1.18 → 1.00,
  r2c 20480 × 21 0.95 → 0.87, r2c 18432 0.85 → 0.78, c2r 18432 0.94 → 0.87.

### Rows beyond 2^20 points (run D2)

Run 20261125-010730-90ff (Slurm 17019, as D1; 8 rounds, none under load, band ±1.00 %), arms N
and D on the catalogue's rows of 2^21–2^23 points (half lengths of 16–64 MiB, the top of the
c2c rule's range): r2c 0.728, 0.745, 0.664, c2r 0.893, 0.811, 0.772. The route gains over the
whole range, so arm A takes it all.

## Hypothesis

- As on Zen 3 (X72), a real row's half-length transform costs what a c2c row of that length
  costs, so it gains wherever the c2c row's depth-first chain does: on the Xeon 6240 from L2/4
  to 64 L2 (X46, X51), where the rules today give it the breadth-first chain (to L2/2) or the
  half-length grid. The diagnosis shows 0.59–0.89 over that range.
- From L2/8 to L2/4 the real rows gain too (0.73–0.92), unlike c2c rows of those lengths (X46),
  because the breadth-first plan they leave is a split run with the pairing separate or fused
  into a leading pass, slower than the c2c row's.

## Change under test

Two arms, each a commit on `exp/x75-xeon-real-rows`, cumulative, on the build of X72's branch
(PR 106, not yet merged):

- **Arm A** (c6c85d3): `machine_profile::depth_real_max_bytes = depth_max_bytes` (64 L2 =
  64 MiB) in `set_grid_sizes`' branch of the Xeon 6240's class (x86-64, private L2 under 2 MiB,
  less than 2 MiB of L3 per CPU). An even real row of one axis, out of place, contiguous on both
  sides, whose half length the c2c depth-first rule takes there (h of 256 KiB–64 MiB in double,
  to 8 MiB in single precision, tops 16, 20, 10 on 32 vector registers), runs h as that chain
  with the pairing as a separate pass (X72's route). Before: the breadth-first chain (h below
  L2/2) or the half-length grid.
- **Arm B** (82e4fdb, on A): `machine_profile::depth_real_min_bytes` (new; default the largest
  value, no limit of its own) = L2/8 in the same branch: real rows' half lengths take the chain
  from 128 KiB instead of the c2c rows' 256 KiB (`real_depth_tops_of` passes it to
  `depth_tops_for` as a lower limit where it is the smaller).

Reach (AGENTS section 6): the key is the branch's, the one X46's and X51's c2c chains ride on:
the Xeon 6240 (Cascade Lake) and other x86-64 cores with a private L2 under 2 MiB and little L3
per CPU. The route needs a top of 16, 20 or 10, which only `avx512` (32 registers) takes there,
so `avx2` plans on that key keep theirs. Ice Lake-SP, which the reach rule gives the Xeon 6240's
rules, falls in X58's branch (6.9 MiB of L3 per CPU on the VM), where this experiment sets
nothing: it keeps its plans, unmeasured. The 2 MiB-L2 key (Sapphire, Emerald, Granite Rapids),
Zen 3 (X72 already), Zen 4 and 5 and Apple keep theirs (profile test). AMD cores with `avx512`
that report a small L3 per CPU (Zen 4c VMs, X63's finding) match the key as they match its c2c
chains: unmeasured.

What each arm can move, by its mechanism:

- **A**: even real rows of one axis out of place, contiguous, any batch, both kinds, double and
  single, whose half length is within the c2c depth-first range: in the catalogue the rows of
  32768–8388608 points (single precision 65536–2^20). In place, strided, odd lengths, rows of
  planes and volumes, and every c2c plan keep theirs.
- **B**: the same rows whose half length is from 128 KiB to 256 KiB: double 16384–24576 points
  (with 18432 and 20480), single 32768–49152.

## Method

- Machine: the Xeon 6240 (Slurm, `-c 8`, shared, `--serial`), one thread under `numactl` on
  four idle whole cores of the least busy socket and its memory (the profile `pick.py` of the
  client check), GCC 14.4 Release, `avx512`, hot cases, statistic median.
- **Run P** (plan check, one round, no timing claims): arms O = 39fd467 (`main`'s plans on the
  Xeon: the X72 branch with this experiment's catalogue, options and test), A = c6c85d3,
  B = 82e4fdb, on catalogue `x75` (69 cases), the standard catalogue's real cases (rows,
  batches, in place, planes and volumes) and the client catalogue's tracking tier. **A case
  moves under an arm when its plan note differs from the arm before's.** Run before these
  rules are committed or after, it decides nothing about time.
- **Run J** (judging): catalogue `x75`, `--tier full` (69 cases), arms O, A, B with the A/A
  control O~, 12–16 rounds, `--serial`.
- **Run K** (informative, decides nothing): the client catalogue's tracking tier (52 shapes) at
  one thread, arms O, B and FFTW 3.3.11 MEASURE, scored as the client check scored its one-thread
  mode (weighted geometric mean of FFTN ÷ FFTW).
- Metric: arm ÷ the arm before, per moved case the median over valid rounds with its 95 %
  interval; over the moved cases the geometric mean with its interval. The band is run J's A/A
  control.

### Decision rules (written before run J)

- **Arm A is adopted** on the Xeon 6240's class if, over run J's cases whose notes differ
  between O and A, the geometric mean of A ÷ O is below 1 − band with its interval's upper end
  below 1, and no such case is slower than 1 + max(band, 3 %) (its whole interval above).
- **Arm B is adopted** there if, over run J's cases whose notes differ between A and B, the
  geometric mean of B ÷ A is below 1 − band with its interval's upper end below 1, and no such
  case is slower than 1 + max(band, 3 %) (whole interval above). B is judged against A, so it is
  adopted only with A; if A is rejected, B is re-judged against O in a declared run on code
  without A.
- **Moves nothing else**: run P shows no note differing between O and A outside arm A's family,
  nor between A and B outside arm B's (none expected on the standard catalogue's real planes,
  volumes, in-place and short rows). A failure is a bug to fix before adoption, not a verdict.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one `--serial` re-run of run J, which decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, arm A may be narrowed by
  `depth_real_max_bytes` and arm B by raising `depth_real_min_bytes` (to exclude 16384, say),
  or either to double precision, without a new experiment; a run on the final code is then
  declared here before it starts, and must show the excluded cases on their old plans and no
  remaining moved case slower than 1 + band.
- Run K and FFTN ÷ FFTW decide nothing.

### Not measured

Threaded plans take the same routes (a team shares the depth-first chain's stages, as for X72;
tests hold bitwise equality with the serial plan); the client runs 32768 at 4 threads and 40960
× 6 at 6. Ice Lake-SP and the other machines above keep their plans or are reached unmeasured.

### Run R (declared after run J, before it starts; informative, decides nothing)

Run J's band was ±3.87 %, wide for this machine, with the shortest single rows far apart
between arms that run identical code (r2c 32768 in single precision: A, which keeps O's plan
there, 1.084 of O). So arm B's 18 cases are timed once more, arms O, A and B, `--serial`, 16–20
rounds, to report B against O and A where the noise is lower. It changes no verdict.

## Results

The catalogue has 68 cases, not the 69 the method says (a miscount); every one moves under one
arm.

**Run P** (plan check, Slurm 17030–17032, one round each, every output checked and correct):
`x75` 20261125-013000-88f2, A moves exactly arm A's family (50 cases: the double rows of
32768–8388608 with the batches of 40960 × 6 and 65536 × 4, single precision 65536–2^20), B
exactly arm B's (18: double 16384, 18432, 20480, 20480 × 21 and × 3, 24576, 24576 × 4; single
32768, 40960, 49152); the standard catalogue's 197 real cases 20261125-013000-05fe, A moves the
real rows of 65536 (× 1, × 2), 262144 and 2^20, B those of 16384, and no plane, volume or
in-place row; the client catalogue's tracking tier 20261125-013000-1da5, A moves r2c and c2r
32768 and 40960 × 6, B r2c and c2r 18432, r2c 20480 × 21 and c2r 20480 × 3. Nothing outside the
families moves.

**Run J** 20261125-023550-8fee (Slurm 17033, `--serial`, pinned as in the method; 16 rounds,
none under load, load average median 4.8; band ±3.87 %, wide for this machine: the shortest
rows differ by up to 17 % between arms running the same plan and code).

- **Arm A ÷ O** over the 50 cases whose notes differ: geometric mean **0.748** [0.735, 0.765],
  none slower than 1.0387 (the worst interval's lower end 0.873, r2c 122880). Per case
  0.515–0.900: r2c 32768 0.634, c2r 32768 0.745, r2c 40960 × 6 0.784, c2r 40960 × 6 0.836,
  36864–61440 0.58–0.64 (r2c) and 0.70–0.83 (c2r), 2^20 0.694 / 0.862, 2^21–2^23 0.52–0.67
  (r2c) and 0.80–0.88 (c2r), single precision 0.68–0.83, batches 65536 × 4 0.874–0.900.
- **Arm B ÷ A** over the 18 cases whose notes differ: geometric mean **0.796** [0.718, 0.835],
  none slower than 1.0387 (worst lower end 0.793, c2r 20480 × 3). Per case 0.61–0.93, with
  16384 0.847–0.871 and single precision r2c 32768 0.852 [0.756, 1.053], both inflated by A's
  slow processes there: against O, which runs the same plans as A on these cases, B took 0.94–0.95
  on 16384 and 1.089 [0.877, 1.110] on single precision r2c 32768.

By the rules both arms pass.

**Run R** 20261125-040403-2740 (Slurm 17035, informative as declared; 20 rounds, none under
load, band ±2.41 %): B ÷ A over arm B's 18 cases 0.838 [0.832, 0.841], but single precision
r2c 32768 **1.095** [1.069, 1.111] (B ÷ O 1.108), past 1 + max(band, 3 %); 16384 0.996 and 0.966;
the rest 0.61–0.95 (r2c and c2r 18432 0.917 / 0.947, r2c 20480 × 21 0.917, c2r 20480 × 3 0.838,
24576 0.73–0.74, single precision c2r 32768 0.670, 40960 0.949 / 0.644, 49152 0.823 / 0.611).

### Narrowing of arm B (AGENTS section 6; declared before run N starts)

The half lengths of exactly L2/8 (128 KiB: double 16384, single 32768) either tie or lose
(single r2c 32768 1.095 in run R) while every longer one gains, so arm B is narrowed to half
lengths above L2/8: `depth_real_min_bytes = L2/8 + 1` (commit 36e53f1, arm B′). **Run N, the
deciding run of the narrowed arm**: catalogue `x75` filtered to arm B's 18 cases, arms A =
c6c85d3 and B′ = 36e53f1 with the A/A control A~, `--serial`, 16–20 rounds, pinned as in the
method. B′ is adopted if run N shows the four excluded cases (r2c and c2r 16384, single r2c and
c2r 32768) with A's plan notes, and none of the 14 cases B′ still moves slower than 1 + band
(its whole interval above).

**Run K** 20261125-032026-36ec (Slurm 17034, informative; 16 rounds, none under load, band
±1.24 %): the client catalogue's tracking tier at one thread, scored as the client check
scored it. Weighted geometric mean of FFTN ÷ FFTW **0.896** [0.891, 0.902] with O → **0.863**
[0.859, 0.866] with B (unweighted 0.908 → 0.878). Shapes above 1.02: 8 → 5; c2r
32768 1.530 → 1.067, r2c 32768 1.400 → below 1.02, c2r 40960 × 6 1.382 → 1.129, r2c 40960 × 6
1.288 → below 1.02, c2r 20480 × 3 1.171 → below 1.02. Left: c2r 40960 × 6 1.129, c2r 864 × 384
1.124 (not moved), c2r 32768 1.067. Planning time of the moved shapes is unchanged (32768 21 ms,
40960 × 6 27 ms; this build has the twiddle tables X71 replaced on `main`).

**Run N** 20261125-041502-3d25 (Slurm 17037, the declared deciding run of B′; 20 rounds, none
under load, band ±2.53 %): the four excluded cases have A's plan notes (r2c and c2r 16384, single
precision r2c and c2r 32768); B′ ÷ A over the 14 cases it still moves **0.821** [0.815, 0.826],
none slower (worst: single precision r2c 40960 0.955 [0.930, 0.986]); r2c and c2r 18432 0.929 /
0.949, r2c 20480 × 21 0.917, c2r 20480 × 3 0.846, 20480 0.877 / 0.858, 24576 0.766 / 0.735,
24576 × 4 0.885 / 0.809, single precision c2r 40960 0.635, 49152 0.825 / 0.608.

## Verdict

- **Arm A adopted** by its rule, no clause used: on the Xeon 6240's class an even real row out of
  place runs its half length on the c2c depth-first chain over that chain's range
  (`depth_real_max_bytes = depth_max_bytes`, 64 L2): 0.748 over the 50 rows it moves
  (32768–2^23 points, single precision 65536–2^20), none slower.
- **Arm B adopted narrowed** (the narrowing clause of 2026-11-23): arm B passed its rule in run J
  (0.796 over 18 cases), but the informative run R found single precision r2c 32768, whose half
  length is L2/8 itself, 1.095 of A; so real rows take the chain from half lengths above L2/8
  (`depth_real_min_bytes = L2/8 + 1`), and run N, declared before it ran, shows the excluded
  rows on their old plans and the 14 still moved at 0.821, none slower.

The client catalogue at one thread on the Xeon (run K, with the arm as judged in run J, which
also moved 16384, not in that catalogue): 0.896 → 0.863 of FFTW; c2r 32768 1.53 → 1.07, c2r 40960
× 6 1.38 → 1.13 are left above 1.02 among the shapes moved.

Recorded in DESIGN 8.5 (the Xeon profile's depth-first bullet), 9.1 (the *Depth-first* real
route) and the decision log; lesson in LESSONS 4.21. Reach: the Xeon 6240's branch on `avx512`
only (its tops); Ice Lake-SP (X58's branch) keeps its plans, unmeasured, and AMD `avx512` cores
that match the key (X63's Zen 4 finding) are reached unmeasured, as by its c2c chains.
Follow-ups: c2r 40960 × 6 (1.13 of FFTW; FFTW's plan pairs in a radix-2 hc2c pass and then runs
8 → 16 → 20 → 16) and c2r 32768 (1.07); threaded long real rows (the client runs 32768 at 4
threads), not judged here; Ice Lake-SP in the next cloud screen.

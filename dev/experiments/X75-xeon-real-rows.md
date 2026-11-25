# X75: long real rows on the depth-first chain on the Xeon 6240's class

- **Status:** running (diagnosis)
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

# X82: X75's and X78's real rows on Ice Lake-SP (cloud screen of 2026-11-25)

- **Status:** done: X75 and X78 adopted on X58's branch (Intel, a private L2 above 1 MiB and
  below 2 MiB with a large L3 per CPU)
- **Owner:** lead session, 2026-11-25 (prepared by a sub-agent)
- **Phase / task:** 1 (the reach rule, AGENTS section 6)

## Question

The reach rule gives Ice Lake-SP the Xeon 6240's rules, but Google Cloud's N2 Ice Lake VM reports
a large L3 per CPU (54 MiB for 8 CPUs, 6.75 MiB each), so it plans in X58's branch of
`set_grid_sizes` (x86-64, a private L2 above 1 MiB and below 2 MiB, a large L3 per CPU), which
has the Xeon 6240's c2c depth-first chain (L2/4 to 64 L2, tops 16, 20, 10) but sets neither X75
(real rows' half length on that chain, `depth_real_max_bytes` and `depth_real_min_bytes`) nor X78
(the c2r pairing fused into the chain's first pass, `depth_c2r_fused_min_bytes` and
`_max_bytes`). Should that branch take them?

## Hypothesis

A real row's half-length transform costs what a c2c row of that length costs (X72, X75), and the
c2c depth-first chain already wins on this VM (X58: one top of 16 took 0.50–0.68 of the time from
32768 to 2^22 points). So the real rows should gain as on the Xeon 6240 (X75: 0.748 over 50 rows,
0.821 over 14 from above L2/8), and the fused c2r pairing, which won there with a 1 MiB L2
(X78 arm B4n: 0.853 over 23 rows), should win with Ice Lake's 1.25 MiB.

## Change under test

Campaign branch `exp/cloud-screen-1125` (cece1c5: `main` a696bfd plus a campaign-only fbench
option this experiment does not use; not for merging). Every arm is that one binary, and the
arms set the Xeon 6240's values from the VM's own L2 (1.25 MiB = 1310720 bytes on N2) through
existing fbench options:

- **B** = `fftn:cece1c5`, `main`'s plans on the VM (neither rule);
- **R** = B + `DEPTH_REAL_MAX_BYTES=<64 L2>` + `DEPTH_REAL_MIN_BYTES=<L2/8 + 1>`: X75 as on the
  Xeon 6240's class (half lengths above L2/8 up to 64 L2);
- **RF** = R + `DEPTH_C2R_FUSED_MIN_BYTES=<L2/4>` + `DEPTH_C2R_FUSED_MAX_BYTES=<16 L2>`: X78 as
  on the Xeon 6240's class, on top of X75.

X75 is judged as R against B, X78 as RF against R. Adopting either means setting those fields in
X58's branch to those values: a plain profile change whose plans are the arm's here.

What each can move, by its mechanism:

- **X75 (R)**: even real rows of one axis, out of place, contiguous on both sides, in double,
  whose half length is above L2/8 and within the c2c depth-first range (160 KiB to 80 MiB on N2;
  in catalogue `x75` the double rows of 24576–2^21 points with the batches 24576 × 4, 40960 × 6
  and 65536 × 4: 18432 and 20480, whose half lengths are 144 and 160 KiB, are not above L2/8 + 1
  on a 1.25 MiB L2; the notes decide). Single precision cannot move: X58's branch gives float no depth-first chain
  (`depth_float_max_bytes` = 0). In place, strided, odd lengths, planes, volumes and every c2c
  plan keep theirs.
- **X78 (RF against R)**: the c2r rows among those whose half length is from L2/4 to 16 L2 (320
  KiB to 20 MiB) and whose sub-row count is a multiple of the target's complex lanes: in `x75`
  the c2r rows of 40960–2^21 points with 40960 × 6 and 65536 × 4. Nothing else.

## Method

- Machine: Google Cloud Spot VM `n2-standard-16 --threads-per-core=1 --min-cpu-platform="Intel
  Ice Lake"` (Ice Lake-SP, model 106: 8 cores, private L2 1.25 MiB), Ubuntu 24.04, GCC 14.2
  Release, `avx512`, one thread pinned to one CPU, hot cases, statistic median. The job checks
  the CPU model and that the VM's caches give X58's branch (and not the Xeon 6240's class, which
  has both rules already), and stops otherwise.
- **Run I1** (judging): catalogue `x75`, `--tier full`, without its single-precision cases (they
  cannot move, above) and without the rows of 2^22 and 2^23 points (about 4 s of planning per
  process and arm), 48 cases; arms B (with the A/A control B~), R, RF; 8–12 rounds within a
  15-minute budget.
- **Run I0** (plan check, one round, no timing claims): the standard catalogue's real cases
  (`^(r2c|c2r)`), arms B and RF: notes may differ only on even real rows out of place whose half
  length is above L2/8.
- **Moved cases**: X75, the cases of I1 whose plan notes differ between B and R; X78, between R
  and RF. A case whose notes agree runs the same plan in the same binary and does not decide.
- Plan check before these rules (no timing claims): cece1c5 on the Xeon 6240 (Slurm 17102 and
  17103, GCC 14.4, `avx512`) planning for the topology the N2 VM recorded in X64's screen
  (`FFTN_TOPOLOGY`), one round. On `x75` without 2^22 and 2^23 (run 20261125-093714-695a, 64
  cases with float) R differs from B in 40 cases, exactly the double rows of 24576–2^21 points,
  r2c and c2r, with the three batches; RF differs from R in 16, exactly the c2r rows of
  40960–2^21 with 40960 × 6 and 65536 × 4; no float case moves. So I1 has 40 X75 cases and 16
  X78 cases of 48. On the standard catalogue's 197 real cases (run 20261125-093719-e6a2) R moves
  the rows of 65536 (× 1, × 2), 262144 and 2^20, r2c and c2r, and RF their c2r rows; nothing
  else. Every output was checked against the reference.
- Metric: R ÷ B and RF ÷ R per moved case, the median over valid rounds with its 95 % interval;
  over the moved cases the geometric mean with its interval. The band is I1's A/A control.

### Decision rules (written before any judging run)

- **X75 is adopted** on X58's branch if, over I1's cases whose notes differ between B and R, the
  geometric mean of R ÷ B is below 1 − band and no such case is slower than 1 + max(band, 3 %)
  (its whole interval above). Otherwise the branch keeps its plans.
- **X78 is adopted** there if X75 is, and, over I1's cases whose notes differ between R and RF,
  the geometric mean of RF ÷ R is below 1 − band and no such case is slower than
  1 + max(band, 3 %) (whole interval above). Without X75 the route X78 fuses into does not exist
  on the branch, and X78 is not adopted.
- **Moves nothing else** (not a verdict): I0 and I1 show no note differing outside the families
  above. A failure is a bug to fix before adoption.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the band gets
  one `--serial` re-run of I1, which decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, X75 may be narrowed by its
  limits (`depth_real_min_bytes` raised, `depth_real_max_bytes` lowered) or by kind, and X78 by
  its limits, without a new experiment; a run on
  the final code is then declared here before it starts, and must show the excluded cases on
  their old plans and no case the rule still moves slower than 1 + band.
- The rows of 2^22 and 2^23 points lie within 64 L2, so adopting X75 at that limit gives them the
  route unmeasured on this VM; X75's run D2 measured them on the Xeon 6240 (0.66–0.89), and the
  verdict records them as unmeasured here.

Reach of the verdicts: X58's branch, i.e. x86-64 cores with a private L2 above 1 MiB and below
2 MiB that report at least 2 MiB of L3 per CPU (Ice Lake-SP as N2 reports it), within `avx512`
and 32 registers. Bare-metal Ice Lake-SP with SMT reports under 2 MiB of L3 per CPU and is in
the Xeon 6240's class, which has both rules already.

### Not measured

Threaded plans (X77 measured the route at the client's widths on the Xeon 6240 and Zen 3 only),
single precision (no depth-first chain on the branch), FFTW and oneMKL (the rules compare FFTN's
arms only).

- Budget: about USD 0.35 for this VM, within the round's USD 1.5; rentals in
  [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md).

## Results

**When the rules were fixed.** The rules above were committed and pushed in c2f11e4
(2026-11-25 09:39:49 UTC; PR 120 opened 09:40:27) before either VM was created (09:43:39 and
09:43:52). PR 120 was **not yet merged** when the VMs started: the lead started them without
waiting for its ASan CI check, expected to take about 80 minutes (it took 13 on that
documents-only change). It merged at 09:53:29 with the rules' text unchanged from c2f11e4, 48
seconds after I1 started (09:52:41) and before any run was scored.

Google Cloud Spot VM `fftn-s1125-n2` (ID 6475811611598328631, n2-standard-16 with one thread
per core, us-central1-a, 09:43:52–10:03:07 UTC): Intel Xeon @ 2.60GHz, model 106 (Ice Lake-SP),
8 CPUs, 48 KiB of L1, a private L2 of 1.25 MiB (1310720 bytes), 54 MiB of L3 shared by the 8
CPUs, so X58's branch (the job checked it: `x58_branch`, `depth_max_bytes` 83886080 = 64 L2,
neither rule set). Ubuntu 24.04, Linux 7.0.0-1017-gcp, GCC 14.2.0 Release, `avx512`, double,
one thread pinned to CPU 2, campaign binary cece1c5, hot cases, statistic median. The arms' values
from that L2: R `DEPTH_REAL_MAX_BYTES=83886080` (64 L2) and `DEPTH_REAL_MIN_BYTES=163841`
(L2/8 + 1); RF adds `DEPTH_C2R_FUSED_MIN_BYTES=327680` (L2/4) and
`DEPTH_C2R_FUSED_MAX_BYTES=20971520` (16 L2). Scores re-computed from the run store by the
screen's scorer (`--rule adopt-mean`) and checked against `fbench show` of the same store: per
moved case the ratio with its 95 % interval; over the moved cases the per-round geometric mean,
its median and 95 % interval. "Slower" counts moved cases whose whole interval lies above
1 + max(band, 3 %).

| Run | ID | Cases | Rounds valid | Band (A/A) | Load average median, max | Rounds under load |
|---|---|---:|---:|---:|---|---:|
| I1 x75 | 20261125-095241-aafe | 48 | 12 | ±2.73 % (B~) | 1.15, 5.21 | 1 of 12 |
| I0 notes | 20261125-100000-ffce | 197 | 1 | (no timing) | 1.00, 1.00 | 0 of 1 |

**Plan checks.** I1: R differs from B in exactly the 40 expected cases (the double rows of
24576–2^21 points, r2c and c2r, with 24576 × 4, 40960 × 6 and 65536 × 4); RF differs from R in
exactly the 16 expected (the c2r rows of 40960–2^21 with 40960 × 6 and 65536 × 4). 16384,
18432 and 20480 (× 1, r2c × 21, c2r × 3) keep their plans: their half lengths are not above
L2/8 + 1. I0: B and RF differ in 8 of the standard catalogue's 197 real cases, exactly the
expected r2c and c2r 65536 (× 1, × 2), 262144 and 2^20; nothing else.

**X75** (R ÷ B; 40 moved: the half-length transform on X58's c2c depth-first chain against the
breadth-first chain or the half-length grid):

| Moved cases | Count | Ratio range | Worst case [95 %] |
|---|---:|---|---|
| r2c rows of 24576–2^21 | 17 | 0.560–0.889 | 24576: 0.889 [0.871, 0.905] |
| c2r rows of 24576–2^21 | 17 | 0.683–0.958 | 65536: 0.958 [0.930, 1.010] |
| r2c 24576 × 4, 40960 × 6, 65536 × 4 | 3 | 0.825–1.026 | 24576 × 4: 1.026 [1.005, 1.042] |
| c2r 24576 × 4, 40960 × 6, 65536 × 4 | 3 | 0.861–0.960 | 65536 × 4: 0.960 [0.923, 0.988] |
| **all moved** | **40** | **geomean 0.745 [0.737, 0.760]** | slower than 1.030: 0 |

r2c 24576 × 4 is the one moved case above 1: 1.026 [1.005, 1.042] (fbench: "no evidence",
within the band), not slower by the rule, whose limit its interval does not clear. c2r 65536
(0.958) and c2r 65536 × 4 (0.960) gain least among the rows; X78 below takes them further.

**X78** (RF ÷ R; 16 moved: the c2r pairing fused into the chain's first pass against the
separate pairing pass):

| Moved cases | Count | Ratio range | Worst case [95 %] |
|---|---:|---|---|
| c2r rows of 40960–2^21 | 14 | 0.719–0.812 | 40960: 0.812 [0.793, 0.818] |
| c2r 40960 × 6, 65536 × 4 | 2 | 0.824–0.833 | 65536 × 4: 0.833 [0.816, 0.858] |
| **all moved** | **16** | **geomean 0.772 [0.762, 0.782]** | slower than 1.030: 0; every case faster beyond the band |

The 8 cases neither arm moves ran the same plan in the same binary and do not decide; they read
0.977–1.013 in R ÷ B, all within the band. Both together (RF ÷ B, `fbench show`, informative):
c2r 65536 0.724 (0.958 with X75 alone); RF's r2c rows run R's plans.

## Verdict

- **X75 adopted** on X58's branch: 0.745 [0.737, 0.760] over the 40 real rows it moves, below
  1 − band (0.973), none slower (the Xeon 6240 in X75: 0.748 over 50).
- **X78 adopted** there (X75 being adopted): 0.772 [0.762, 0.782] over the 16 c2r rows it
  moves, none slower (the Xeon 6240 in X78: 0.853 over 23).
- **Moves nothing else**: I0's and I1's notes as expected (above). No near-miss re-run and no
  narrowing were used.
- **Code**: `set_grid_sizes`' X58 branch (`src/threading/schedule.cpp`) sets
  `depth_real_max_bytes` = its `depth_max_bytes` (64 L2), `depth_real_min_bytes` = L2/8 + 1,
  `depth_c2r_fused_min_bytes` = L2/4 and `depth_c2r_fused_max_bytes` = 16 L2, derived from the
  L2: on the N2's 1.25 MiB exactly arm RF's 83886080, 163841, 327680 and 20971520, so its plans
  are RF's here. Intel only, as the branch (DESIGN 13.5).
- **Reached unmeasured**: the real rows of 2^22 and 2^23 points (half lengths of 32 and 64 MiB,
  within 64 L2 = 80 MiB) take X75's route on this branch without having been timed on Ice Lake;
  X75's run D2 measured them on the Xeon 6240 (0.66–0.89). Their c2r rows keep the separate
  pairing pass (above 16 L2 = 20 MiB), as on the Xeon 6240. Single precision keeps its plans (no
  depth-first chain on the branch), and threaded plans are unmeasured here (X77 measured the
  route at the client's widths on the Xeon 6240 and Zen 3 only).
- **Reach**: X58's branch, i.e. Intel x86-64 cores with a private L2 above 1 MiB and below 2 MiB
  (Ice Lake-SP as N2 reports it, with a large L3 per CPU), within `avx512` and 32 registers.
  Bare-metal Ice Lake-SP with SMT is in the Xeon 6240's class, which had both rules already.
- Spend: USD 0.15 for this VM (19.3 minutes at USD 0.460/h with disk and IP); the screen's two
  VMs USD 0.40 ([cloud-ledger-2026-11.md](cloud-ledger-2026-11.md)).

# X82: X75's and X78's real rows on Ice Lake-SP (cloud screen of 2026-11-25)

- **Status:** planned (rules only; the cloud run `screen-1125` has not started)
- **Owner:** lead session, 2026-11-25 (prepared by a sub-agent)
- **Phase / task:** 1 (the reach rule, AGENTS section 6)

## Question

The reach rule gives Ice Lake-SP the Xeon 6240's rules, but Google Cloud's N2 Ice Lake VM reports
a large L3 per CPU (54 MiB for 8 CPUs, 6.9 MiB each), so it plans in X58's branch of
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
  in catalogue `x75` the double rows of 18432–2^21 points, with the batches 20480 × 21, × 3,
  24576 × 4, 40960 × 6 and 65536 × 4; the notes decide which, since the chain's own limits
  apply). Single precision cannot move: X58's branch gives float no depth-first chain
  (`depth_float_max_bytes` = 0). In place, strided, odd lengths, planes, volumes and every c2c
  plan keep theirs.
- **X78 (RF against R)**: the c2r rows among those whose half length is from L2/4 to 16 L2 (320
  KiB to 20 MiB) and whose sub-row count is a multiple of the target's complex lanes. Nothing
  else.

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

## Verdict

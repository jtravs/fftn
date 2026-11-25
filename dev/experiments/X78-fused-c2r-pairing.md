# X78: the c2r pairing fused into the depth-first chain's first pass

- **Status:** running
- **Owner:** x78 session, 2026-11-25 (overnight)
- **Phase / task:** 1 (follow-up of [X77](X77-real-rows-threaded.md), on [X72](X72-zen3-real-rows.md)
  and [X75](X75-xeon-real-rows.md))

## Question

The depth-first real route (X72 on Zen 3's class, X75 on the Xeon 6240's) runs a c2r row as a
separate pairing pass from the input into the workspace block, then the half-length chain,
whose first pass reads the block. The breadth-first chain it replaced fused the pairing into its
first pass. X77 found the c2r route keeping less of its gain in teams (Xeon c2r 32768 at 4
threads 1.031 of the old plan, against 0.70 at one thread) and suspected the extra phase. Does
fusing the pairing into the depth-first chain's first pass make the c2r route faster at one
thread and at the client catalogue's widths?

## Hypothesis

The separate pass reads the h + 1 input bins and half a pairing table, writes h complex values
to the workspace block, and the first pass reads them back: about 3.5 h values of traffic and one
more phase (one more barrier in a team, every member writing what every member then reads). The
first pass can form each Z_e it loads from X_e and X_(h−e) of the input instead (the c2r pairing
Z_e = A + conj(v_e) B, with A = X_e + conj X_(h−e), B = X_e − conj X_(h−e) and v_e = −i w^e,
holds for every e from 1 to h − 1 with one table of v_e, since v_(h−e) = conj v_e): two reads of
the input and one of a table of h entries, no writes, no extra phase, no workspace block. That
should take a few per cent off at one thread and more in teams, where the phase also costs a
barrier.

## Change under test

One commit on `exp/x78-fused-c2r` (B), against the commit before it (O, `main` with this file):

- **Arm B**: on the depth-first real route, a c2r row whose sub-row count S (the tops' product)
  is a multiple of the target's complex lanes C runs the chain's first pass with the pairing in
  its loads (`depth_first_pass` with the c2r input, passes/depth.inl): each run of C sub-rows'
  elements e reads X_e upwards and X_(h−e) downwards from the input and a per-element table of
  v_e (h entries, built from the root table of n), forms Z_e, and goes on as the plain first pass.
  Z_0 is formed from the real parts of X_0 and X_h, as before. The separate pass and the
  workspace block go. Rows with S not a multiple of C keep the separate pass.
- r2c keeps its separate pass (an r2c arm, the pairing fused into the last top pass, is not part
  of this experiment).

What B can move, by its mechanism: c2r rows that take the depth-first real route, i.e. even real
rows of one axis, out of place, contiguous on both sides, whose half length the route's rule
takes on Zen 3's class (`depth_real_max_bytes`, X72) or the Xeon 6240's (X75, from above L2/8),
double and single precision, any batch, any team. Nothing else: r2c, c2c, in place, strided,
planes and volumes, and every class without the route keep their plans and code. A case
moves when its plan note differs between O and B.

Reach (AGENTS section 6): the classes the route already has (Zen 3's `zen3_like` key, which also
reaches Zen 2, and the Xeon 6240's), within each target's ISA and register file; no new key.

## Method

- Catalogue `x78`: every c2r case of catalogues `x72` and `x75` (both judged at one thread
  there), 41 cases, double and single precision.
- **EPYC 7763** (DMOG, Slurm, shared nodes 16–21, `-c 8`, `--hint=nomultithread
  --distribution=block:block`, never `--exclusive`), GCC 16 Release, `avx2`.
- **Xeon 6240** (Slurm `--serial`, `-c 8` for one thread, the client check's `xeoncc8` (`-c 16`)
  for widths 4 and 6: `numactl` on the idlest whole cores of the least busy socket and its
  memory), GCC 14.4 Release, `avx512`. The Xeon's load (`uptime`, `squeue`) is recorded before
  each job; the X76 session may time there too.
- **Run J** (judging, one thread, each machine): catalogue `x78` `--tier full`, arms O and B with
  the A/A control O~, 12–16 rounds, hot cases, statistic median.
- **Run W** (judging, the client's widths, each machine): `fftn_omp+THREADS=k` for O and B with
  O~, and FFTW 3.3.11 MEASURE with `THREADS=k` (F, informative); width 4: c2r 18432, c2r 32768,
  c2r 20480 × 3; width 6: c2r 40960 × 6 (the client catalogue's c2r rows on the route and the
  widths the client's rule gives them, as in X77). 12–20 rounds.
- **Run C** (collateral, informative, each machine): the standard catalogue's real cases
  (`^(r2c|c2r)`), arms O and B, 6 rounds: notes must differ only on c2r rows on the route.
- Metric: B ÷ O per moved case, the median over valid rounds with its 95 % paired interval; over
  the moved cases of a run the geometric mean with its interval. The band is the run's A/A
  control.

### Decision rules (written before runs J and W)

Per class (Zen 3's on the EPYC, the Xeon 6240's on the Xeon); X77's drafted rules, unchanged:

- **B is adopted on a class** if both hold:
  1. at one thread (run J), the geometric mean of B ÷ O over the cases whose notes differ is
     below 1 − band with its interval's upper end below 1, and no such case is slower than
     1 + max(band, 3 %) (its whole interval above);
  2. at the client's widths (run W), the geometric mean of B ÷ O over the moved cases has its
     interval's upper end at most 1 + band, and no case is slower than 1 + max(band, 3 %)
     (whole interval above).
- **Moves nothing else**: in runs J, W and C, no note differs between O and B outside the c2r
  rows on the depth-first real route. A failure is a bug to fix before adoption, not a verdict.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one `--serial` re-run of that run, which decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, B may be narrowed by class
  (a profile field keeping the separate pass on the losing class), by length (a byte limit on
  the half length) or to double precision, without a new experiment; a run on the final code,
  declared here before it starts, must then show the excluded cases on their old plans and no
  case B still moves slower than 1 + band.
- If B is adopted on one class and not the other, it is narrowed to the winning class as above.
  If it is adopted on neither, the PR carries this file with the negative verdict and the code
  is left out.

Correctness, not a decision rule: the accuracy tests and the determinism tests (a team's result
bitwise equal to the serial result) hold on B; bit identity with O is not required.

### Amendment 1: arm B4n (written after arm B's runs and diagnosis D1, D2; before runs J4, W4n, C4, E)

Arm B failed rule 1 on both classes (Results below: EPYC 1.151, Xeon 1.064 over the moved cases),
so it is rejected by its rule. Diagnosis (runs D1, D2, informative) found a variant that wins on
the Xeon: **B4** (d92558d) reads the pairing twiddle of e = i + leg j as v_i times a broadcast
root w_n^(leg j), one table stream of h / R entries instead of R streams over a table of h; on
the Xeon it took 0.887 of O's time over the 33 moved cases, losing only at half lengths of 32 and
64 MiB (1.27–1.30) and in single precision below L2/4 (1.06–1.07). On the EPYC every variant lost
(B 1.12, B2 1.17, B3 1.20, B4 1.15).

- **Arm B4n** (a02183d): B4 on the Xeon 6240's class only, for half lengths from L2/4 to 16 L2
  (`machine_profile::depth_c2r_fused_min_bytes`, `_max_bytes`, set in `set_grid_sizes`' branch of
  that class; default 0: never). Against O (65f4fe4) as before.
- What B4n can move: on the Xeon 6240's class, c2r rows on the depth-first real route whose half
  length is from 256 KiB to 16 MiB (double 32768–2^21 points, single 65536–2^20); the client's
  c2r 32768 and 40960 × 6, not 18432 or 20480 × 3. Nothing on Zen 3's class or any other.
- **Run J4** (judging, Xeon, one thread): catalogue `x78`, arms O and B4n with O~, 12–16 rounds,
  `--serial`, as run J. **Run W4n** (judging, Xeon): widths 4 and 6 as run W (FFTW informative).
  **Run C4** (Xeon): run C with B4n. **Run E** (declared deciding for Zen 3's class under the
  narrowing clause, EPYC): catalogue `x78`, arms O and B4n, 6 rounds: every case on its old plan.
- **Rules**, unchanged from those above: B4n is adopted on the Xeon 6240's class if, at one thread
  (run J4), the geometric mean of B4n ÷ O over the cases whose notes differ is below 1 − band with
  its interval's upper end below 1 and no such case is slower than 1 + max(band, 3 %) (whole
  interval above), and at the client's widths (run W4n) the geometric mean over the moved cases
  has its upper end at most 1 + band and no case is slower than 1 + max(band, 3 %). Runs J4, W4n
  and C4 show no note differing outside the moved family; run E shows no note differing at all.
  The near-miss and narrowing clauses apply as above.

## Results

(to come)

## Verdict

(to come)

# X78: the c2r pairing fused into the depth-first chain's first pass

- **Status:** done (arm B rejected; arm B4n adopted on the Xeon 6240's class)
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

Every output in every run was checked against the reference by fbench and was correct. Builds:
O 65f4fe4, B 5533759, B2 33671de, B3 eb53866, B4 d92558d, B4n a02183d; GCC 16.2 Release `avx2`
on the EPYC 7763 (DMOG node19, named with `--nodelist`, idle: load average 0.8–4, no round
loaded), GCC 14.4 Release `avx512` on the Xeon 6240 (`--serial`, pinned to idle whole cores of
the least busy socket by the client check's `pick.py`). The X76 session's 16-thread runs held the
Xeon's other socket during most Xeon runs ("other processes" 14–19 cores), and this session's
build jobs overlapped X76's run 17062 and this experiment's runs C (both machines) and the
start of D2: the bands below are each run's own.

**Plans.** A case moves when its note differs between the arms. Under B the EPYC moves the 15
c2r rows on Zen 3's route (18432–245760 points, half lengths not powers of two, double); its 26
others (powers of two, below the range, single precision) keep their plans. The Xeon moves 33 of
41 (16384 and below keep theirs). Every moved note reads "the pairing fused into its first pass"
instead of "after a pairing pass into the workspace block"; nothing else differs.

### Arm B (judged by the rules)

| run | machine, width | Slurm | rounds, band | B ÷ O over the moved cases |
|---|---|---|---|---|
| J 20261125-053120-822c | EPYC, 1 | 518333 | 16, ±1.00 % | **1.151** [1.146, 1.156], 15 moved, all slower (1.065–1.299) |
| J 20261125-053529-aac6 | Xeon, 1 | 17058 | 16 (10 loaded), ±2.13 % | **1.064** [1.057, 1.070], 33 moved, 12 slower |
| W 20261125-053923-fe4f | EPYC, 4 | 518334 | 12, ±1.00 % | c2r 18432 1.110, c2r 20480 × 3 1.124 (32768 not moved) |
| W 20261125-054019-1a31 | EPYC, 6 | 518335 | 12, ±1.24 % | c2r 40960 × 6 1.049 [1.044, 1.063] |
| W 20261125-054610-426f | Xeon, 4 | 17059 | 13, ±1.00 % | 0.884 [0.878, 0.893]: 18432 0.951, 32768 **0.748**, 20480 × 3 0.970 |
| W 20261125-054646-870c | Xeon, 6 | 17060 | 12 (10 loaded), ±1.00 % | c2r 40960 × 6 1.130 [1.049, 1.226] |

On the Xeon at one thread B gained on the shorter rows that are not powers of two (18432–49152,
122880, 163840: 0.91–0.95) and lost on the long ones (524288–2^23 points 1.32–1.63, single
precision 2^20 1.25), on powers of two (131072 1.05, 262144 1.04) and on batches (24576 × 4
1.09, 65536 × 4 1.08, 40960 × 6 1.07). Arm B fails rule 1 on both classes: **rejected**.

Runs C (the standard catalogue's 197 real cases, 4 rounds, informative): EPYC
20261125-054039-331a (518338; three rounds loaded by this session's D1 build, band ±4.23 %): no
note differs. Xeon 20261125-054704-775f (17066, band ±6.91 %): four notes differ, all c2r rows on
the route (65536 × 1 and × 2, 262144, 2^20; 2^20 1.52); nothing else moves.

### Diagnosis (informative)

- **Not register pressure.** B2 (the unit's R runs paired into an R × C tile, then the codelet
  from the tile) lost more than B on the EPYC: run D1 20261125-054628-68ff (518340, 10 rounds,
  ±1.10 %) B 1.121, B2 1.168 over the 15 moved cases.
- **The table streams, on the Xeon.** B reads its pairing twiddles from a table of h entries
  at the R legs of each unit: R more streams beside the R upward and R downward input streams,
  on top of the input's power-of-two leg spacing on rows of 2^k points. B3 (B2 with the twiddle
  of e = i + leg j taken as v_i · w_n^(leg j): one table stream of h / R entries and a broadcast
  root, one more complex product per leg) and B4 (the same without B2's tile): run D2 on the
  Xeon 20261125-060137-dd4d (17072, 12 rounds, 5 loaded, ±1.69 %), over the 33 moved cases, B
  1.059, B3 0.889, **B4 0.887**: B4 0.69–0.80 on 36864–262144 and 2^21, 0.84–0.97 on most of the
  rest; it lost only at half lengths of 32 and 64 MiB (2^22 1.300, 2^23 1.273) and in single
  precision below L2/4 (40960 1.060, 49152 1.068); 18432–25600 tied (0.97–1.00).
- **The EPYC loses whatever the variant.** Run D2 on the EPYC 20261125-055717-05b5 (518342,
  10 rounds, ±1.15 %): B 1.119, B3 1.195, B4 1.153, every moved case slower. On Zen 3 the extra
  arithmetic in the first pass (the pairing, the mirror's reversed load through a tile, and in
  B4 the second product) costs more than the separate pass's sequential sweep and its write and
  re-read of the block, which stay in the 512 KiB L2 and the CCX's L3 there.

### Arm B4n (judged by amendment 1's rules)

| run | machine, width | Slurm | rounds, band | B4n ÷ O over the moved cases |
|---|---|---|---|---|
| J4 20261125-062612-079d | Xeon, 1 | 17079 | 16 (15 loaded), ±4.22 % | **0.853** [0.841, 0.865], 23 moved, none slower |
| W4n 20261125-063649-3e6d | Xeon, 4 | 17080 | 20, ±1.00 % | c2r 32768 **0.740** [0.730, 0.752] (18432 1.002, 20480 × 3 1.004: not moved) |
| W4n 20261125-063741-028a | Xeon, 6 | 17081 | 12, ±1.00 % | c2r 40960 × 6 **0.886** [0.869, 0.899] |
| E 20261125-062442-ddf9 | EPYC, 1 | 518347 | 6, ±2.12 % | no case moved (41 of 41 on their old plans), all within the band |

- Run J4 was loaded by another session's build on the Xeon (load average 62, the X76 run on the
  other socket), hence its band. Moved: the 23 c2r rows with half lengths from 256 KiB to 16 MiB
  (32768–2^21 points, single precision 65536–2^20); 0.74–0.90 for double 32768–262144, 2^21 and
  the batches, single precision 0.86–0.97, 524288 1.009 and 2^20 1.043 [1.002, 1.116] (in run D2
  0.923 and 0.972), no whole interval above 1 + max(band, 3 %) = 1.042. The 18 others kept
  their plans: 16384 and below, 18432–25600, 20480 × 3, 24576 × 4, single 32768–49152 (below
  L2/4), 2^22 and 2^23 (above 16 L2).
- FFTN ÷ FFTW (run W4n, FFTW 3.3.11 MEASURE with its threads): c2r 32768 at 4 threads 0.668 → 0.495,
  c2r 40960 × 6 at 6 threads 0.794 → 0.704 (O ÷ F → B4n ÷ F).
- Run C4 20261125-063758-67e0 (Xeon, 17082, the standard catalogue's 197 real cases, 4 rounds, band ±4.57 %): four notes differ, all c2r rows on the route within the range (65536 × 1 and × 2, 262144, 2^20: 0.74–0.95); nothing else moves.

## Verdict

- **Arm B rejected** by its rule on both classes (one thread: EPYC 1.151, Xeon 1.064 over the
  moved cases; the client's widths: EPYC 1.05–1.12, Xeon 6 threads 1.13).
- **Arm B4n adopted on the Xeon 6240's class** by amendment 1's rules, committed (29bb06e) before
  its runs: one thread 0.853 [0.841, 0.865] over the 23 rows it moves, none slower than
  1 + max(band, 3 %); the client's widths c2r 32768 at 4 threads 0.740, c2r 40960 × 6 at 6
  threads 0.886; runs C4 and J4 show nothing else moving. No near-miss re-run was needed.
- **Zen 3's class keeps the separate pass** (`depth_c2r_fused_max_bytes` 0): the narrowing by class
  that rules and amendment named, with run E (declared before it started) showing all 41 cases on
  their old plans. Every fused variant lost there.
- For the maintainer's review: B4n is an arm defined after arm B's judging runs from a diagnosis
  (D1, D2), with its limits (L2/4 to 16 L2) read off that diagnosis, then judged under rules
  committed before its own runs, as X75's arm B was. Run J4 was loaded (band ±4.22 %) by another
  session's build. The client's c2r 18432 and 20480 × 3 (half lengths below L2/4) keep the
  separate pass and X77's 1.019 and 0.92 at 4 threads.
- Reach: the Xeon 6240's key (`set_grid_sizes`' branch of X46, X51 and X75). Ice Lake-SP keeps its
  plans (X58's branch), the 2 MiB-L2 key, Zen 4 and 5 and Apple have no depth-first real route.
- Not done: the r2c arm (the pairing fused into the last top pass). The last top pass's outputs
  k + M t pair with M − k + M (F1 − 1 − t), so the units would pair across blocks as r2c_last
  does; given Zen 3's result, it should be tried on the Xeon class only.
- Code: `run_in_c2r` and `first_input` in passes/depth.inl, `chain_builder::c2r_depth`,
  `c2r_depth_fits` and `depth_c2r_twiddles`, the profile fields and fbench options
  `DEPTH_C2R_FUSED_MIN_BYTES` and `_MAX_BYTES`. The intermediate arms B, B2 and B3 are in the
  branch's history (5533759, 33671de, eb53866). DESIGN 9.1 and the decision log, LESSONS 4.27.
- Tests: Mac (Clang 23, `neon_fcma`, `neon`, scalar) Release ctest 2665 of 2665 on a02183d; DMOG
  (GCC 16, `avx2`, `avx`, `x86_v2`, scalar) Release ctest 3240 of 3240 on B and on the final code;
  the Mac's GCC 16 `-Werror` build clean. `avx512` ran only through fbench's correctness checks on
  the Xeon (every case of every run) and CI; an ASan build on DMOG ran out of memory and was
  not repeated, so ASan rests on CI.

# X85: real-to-real speed (the folded DCT-I and DST-I, lanes over adjacent lines, fused stages)

- **Status:** running (rules committed before any judging run)
- **Owner:** task AK session, 2026-11-27
- **Phase / task:** 6.12 ([PLAN.md](../PLAN.md), phase 6 table); DESIGN.md 5.7 and 18.2 "Phase 6"
  Q14 and Q15

## Question

Which of three changes to the real-to-real plans (DESIGN.md 5.7) make them faster, on which
machines, without leaving the accuracy contract (DESIGN.md 15.2): (D) a cheaper route for odd
DCT-I and DST-I, (L) the strided axes of planes and volumes as lanes over blocks of adjacent
lines, (F) the read and store stages fused into the chain's first and last passes? And where do
FFTN's r2r plans stand against FFTW's r2r and pocketfft (Q15's proposed target: at or below FFTW
as a geometric mean over an `r2r` catalogue, one thread, with the tail rule of DESIGN.md 1.3)?

## Standing before the arms (informative, not a judging run)

Run 20261127-064235-0a8f (EPYC 7763, DMOG node shared: load average 34 of 128 CPUs, every round
"loaded"; GCC 16.2, `avx2`; band ±1.05 %), `r2r` catalogue's quick tier (51 cases), the commit of
task 6.12's first PR (c2a6f672): FFTN ÷ FFTW geometric mean 0.784, 35 faster, 15 slower;
pocketfft ÷ FFTW 0.778 (pocketfft is slow on short rows, 1.7–3.4, fast on long DCT-I/DST-I, whose
FFTW plans are slow). FFTN's losses to FFTW:

- DCT-III and DST-III everywhere (the inverse route: twiddle and pairing in the read stage, the
  Makhoul order back in the store stage): rows 1.11–1.16, 64-point batches 1.60, planes 1.70 (64²)
  and 1.98 (256²), strided columns 1.47;
- planes of DCT-II and the DHT 1.19–1.30, the DHT of 64 points 1.41, batches of 64 points
  1.18–1.29: short lines and the planes' strided axes, one line at a time;
- every DCT-I and DST-I is 0.23–0.91 of FFTW's time (FFTW's REDFT00 and RODFT00 of even n are
  slow), DCT-IV and DST-IV 0.51–0.96.

The read and store stages are baseline code, one line at a time (DESIGN.md 5.7, "As built"), and
the strided axis of a plane reads each line with its stride into scratch.

## Hypotheses

- **D.** For odd n the real DFT of n ∓ 1 points (a complex chain of (n ∓ 1)/2) replaces the
  extension's 2(n ∓ 1) (a chain of n ∓ 1): about half the chain's work, so 0.6–0.8 of the time
  where the chain dominates. Its odd outputs are running sums of the spectrum (Numerical Recipes'
  cosft1 and sinft, Press et al. section 12.3), so the transform's rounding errors add along them
  and the error grows like √n instead of √log n: FFTW's documentation warns that its REDFT00 by
  an r2hc of n − 1 is less accurate. A check on the Mac (fbench `--check-only`, not timing) found
  the folded route's relative RMS error 1.0e-16 at n = 5, 2.9e-15 at 1025, 2.0e-14 at 65537 in
  double, against the extension's 1–3e-16: so the route can be adopted only up to a length where
  it stays inside the contract, and Q14 asks for exactly that judgement.
- **L.** The c2c planes gained from lanes over adjacent columns (X40). For r2r the read and
  store stages stay scalar per line, but a block's lines then go through one lanes chain (one
  vector lane per line) instead of one chain per line, and the strided reads of adjacent lines
  hit the same cache lines back to back: 0.7–0.9 of the time on the planes' strided axes.
- **F.** Fusing the stages into the passes saves a sweep over scratch on each side; it needs
  new pass kinds in the generator's first and last passes per route (as the r2c pairing has,
  DESIGN.md 9.1).

## Change under test

One commit with both switches, judged as fbench options on one build (as X66 and X67 did):

- **Arm D** (`machine_profile::r2r_folded_max`, fbench `R2R_FOLDED_MAX=N`): odd DCT-I and DST-I
  axes up to N points take the folded route (core/r2r.cpp, `folded_read`, `folded_v0`,
  `folded_store`): the read stage folds the line into y_j (DCT-I: u_j/2 − s_j v_j; DST-I:
  s_j u_j + v_j/2, u and v the line's sums and differences about its middle, s_j = sin(πj/N)) and
  packs it as the r2c packs a line, the forward chain of N/2 runs in scratch, and the store stage
  (one unit) forms the spectrum by the r2c's pairing and writes the even outputs from it and the
  odd ones as running sums; DCT-I's v_0 = x_0 − x_(n−1) goes to the output's place of Y_1 by a
  stage of its own, so plans with accumulation keep the extension. Arm: `R2R_FOLDED_MAX=65535`;
  baseline: 0 (main's extensions).
- **Arm L** (`machine_profile::r2r_lanes_columns`, fbench `R2R_LANES_COLUMNS=1`): a phase whose
  axis is not the contiguous one, where the contiguous axis holds the lines one real apart on
  both sides, no factor edge applies in the phase, the route is not the folded one and every pass
  of the chain has a lane kernel, runs as blocks of B adjacent lines (B a cache line of reals,
  rounded up to C): the line's read stage for each line into a tile in scratch (element i of line
  l at i B + l; the read and store stages take scratch strides for this), the chain of m as lanes
  passes on the tile (strategies/lanes.cpp, `c2c_lanes`), the store stage for each line from the
  tile; the lines left over as one narrower block. Arm: `R2R_LANES_COLUMNS=1`; baseline: 0.
- **Arm F** is not built in this experiment (see "Method", F).

Both defaults are off, so the baseline arm is main's plans; the commit also changes the read and
store stages to take a scratch stride (1 for a line alone), checked by rule C.

## Method

- **Machines:** the EPYC 7763 on DMOG (Zen 3's class, AMD; GCC 16.2, `avx2`; Slurm, 8 CPUs of a
  shared node, never `--exclusive`; rounds added rather than waiting for quiet) and the Xeon Gold
  6240 (the Xeon 6240's class, Intel; GCC 14.4, `avx512`; Slurm through the `fbench-serial`
  singleton chain, behind every other timing job). No Mac timing tonight: the Mac leg (Apple's
  class) is a daytime follow-up, on AC power with `pmset -g batt` before and after.
- **Cases:** catalogue `x85` (groups `d-*`: DCT-I and DST-I of odd lengths, rows, batches, planes,
  float; groups `l-*`: planes in and out of place and volumes of every kind, float planes); for
  rule C also the `r2r` catalogue's quick tier.
- **Runs:** `fbench_remote.py submit <epyc|xeon --serial> A=fftn:<sha> D=fftn:<sha>+R2R_FOLDED_MAX=65535
  L=fftn:<sha>+R2R_LANES_COLUMNS=1 -- --catalogue x85 --tier full` (fbench's A/A control, one
  thread, hot, median statistic); for rule C `B=fftn:c2a6f672 A=fftn:<sha> -- --catalogue r2r
  --tier quick` plus the `x85` catalogue.
- **Accuracy (rule D1):** `bench/x85/accuracy.py <build>/bench/fbench-fftn R2R_FOLDED_MAX=65535`
  on each machine (its instruction set), and once with no option (the extensions, for comparison).
- **Metric:** per case the arm ÷ A ratio with fbench's paired bootstrap interval (95 %) and the
  run's A/A band; geometric means over a rule's target cases.

**Decision rules** (written before any judging run; a rule measured on one vendor's processors is
keyed to that vendor's class; plans outside a rule's reach stay as they were):

- **D1 (accuracy, per instruction set measured).** N_acc is the largest swept odd length n
  (2^k + 1 and 3·2^k + 1 up to 65535) such that, for every swept n up to it, both kinds and both
  precisions, the folded route's relative RMS error (fbench's check against the double-double
  reference, two rows) is at most half the rms tolerance the r2r tests apply to these kinds,
  ε (3 √L + 11) with L = log₂ N (N = 2(n − 1) for DCT-I, 2(n + 1) for DST-I): a margin of two for
  inputs other than fbench's. N_acc is the smallest over the instruction sets measured. The
  largest error is not judged separately (fbench normalises it by the rms of the output, the tests
  by the largest output); the tests added with an adoption check both metrics at the adopted
  lengths with the r2r tests' tolerance, unchanged.
- **D2 (speed, per machine).** Over the `d-*` cases whose every odd DCT-I or DST-I axis is at
  most N_acc (the cases the setting moves): geometric mean D ÷ A at most 0.90, and no such case
  slower than A beyond the band (its whole interval above 1 + band). If D2 passes, that machine's
  class takes `r2r_folded_max` = N_acc (Zen 3's class, AMD's cores; the Xeon 6240's class,
  Intel's cores); otherwise it keeps 0.
- **L1 (speed, per machine).** Over the `l-*` cases whose plan notes show lanes in arm L (the
  cases it moves): geometric mean L ÷ A at most 0.95, and no such case slower beyond the band. If
  L1 passes, that machine's class takes `r2r_lanes_columns`; if it fails only on some cases, the
  change may be narrowed (AGENTS.md section 6) by a rule the plans can see (kind, length, block
  bytes, precision) with one deciding run declared here first.
- **L2 (bits).** Every plan arm L moves gives output bitwise equal to its per-line plan (tests:
  every kind, planes and volumes, in place, both precisions, serial, teams and executors). Not
  met: L is rejected, whatever L1 says.
- **C (collateral).** The commit with both switches off against the base commit c2a6f672 (task
  6.12's first PR): over the `r2r` quick tier and the `x85` cases, no case slower than the base
  beyond the band, on each machine; a case past it is re-run and, if confirmed, the scratch
  stride of the stages is reworked before any adoption.
- **Near misses** of any criterion by less than the run's band get one `--serial` re-run, which
  decides (the maintainer, 2026-11-23).
- **F (not judged here).** Drafted for the follow-up: a fused first or last pass per route
  (DHT and DCT-II: the store fused into the r2c's fused last pass; DCT-III: the read fused into
  the c2r's first pass) against the separate stages, on the r2r rows and batches whose chain has
  a codelet last (first) pass, rule as L1 with 0.95, per machine.

## Results

**Judging run, EPYC 7763** (20261127-070028-304b; DMOG node17 shared, every round loaded, load
average 44–60 of 128; GCC 16.2, `avx2`; band ±1.73 %; commit 0a1e55a1; arms A, A~, B = c2a6f672,
D = `R2R_FOLDED_MAX=65535`, L = `R2R_LANES_COLUMNS=1`):

- **D:** over the 46 `d-*` cases with every axis odd and at most 257 points (N_acc below), D ÷ A
  0.787; all faster but DCT-I of 5 points, 1.041 [1.023, 1.043], past the band by 0.6 %, a near
  miss. Above 257 (not judged: outside N_acc) 0.57–0.82.
- **L:** over the 270 planes and volumes L ÷ A 1.021, 100 slower (DCT-IV 384² 1.16, DCT-II 256² in
  place 1.10, the DHT of 1024² in place 1.09), 27 faster: rule L1 fails. Rule L2 fails too: on
  `avx2` the lanes chain gives other bits than the line's scratch chain at line lengths 32 and 48
  (test run below; scalar, `x86_v2` and `avx` agree bitwise).
- **C:** B ÷ A geometric mean 0.990, 68 cases where the base was faster beyond the band (small
  planes 0.85–0.95): the read and store stages' scratch stride, added for L only, costs.

**Accuracy (rule D1), EPYC `avx2`** (DMOG job 520982, `bench/x85/accuracy.py`): folded, the
largest swept n with every swept n up to it within half the bound is **257**; the first miss is
DST-I of 385 points in float (1.07 of half the bound), then DST-I 513 in double (1.17), DCT-I
1537 (1.0 and 1.22), and the ratio grows like √n to 4–6 at 24577–49153. The extensions stay
within half the bound to 49153. The tests' clause: the folded test (both metrics, the r2r tests'
tolerance) passes at every swept n up to 385 on every x86 target and on the Mac's, and fails from
513 (the largest error of the kind twice, 1.08–2.5 of its tolerance).

**Tests, EPYC** (DMOG job 520982, GCC 16.2 Release `-Werror`, full ctest 3780 of 3782): the two
failures were L2's bitwise test (avx2 double and float).

**Action on L** (commit 363ed5aa): arm L and the stages' scratch stride deleted (AGENTS.md section
6: a loss is deleted); the stages are main's again.

**Deciding run, EPYC (declared before it starts).** On the final code 363ed5aa, arms A, B =
c2a6f672 and D = `R2R_FOLDED_MAX=257` (the setting rule D2 would adopt), catalogue `x85`:
rule D2 over the `d-*` cases with every axis odd and at most 257 (the near miss DCT-I 5 is
re-run here and this run decides it); every case the setting does not move (above 257, planes and
volumes of other kinds) on its old plan and not slower beyond the band; rule C on every case
(B ÷ A).

**Deciding run, EPYC** (20261127-072527-8c75, stopped after 6 rounds; node17 shared, loaded,
load average 34; band ±1.60 %; final code 363ed5aa): D = `R2R_FOLDED_MAX=257` over the 46 target
cases 0.793, but DCT-I of 5 points 1.078 [1.076, 1.108] of A: the near miss is confirmed and rule
D2's second clause fails on that one case (DCT-I of 9: 0.977 [0.949, 0.985]; DST-I of 5 0.918). No
case outside the targets moved (D ÷ A 0.998 over the other 294, none slower beyond the band). Rule
C (B = c2a6f672 ÷ A): 15 cases where the base was faster beyond the band (0.92–0.98: DCT-I rows of
33–129 points on the extension, which A runs with the same plans) and 15 where it was slower
(1.03–1.25: DST-II planes of 16–128 points), with the same plans (describe identical) in both
directions: the TU's code moved (the folded kernels added to src/core/r2r.cpp), not the stages.

**Narrowing (AGENTS.md section 6, commit a997ebb8):** DCT-I takes the folded route from 9 points
only (its extension of 5 points is one leaf of 4, which the folded route's extra stage outweighs).
**Narrowed deciding run, EPYC (declared before it starts):** code a997ebb8, arms A, D =
`R2R_FOLDED_MAX=257`, catalogue `x85`: every excluded case (DCT-I of 5) on its old plan (notes
identical to A's), and no case D moves (odd DCT-I from 9 and odd DST-I up to 257) slower than A
beyond the band; the geometric mean over the moved cases is reported against D2's 0.90.

(Xeon 6240 pending: judging run Slurm 17556 on 363ed5aa behind other tasks' jobs in the
`fbench-serial` chain; the narrowing applies there too.)

## Verdict

(Pending.)

# X77: X72's and X75's real rows at the client's thread widths

- **Status:** running
- **Owner:** x77 session, 2026-11-25 (overnight)
- **Phase / task:** 1 (follow-up of [X72](X72-zen3-real-rows.md) and [X75](X75-xeon-real-rows.md))

## Question

X72 (Zen 3's class) and X75 (the Xeon 6240's class) put an even real row's half-length
transform on the c2c depth-first chain, and both were judged at one thread. The client
catalogue runs these rows threaded: real 32768 at 4 threads, 40960 × 6 at 6, and the others
at the widths its own rule gives them. The tests hold the threaded plans bitwise equal to the
serial ones, but nobody has timed them. Is the route still faster than the plans it replaced
at those widths, and how do the rows stand against FFTW there? A case slower beyond the band
would call for narrowing the route by thread count. This experiment is **informative**: it
changes no default, and a finding ends in a proposal with drafted decision rules for the
maintainer.

## Hypothesis

The team shares the depth-first chain's stages (DESIGN 8.5, 13), so a route that wins at one
thread should still win at 4–16 threads. The gain may shrink where the plan the route replaced
splits better over the team: the half-length grid of the longer rows has phases of many
independent lines, while the depth-first chain's top passes have only 16 or 20 (the Xeon) or
4 (Zen 3) sub-rows to share, so a row of one transform at 4–6 threads may lose part of its
gain, and a batch spread over 8 or 16 members (r2c 20480 × 21) should keep it.

## Change under test

No library change. Both arms run `main` e2a4413 (PRs 105–111):

- **A**: `main`'s rules (the route on where X72 or X75 put it).
- **O**: the same build with fbench's `DEPTH_REAL_MAX_BYTES=1` (bench/backends/fftn_tuning.hpp:
  a size field takes a whole number from 1, so 1, not 0, keeps every real row off the route;
  `depth_real_min_bytes` then does nothing). That is `main`'s plans before X72 and X75.
- **F**: FFTW 3.3.11 MEASURE with `THREADS=k` (POSIX threads, fresh wisdom per run), as the
  client check ran it.

## Cases and widths

The client catalogue's real rows that X72 or X75 move at one thread, at the widths the client's
rule gives them on a Linux budget of 8 or 16 (the client check's "CW" width rule, scored by its
script): the rows are in catalogue `x75` under the same names.

| case | width | moved at one thread by |
|---|---|---|
| r2c and c2r 18432 | 4 | X72 (Zen 3), X75 arm B (Xeon) |
| r2c and c2r 32768 | 4 | X75 arm A (Xeon); Zen 3 keeps its plan (a power-of-two half length) |
| c2r 20480 × 3 | 4 | X72, X75 arm B |
| r2c and c2r 40960 × 6 | 6 | X72, X75 arm A |
| r2c 20480 × 21 | 8 and 16 (the full budget) | X72, X75 arm B |

The client's other real rows (6144–10240, 864 × 384 and the rest) are outside both routes'
ranges; X72's arm B (`avx2` radix overrides on AMD for 3072–5120) is not switched off by this
option and is not judged here.

## Method

- **EPYC 7763** (DMOG, Slurm, shared nodes 16–21, `-c 8` for widths 4–8 and `-c 16` for 16,
  `--hint=nomultithread --distribution=block:block`, unpinned inside the allocation as the
  client runs; the client check's profiles `epyccc8`, `epyccc16`), GCC 16 Release, `avx2`.
- **Xeon 6240** (Slurm `--serial`, the client check's profiles `xeoncc8` (-c 16) and `xeoncc16`
  (-c 32): `numactl` on the 8 or 16 idlest whole cores of the least busy socket and its memory),
  GCC 14.4 Release, `avx512`. The Xeon's load (`uptime`, `squeue`) is recorded before each job;
  the X76 session times there too.
- Four fbench runs per machine, one per width (4, 6, 8, 16), arms A, O, F at that width with the
  A/A control A~, `--tier full` on catalogue `x75` filtered to the width's cases, 12–20 rounds,
  hot cases, statistic median. FFTN arms are `fftn_omp+THREADS=k` (the `omp::executor` on
  libgomp).
- **A case moves** at a width when its plan note differs between A and O in that run; a case
  whose notes agree runs the same plan and is reported, not judged.
- Metric: A ÷ O per moved case, the median over valid rounds with its 95 % paired interval;
  FFTN ÷ FFTW (A ÷ F, and O ÷ F); the band is the run's A/A control.

### Reporting rules (written before the runs)

- A moved case is **slower beyond the band** if A ÷ O's whole interval lies above
  1 + max(band, 3 %) (X72's and X75's limit). A case past 1 + band but not that limit, or
  missed by less than the band, gets one `--serial` re-run of its width, which decides what is
  reported (the near-miss clause of AGENTS section 6, used informatively).
- A case slower beyond the band is diagnosed (plans and per-stage split over the team) and a
  narrowing is proposed (for example by thread count) with drafted decision rules. Nothing is
  adopted.
- FFTN ÷ FFTW decides nothing; it is reported beside the client check's numbers.

## Results

Every output was checked against the reference and correct. On the EPYC the first four runs
(20261125-044540-c31d, -7abc, -d074, -dcec; Slurm 518314–518317) ran at the same time on one
node at load 45–56 with each other and are void (bimodal between processes, 90–370 µs for one
40960 × 6 batch); they were repeated one at a time on idle node18. On the Xeon the first runs
overlapped the X76 session's build (other processes 9–10 cores), and the first 16-thread run
(20261125-044951-f271, Slurm 17042) picked cores already 2 cores busy: both FFTN arms collapsed
to 6–9 ms per execution (scheduler ticks while a spinning member waits), FFTW was unaffected, so
that run is void and was repeated on an idle socket.

Departures from the method: the repeated EPYC runs ran one after another (`--serial`) on
node18, named with `--nodelist` (the client check's profiles with that one line added), and every
repeated or re-run job took 16–24 rounds instead of 12–20.

**Plans.** Every case the table names moves at its width (its notes differ between A and O),
with the route's depth-first chains (EPYC: depth-first 4 × 2304, 4 × 2560, 4 × 5120; Xeon:
16 × 576, 16 × 640, 16 × 1024, 16 × 1280), except the EPYC's r2c and c2r 32768, as expected
(both arms: the breadth-first chain 8 × 16 × 16 × 8).

**Runs.** EPYC 7763, node18 idle (load average 0.9–1.8, no round loaded), band ±1.00 % in each:
width 4 20261125-044813-2a90 (Slurm 518318, 24 rounds) and its re-run 20261125-045446-be3f
(518322, 24), width 6 20261125-044930-3bd3 (518319, 16), width 8 20261125-045000-54f9 (518320,
16), width 16 20261125-045030-1117 (518321, 16). Xeon 6240 (`--serial`, pinned as in the method):
width 4 20261125-044739-fd3c (Slurm 17039, 20 rounds, all loaded by the X76 build, band ±1.94 %)
and its re-run 20261125-045511-181a (17045, 24 rounds, 2 loaded, idle socket 1, ±1.00 %), width 6
20261125-044856-f373 (17040, 19 rounds loaded, ±1.00 %), width 8 20261125-044934-2b6b (17041, 12
rounds loaded, ±1.00 %), width 16 20261125-045638-64ec (17046, 16 rounds, idle socket 1, ±1.00 %).

A ÷ O (the route against the plans it replaced) and A ÷ F (FFTN ÷ FFTW), medians with 95 %
intervals:

| machine | case | width | A ÷ O | A ÷ F |
|---|---|---|---|---|
| EPYC | r2c 18432 | 4 | 0.676 [0.672, 0.684] | 0.570 |
| EPYC | c2r 18432 | 4 | 0.799 [0.794, 0.803] | 0.440 |
| EPYC | c2r 20480 × 3 | 4 | 1.021 [1.013, 1.027]; re-run 1.015 [1.005, 1.036] | 0.807; 0.710 |
| EPYC | r2c 40960 × 6 | 6 | 0.660 [0.634, 0.681] | 0.824 |
| EPYC | c2r 40960 × 6 | 6 | 0.665 [0.652, 0.682] | 0.952 |
| EPYC | r2c 20480 × 21 | 8 | 0.840 [0.831, 0.865] | 0.625 |
| EPYC | r2c 20480 × 21 | 16 | 0.653 [0.624, 0.666] | 0.668 |
| EPYC | r2c, c2r 32768 (not moved) | 4 | 1.000, 0.992 | 0.791, 0.612 |
| Xeon | r2c 18432 | 4 | 0.762 [0.742, 0.784]; re-run 0.743 [0.736, 0.752] | 0.769 |
| Xeon | c2r 18432 | 4 | 1.035 [0.996, 1.083]; re-run 1.019 [1.005, 1.038] | 0.563 |
| Xeon | r2c 32768 | 4 | 0.870 [0.841, 0.886]; re-run 0.870 [0.855, 0.880] | 0.874 |
| Xeon | c2r 32768 | 4 | 1.058 [1.027, 1.091]; re-run 1.031 [1.010, 1.047] | 0.612 |
| Xeon | c2r 20480 × 3 | 4 | 0.978 [0.963, 1.008]; re-run 0.916 [0.884, 0.948] | 0.530 |
| Xeon | r2c 40960 × 6 | 6 | 0.898 [0.884, 0.917] | **1.042** [1.022, 1.056] |
| Xeon | c2r 40960 × 6 | 6 | 0.879 [0.870, 0.883] | 0.849 |
| Xeon | r2c 20480 × 21 | 8 | 0.937 [0.912, 0.973] | 0.516 |
| Xeon | r2c 20480 × 21 | 16 | 0.884 [0.871, 0.903] | 0.583 |

(A ÷ F of the width-4 rows on the Xeon from the re-run.)

- **No case is slower beyond the band** (whole interval above 1 + max(band, 3 %)). Two widths
  had a case past 1 + band and got their re-run, which decides: the EPYC's c2r 20480 × 3
  (1.021, re-run 1.015 [1.005, 1.036]: within the band) and the Xeon's c2r 32768 (1.058 in the
  loaded run, re-run 1.031 [1.010, 1.047]: past the band, under 1.03 at its lower end). The
  Xeon's c2r 18432 is 1.019–1.035 in both runs, its intervals reaching 1.005 and 0.996.
- **r2c keeps its gain in teams; c2r keeps less at 4 threads.** At one thread (the diagnosis
  runs below) c2r 18432 takes 0.93–0.96, c2r 32768 0.70 and c2r 20480 × 3 0.83–0.95; at 4
  threads the EPYC's c2r 18432 gains more (0.79), the Xeon's c2r 20480 × 3 keeps part (0.92),
  and the Xeon's c2r 18432 and 32768 and the EPYC's c2r 20480 × 3 lose it (1.015–1.031). Every
  r2c row and every batch at 6–16 threads gains 0.65–0.94, both kinds of 40960 × 6 included.
- **FFTN ÷ FFTW** at the client's widths: every moved row below 1 except the Xeon's r2c 40960 ×
  6 at 6 threads (1.042); against the client check's numbers (`main` d4b169f, before X72 and
  X75) the EPYC's 40960 × 6 at 6 threads went from 1.22–1.28 to 0.82 (r2c) and 0.95 (c2r).

### Diagnosis: how each plan scales with the team (informative)

Runs 20261125-045546-b4fa (EPYC node18, Slurm 518323, 16 rounds, ±1.00 %) and
20261125-045700-b435 (Xeon socket 1, Slurm 17047, 20 rounds, none loaded, ±1.00 %): arms A and O
at 1, 2 and 4 threads on the width-4 rows. Time at 4 threads ÷ time at 1 thread, per plan:

| machine | case | A (route) | O (old plan) | A ÷ O at 1 / 4 threads |
|---|---|---|---|---|
| EPYC | r2c 18432 | 0.479 | 0.645 | 0.902 / 0.670 |
| EPYC | c2r 18432 | 0.516 | 0.622 | 0.958 / 0.795 |
| EPYC | c2r 20480 × 3 | 0.383 | 0.356 | 0.945 / 1.016 |
| Xeon | r2c 18432 | 1.002 | 1.224 | 0.906 / 0.742 |
| Xeon | c2r 18432 | 1.007 | 0.919 | 0.928 / 1.016 |
| Xeon | r2c 32768 | 0.923 | 0.661 | 0.613 / 0.856 |
| Xeon | c2r 32768 | 0.756 | 0.517 | 0.702 / 1.027 |
| Xeon | c2r 20480 × 3 | 0.383 | 0.345 | 0.829 / 0.921 |

- Where the c2r route loses its gain, the depth-first chain scales worse over the team than
  the breadth-first chain it replaced: Xeon c2r 32768 1.32 times faster at 4 threads against
  1.93 for the old plan, EPYC c2r 20480 × 3 2.61 against 2.81. The old c2r plans fuse the
  pairing into their first pass (notes: "the pairing fused into its first pass"); the route
  pairs in a pass of its own into the owner's workspace block, from which the chain's first
  pass reads the whole line: one more phase per transform and one more round trip of h complex
  values, written by every member and read by every member. The r2c plans pair in a separate
  pass under both arms, and r2c keeps its gain. That fits the c2r route's extra phase, but it is
  not proven: no arm here fuses the pairing into the depth-first chain.
- **On the Xeon, single real rows of 18432–32768 points gain little or nothing from a team of
  4 under either plan** (A at 4 threads 0.76–1.01 of its one-thread time; O's r2c 18432 1.22,
  slower than one thread), while on the EPYC they run in 0.48–0.65 of it. FFTW's 4 threads gain
  little there too (FFTN stays at 0.56–0.87 of FFTW's time), so this is the split rule for
  single rows on the Xeon's mesh (LESSONS 5.2: a single row pays from about 8k points there,
  more for these plans), not the route.

## Verdict

Informative; nothing is adopted or changed.

- **No narrowing is proposed.** No case is slower beyond the band at the client's widths. The
  losses are small (c2r single rows at 4 threads: Xeon 32768 1.031, 18432 1.019; EPYC c2r 20480
  × 3 1.015, all after re-runs) against one-thread gains of 0.70–0.95 on the same rows. A
  narrowing by thread count is not available inside one plan: a plan does not know at
  construction whether teams will run it, and a team program must compute the serial program's
  bits (DESIGN 13.2, the determinism guarantee), which a different chain would not. A narrowing
  by kind (c2r single rows off the route on the class) would trade the one-thread gain for 2–3 %
  at 4 threads, on the client catalogue only for c2r 32768 (weight 0.029, run at width 4).
- **Proposed follow-up (a library change, its own experiment):** fuse the c2r pairing into the
  depth-first chain's first pass, as the breadth-first chain does, so the route drops its extra
  phase and round trip. Drafted decision rules:
  - Arms O = `main`, B = the fused pairing, on both classes; cases: every c2r row the route
    moves in catalogues `x72` and `x75`, at one thread, and the client's c2r rows at their widths
    (18432, 32768 and 20480 × 3 at 4 threads, 40960 × 6 at 6).
  - B is adopted on a class if, at one thread, the geometric mean of B ÷ O over the cases whose
    notes differ is below 1 − band with its interval's upper end below 1 and no such case is
    slower than 1 + max(band, 3 %) (whole interval above); and at the client's widths the
    geometric mean's upper end is at most 1 + band and no case is slower than 1 + max(band, 3 %).
  - The near-miss and narrowing clauses of 2026-11-23 apply (narrowing by length or class).
- **If the maintainer prefers the client's widths to one thread** (not recommended): narrow the
  route to r2c for single rows of the Xeon 6240's class below L2/2 half length. Drafted rule: a
  run on the final code, declared deciding before it starts, shows c2r 18432 and 32768 on their
  old plans at 1 and 4 threads and no case the route still moves slower than 1 + band; the
  one-thread loss on those two rows (1.08–1.42 of the route's time, run 20261125-045700-b435) is
  recorded as the price.
- **Other findings for follow-up:** the Xeon's r2c 40960 × 6 at 6 threads is 1.042 of FFTW's
  time (the only moved case above 1.02); single real rows of 18432–32768 points gain little from
  a team of 4 on the Xeon under either plan (above), which the client's width rule could take
  into account (run them at width 1 there) or a split rule for single rows on the Xeon's mesh
  could; a spinning team on cores that other processes share collapses to scheduler ticks
  (void run 20261125-044951-f271), as fbench's pinning to idle cores assumes.

Recorded in DESIGN 9.1 (the *Depth-first* real route) and LESSONS 5.20.

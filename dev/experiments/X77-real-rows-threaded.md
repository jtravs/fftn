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

(pending)

## Verdict

(pending)

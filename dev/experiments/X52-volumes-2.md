# X52: volumes and threaded planes against MKL and FFTW

- **Status:** running
- **Owner:** X52 agent, 2026-11-18 (overnight round)
- **Phase / task:** 4.1 (threaded planes), 4.2 (volumes)

## Question

Which change to the column phases or to how a team shares a transform closes the volume and
threaded-plane cases that X43, X43b and X44 left over the reference: the Mac's 16 × 256 ×
256 (1.4–1.6 of FFTW), 512 × 64 × 64 and 64 × 64 × 512 (1.1–1.2), the EPYC's 8 lanes (17 of
38 volumes over FFTW's better plan: 128³ 1.33, 256³ 1.10; 2048 × 36 about 1.7, 2048²
1.07–1.11), and the Xeon's volumes against oneMKL (1.15 one thread, 1.24 at 8 lanes)?

## Hypothesis

- **Leaf columns whose legs share an L1 set** (X43b's lead). A volume's outer axis of 16
  points runs as one leaf of 16 over blocks of adjacent pencils; at 16 × 256 × 256 its legs
  lie 1 MiB apart, so all 16 (and the 16 lines they store) fall into one L1 set, more than
  the L1 has ways. `column_radices` (DESIGN 9.3, X40) replaces a last pass whose legs crowd a
  set, but `ranked_radices` offers no other sequence for a leaf length. Two lane passes
  (4 × 4) put 4 legs in a set per pass. Expected: 16 × 256 × 256 near FFTW's time on the
  Mac; on x86 also 16³ (legs 4 KiB apart, one set of a 32–48 KiB L1).
- **Column blocks on the Mac.** The block rule (X40) gives 32 columns (4 Apple lines) where a
  column's elements lie a page apart and 16 elsewhere; FFTW's plans for 512 × 64 × 64 and
  64 × 64 × 512 buffer 64 to 256 pencils at a time (`dft-buffered-512-x64`,
  `dft-buffered-64-x256`).
- **Team work without affinity** (X43's lead) and the per-chunk claims of a split task: a
  split task's chunks are claimed one at a time from one shared counter, so every chunk
  costs a contended atomic and lands on whichever member arrives.

## Change under test

Screens first (no library change), then one commit per arm, each with its rule written
here and committed before its judging run.

- **Arm L:** in `column_radices`, a column of a leaf length whose legs crowd one L1 set
  (the existing `fits` test) runs as two lane passes a × b = n, the most balanced that fit
  (16 = 4 × 4, 32 = 4 × 8, 64 = 8 × 8). Columns that fit are unchanged; rows are unchanged.
  Bits: the changed columns move within the derived tolerances (each column still a lane
  of its own chain; tests/core/columns_test.cpp compares with the chain's columns).

- **Arm W1 (Mac):** a column phase whose column elements lie a page or more apart takes
  blocks of at least 8 L1 lines of columns where a cluster shares L2 (Apple: 64 complex
  doubles), not X40's 4 (`machine_profile::column_page_lines`); x86 unchanged. On top of L.
  Bits unchanged (a block's width does not change what a column computes, X40).
- **Arm G (screen first):** a split task's chunks are claimed in runs within one phase, each
  a share of the phase's chunks left (left / (cap × claim_divisor), at least one), as whole
  tasks already are (`threading/engine.hpp`, `join`): one claim and one count per run, a
  member's chunks of a phase adjacent. Bits unchanged (a chunk computes the same whoever runs
  it, DESIGN 15.5).

## Method

- Cases: the volumes catalogue (42 cases, tier full) and the standard catalogue's planes
  and volumes (`^(c2c|r2c|c2r)-[0-9]+x[0-9]+(x[0-9]+)?-b[0-9]+-(op|ip)$`).
- Machines: Mac M5 Max (Homebrew Clang, `neon_fcma`, FFTW's NEON build), EPYC 7763 (GCC 16,
  `avx2`, 8 whole cores, `-c 8 --hint=nomultithread --distribution=block:block`), Xeon 6240
  (GCC 14, `avx512`, oneMKL, 8 whole cores of one socket, the same options). Judging runs on
  the EPYC and the Xeon through `fbench_remote.py submit --serial`; the Mac locally, one
  fbench run at a time. Threads: FFTN's `thread_pool` at 8 lanes, FFTW and oneMKL at 8
  threads; competitors by their better plan per case (LESSONS 5.9).
- Metric: per case the median paired ratio arm ÷ base, 95 % interval; geometric means.
- Decision rule for **L**, written before its judging runs, per machine-target (release
  toolchains: GCC on x86-64, Clang on the Mac): adopt L if (1) the geometric mean of L ÷
  base at one thread over the cases of the case set whose program L changes (by describe)
  is at most 0.95, (2) no case of the case set is slower than base by more than the larger
  of 2 % and the run's A/A band, at one thread or (EPYC, Xeon) at 8 lanes (a case just past
  it on a loaded machine is re-run once before it decides), and (3) one-dimensional plans
  are unchanged (by construction: rows never take a column phase). A machine where L
  changes no case keeps it by construction.

- Decision rule for **W1**, written before its judging run, on the Mac: adopt W1 if (1) the
  geometric mean of W1 ÷ L at one thread over the cases of the case set whose program W1
  changes (by describe) is at most 0.97, and (2) no case of the case set is slower than L by
  more than the larger of 2 % and the run's A/A band (a case just past it on a loaded machine
  is re-run once before it decides).

- Decision rule for **G**, written before its judging runs, per machine (EPYC, Xeon; GCC):
  adopt G if (1) the geometric mean of G ÷ L at 8 lanes over the cases of the case set that
  a team runs split (not inline, not as whole transforms only: the planes and volumes whose
  single transform the schedule splits) is at most 0.97, (2) no case of the case set is
  slower than L at 8 lanes by more than the larger of 2 % and the run's A/A band (a case just
  past it on a loaded machine is re-run once before it decides), and (3) one-thread
  programs are unchanged (by construction: the engine runs only for teams). Where G is not
  adopted on a machine it is compiled out or gated off there. The Mac's teams are not
  measured in this experiment; G is adopted there only if a Mac run at 6 lanes meets the same
  rule.

## Results

### Screens (not judging)

- Mac, run 20261117-234022-6925 (`main` a0dffe1; 5 rounds, all loaded, band ±6.1 %): forced
  column radices 4 × 4 for every column of 16 took 16 × 256 × 256 to 0.67–0.70 of `main`'s
  time (FFTW 0.79–0.80), 8 × 2 0.70–0.82; but 16³, whose legs fit the M5 Max's L1, lost
  1.35–1.53 with them. 256 × 256 × 16, 64 × 64 × 512 and 512 × 64 × 64 unchanged.
- Mac, run 20261117-234202-9036 (`main` a0dffe1; 6 rounds, all loaded, band ±2.0 %), 512 ×
  64 × 64 and 64 × 64 × 512: every column block forced to 64 columns 0.85–0.91 (16: 0.94–0.99;
  8: 1.15–1.28); other column radices lost (512 as 8 × 64 2.4, 16 × 32 2.1; 64 as one pass
  2.9–3.3, 4 × 16 1.8); FFTW 0.77–0.83.

## Verdict

Pending.

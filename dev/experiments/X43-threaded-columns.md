# X43: column blocks and phases of planes under threads

- **Status:** running
- **Owner:** task 4.2 agent, 2026-11-17
- **Phase / task:** 4.1 (threaded planes), feeding 4.2 (volumes)

## Question

Why do FFTN's largest c2c planes scale 2.3–4.7 times on 8 cores where FFTW's threads scale
4–6 (X41, EPYC run 20261117-122327-970c), and why do oneMKL's threads beat FFTN's from 16²
to 1024² on the Xeon (run 20261117-131055-a1d0)? The answer decides how a threaded column
phase is cut into chunks and how wide its blocks are, for planes now and for the volumes
of task 4.2.

## Hypothesis

Two suspects, both in what a team gets to share, not in what a chunk computes:

1. **The column phase is cut too coarsely** (checked against the code before any run, and
   ruled out): the work graph (`src/threading/work_graph.cpp`) cuts a phase of lines into
   runs of about `chunk_points` (2048) points by the length its stages record, and a lanes
   pass records n · B for a block of B columns (`strategies/lanes.cpp`, `lanes_body`), so a
   column phase already gets one block per chunk from 256² up (512²: 32 chunks, 2048²: 128).
   The screen's `CHUNK_POINTS=256` arm still shows whether finer row chunks matter.
2. **The column blocks are sized for one thread** (X41's diagnosis). Where a column's
   elements lie a page or more apart the block is X40's wide one (16 complex doubles on
   x86), whatever its tiles cost: at 2048² each member's two tiles take 1 MiB against the
   EPYC 7763's 512 KiB private L2; at 1000² 512 KiB. One thread had the whole 32 MiB L3
   behind that L2; eight members share it with the plane. Expected: narrower blocks under
   threads (tiles within a member's L2) cut the column phase's L3 traffic at 1000²–2048²,
   while serial plans keep X40's wide blocks, which won there for one thread (2048²: 0.126
   of the chain's time with 16 columns, 0.183 with 4).

## Change under test

- **Screen (diagnosis, no library change; not a judging run):** `main` 310d8d6 at 8 lanes
  with fbench's existing knobs: `CHUNK_POINTS=256` (finer chunks everywhere), `COLUMN_BLOCK=8` and `4`
  (narrower blocks forced, serial and threaded), against the default, FFTN serial, FFTW at
  8 threads and serial, and on the Xeon oneMKL at 8 threads (`mkl_omp`) and serial.
- **Arm B1 (written after the screen, before its judging runs):** a *team program*. A plan
  can be executed serially and by teams, and must give the same bits either way; the width
  of a column block does not change what any column computes (each column is a lane of its
  own chain, X40: the tests hold lanes columns of any block and chain columns byte for byte
  equal where the radices agree), so a plan may run different blocks by team. A c2c plan of
  two or three axes builds, beside its program, a second whole plan core whose column
  blocks are at most `machine_profile::column_team_bytes` of columns (128 bytes: 8 complex
  doubles, 16 complex floats; set on x86-64, 0 elsewhere, which builds none), with the same
  radices per column, and keeps it only if some block differs. Every execution by a team of
  two or more members (team overload, executors) runs it, with its own work graph and
  scratch layout; serial executions and the one-thread programs are unchanged, and
  `workspace_bytes(lanes > 1)` covers the team core. Columns that the one-thread rule sends
  alone through the chain stay there (2048 × 36: the Xeon's forced blocks of 8 took 2.0 of
  the chain's time; not changed by B1). The rule is X43's decision rule above, the target
  cases the threaded c2c planes and volumes whose team program differs from the program.
- **Arm B2 (written before its runs, on top of B1):** in the team program, c2c columns that
  the one-thread rule runs alone through the chain because even a line of them would put
  the tiles past the budget (2048 × 36) run as lanes blocks of one line (4 complex doubles on
  x86), with the chain's radices so the bits stay the program's. Members running
  neighbouring columns alone write the same cache lines; the EPYC screen's blocks of 4 took
  0.72–0.77 of the chain columns' time at 8 lanes there (blocks of 8: 1.06, and 2.0 on the
  Xeon). Rule: X43's decision rule, against B1 at 8 lanes, the target cases the threaded
  planes whose team program B2 changes (2048 × 36 in and out of place in the case set).
- Rows-against-columns split: the screen shows how the two phases scale; an arm only if it
  points at one.

## Method

- Cases: the standard catalogue, tier full: the 37 c2c planes and the 71 real planes,
  `^(c2c|r2c|c2r)-[0-9]+x[0-9]+-b[0-9]+-(op|ip)$` (X41's threaded set), plus the c2c
  volumes `^c2c-[0-9]+x[0-9]+x[0-9]+-b1-op$` for the record (task 4.2 judges those).
- Machines: EPYC 7763 (GCC 16, `avx2`; 8 whole cores, `-c 8 --hint=nomultithread
  --distribution=block:block`, X41's allocation) and Xeon 6240 (GCC 14, `avx512`, with
  oneMKL; 8 whole cores of one socket, the same options). The Mac only for a Mac-specific
  arm. Timing through `fbench_remote.py submit --serial`, interleaved rounds, A/A control.
- Threads: FFTN's `thread_pool` at 8 lanes (`THREADS=8`); FFTW `THREADS=8`
  (`fftw_plan_with_nthreads`); oneMKL `mkl_omp` at 8 threads. FFTN is compared with each
  competitor's better plan per case, threaded or serial (LESSONS 5.9, as X41 did).
- Metric: per case the median paired ratio arm ÷ base at 8 lanes, 95 % interval; geometric
  means over the target cases; FFTN ÷ competitor for the record.
- Decision rule, written before any judging run, per arm and machine-target (release
  toolchains: GCC on x86-64): adopt the arm if (1) the geometric mean of arm ÷ base at 8
  lanes over the threaded planes whose work graph or team program the arm changes is at
  most 0.95, (2) no threaded plane of the case set is slower than base by more than the
  larger of 2 % and the run's A/A band (a case just past it on a loaded machine is re-run
  once before it decides), and (3) one-thread programs are unchanged (checked by
  construction and by the tests), or, if an arm changes them, the same rule (1)–(2) holds
  at one thread over the 108 planes. A win on one machine and a loss on the other is
  adopted per machine.

## Results

### Screen (diagnosis, `main` 310d8d6, not judging)

27 cases: c2c planes 128²–2048², 480 × 640, 1000², 36 × 2048, 2048 × 36 in and out of place,
128² × 16, the standard catalogue's c2c volumes; 6 rounds. Ratios are times over T8
(FFTN at 8 lanes, the rule's blocks).

- EPYC run 20261117-142758-6d63 (Slurm 511965; all 6 rounds loaded, other processes 56
  cores of 128, band ±10.9 %). Forced blocks of 8 at 8 lanes (TB8): 0.892 over the 27;
  1000² 0.67, 2048² 0.78–0.80, 1024² 0.80–0.86, 480 × 640 0.76–0.83, 128² 0.73–0.79, 256²
  0.80–0.91, 512² 1.00, 36 × 2048 0.96, 100³ 0.78, other volumes 0.93–1.08; 2048 × 36 (the
  rule's columns alone through the chain) 1.05–1.06, but blocks of 4 (TB4) 0.72–0.77 there,
  while TB4 lost elsewhere (1.08 overall; 1024² 1.33). The same blocks of 8 at one thread
  (SB8 against S1): 2048² 1.42, 256² 1.17, 512² 1.12, 36 × 2048 1.16, 64³ 1.24, but 1000²
  0.90, 1024² 0.95: one thread wants X40's wide blocks where a team wants narrow ones.
  `CHUNK_POINTS=256` (TC) lost 1.125 (36 × 2048 1.45–1.51, volumes 1.3–1.7). FFTN's 8 lanes
  against its one: 2.3 (128² 1.8, 2048² 2.6, 1024² 3.4–3.6, 2048 × 36 1.25); FFTW's 8 threads
  over FFTN's 8 lanes: 1.57 overall but 0.67–0.70 at 2048², 0.39–0.40 at 2048 × 36, 0.51 at
  100³, 0.65 at 128³.
- Xeon run 20261117-142924-e025 (Slurm 16321; 2 of 6 rounds loaded). TB8: 0.973 over the 27;
  128² 0.65–0.82, 256² 0.82–0.84, 1000² 0.78–0.80, 36 × 2048 0.90, 480 × 640 0.94, volumes
  0.80–0.96 (16³ 1.21, wide interval), 512²–2048² 1.01–1.13; 2048 × 36 2.0 (blocks of 8 in
  place of the chain). TC 1.107. oneMKL's 8 threads over FFTN's 8 lanes: 1.048 overall, but
  0.48–0.83 on every plane but 2048² (1.10–1.15) and 0.39–0.53 on the volumes from 32³; FFTN's
  8 lanes run 128² no faster than its one (S1 ÷ T8 0.71–0.94), and 128² × 16 (whole planes
  per member) only 3.8 times faster where oneMKL's threads take 0.40 of FFTN's time.

Reading: the chunks are not the problem (finer ones lose everywhere). Narrower blocks help
a team on both machines (L2 per member at 1000²–2048² on the EPYC; more blocks than members
at 128²–256²), but hurt one thread, so they belong in a team program (arm B1). The
remaining gap to oneMKL's threads on the Xeon (small planes, whole planes per member,
volumes) is not in the column blocks: it is the cost of a team execution itself, outside
X43's arms.

### Arm B1 (team program, column blocks of at most 128 bytes)

- EPYC run 20261117-162414-89ca (Slurm 512008; A 9323b8b and B1 5eb077f at 8 lanes, FFTW at
  8 threads and serial; 8 rounds, all under load, other processes about 35 cores of 128,
  band ±2.5 %). B1 ÷ A over the 39 threaded c2c planes and volumes whose team program
  differs: 0.887 (1000² 0.66–0.67, 480 × 640 0.71–0.73, 128² 0.75–0.76, 96² 0.78–0.81, 100²
  0.83, 1024² 0.82–0.84, 256² 0.82–0.91, 512² 0.87–0.89, 2048² 0.87–0.90, 36 × 2048 0.91–0.96,
  volumes 0.75–0.97 with V's plane-at-a-time order, the base's). Worst c2c 1.017 (32² op,
  inline: no team runs it). The real planes, whose programs B1 cannot change (no team core
  for real kinds), came out 1.010 overall with nine cases at 1.026–1.050, past the band:
  re-run below before rule (2) decides. FFTN at 8 lanes over FFTW's better plan, the 37 c2c
  planes: 0.953 → 0.852 (still over: 2048 × 36 2.3–2.5, 32² 1.40 inline, 2048² 1.07–1.11,
  1024² out of place 1.06).

### Arm B2 (team program, chain-only columns as blocks of a line)

- EPYC run 20261117-163827-f12d (Slurm 512041; B1 5eb077f against B2 de05523 at 8 lanes, the
  37 c2c planes; 8 rounds, all under load, band ±3.9 %): 2048 × 36 0.728 and 0.714, no other
  team program changes; worst 1.042 (32² in place, whose program and team program B2 does
  not change), inside the band. Rule met on the EPYC.

## Verdict

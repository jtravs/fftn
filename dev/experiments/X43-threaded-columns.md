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
- **Arm B1 (only if the screen supports it):** a threaded column block sized to a member's
  share of L2, kept apart from the one-thread rule (its own profile field or branch). Its
  exact form and its rule are written here and committed before its judging runs.
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

## Verdict

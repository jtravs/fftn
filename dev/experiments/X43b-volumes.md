# X43b: volumes plane at a time, the outer axis over blocks of pencils

- **Status:** running
- **Owner:** task 4.2 agent, 2026-11-17
- **Phase / task:** 4.2

## Question

Does a c2c volume run faster with its two inner axes plane at a time (each plane finished
while it is in cache, whole planes per member under threads) and its outer axis over
blocks of adjacent pencils, than axis by axis over the whole volume as `main` runs it? The
answer is task 4.2's program for 3-D c2c (DESIGN.md 9.3, "Volumes").

## Hypothesis

`main` (X40) runs a volume n0 × n1 × n2 axis by axis: the rows of every plane, then the
columns of axis 1 over every plane, then the pencils of axis 0, three sweeps over the
whole volume. Once the volume leaves the caches each sweep streams it from memory and back.
Plane at a time touches the volume twice: each plane's rows and columns while the plane
(n1 n2 points, 64 KiB at 64², 1 MiB at 256²) is in L2 or L3, then the outer axis. FFTW's
rank-3 plans make the same split (a rank-2 plan in a loop over n0, then a vector loop of
rank-1 transforms over the n1 n2 pencils). Expected: volumes beyond L2 (from 64³) take
0.7–0.9 of `main`'s time at one thread, small volumes unchanged; under threads whole planes
are the members' work instead of lines of each axis, one phase boundary fewer.

## Change under test

- Arm V: in a c2c plan of three axes with n0 > 1, a loop over the n0 planes whose body is
  the plane's row phase (the chain, or a loop of leaves) and its column phase (X40's lanes
  over blocks of adjacent columns where they fit, else the chain), then the outer axis as
  before (`add_volume_planes`, `core/plan_core.cpp`). The work graph takes the loop over the
  planes as one phase whose lines are whole planes (`threading/work_graph.cpp`). Every line
  computes what it computed axis by axis, so the bits do not change (tests compare lanes and
  chain columns byte for byte and team and serial runs byte for byte); the scratch per
  member stays a line, a tile of columns or a tile of pencils (tests/core/volumes_test.cpp
  checks the bound). Planes, rows and real volumes are unchanged.
- The outer axis's blocks of pencils follow `column_block` (X40, X41) at one thread and
  X43's team rule under threads if X43 adopts one.

## Method

- Cases: the new `volumes` catalogue (bench/catalogues/volumes.toml): cubes 16³–256³, odd
  and prime cubes (48³, 81³, 97³, 100³, 120³), non-cubes with a short or long axis at either
  end, in place and out of place, two backward; plus the standard catalogue's six c2c
  volumes, `^c2c-[0-9]+x[0-9]+x[0-9]+-b1-op$`.
- Machines: EPYC 7763 (GCC 16, `avx2`) and Xeon 6240 (GCC 14, `avx512`, with oneMKL), 8 whole
  cores each (X41's allocation), through `fbench_remote.py submit --serial`; the Mac M5 Max
  (Homebrew Clang, `neon_fcma`, FFTW's NEON build) for the one-thread rule, one run.
- Arms: V and its parent at one thread and at 8 lanes (`THREADS=8`), FFTW serial and at 8
  threads, oneMKL serial and at 8 threads (`mkl_omp`) on the Xeon. Competitors are compared
  by their better plan per case, threaded or serial (LESSONS 5.9).
- Metric: per case the median paired ratio V ÷ parent, 95 % interval; geometric means.
- Decision rule, written before any judging run, per machine-target (release toolchains:
  GCC on x86-64, Clang on the Mac): adopt V if (1) the geometric mean of V ÷ parent over the
  catalogue's volumes whose arrays exceed the L2 (64³ and larger, and the non-cubes of 1 MiB
  or more) is at most 0.95 at one thread, (2) no volume of the case set is slower than the
  parent by more than the larger of 2 % and the run's A/A band, at one thread or at 8 lanes
  (a case just past it on a loaded machine is re-run once before it decides), and (3)
  planes, rows and real transforms are unchanged (by construction; their programs are
  identical). Task 4.2's acceptance is judged separately: FFTN ÷ FFTW and ÷ oneMKL per case,
  one thread and threaded.

## Results

### Mac M5 Max, one thread

Run 20261117-153528-4b71 (P bce5979 against V 9323b8b, FFTW NEON; 6 rounds, the first under
load, band ±3.6 %; Homebrew Clang, `neon_fcma`): V ÷ P 0.963 over the 38 volumes, none
slower than 1.004; 128³ 0.87–0.89, 64 × 64 × 512 0.83 out of place, 512 × 64 × 64 0.92,
256 × 256 × 16 0.92–0.97, 256³ 0.95–0.98, 32³ 0.93, the rest 0.95–1.00. Over the volumes the
rule names by size (64³ and larger, the non-cubes of 1 MiB or more) 0.96: rule (1) fails as
written. The rule's sizes were x86's (512 KiB–1 MiB of private L2); the M5 Max's L2 is 16 MiB
per cluster, and over the volumes beyond it (128³, 256³, 120³, 64 × 64 × 512, 512 × 64 × 64)
the same run gives 0.925. Reading the rule by its words rather than its listed sizes would
be a rule changed after seeing results, so the Mac takes a new arm:

**Arm VM (written before its run; the same code as V):** on the Mac, adopt V if (1) the
geometric mean of V ÷ P over the catalogue's volumes whose arrays exceed the M5 Max's
16 MiB cluster L2 (128³, 256³, 120³, 64 × 64 × 512, 512 × 64 × 64, in and out of place) is at
most 0.95 at one thread, and (2) no volume is slower than P by more than the larger of 2 %
and the band, in a fresh run.

VM, run 20261117-154622-1813 (8 rounds, all under load, other processes about 4 cores, band
±5.2 %): 0.962 over those ten (128³ 0.96–0.97, 256³ 0.95–0.97, 120³ 0.94–0.98, 64 × 64 × 512
0.91–0.95, 512 × 64 × 64 0.97 and 1.04); 0.977 over the 38; worst 1.036 (512 × 64 × 64 in
place, inside the band). Rule (1) fails: VM rejected, the Mac keeps the axes one by one.

## Verdict

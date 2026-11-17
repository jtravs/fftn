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

### EPYC 7763 and Xeon 6240, one thread

- EPYC, run 20261117-155546-f3c5 (Slurm 512004; 8 rounds, 4 under load, band ±1.1 %): V ÷ P
  0.980 over the 38, 0.979 over the 30 target volumes; 120³ in place 0.92, 512 × 64 × 64 and
  64 × 64 × 512 in place 0.90–0.91, 128³ 0.95–0.97, most others 0.96–1.00, but 256³ out of
  place 1.065. Rule (1) fails (and (2) at 256³): V rejected on the EPYC. Its 32 MiB L3 per
  CCX holds every volume up to 128³, so a third sweep costs little there.
- Xeon, run 20261117-154354-7453 (Slurm 16339; 8 rounds, 2 under load, band ±4.1 %): V ÷ P
  0.945 over the 38, 0.934 over the 30 targets; 64 × 64 × 512 0.75–0.78, 128³ 0.84–0.91,
  256³ 0.87–0.97, 120³ 0.88–0.91, 100³ backward 0.92; 97³ out of place 1.092, past the band
  (re-run below, rule (2)). The Xeon has 1.4 MiB of L3 per core behind its 1 MiB L2.

### EPYC 7763, 8 lanes

Run 20261117-160908-2b72 (Slurm 512006; 8 rounds, all under load, other processes about 35
cores of 128, band ±2.4 %; P and V at 8 lanes, FFTW at 8 threads and serial): V ÷ P 0.778
over the 38 volumes (0.66–0.95; 16³ 0.99, which runs inline); FFTN at 8 lanes over FFTW's
better plan 1.219 → 0.950 (worst 16 × 256 × 256 1.53, 100³ 1.39–1.48, 120³ out of place
1.43, 512 × 64 × 64 out of place 1.26, 128³ out of place 1.26, 128 × 64 × 32 1.23–1.26).
Whole planes per member, one phase boundary fewer and each plane's rows and columns on one
core, are worth far more to a team than the third sweep is to one thread.

### Xeon 6240, 8 lanes

Run 20261117-160719-6a3c (Slurm 16341; 8 rounds, 2 under load, band ±4.7 %; P and V at 8 lanes,
FFTW and oneMKL at 8 threads and serial): V ÷ P 0.788 over the 38 (0.66–0.96), none slower.
FFTN at 8 lanes over FFTW's better plan 0.78 with V (worst 1.25); over oneMKL's better plan 1.65 → 1.28
(worst 1.82): oneMKL's threads stay well ahead on the Xeon.

**Arm VT (written before its runs):** V's order in the team program only (X43's team core,
`plan_core::team`): plane at a time changes no bits, so a team may run it while serial
executions keep the axes one by one. A profile field `team_volume_planes`, set on x86-64
where each core has its own L2 (the EPYC; the Xeon's program already has V); the team core
is then built for any c2c volume and kept where its program differs. Rule: on the EPYC, adopt
VT if (1) the geometric mean of VT ÷ base at 8 lanes over the volumes catalogue's volumes
that run threaded (not inline) is at most 0.95, and (2) no volume is slower than the base by
more than the larger of 2 % and the band at 8 lanes, in a fresh run; one-thread programs are
unchanged by construction (the base is 06229fa, the axes one by one on the EPYC).

## Verdict

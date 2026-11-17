# X43b: volumes plane at a time, the outer axis over blocks of pencils

- **Status:** done (plane at a time rejected for one thread on all three machines; adopted for teams on the EPYC and the Xeon)
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
- Xeon re-run of 97³, run 20261117-184525-337a (Slurm 16382; 12 rounds, none under load, band
  ±1.0 %): out of place 1.048 (interval 1.035–1.088), in place 1.012. Rule (2) fails: V
  rejected on the Xeon too (commit 06229fa had adopted it there before this re-run; reverted
  below). No machine runs volumes plane at a time in its one-thread program.

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

#VT, EPYC run 20261117-190518-b8a6 (Slurm 512116; 06229fa against ca43a10, both at 8 lanes; 8
rounds, all under load, other processes about 24 cores, band ±1.0 %): 0.775 over the 38, 0.77
over the 36 that run threaded (48³ 0.69–0.70, 100³ 0.69–0.71, 256³ 0.71–0.74, 128³ 0.77, 97³
0.85–0.86, 16 × 256 × 256 0.77–0.83); 16³ (inline) 1.000–1.012, inside the 2 %. Rule met: VT
adopted on the EPYC.

**Arm VTX (written before its run, after V's rejection on the Xeon):** VT on the Xeon's
profile too (`team_volume_planes` on x86-64 with a private L2 below 2 MiB, both the EPYC's and
the Xeon's classes; the Xeon builds no team column blocks, B1). Rule: on the Xeon, VT's rule
(threaded volumes' geometric mean at most 0.95, none slower beyond the larger of 2 % and the
band at 8 lanes) in a fresh run against the axes one by one at 8 lanes.

VTX, Xeon run 20261117-193858-1598 (Slurm 16403; bce5979 against c4cec37 at 8 lanes, oneMKL at
8 threads and serial, c4cec37 serial; 8 rounds, 6 under load, band ±11.8 %): 0.783 over the
38, 0.771 over the 36 that run threaded; worst 16³ in place 1.048 (inline: its team program is
never run). Rule met: VTX adopted on the Xeon. FFTN ÷ oneMKL's better plan in the same run:
8 lanes 1.24 (worst 1.74), one thread 1.15 (worst 1.77).

## Task 4.2's acceptance (FFTN ÷ the competitor's better plan, the 38 volumes)

- Mac, one thread (run 20261117-154622-1813, P, the Mac's program): ÷ FFTW 0.804; over it
  16 × 256 × 256 1.57–1.63, 512 × 64 × 64 1.19–1.24, 64 × 64 × 512 1.10–1.21, 64³ out of
  place 1.02, 32³ out of place 1.01, 32 × 64 × 128 in place 1.02.
- Mac, 6 lanes of one cluster (run 20261117-171803-2bd5, ce01908, FFTW at 6 threads and
  serial; 5 rounds, all under load, band ±12.4 %): 0.755; over 16 × 256 × 256 1.35–1.40,
  512 × 64 × 64 1.19–1.20.
- EPYC, one thread (run 20261117-155546-f3c5, P, the EPYC's program): ÷ FFTW 0.848; over it
  16³ 1.13, 128 × 64 × 32 1.08–1.12, 32³ 1.04, 16 × 256 × 256 out of place 1.03, 512 × 64 ×
  64 and 64 × 64 × 512 in place 1.02.
- EPYC, final programs (run 20261117-191501-efe5, ce01908: one thread the axes one by one,
  teams VT with B1's blocks; FFTW at 8 threads and serial; 6 rounds, all under load, band
  ±1.9 %): one thread ÷ FFTW 0.845 (worst 16³ 1.15); 8 lanes ÷ FFTW's better plan 0.904, 17
  of 38 over it: 16 × 256 × 256 1.37–1.58, 512 × 64 × 64 1.11–1.37, 128³ out of place 1.33,
  64 × 64 × 512 out of place 1.20, 256 × 256 × 16 1.06–1.20, 120³ out of place 1.16, 16³ 1.13–
  1.15 (inline), 256³ 1.10–1.12, 100³ backward 1.12.
- Xeon, one thread (run 20261117-154354-7453, P, the Xeon's program): ÷ FFTW 0.733 (over:
  64 × 64 × 512 1.13–1.24, 512 × 64 × 64 in place 1.04); ÷ oneMKL 1.118 (worst 1.51; 16³
  1.42–1.46, 128³ 1.43–1.49, 100³ 1.35–1.38).
- Xeon, 8 lanes (run 20261117-184612-d6bc, ce01908, whose teams run what c4cec37's do: plane
  at a time, no team blocks; 6 rounds, 3 under load, band ±3.1 %): ÷ FFTW's better plan 0.820
  (over: 16 × 256 × 256 out of place 1.26, 512 × 64 × 64 out of place 1.07, 32 × 64 × 128 out of
  place 1.06, 81³ 1.01–1.04, 64 × 64 × 512 out of place 1.02); ÷ oneMKL's better plan 1.268
  (worst 1.77).
- The 16-point outer axes of 16 × 256 × 256 (and 16³ on x86) run as one leaf of 16 whose 16
  legs, a plane apart, share one L1 set; `column_radices` offers no other sequence for a
  leaf length. A lead for the column radices' owner, not changed here.

## Verdict

- **V (one thread) rejected** on every machine: EPYC 0.979, Mac 0.962 (VM), Xeon 0.934 but 97³
  out of place 1.048 on re-run (rule (2)). One thread keeps the axes one by one (c4cec37).
- **VT adopted on the EPYC** (0.775 at 8 lanes) and **VTX on the Xeon** (0.771): teams run a
  team program with the inner axes plane at a time (`team_volume_planes`, x86-64 with a
  private L2 below 2 MiB; commits ca43a10, c4cec37). The Mac's teams are unmeasured and keep
  the axes one by one.
- The scratch bound is tested (tests/core/volumes_test.cpp); the outer axis's pencil blocks
  follow `column_block` (and B1's team blocks on the EPYC). DESIGN 9.3, decision log;
  LESSONS 5.11.
- Task 4.2's acceptance is not met case by case (section above): against FFTW the Mac misses
  16 × 256 × 256, 512 × 64 × 64 and 64 × 64 × 512 at one thread and threaded, the EPYC 16³,
  128 × 64 × 32 and 32³ at one thread and 17 of 38 volumes at 8 lanes, the Xeon 64 × 64 × 512
  at one thread and five volumes at 8 lanes; oneMKL stays ahead on the Xeon (1.15 one thread,
  1.24 at 8 lanes). Leads: the 16-point outer leaf whose legs share an L1 set (column radices),
  the outer axis of 512 × 64 × 64 (long pencils, wide tiles), and the team engine's lack of
  affinity in hot loops (X43's verdict).
- **Few planes (second reader, after the verdict):** the team program's plane phase has one
  line per plane, so a volume of fewer planes than members leaves members idle. Team plane
  order is therefore limited to volumes of at least 8 planes (no measured case changes: the
  catalogue's thinnest had 16; commit 135b63f); 4 × 512 × 512 and 2 × 1024 × 1024 were added
  to the volumes catalogue, unmeasured. Teams of more than 8 members on volumes of 8 to 63
  planes remain a known limit (not measured).

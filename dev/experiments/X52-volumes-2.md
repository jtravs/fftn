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

- **Arm W4 (x86-64 with a private L2):** a c2c lanes column phase widens its block to 1 KiB of
  columns (64 complex doubles) where both tiles fit `column_tile_bytes` (a quarter of L2):
  columns of up to 64 points on the EPYC, 128 on the Xeon (`machine_profile::
  column_wide_bytes`). The team program's blocks are clamped after it as before (B1). Bits
  unchanged. Decision rule, written before its judging runs, per machine (EPYC, Xeon; GCC):
  adopt W4 if (1) the geometric mean of W4 ÷ L at one thread over the cases of the case set
  whose program W4 changes (by describe) is at most 0.97, and (2) no case of the case set is
  slower than L by more than the larger of 2 % and the run's A/A band, at one thread or at 8
  lanes (a case just past it on a loaded machine is re-run once before it decides).

- **Arm S (written after the pathology below, before its judging runs; on top of G):** (S1)
  a real plane or volume is split by its complex points (the last axis n / 2 + 1 long), not
  its real points, against the split thresholds; (S2) where the cache group is a mesh (the
  Xeon), a plane or volume (a work graph with a phase of lines) is split only from 32768
  points. Rows unchanged; one-thread programs unchanged; bits unchanged (only who runs what).
  Decision rule per machine (EPYC, Xeon at 8 lanes; Mac at 6 lanes): adopt S if (1) the
  geometric mean of S ÷ G over the cases of the case set whose split decision S changes (by
  the thresholds' arithmetic) is at most 0.95, (2) no case of the case set is slower than G by
  more than the larger of 2 % and the run's A/A band (a case just past it on a loaded machine
  is re-run once), and (3) one-thread programs unchanged (by construction). The parts are
  adopted together; a machine where S changes no case keeps it by construction.

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

- EPYC, run 20261117-234518-1124 (Slurm 512510; `main` at 8 lanes; 6 rounds, all on a node
  whose other jobs used about 71 of 128 cores, band ±3.6 %), the volumes: FFTN at 8 lanes ÷
  FFTW's better plan (8 threads or serial) 0.687, over 1.05 only 16³ (1.13–1.14, which runs
  inline) and 4 × 512 × 512 out of place (1.06); X43b's 17 of 38 did not recur on this
  loaded node. Forced 4 × 4 columns of 16 took 16³ to 0.68 and 16 × 256 × 256 to 0.87–0.93.
  Coarser chunks (`CHUNK_POINTS=8192`) took 128³ to 0.84–0.92, 128 × 64 × 32 0.89, 16 × 256 ×
  256 0.91 but 32³ 1.52–1.54 and 17 × 15 × 37 2.4–2.5 (32768: 1.25 overall).
- EPYC, run 20261117-234518-a518 (Slurm 512512; the same node load, band ±3.4 %), the 37 c2c
  planes: FFTN at 8 lanes ÷ FFTW's better plan 0.64, over 1.05 only 2048 × 36 (1.10–1.14);
  coarser chunks lost (8192: 1.17, 32768: 1.41).
- Xeon, run 20261117-234934-c994 (Slurm 16423; 6 rounds, 3 loaded, band ±13.5 %), the
  volumes: one thread ÷ oneMKL 1.05 (26 of 42 over 1.05: 16³ 1.35–1.41, 64 × 64 × 512
  1.29–1.49, 128³ 1.25–1.46, 512 × 64 × 64 1.32–1.35, 81³ 1.22–1.27, 120³ 1.22–1.31); 8 lanes
  ÷ oneMKL's better plan 0.97 (18 over 1.05: 16³ 1.41–1.43 inline, 30 × 40 × 50 1.52–1.64,
  81³ 1.26–1.51, 32³ 1.31–1.40, 48³ 1.32–1.33, 100³ and 120³ out of place 1.26–1.27). FFTN's
  8 lanes take 0.60–0.69 of its one thread at 32³ and 0.44–0.52 at 30 × 40 × 50, where
  oneMKL's 8 threads take 0.51 and 0.29 of its own: the small volumes scale poorly. Forced
  4 × 4 columns of 16: 16³ 0.93–0.97 (they also change axis 1 there).
- EPYC, run 20261118-000233-c39a (Slurm 512521; L and G at 8 lanes, the standard
  catalogue's 132 c2c planes and volumes; 8 rounds, all loaded, band ±2.5 %): G ÷ L 0.984
  (128³ 0.87, 36 × 2048 0.90, 1024² 0.94, 48³ 0.94, 100³ 0.94, 480 × 640 0.95; 2048² 1.02–1.04,
  2048 × 36 in place 1.03, inside the band).

### L and W1 on the Mac (judging)

Run 20261118-000801-5e21 (L 446be06 against `main` a0dffe1 and W1 c3d80f8, FFTW; 10 rounds, 9
loaded, band ±6.4 %), the volumes catalogue. L changes 16 × 256 × 256 alone on the Mac (by
describe; none of the standard catalogue's planes and volumes): L ÷ `main` 0.57 out of place,
0.54 in place (`main` 1.76–1.85 of L); FFTN ÷ FFTW there 1.41–1.42 → 0.76–0.81. No case slower
than `main` beyond 1.058 (81³, unchanged program), inside the band. **L adopted on the Mac.**
W1 changes 30 of the 42 volumes: W1 ÷ L 0.993 over them (worst 100³ 1.06, 2 × 1024 × 1024
1.03–1.05), against the rule's 0.97: **W1 rejected** (reverted).

### Screens after the Mac verdicts (not judging)

- W3 (Mac; column blocks of 1 KiB where their tiles fit twice L1): runs 20261118-002050-0efb
  (volumes, band ±7.7 %) and 20261118-002621-c540 (the 132 planes and volumes, band ±4.8 %):
  0.985 over the 34 volumes and 1.031 over the 22 planes it changes (64 × 64 × 512 and 512 ×
  64 × 64 0.87–0.94, but 96² 1.10–1.11, 64² 1.06–1.08): dropped without a judging run.
- EPYC one thread, run 20261118-002217-a671 (Slurm 512541; band ±2.6 %), every column block
  forced: 8 columns 1.12; 32 0.99; 64 0.974, with short columns gaining (64³ 0.87–0.96, 48³
  0.86–0.90, 120³ in place 0.77, 128 × 64 × 32 0.88–0.89, 64 × 64 × 512 0.91–0.94) and long
  ones losing (4 × 512 × 512 1.15–1.19, 512 × 64 × 64 1.07–1.10, 256³ 1.05–1.06). Hence arm W4
  (x86-64 with a private L2): blocks widen to 1 KiB of columns where both tiles fit
  `column_tile_bytes` (a quarter of L2); screened before any rule.
- EPYC, run 20261118-000233-49d2 (Slurm 512519; L and G at 8 lanes, the volumes; 8 rounds, all
  overloaded, band ±6.8 %): G ÷ L 0.904 (128 × 64 × 32 0.71–0.76, 256 × 256 × 16 0.70–0.78,
  512 × 64 × 64 0.77–0.78, 4 × 512 × 512 0.78–0.84, 128³ 0.83–0.88, 64³ 0.86; 97³ in place
  1.17, 64 × 64 × 512 1.09–1.10, past or near the band).

- EPYC W4 screens, runs 20261118-010214-dc58 (volumes; Slurm 512556; band ±2.3 %) and
  20261118-010214-5323 (the 132 planes and volumes; Slurm 512558; band ±2.0 %), against L at one
  thread and at 8 lanes: W4 ÷ L 0.932 over the 25 volumes it changes (64³ 0.88–0.89, 48³
  0.88–0.89, 512 × 64 × 64 0.88–0.93, 64 × 64 × 512 0.89–0.94), none slower; 0.959 over the
  13 planes it changes; at 8 lanes 1.001 (64³ 1.07–1.08, past the band; the team program's
  blocks are the same).

### G on the Mac at 6 lanes (judging)

Runs 20261118-010152-2709 (volumes; 8 rounds, all loaded, band ±12.0 %), 20261118-010632-1cd0
(the 45 c2c planes and volumes of the standard catalogue; band ±9.1 %) and
20261118-011212-0662 (its 87 real planes and volumes; band ±14.6 %), L 446be06 against G
f6f9b4d, both at 6 lanes. Over the cases whose single transform reaches the Mac's split
threshold (8192 points; the smaller ones run inline or whole, where G changes nothing):
volumes 0.859 (81³ 0.70–0.75, 120³ 0.72–0.75, 64 × 64 × 512 and 512 × 64 × 64 out of place
0.72, 100³ in place 0.75), c2c planes 0.937, real planes 0.828; worst c2r 96² out of place
1.12, 256² in place 1.055, inside their bands. Rule met: **G adopted on the Mac** (the Mac
was loaded by other work throughout; the bands are wide).

### L and G on the EPYC and the Xeon (judging), the volumes

Arms L1 (446be06) and A1 (`main` a0dffe1) at one thread, L8, A8 and G8 (f6f9b4d, L with G) at
8 lanes, and the competitors serial and at 8 threads, 8 rounds each.

- EPYC, run 20261118-005711-d018 (Slurm 512538; all rounds on a node with other work, band
  ±2.7 % from the one-thread A/A control). L changes 16³ and 16 × 256 × 256 (by describe): L ÷
  `main` 0.755 over them (16³ 0.66–0.67, 16 × 256 × 256 0.85–0.87). Unchanged programs
  measured up to 1.049 at one thread (100³ backward) and 1.125 at 8 lanes (48³ in place;
  programs identical by describe, so this is the 8-lane noise of the node, which the one-thread
  band understates). G ÷ L at 8 lanes over the 40 volumes that split: 0.877 (256 × 256 × 16
  0.65–0.75, 512 × 64 × 64 0.71, 128 × 64 × 32 0.72, 4 × 512 × 512 0.74, 16 × 256 × 256 0.78);
  worst 1.021 (17 × 15 × 37 in place), inside the band. FFTN ÷ FFTW's better plan: one thread
  0.786 (`main`, 16³ 1.12–1.14) → 0.764 (L, none over 1.05); 8 lanes 0.857 with 10 over 1.05
  (`main`) → 0.755 with one (G: 120³ out of place 1.17).
- Xeon, run 20261118-005716-711d (Slurm 16439; 5 of 8 rounds loaded, band ±1.9 %). L ÷ `main`
  0.966 over the four cases it changes (16³ 0.94–0.95, 16 × 256 × 256 0.97–1.00): rule (1)
  fails, **L rejected on `avx512`** (gated off there, commit after this run). G ÷ L at 8 lanes
  over the 40 that split 0.910 (32 × 64 × 128 0.76–0.78, 81³ 0.79–0.82, 16 × 256 × 256 0.85);
  worst 0.986 (32³ in place), 16³ (inline) 1.019. FFTN ÷ oneMKL: one thread 1.06 (29 of 42
  over 1.05: 64 × 64 × 512 in place 1.52, 128³ in place 1.48, 120³ 1.34, 512 × 64 × 64 1.25–
  1.33, 16³ 1.29–1.31); 8 lanes ÷ oneMKL's better plan 1.15 (`main`) → 1.055 (G), 29 over
  1.05 (30 × 40 × 50 1.53–1.77, 48³ 1.37–1.45, 32³ 1.41–1.44, 81³ 1.32–1.33).

### L and G on the EPYC and the Xeon (judging), the standard catalogue's planes and volumes

- EPYC, run 20261118-012443-6c99 (Slurm 512539; all rounds loaded, band ±2.6 %), the 132 c2c,
  r2c and c2r planes and volumes. L changes c2c 16³ alone here: 0.657. Unchanged programs up
  to 1.091 at one thread (c2r 3 × 5 × 4) and 1.135 at 8 lanes (c2r 512² in place). G ÷ L at 8
  lanes over the 83 that split 0.882 (real 1000²–1024² 0.45–0.54, c2r 1000² in place 0.50);
  past the limit: r2c 256² out of place 1.142, c2c 2048 × 36 1.05–1.08, r2c 128² × 16 1.052,
  r2c 100² in place 1.044: re-run with the unchanged cases past it (Slurm 512564). FFTN ÷ FFTW's
  better plan at 8 lanes: 0.980 (`main`, 47 over 1.05) → 0.911 (G, 36 over: tiny real batches
  5 × 7 and 8 × 6 at 1.5–3.6, real 2048² 1.40–1.71, c2c 2048 × 36 1.35–1.36).

- Xeon, run 20261118-013706-3586 (Slurm 16441; no round loaded, band ±4.6 %). L changes c2c
  16³ alone here (0.964; gated off `avx512` anyway). G ÷ L at 8 lanes over the 83 that split
  0.913 (real 128³ 0.54–0.61, real 100³ 0.63–0.73, real 1000² in place 0.76); worst c2c 96² in
  place 1.035, inside the band; unchanged programs at 8 lanes measured up to 1.23 (c2r 100² in
  place), the 8-lane noise again. Rule met: **G adopted on the Xeon.** FFTN at 8 lanes ÷
  oneMKL's better plan: 1.35 (`main`) → 1.30 (G), 99 of 132 over 1.05; the real planes are the
  gap (c2r 32³ 3.9, c2r 128² 3.8, r2c 96² 2.9–3.0, c2r 256² 3.2).
- **A pathology the runs show (not an arm of this experiment's rules):** single planes just
  above the split threshold run slower on a team than on one thread. With G at 8 lanes ÷ one
  thread, Xeon: c2c 96² 1.69–1.96, 100² 1.37–1.52, 128² 0.98–1.07; real 96² 2.3–2.6, 100²
  1.4–1.7, 128² 1.5–1.9, 32³ 1.06–1.15, 256² in place 1.00–1.06. EPYC: real 96² 1.18–1.34,
  100² 1.09–1.19 (c2c fine: 0.78–0.90). oneMKL's 8 threads take c2r 256² to 0.28 of its one.
  The split rule (T5's thresholds, 6144 points on the CCX, 8192 on the mesh) was fitted on
  rows; a plane's phases move it between cores' caches. Screens of higher thresholds for the
  planes and volumes: EPYC Slurm 512568, Xeon Slurm 16460.

### S (judging on the Mac; screens on the EPYC)

- EPYC split-threshold screen, run 20261118-021016-b769 (Slurm 512568; G at 8 lanes with
  `SPLIT_MIN_POINTS` 16384, 32768, 65536; band ±2.6 %): unsplit, c2c 96²–100² take 1.29–1.41 of
  the split time but real 96²–100² 0.77–0.93; unsplit 128² loses for every kind (1.18–2.63).
  So on the EPYC the real planes alone are split too early, by about their real/complex ratio:
  hence S1.
- EPYC S1 screen, run 20261118-023720-caa7 (Slurm 512574; band ±2.6 %): S ÷ G 0.887 over the
  eight real 96² and 100² planes it changes (0.78–0.99), 0.993 over the 87 real planes and
  volumes.
- Mac, run 20261118-023738-b4bf (G and S at 6 lanes, G serial; 6 of 8 rounds loaded, band
  ±14.7 %): S changes the real 96² and 100² planes (split at 6 lanes, unsplit with S1), and
  they take 1.01–1.49 of G's time unsplit (geometric mean about 1.29); on the M5 Max, whose
  cluster shares L2, splitting them pays. **S rejected on the Mac**: S1 applies only where each
  core has its own L2 (S2 never applies there: no mesh).

### W4 on the EPYC (judging)

- Run 20261118-032100-228a (Slurm 512561; L1 446be06 and W41 e9160a0 at one thread, both at 8
  lanes; all rounds loaded, band ±4.7 %), the volumes: W4 ÷ L 0.926 at one thread over the 25
  volumes it changes (64 × 64 × 512 0.87–0.88, 48³ 0.87–0.89, 64³ backward 0.87, 512 × 64 × 64
  out of place 0.88), worst 0.994; none of the 42 slower than 1.013. At 8 lanes 1.002, worst
  30 × 40 × 50 1.093 and 81³ 1.08; the EPYC's team programs are the same with and without W4
  (B1 clamps their blocks to 8 columns after it), so these are the 8-lane noise.

- Run 20261118-033804-a8a8 (Slurm 512562; band ±1.6 %), the 132 planes and volumes: W4 ÷ L
  0.947 at one thread over the 13 it changes (64³ 0.89, 48³ 0.89, 64² × 16 0.92, 64² 0.93),
  worst 36 × 2048 out of place 1.017; with the volumes, 0.933 over the 38 changed cases.
  Unchanged programs up to 1.026 (512² in place), just past the 2 % limit on a loaded node:
  re-run (Slurm 512620). At 8 lanes 1.000 (team programs unchanged on the EPYC).

## Verdict

Pending.

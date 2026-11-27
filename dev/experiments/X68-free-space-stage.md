# X68: the free-space stage as three objects against today's route

- **Status:** done 2026-11-27 on the Xeon 6240 and the EPYC 7763; the Mac leg and arm W are daytime
  follow-ups. The decision rules were drafted with 5.3's design (2026-11-26), before anything was
  built, and amended before the first judging run (below).
- **Owner:** X68 agent of the lead's night session, 2026-11-27 (task 5.3h)
- **Phase / task:** 5.3h, on 5.3a–5.3f as built

## Question

Two questions, each with a decision:

1. Does the client's free-space stage (DESIGN 12.5) meet step 5.3's acceptance in time and in
   DRAM traffic, at 384² at full width, against today's route? The stage is built as three
   objects: inverse planes with a prologue callback, spectral pairs on the ω pencils, and
   forward planes with an epilogue callback.
2. Does grouping the axes by (x, ω) slabs save enough over the three objects to justify building
   a multi-axis pipeline?

## Hypothesis

- **Today's route** makes 9 sweeps of the volume, about 18 V of traffic (12.5).
- **The three objects** should make 3 sweeps where a block of adjacent ω pencils over a whole
  plane fits a cache level, and 5 where it does not. 384² on x86 is a case of 5: a block of 4
  pencils is 9.4 MB.
  - At full width the stage is bandwidth-bound, so time should follow traffic: O/T about
    0.55–0.70 at 384².
  - At one thread it is less bound: O/T about 0.75–0.90.
- **The slab grouping** (G) makes 3 sweeps wherever a slab stays in a member's cache.
  - A slab at 384² with 750 stored frequencies is 384 × 750 complex doubles, 4.6 MB, beyond
    every machine's private L2. So G should gain little over O there: G/O about 0.9–1.0.
  - G should gain more at the smaller sizes, where a slab fits a member's share of the L2 or L3.
- **The propagator.** As a precomputed factor (O_f), it adds a read of an array as large as the
  field to the first sweep. So O_f/O should be above 1 at full width.

## Change under test

There is no library change. X68 measures the stage as built (5.3a–5.3f), composed in fbench as a
stage case (5.3h), against the same stage written as today's route. The arms, in one fbench
session per machine:

- **T:** today's route, made of plain FFTN plans and the application's separate loops, in
  today's order. The case's note counts its sweeps. The fbench case writes it from the stage's
  description before any run.
- **O:** the three objects (12.5):
  1. the inverse (y, x) plan, in place, ω a stride-1 batch, with a prologue callback computing
     the propagator per element;
  2. a `<C, T, C>` spectral pair, inverse first, on the ω pencils, with the band as runs on the
     half spectrum and a middle callback for the nonlinear term (`group` for two components in
     the coupled cases);
  3. the forward (y, x) plan, in place, with an epilogue callback for the finish.
- **O_f:** O with the propagator as a complex factor array (the vocabulary) instead of the
  prologue callback.
- **G:** the slab grouping, emulated with the same objects:
  1. the inverse y plan with the prologue callback;
  2. for each y, run serially by the member that owns that y (`transforms_of` over y, inside
     one parallel region): the inverse x plan on the slab, the ω pair, and the forward x plan;
  3. the forward y plan with the epilogue callback.

  It is what a multi-axis pipeline would do, less its per-call overhead (12.5).
- **W** (informative): today's route with FFTW 3.3.10 plans at MEASURE (NEON build on the Mac),
  and the same loops.

The callbacks and loops do the same arithmetic in every arm:

- the propagator: exp(i kz(kx, ky, ω) Δz) from coordinates;
- the nonlinear term: cubic in the real time field;
- the finish: a pointwise linear update with the stage's normalisation.

## Method

- **Cases:** a catalogue `x68` (`bench/catalogues/x68.toml`), written before any run:
  - 128² × 864, 192² × 384, 256² × 864 and 384² × 750: n_y = n_x, then the stored frequencies.
    The catalogue fixes the time line's length nt and the band's position from the client
    catalogue;
  - double precision;
  - one component, and two coupled components (`group`) at 192² and 384².
- **Machines:**
  - the Mac M5 Max: on AC, one fbench at a time, `pmset -g batt` before and after each run,
    under the maintainer's current guidance on Mac timing;
  - the Xeon 6240 (`--serial`, its load recorded);
  - the EPYC 7763 (DMOG).

  GCC 16 on Linux and Clang on macOS. Thread counts:
  - one thread;
  - the client's widths;
  - the machine's full width, in a whole-socket-sized but shared allocation, never `--exclusive`
    (the maintainer's standing instruction: no exclusive nodes; add rounds instead), with the
    node's load recorded in every round.
- **DRAM traffic** per stage, around each arm's executions:
  - on the Xeon, the uncore memory controllers' CAS counts (`perf stat` with the IMC uncore
    events, or VTune);
  - on the EPYC, perf's data-fabric DRAM events where the cluster allows them;
  - on the Mac, a bandwidth model: the sweeps of each arm's code path times V, checked against
    the time at the bandwidth a copy loop measures at that width.

  Traffic is reported in sweeps: bytes ÷ V, where V is the stored field's size.
- **Metric:** fbench's median paired ratios O/T, O_f/O, G/O and O/W per case, with 95 %
  intervals and an A/A control for the band, and the sweeps per arm.
- **Decision rules,** written before running:
  1. **Acceptance (step 5.3).** At 384² × 750 at full width, on each machine:
     - O/T is at most 0.80;
     - on the Xeon and the EPYC, O's measured traffic is at most 0.70 of T's.

     At one thread, O/T at 384² is at most 0.95.
  2. **No loss.** No case, at any thread count, has O/T above 1 beyond the band. A case that
     loses is fixed within 5.3h, or recorded as a lead with its cause, with the maintainer's
     agreement.
  3. **Traffic check.** If O's traffic at 384² exceeds 6 sweeps (12.5's 5 plus one), the excess
     is diagnosed and its cause recorded before rule 4 is judged. For example, a callback might
     run as a phase of its own, or the planes might take more sweeps than 12.5 counts.
  4. **Multi-axis pipeline:**
     - **Proposed** to the maintainer as a new step with drafted rules (not built by X68) if both
       hold: G/O is at most 0.85 beyond the band at 384² × 750 at full width on at least two of
       the three machines, and G/O is at most 1 + band at every size and width.
     - **Not built, documented** if G/O lies between 0.85 and 1 − band at some client size and
       width: `docs/fusion.md` (5.5) documents G's composition (owner-computes over slabs) as an
       option.
     - **Not built** otherwise, and closed. Saving fork/joins alone is not a reason to build it
       (12.5).
  5. **Propagator** (informative). O_f/O per case goes into `docs/fusion.md`'s advice beside
     X66's rule 5.
  6. **FFTW** (informative). Any case with O/W above 1.05 × (1 + band) is recorded as a lead for
     the performance phase.

  Near misses are re-run (AGENTS.md section 6).

### Amendments before the first judging run (2026-11-27)

Committed before any judging run, each with its reason. The drafted rules above stand except
where an item below makes them precise or changes them.

1. **The time lines and sizes come from the client catalogue,** as the Method asks, which the
   drafted sizes did not all follow:
   - 384² × 750 and 192² × 384 are envelopes: the client's rows there are c2c of 750 and 384
     points, every bin stored (n_t = the stored count). Their pencils are therefore `<C, C, C>`
     spectral pairs, inverse first, in place, with no runs; the "band as runs on the half
     spectrum" does not arise there.
   - The client's 128² case (its Cartesian filament) stores 433 bins of a real time line of 1728
     points (planes of 128² × 433, rows of r2c 1728); the drafted "864" was the length of its
     coarse rows. The case is 128² × 433, n_t = 1728, its pencils `<C, T, C>` with the band as
     runs. The catalogue gives no band position: it is bins [1, 434), clear of bin 0 and of the
     Nyquist bin, whose imaginary parts a c2r transform ignores (so that the stage is still the
     identity in exact arithmetic).
   - 256² has no case in the client catalogue. It keeps its place between 192² and 384² with the
     128² case's time line (256² × 433, n_t = 1728).
   - Two coupled components (`group`) at 192² and 384², as drafted: the field's components lie
     one after another, (c, y, x, ω); the coupled term is v_c (a + b Σ |v_k|²).
2. **The real lines' pencils run out of place** in O and Of: stored runs are refused on an
   in-place pipeline (5.2d). The pair writes a second field-sized array, and the forward planes
   read it and write the field (out of place, with the epilogue). Each is still one sweep (read
   V, write V); the footprint is 2 V, as T's (its scratch volume). In G each member's pencils
   write a slab buffer of its own, which its forward x plan reads back into the slab.
3. **T as built** (bench/backends/fftn_stage.cpp, written from the client's route before any
   run): the stage input crosses the field (y, x, ω) into a scratch volume in plane-major order
   (ω, y, x) with the propagator; the inverse planes run in place on it; then, a node chunk at a
   time, the pack (a transposing gather into spectrum rows, the gaps zeroed), the rows to time
   (c2c, or c2r, out of place), the nonlinear term, the rows back (c2c or r2c, out of place) and
   the crop (a transposing scatter back into the scratch); the forward planes; the finish,
   crossing back into the field. The node chunks are the client catalogue's (4469 pixels at
   384², 7373 at 192², 1261 for the 1728-point line, so also at 256²), with the tails' own plans
   as the client has. The crossings are tiled (8 pixels by 64 bins); every loop is shared over
   the executor's lanes, every plan runs on the executor. That is 9 sweeps, the planes counted
   once each.
4. **W is not built tonight.** FFTW's arm needs the whole route written again against FFTW's API
   and threads; it is informative (rule 6), and the Mac leg, where the NEON FFTW matters most,
   is a daytime follow-up anyway. Rule 6 is not judged; W is left as a follow-up.
5. **Arm C, informative:** one in-place sweep of the field (each value times one, shared over the
   lanes). Its time is a sweep's at the bandwidth that width reaches: `bench/x68/model.py` gives
   every arm's time in sweeps against it (the Mac's bandwidth model, and a check everywhere), and
   its counted traffic calibrates the counters (it should read about one sweep).
6. **Traffic as it can be counted on these machines.** Neither shared machine lets a user read
   the memory controllers: the Xeon runs with `perf_event_paranoid` 1 (no CPU-wide events, so no
   uncore IMC counters through perf) and VTune's sampling driver is restricted to a group we are
   not in; DMOG's nodes run with paranoid 2 and have no perf. Changing either is an
   administrator's decision. So:
   - **Xeon:** `perf stat` around the harness process (all its threads), enabled only over one
     counted execute of the stage (the harness's new `--perf-ctl`), counting the core's offcore
     responses that miss the L3: all data reads (demand and prefetch) and all RFOs. Each is a
     64-byte line brought from memory.
   - **EPYC:** the harness's own user-mode counters (`--pmu`) on its thread, so at one thread
     only: data cache fills from local or remote memory (PMCx044, umask 0x48) plus L2 prefetches
     that miss the L3 (PMCx072, umask 0xff).
   - Neither counts writes. Every sweep of these routes writes what it reads, so traffic is
     reported and judged in **read sweeps**: lines read (or read for ownership) × 64 / V. Rule
     1's "measured traffic at most 0.70 of T's" is the ratio of read sweeps; rule 3's "6 sweeps"
     is 6 read sweeps. On the EPYC, rule 1's traffic leg is judged at one thread (full width
     cannot be counted there).
7. **Threads and placement.** One thread; 8 threads, the client's width as X66 took it; and full
   width, one socket: 18 cores on the Xeon (its Slurm binds no CPUs, X59, so `bench/x68/x68.py`
   confines each run with `taskset` to the socket with the fewest busy cores, idlest cores first,
   and binds memory to that socket's node), 64 cores on the EPYC (a shared 64-core allocation,
   block distribution, memory interleaved over its NUMA nodes). Never `--exclusive`; the load is
   recorded with every round and every traffic count. OpenMP's executor (`fbench-fftn_omp`, the
   client's runtime), `OMP_PLACES=cores`, `OMP_PROC_BIND=close`.
8. **Toolchains.** GCC 16 on the EPYC; GCC 14.4 on the Xeon, the newest its fbench profile has
   (as X65 and X66 ran there). Both are the release toolchain family.
9. **Sessions.** Per machine and thread count, one fbench run over catalogue `x68`, arms T
   (baseline), O, Of, G and C, its A/A control T~, three batches per process (a stage takes up to
   seconds), median statistic, 6 rounds (more where the budget allows).
10. **Thresholds made precise** (as X66 read its rules):
    - a ratio "at most x" holds when its median paired ratio (the point) is at most x;
    - "above 1 beyond the band" (rule 2): the whole 95 % interval above 1 + band;
    - rule 4's "G/O at most 0.85 beyond the band": the whole interval at most 0.85; "at most
      1 + band": the point; "between 0.85 and 1 − band": the point;
    - the band is the run's A/A control (T against its copy); ratios between other arms (O/T,
      Of/O, G/O) come from `bench/x66/score.py`, which pairs any two arms round by round.
11. **Rule 4's machines.** Tonight measures two of the three. If G/O passes rule 4's first leg on
    both, the multi-axis pipeline is proposed whatever the Mac shows; if on neither, it is not
    proposed whatever the Mac shows; if on one, the verdict on that leg waits for the Mac's
    daytime run.

## Results

Every arm passed fbench's identity check on every case, and `STAGE_SELFCHECK` (each arm against
T with work that is not the identity: a cubic term, a propagator that depends on the coordinates,
a window that depends on the bin) agreed to 1e-12 on the test cases (bench/CMakeLists.txt,
`bench.fftn_stage_*`). Recorded times are as fbench stored them; nothing was edited.

**Sessions** (commit 71711f20, `bench/x68/x68.py`, 6 rounds of 3 batches each, median statistic,
arms T, O, Of, G, C and the control T~):

| Machine | Threads | Run | Band | Load |
|---|---|---|---|---|
| EPYC 7763 (DMOG node17, GCC 16.2, `avx2`; a shared 64-core allocation beside another user's 32-core job) | 1 | 20261127-044232-ce8d | ±2.84 % | 33–34 |
| | 8 | 20261127-055642-b144 | ±1.00 % | 34–40 |
| | 64 (one socket, memory interleaved) | 20261127-061851-2eb8 | ±3.36 % | 40–65 |
| Xeon 6240 (GCC 14.4, `avx512`; Slurm job 17547 in the `fbench-serial` chain, `-c 36`, never exclusive; taskset on the idler socket, memory bound to its node) | 1 | 20261127-044138-3acb | ±1.27 % | 2–8 |
| | 8 | 20261127-062235-ce8a | ±1.16 % | 5–65 (2 rounds loaded) |
| | 18 (one socket) | 20261127-065922-020f | ±1.18 % | 6–11 |

**Ratios** (`bench/x66/score.py`; median paired ratio and 95 % interval; the 384² × 750 case,
then the geometric mean over the six cases):

| Machine, threads | O/T at 384² | O/T mean | Of/O mean | G/O at 384² | G/O mean | G/T at 384² |
|---|---|---|---|---|---|---|
| EPYC, 1 | 1.329 [1.324, 1.334] | 1.388 | 0.757 | 0.959 [0.954, 0.962] | 0.931 | 1.273 |
| EPYC, 8 | 1.200 [1.195, 1.207] | 1.194 | 1.036 | 0.660 [0.656, 0.663] | 0.694 | 0.792 |
| EPYC, 64 | 2.587 [2.575, 2.671] | 1.518 | 1.425 | 0.305 [0.293, 0.313] | 0.441 | 0.804 |
| Xeon, 1 | 1.169 [1.145, 1.188] | 1.615 | 0.696 | 0.795 [0.782, 0.815] | 0.637 | 0.934 |
| Xeon, 8 | 2.078 [1.996, 2.112] | 1.909 | 0.829 | 0.532 [0.513, 0.564] | 0.549 | 1.101 |
| Xeon, 18 | 1.885 [1.741, 1.938] | 1.754 | 0.880 | 0.519 [0.505, 0.534] | 0.546 | 0.977 |

O is slower than T beyond the band on 33 of the 36 case-and-width pairs. The three others are the
EPYC's 128² × 433 at one thread (1.049, inside its ±2.84 % band), its 128² × 433 at 8 threads (0.793, faster) and its 192² × 384 with two components at 64
(1.013, inside the band). The worst are the Xeon's 192² cases at one thread (2.47, 2.48) and the
EPYC's 384² at 64 threads (2.59). G is faster than O on every case and width (0.30–0.96). Every
case's per-case ratios are in the runs.

**Time in sweeps** (`bench/x68/model.py`: each arm's time over arm C's, one in-place sweep of the
field at that width). At one thread every route is bound by its arithmetic, not by memory: T takes
53–74 sweeps of time, O 68–135. At full width T comes close to its design: 9.5–12 sweeps of time
at 384² on the EPYC (64 threads) and 19–20 on the Xeon (18), while O takes 16–40. A sweep of the
384² field took 24 ms on the EPYC at 64 threads (147 GB/s with memory interleaved) and 35 ms on
the Xeon at 18 (101 GB/s).

**DRAM traffic** (read sweeps: lines read or read for ownership × 64 / V, one counted execute
after a warm one, one thread unless stated):

| Case | EPYC T | O | Of | G | C | Xeon T | O | Of | G | C |
|---|---|---|---|---|---|---|---|---|---|---|
| 384x384x750-b1 | 17.52 | 4.46 | 8.57 | 3.64 | 1.81 | 16.01 | 7.81 | 9.76 | 4.29 | 1.00 |
| 384x384x750-b2 | 18.08 | 4.54 | 8.52 | 3.83 | 1.77 | 16.42 | 8.54 | 10.73 | 4.75 | 1.00 |
| 256x256x433-b1 | 12.87 | 6.23 | 8.25 | 3.12 | 1.80 | 18.81 | 11.56 | 13.79 | 3.92 | 1.00 |
| 192x192x384-b1 | 17.61 | 4.72 | 8.92 | 3.24 | 1.82 | 15.23 | 16.49 | 23.02 | 3.31 | 1.00 |
| 192x192x384-b2 | 18.07 | 4.95 | 9.24 | 3.33 | 1.82 | 15.34 | 15.91 | 24.00 | 3.57 | 1.00 |
| 128x128x433-b1 | 15.15 | 5.79 | 7.68 | 3.14 | 1.82 | 18.57 | 6.57 | 7.77 | 3.18 | 1.00 |

| Case | Xeon, 18 threads: T | O | Of | G | C |
|---|---|---|---|---|---|
| 384x384x750-b1 | 16.06 | 17.59 | 25.55 | 5.04 | 1.00 |
| 384x384x750-b2 | 16.82 | 18.16 | 25.73 | 5.11 | 1.00 |
| 256x256x433-b1 | 12.50 | 19.46 | 27.28 | 6.02 | 1.00 |
| 192x192x384-b1 | 14.29 | 15.67 | 23.94 | 3.27 | 1.00 |
| 192x192x384-b2 | 14.95 | 16.90 | 24.89 | 3.92 | 1.00 |
| 128x128x433-b1 | 10.61 | 8.56 | 12.39 | 3.97 | 1.00 |

- The EPYC's counters (PMCx044 fills from memory plus PMCx072 L2 prefetches that miss the L3)
  count one in-place sweep (arm C) as 1.77–1.82 sweeps: they count some lines twice, most likely
  an L2 prefetch and the demand fill it serves. Divided by C's count, the EPYC reads T 7.1–10.2
  sweeps, O 2.5–3.5, Of 4.2–5.1 and G 1.7–2.2.
- The Xeon's counters (perf stat over the process's threads, offcore responses missing the L3:
  all data reads and all RFOs; campaign record 20261127-044134) count arm C as 1.00 sweep, so its
  figures are read as they stand. At one thread O read 0.49–0.53 of T's lines at 384² and 0.35 at
  128², but slightly more than T at 192² (16.5 against 15.2); at 18 threads O read more than T on
  five of six cases (384²: 17.6 against 16.1), while T's traffic hardly moved with the width (16.0
  and 16.1 at 384²) and G's stayed at 3.3–6.0. Of read 2–8 sweeps more than O: its factor is read
  in a pass of its own.

**Where O's time goes** (`STAGE_PROFILE=1`, a diagnosis made after the judging runs had started,
on commit ccf0fe84, which adds only the profile's stamps to the judged code: DMOG, one thread and eight, each step's mean over three runs,
campaign record 20261127-051647). At 384² × 750 on the EPYC, one thread:

| Step | T | O | Of | G |
|---|---|---|---|---|
| propagator and inverse planes (T: the input crossing, then the planes) | 2957 + 699 | 5117 | 3176 | 4382 (inverse y) |
| the time line (T: pack, rows, response, rows, crop) | 2029 | 670 (the pencils' pair) | 648 | 1213 (per slab: x, pair, x) |
| forward planes and finish (T: the planes, then the crossing) | 704 + 481 | 3540 | 3562 | 2876 (forward y) |

and at eight threads the forward planes and finish took 806 ms in O against T's 161 + 181. The
pencils do what 12.5 promised: the spectral pair on the ω pencils took 0.33 of T's rows route at
one thread and 0.17 at eight (124 ms against 716). The planes do not: FFTN's (y, x) planes over
the field's own layout (y and x at strides of n_x n_ω and n_ω elements, ω a stride-1 batch, run as
lanes over blocks of 16 adjacent transforms) took 4–5 times the time of the same planes stored
tight, plus the transposes T pays to make them tight. The Xeon's 192² cases (strides of 6144
bytes and 1.18 MB, multiples of 2 KiB) lost most. The propagator as a callback on those planes
cost about 2 s more than the factor array at one thread (O's inverse planes 5117 ms against Of's
3176), so per element it cost about the same as in T's crossing; Of's factor, read as a pass of
its own over the planes' input, cost more than the callback from eight threads up on the EPYC
(Of/O 1.04 at 8, 1.43 at 64).

## Verdict

Judged on the EPYC 7763 (AMD) and the Xeon 6240 (Intel); the Mac leg is a daytime follow-up
(the maintainer's guidance on Mac timing at night).

- **Rule 1, 5.3's acceptance: not met.** At 384² × 750 at full width O took 2.59 of T's time on the
  EPYC and 1.89 on the Xeon (the bar is 0.80); at one thread 1.33 and 1.17 (bar 0.95). The traffic
  leg passes on the EPYC (0.25 of T's lines at one thread, where its counters can count) and fails on
  the Xeon at full width (1.09: 17.6 read sweeps against 16.1; 0.49 at one thread). The stage as
  three objects moves a quarter to a half of today's DRAM lines on one thread and still takes
  longer, because its planes are slow, and at a full Xeon socket the same planes read more lines
  than today's whole route (below).
- **Rule 2, no loss: failed,** on 33 of 36 case-and-width pairs. The cause is one defect, diagnosed
  above: FFTN's planes on the free-space layout (ω the stride-1 batch) run 4–5 times the time of
  the same planes stored tight plus the transposes; the pencils' pair wins (0.17–0.33 of T's rows
  route). Fixing it is a planner and strategy change with an experiment of its own, beyond 5.3h's
  reach, so it is recorded as **lead 1** below; leaving it as a lead needs the maintainer's
  agreement. It extends X66's lead 3 (the same layout's callback blocks).
- **Rule 3, traffic check: triggered on the Xeon, diagnosed.** On the EPYC O read 4.5 sweeps at
  384² (2.5 divided by C's count), within 6. On the Xeon it read 7.8 at one thread and 17.6 at 18
  threads, above 6. The excess is the planes over the free-space layout: of O's three objects only
  they change their access with the width (the pencils are whole lines in the workspace at any
  width), T's and G's traffic barely moves with the width (T 16.0 to 16.1, G 4.3 to 5.0), the
  profile puts O's lost time in the same planes, and at 192² (2 KiB-multiple strides) O re-reads
  even at one thread (16.5 against T's 15.2). Lines of 16 adjacent transforms (256 bytes) at those
  strides are evicted before the next pass over them comes back: lead 1. Rule 4 is judged with
  that cause recorded.
- **Rule 4, the multi-axis pipeline: proposed by the rule's letter, with a caveat for the
  maintainer.** G/O at 384² × 750 at full width: 0.305 [0.293, 0.313] on the EPYC and 0.519
  [0.505, 0.534] on the Xeon, both wholly below 0.85; G/O at every size and width at most 0.961
  (EPYC, one thread), below 1 + band. Both machines pass the first leg, so by amendment 11 the
  Mac cannot change it: **a multi-axis pipeline is proposed to the maintainer as a new step, not
  built.** The caveat: G beats O mostly because G's y and x plans avoid the slow two-axis planes
  of lead 1, not because the slab grouping saves traffic of its own: G read 3.6 sweeps against
  O's 4.5 on the EPYC, and on the Xeon 4.3 against 7.8 at one thread and 5.0 against 17.6 at 18,
  where most of the difference is O's planes re-reading the field. Against today's route G took 0.80 of T's time at 384² at full width on the EPYC and 0.98
  on the Xeon, and 1.27 and 0.93 at one thread. The proposal below therefore asks for lead 1
  first and a re-run of G/O on the fixed planes before anything is built.
- **Rule 5, the propagator (informative):** Of/O 0.76 (EPYC) and 0.70 (Xeon) at one thread, where
  the cos and sin per element dominate; 1.04 and 0.83 at eight; 1.43 (EPYC, 64) and 0.88 (Xeon, 18)
  at full width, where the extra array read costs more than the arithmetic on the EPYC. As
  hypothesised, O_f/O rises above 1 at full width on a machine that is bound by bandwidth there.
  The advice in `docs/fusion.md`: precompute a fixed propagator when a stage runs on few threads;
  at full width on a bandwidth-bound machine computing it per element in the callback can win.
- **Rule 6, FFTW: not judged** (W not built, amendment 4).

**Leads for the maintainer:**

1. **Planes over a stride-1 batch at large strides** (the free-space layout; X66's lead 3). At 384²
   × 750 the field's (y, x) planes took 4–5 times the tight planes' time (EPYC, one thread: 3540
   ms against 704 + 481 with T's transposes; eight threads: 806 against 161 + 181), and the Xeon's
   192² × 384 (strides that are multiples of 2 KiB) 2.5 times T's whole stage at one thread. The
   lanes over blocks of 16 adjacent transforms read 256 bytes per line from lines 12 KB (x) and 4.6
   MB (y) apart: a cache line's worth per DRAM page and per TLB entry. Candidates for its own
   experiment: wider blocks of adjacent transforms (whole cache lines or pages of ω), the x axis
   slab by slab while a slab is in cache and the y axis on wide blocks (what G does by hand), or
   transposing internally as the old route does. The stage, the three objects and G are the
   benchmark for it (`STAGE`, catalogue `x68`).
2. **The multi-axis pipeline** (rule 4's proposal): a new step after lead 1, with drafted rules:
   G (or the pipeline) against O on the fixed planes at the four sizes, 1, 8 and full width, on the
   EPYC, the Xeon and the Mac; build it only if G/O is at most 0.85 at 384² × 750 at full width on
   two of the three machines and at most 1 + band everywhere, measured after lead 1 is adopted; and
   G/T at most 0.80 at 384² at full width, so that it beats today's route too.
3. **A factor on planes' input at full width.** Of's complex factor of the field's size, applied
   as a pass of its own over the planes' input, cost more than computing cos and sin per element
   at 64 threads on the EPYC (1.43). A factor of stride 1 along ω could be fused into the planes'
   first lanes pass, as a row's chain fuses one (12.1).

**Follow-up:** the Mac M5 Max leg (the same campaign, `--place none`, on AC with `pmset -g batt`
before and after, the model of `bench/x68/model.py` as its traffic estimate) and arm W, by day.
The lesson is in [LESSONS](../LESSONS.md) 4.31.

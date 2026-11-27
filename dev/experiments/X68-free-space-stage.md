# X68: the free-space stage as three objects against today's route

- **Status:** running. The decision rules were drafted with 5.3's design (2026-11-26), before
  anything was built, and amended before the first judging run (below).
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

Not run.

## Verdict

Not run.

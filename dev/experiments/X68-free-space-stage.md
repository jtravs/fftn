# X68: the free-space stage as three objects against today's route

- **Status:** planned. The decision rules were drafted with 5.3's design (2026-11-26), before
  anything was built. The agent who runs X68 claims it here and may adjust the rules only
  before the first judging run, stating each change and why.
- **Owner:** unclaimed (task 5.3h)
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
  - the machine's full width. On DMOG that needs a node's cores: the agent asks the maintainer
    for an exclusive node (AGENTS.md section 6 allows one for a threading experiment that needs
    every core). Without one, the client's widths stand in, and the file says so.
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

## Results

Not run.

## Verdict

Not run.

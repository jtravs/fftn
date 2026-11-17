<!-- SPDX-FileCopyrightText: 2026 John C. Travers and contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Performance lessons

What building FFTN (and its prototype before it) taught about making FFTs, and numerical
kernels in general, fast on current CPUs. Each lesson states the claim, the evidence (what
was measured, on which machine) and where the details are. They are written to be reused
in other projects, so they say what generalises and where it stops.

Machines named below: **Mac** = Apple M5 Max (NEON with FCMA); **EPYC** = AMD EPYC 7763
(Zen 3, AVX2); **Xeon** = Intel Xeon Gold 6240 (Cascade Lake, AVX-512). Ratios are time ÷
reference time (below 1 is faster) unless stated. "The prototype" is FFTN's predecessor,
whose measured lessons are summarised in [DESIGN.md](DESIGN.md) and the research notes it
cites.

**Keeping this file current.** When an experiment closes with a verdict that teaches
something beyond FFTN's own constants (a cost, a pitfall, a method), add or update its
lesson here with the numbers and a link to the experiment file (AGENTS.md section 6, step 7). A
lesson that a later measurement contradicts is corrected, not deleted, with both results.

---

## 1. Measuring

1.1 **On shared machines, add rounds instead of waiting for quiet.** Interleave arms in
balanced orders (each arm in each position and after each other arm equally often), run
each round as fresh processes, include an identical-arm control (A/A) in every run, and
judge a difference only against that run's A/A noise band. Verdicts held on machines with
load averages of 6–17: bands of ±1.0–2.7 % on the loaded Mac, ±1.0–4 % on the shared Xeon.
Waiting for a quiet or exclusive machine cost far more wall time than extra rounds.
[bench/README.md](../bench/README.md) ("Reading a verdict").

1.2 **One process timeout must not kill a run.** A one-hour process limit killed whole
baseline runs (FFTW PATIENT planning in every round) until timeouts lost only their own
cases. Store per-case results as they arrive.

1.3 **Fewer instructions is not faster.** X10's leg-offset tables cut instructions per
transform by 5–15 % on the Xeon (2825 → 2293 at 256 points) and changed time by about
nothing in geometric mean. The bottleneck was one execution port (port 5, shuffles: 73 %
busy against FFTW's 55 %). Profile ports or cycles, not only instruction counts.
[X10](experiments/X10-pass-loops.md).

1.4 **Check what the competitor actually runs.** FFTW's planner on the Xeon mixes widths
within one plan (AVX2 codelets for untwiddled passes, AVX-512 for twiddled ones). A first
diagnosis read another copy of the same symbol names and drew the wrong "same width"
conclusion. Look at the plan the reference chose, and at the code it executed, before
comparing. [X18](experiments/X18-x86-kernels.md).

1.5 **Baselines need the competitor at its best.** Homebrew's FFTW has no SIMD on arm64
(compare against a NEON build); pocketfft built without `-march` lost a factor it should
not have; MKL needs `MKL_ENABLE_INSTRUCTIONS` and `MKL_CBWR` unset. A handicapped
competitor makes every result look better than it is.

1.6 **Bake the build's identity at build time, and refuse identical arms.** fbench rejects
two arms meant to differ that report the same configuration; that caught two A/Bs that
would have compared a build against itself (the identity is captured at configure time,
so a reconfigure is needed after a commit).

1.7 **A per-length search needs its noise handled per stage.** Screening many candidates
with two rounds, then refining the best four with six, then confirming the winner against
the default with up to 30 kept the cost down; but a two-round screen can call ties that a
paired A/B later separates by 25–40 % (X2's lengths 490, 686, 882 on NEON). Confirm
adopted choices with a paired run. [X2](experiments/X2-radix-sets.md).

1.8 **Don't let your own builds become the load your timings tolerate.** On the shared
Xeon, X23's concurrent fbench jobs each built every arm in their own directory: two jobs
with the same compiler built the same five commits, GCC took about 13 minutes per arm on
8 cores, so a job spent its first hour compiling, and those compiles were most of the
"other processes" (about 34 cores) the other jobs' timing rounds had to tolerate (noise
band 15–24 % after 9 rounds, against 5 % for a job that started timing while it was
quieter). Build each arm once per machine into a shared cache, in a short, wide build job,
and time afterwards.

1.9 **Hardware counters need neither `perf` nor root, and the competitor's plan must be
pinned before counting it.** With `perf_event_paranoid` at 2 (the default on many
clusters) a process may count its own user-mode events through `perf_event_open`. About
100 lines in the benchmark harness (a group of raw events enabled around the execute
loop, read once, divided by the calls) gave cycles, instructions, FP-pipe uops, loads,
stores, L1 and L2 misses and TLB misses on a Zen 3 cluster that has no `perf`, so no cloud
machine was rented for it. Check that the group stayed scheduled (time running = time
enabled): the NMI watchdog holds one counter, so five events fit in a group on Zen 3.
FFTW's MEASURE plans changed from process to process on a loaded machine (60 points:
15 × 4, then 10 × 6; 806 against 1211 instructions per call), so counters of separate
processes describe different code unless the plan is shared through wisdom.
[X25](experiments/X25-x86-diagnosis.md).

1.10 **A measuring planner on a shared machine needs a budget for building, a fixed
reproduction path, and its own check against the benchmark.** FFTN's `planning::measure`,
first pass over 61 cases on three loaded machines:
- only 43–66 % of cases kept one program over 12 process rounds; the choices that changed
  were within noise of each other;
- building plans, not timing them, set the cost of large transforms (one 2²² plan took
  0.5–1.7 s to build); building all 16 candidates before timing any spent a 10 s budget and
  left the rules' misfiring choice in place;
- on the EPYC, choices that read 1.1–2.5 times faster inside planning were 0.90–0.94 of the
  rules' speed in the benchmark, whose buffers sit at random 64-byte offsets where planning's
  are page-aligned.

So: build and time each candidate in turn, keep the rules' choice unless clearly beaten
(1 %), record the winner as wisdom checked by a fingerprint of the program, and judge the
planner by the benchmark rather than by its own timings. Where measurement did find real
gains, they were stable across rounds: planes' radices (480 × 640 0.81–0.83 of the rules'
time on the Mac) and Granite Rapids' misfiring chain rules forced on a Xeon (10 of 11
recovered). [P1](experiments/P1-rules-vs-measure.md), [P2](experiments/P2-measure-budget.md).

1.11 **Time the call a user makes, not the one next to it.** fbench's FFTN backend called
`transform_count` and then `execute_core`, two calls into the library, where
`plan::execute` makes one (`execute_all`). The extra call was 4–20 % of a single row of 2–6
points: switching the backend took them to 0.80–0.82 of their time on the EPYC 7763,
0.86–0.91 on the Xeon 6240 and 0.96 on the M5 Max (2 points: 1.22 → 1.15 of FFTW on the
Mac, 0.86 → 0.69 on the EPYC). At a few nanoseconds per call the harness's own path is part
of the result; check it against the public entry whenever the entry changes. And keep
every arm on one base: merging `main` into the branch mid-experiment moved the same rows by
1.14–1.16 on the EPYC (the merged `main` alone did), which first read as a loss of the arm
being timed. [X37](experiments/X37-short-batches.md).

1.12 **Across cloud VMs, compare ratios within a run, and distrust single long rows even
then.** Two Granite Rapids VMs of one type (`c4-standard-16`), a day apart, ran both FFTN and
FFTW about 6 % faster on the second over 35 shared rows, so times across VMs mislead. The
ratio held over the set (FFTN ÷ FFTW 0.778 → 0.784), but single long rows did not: FFTW took
0.46 of its earlier time at 2²² (FFTN 0.66) and 0.80 at 65536 (FFTN 0.99), turning 0.76 →
1.10 and 1.24 → 1.53 with FFTN's plans unchanged; the competitor's MEASURE plans and the
VM's memory placement both move. Judge a row by a same-run A/B on one VM, and a VM's
standing by the geometric mean. [X45](experiments/X45-cloud-exit.md).

## 2. The instruction set and the machine

2.1 **The data layout that wins depends on the ISA.** Interleaved complex numbers suit
NEON with FCMA, which multiplies and rotates complex values in one instruction; FFTN beats
NEON FFTW on smooth rows (0.75–1.0 for 128–100000 points) in that form. On x86 every
multiplication by ±i in interleaved form is a shuffle; the prototype's main x86 path used
split re/im instead. Split form doubles register use (radix R needs 2R vectors).
[X18](experiments/X18-x86-kernels.md), [X10](experiments/X10-pass-loops.md).

2.1a **On x86, split re/im lane passes pay even with both conversions.** A chain whose
last two passes run on separate real and imaginary registers (deinterleave on the first
one's loads, interleave on the last one's stores, no shuffle in the arithmetic, twiddle
parts broadcast) beat the same passes on interleaved registers by 7–11 % at 1024–2048
points in double and 18–23 % in float on the Xeon (AVX-512 and AVX2), and by 7–15 % at 2048
on the EPYC (Zen 3; 1.3 % at 1024, and 2.5 % slower with a radix-12 pass); on the Mac (NEON
with FCMA) it was 10–25 % slower. On the Xeon's AVX-512 it cut port-5 uops by 12 % and
instructions by 17 % at 1024, yet port 5 stayed 58 % busy: the conversions and the first
pass's transposes remain. A radix-16 split codelet holds 34–40 live reals, so AVX2's 16
registers spill, and it still won there. A rule that the planner's two targets must round
alike constrains where a split run may start (the wider target's lane count on both), which
costs the narrower one some lengths. Judge such a change on the cases it touches: over a
tracking set where few cases had a run the geometric mean moved by 1 %.
[X1](experiments/X1-split-lowering.md).

2.2 **Count the shuffles your generator emits.** FFTN's generated radix-16 codelet
needed 19 distinct ±i swaps where FFTW's has 7 (radix 8: 6 against 3). A tie-break in the
lowering changed nothing (the stores' rotations decide every choice). What worked is
moving the rotations to the outputs: rebuild every node as R + i·I with real-coefficient
parts, so each output pair X_k, X_{n−k} costs one swap, ⌊(R − 1)/2⌋ per codelet (16: 7,
32: 15, 64: 31, FFTW's counts), with no extra operations (fewer on fused targets) and the
same unfused arithmetic. x86 0.976 of the old codelets, NEON 0.991; port-5 uops −4 to
−16 %. [X10](experiments/X10-pass-loops.md), [X19](experiments/X19-generator-rotations.md).

2.3 **Wider is not automatically faster.** On the Xeon (Cascade Lake) AVX-512 against AVX2
for the same plans: double 0.87 (batch 1) and 0.92 (batch 64), but float 1.11 and 1.13,
1.43 at small sizes, where eight complex floats per register leave many partially filled
vectors (tail tiles). The verdict is per machine generation (Cascade Lake lowers its clock
for 512-bit code) and per precision. [X16](experiments/X16-avx512-table.md).
Off Cascade Lake the double result holds and the float result depends on the core: over
c2c rows of 16–65536 points, AVX-512 ÷ AVX2 was 0.83–0.86 in double and 0.95–0.99 in float
on an Emerald Rapids Xeon (Google C4), and 0.87 in double but 1.03–1.07 in float on Zen 5
(C4D). Per length it swung from 0.52 (1024 points, float) to 3.1 (60 points, float, on both
machines): no machine-wide preference is right, so price both targets per plan.
[X21](experiments/X21-phase1-cloud.md).
On Cascade Lake, with split runs and refitted tables, the dividing line is the tail, not
the size: float rows whose length 8 divides took 0.78–0.85 of AVX2's time, the others 1.20–
1.23 (double: 0.71–0.78 against 1.00–1.05 where 4 does not divide n). A key for the
per-plan choice is C | n before any size threshold.
[X16](experiments/X16-avx512-table.md).
For the smallest transforms the frequency matters as well as the cycles. On Cascade Lake,
FFTN's 8-point transform in 512-bit code spent its cycles at AVX licence 1, while FFTW's
128-bit SSE2 codelet stayed at licence 0: 1.62 times the cycles became 1.84 times the
time. FFTW chooses 128-bit codelets for single rows of 8 and 16 points on both the Xeon
and Zen 3. [X25](experiments/X25-x86-diagnosis.md).
A transform that broadcasts one line into every lane gains nothing from width: FFTN's
leaves compiled at 128 bits with the same operations (bit-identical) ran at licence 0 and
took the Xeon's 8- and 16-point rows from 65 to 47 and 110 to 85 cycles per call, and
were also faster on Zen 3, where width costs no clock (44 → 35 cycles at 8 points, with
the lighter entry). [X28](experiments/X28-short-rows.md).

2.4 **Integer division is expensive enough to matter in bookkeeping.** A 64-bit `div` costs
40–90 cycles on Cascade Lake; four per chunk in the driver were about 37 % of a 64-point
transform there. Computing chunk boundaries without dividing in the common case made the
Xeon 15 % faster in geometric mean (64 points: 0.59 of the time) with identical bits. Look
for divisions and modulos in per-call and per-chunk paths, not only in kernels.
[X18](experiments/X18-x86-kernels.md).

2.5 **The compiler is rarely the explanation for the main loops, but often for tails and
spills.** FFTN built with Clang 23 and GCC 14/16 was within 1 % on x86 after the X18 fixes
(tracking rows, mostly powers of two) and within 7 % on NEON (two cases at 1.4 with GCC).
On the Xeon's 7-smooth rows of 64–1024 points, though, Clang took 0.90 of GCC's time in
double and 0.84 in float, and Intel's icpx 0.67 in float: element-by-element tail loops
(fewer copies left than lanes) became masked AVX-512 loads and stores with icpx, partly
with Clang, and scalar stores into a stack tile followed by a vector load (a
store-forwarding stall per leg) with GCC; in full 32-point codelets the stack references
were 139 (icpx), 169 (GCC) and 215 (Clang). Test the compiler early on lengths with tails
and big codelets too, and when one compiler wins, write what it found (masks, a smaller
schedule) into the code rather than depending on it.
[X18](experiments/X18-x86-kernels.md), [X22](experiments/X22-icpx.md).

2.6 **Compiler auto-vectorisation does not replace a hand vocabulary.** `omp simd` on the
scalar pass loops gave compiler-dependent results, and `-march=native` closed no per-pass
gap against a hand-written vocabulary; older ISA targets (SSE-only, AVX without AVX2) need
their own vocabularies. [X15](experiments/X15-compiler-vectorised-scalar.md).

2.7 **Planning tables measured on one machine travel badly.** FFTL.jl-style radix rules
were 1.25–1.44 times slower than the best sequence on the EPYC but only 1.07–1.12 on the
Mac; AVX2 tables fitted on the EPYC picked a sequence within 2 % of the best for only
48–65 % of lengths on the Mac, against 82–83 % for tables fitted there (3.6 % faster
overall). On the Xeon's AVX-512 the borrowed tables were already within 0.6–2.1 % of the
best. [X2](experiments/X2-radix-sets.md), [X16](experiments/X16-avx512-table.md).
They also go stale when the kernels change. After split re/im runs (X1) the same AVX-512
float table was 3.5 % from the best up to 1024 points and 8–10 % above, and single lengths
were off by 1.7–2.1 times (700 points: 4·7·5·5 against 14·10·5), because a run starts only
after a prefix the split block divides, so the best first radix moved to 8 or 16; AVX2 and
NEON tables, whose kernels changed less, stayed within 1–2 %. Re-run the search after every
kernel or lowering change, not only on a new machine.
[X2](experiments/X2-radix-sets.md) (refresh).

2.8 **A fitted cost table must not decide where it never measured.** Fitted to every
candidate up to 1024 points and to the cost estimate's best 20 above, the AVX-512 float
costs chose, for 4096–32768 points, five- and six-pass chains of radix 4 that no campaign had
timed (32768: 16·4·4·4·4·8, against the measured best 8·16·16·16); within the measured
candidates it reproduced 97 % of winners. Check whether the model's choice at each measured
length was itself timed, and pin the measured best where it was not; treat the
unmeasured lengths in between as the model's real test.
[X2](experiments/X2-radix-sets.md) (refresh).

2.9 **A ratio to a competitor is a property of the pair on that machine, not of your code.**
FFTN ÷ FFTW at ≥ 16384 points was 1.19 on the Mac but 0.75 on Google Axion (Neoverse V2),
same instruction set and same library: per point FFTN was 1.4 times slower on Axion than on
the Mac at 2²⁰, FFTW 2.6 times. A deficit seen on one machine per architecture can be the
competitor's strength there; check the absolute times on a second core before spending on
the gap, and judge per-core rules on several cores of one ISA (Zen 3 against Zen 5 and
Cascade Lake against Granite Rapids differed by up to 0.3–0.6 in single cases, not in which
cases lose). [X31](experiments/X31-cloud-1d.md).

2.10 **Without FMA, width still pays and split arithmetic stops paying.** Forced on a Zen 3
(EPYC 7763) and a Cascade Lake Xeon, a 256-bit target without FMA or AVX2 ran 0.68–0.73 of
the time of the same kernels on 128-bit SSE4.2 registers, and 0.90–0.94 of FFTW built with
SSE2 and AVX; the 128-bit target ran 0.96–0.98 of FFTW's SSE2 build. Split re/im passes,
7–23 % faster than interleaved ones with FMA (X1), gained only 0.5–2.4 % without it: the
FMA-shaped split text, de-fused, needs up to 17 % more multiplies (3 % over radices 2–16)
than the complex-register text, whose lowering never fuses a shared scale. Keep the
unfused order in the generator rather than de-fusing an FMA schedule, and re-measure layout
choices per ISA level, not per register width. Caveat: modern cores, not Sandy Bridge,
whose 256-bit loads take two cycles. [X14](experiments/X14-old-x86.md).

2.11 **A code-placement jump on Cascade Lake is a branch-alignment effect; pad the AVX-512
units.** Single c2c double rows of 160, 180 and 240 points on the Xeon 6240 went to 1.07–1.10
of their time (360 and 384: 1.04–1.06) between main 304a8c5 and 509d123 with the same radices,
the same kernels and the same execute path; bisecting the first-parent merges put the jump at
X40's merge (20261117-184544-8056, band ±1.0 %), whose diff touches no code those rows run
(X40 had seen c2c 128 do the same and X37 a jump at 2–8 points on the EPYC). Assembling the
library with GCC's `-Wa,-mbranches-within-32B-boundaries` (no branch crosses or ends on a
32-byte boundary, the Skylake-family JCC erratum's mitigation) took them back to 0.89–0.94 of
the unpadded time on the Xeon and also sped up 360 and 384; the flag on `driver.cpp` alone did
nothing, on the avx512 units alone it did all of it (20261117-193445-016d; the first runs
were noisy but paired). The same flag on every unit slowed 180 and 300 points by 5–11 % on the
EPYC (Zen 3, 20261117-190948-906d): code placement is luck there too, so only the units
that run on the affected cores take it. A sudden single-row jump with an unchanged plan is
this before it is a code change: bisect, then try the flag on the units that run.

## 3. Memory and caches

3.1 **Buffers a power of two apart conflict in cache; offset them.** Staggering the
grid's second scratch buffer by seven cache lines made 2²² points about 8 % faster on the
Mac; padding rows of the intermediate array did nothing on top of it. The prototype found
the same with streams N/R apart at 4 KiB multiples: stagger work buffers by an odd number
of cache lines, avoid radix ≥ 16 at large power-of-two strides.
[X4](experiments/X4-padding.md). **Revised by X26b:** with more rounds the padding did pay on
the Mac on top of the stagger (long rows 0.976 of the unpadded time, 2^18 0.94), as long as
it pads only an intermediate the plan already keeps in its workspace: moving one there to
pad it cost 1–9 %. On x86 (Xeon, EPYC) it gained nothing.
[X26b](experiments/X26b-padded-grid.md).

3.2 **Padding costs address arithmetic.** About 4 % on short rows on Apple in the
prototype; pad only where conflicts occur, or compute padded addresses incrementally.

3.3 **Fusing copies into compute passes stops paying beyond L2.** Folding a blocked
algorithm's gather/scatter copies into its first and last passes gained 10 % at 2¹⁸ points
and nothing at 2²⁰–2²²: there the cost is memory traffic and page walks, which a fused pass
pays too. Fusing only one end of a phase lost 9–10 %. [X12](experiments/X12-beyond-l2.md).

3.4 **Count the lines a pass sequence touches.** Using the destination as a chain's
second scratch line (three lines instead of four) kept 512–1024-point rows in a 32 KiB L1
longer: 6–15 % faster on the Xeon. [X18](experiments/X18-x86-kernels.md).

3.5 **Long rows need a blocked algorithm; L2 capacity sets where.** A chain of whole-row
passes ran 1.9–2.8 times FFTW's time from 2¹⁴ to 2²² points on the Mac; a two- or
three-phase grid ran at 0.47–0.71 of the chain's time. Where to switch to three phases is
machine-specific: on the EPYC a switch at 1 MiB was 3 % faster than the default and never
switching was 23 % slower; on the x86 machines it belongs at L2, on the Mac at L2/8 (3.8). Inside L2 the pass count (L2 bandwidth) decides, which is why
FFTW's radix-32/64 three-pass plans win there. [X12](experiments/X12-beyond-l2.md). Correction (X29): this holds for powers of two on the Mac
and beyond L2/2 on the Xeon, not in general; see 3.11 and 3.12.

3.6 **Misaligned wide accesses split cache lines.** With four complex doubles per AVX-512
vector, a stride ≡ 2 mod 4 makes every other access split a 64-byte line (prototype).

3.7 **Prefetchers differ by vendor.** Software next-row prefetch helped the Xeon by 25 % on
45 MB batches and cost Zen 3 and Apple a few percent (prototype). In FFTN's grid copies the
prefetch of rows eight ahead cost Zen 3 8 % (0.92 of the time without it), helped the Xeon
and the Mac by 4–20 %, and the Xeon forced to AVX2 behaved like the Xeon, not like Zen 3
(2^22 0.84 with it): the vendor decides, not the instruction set. But on the same Intel core
it took float rows of 100000 points, whose copied rows are 32 bytes, to 1.27: a prefetch per
row narrower than a cache line costs more than it fetches, so such a rule needs the row
width as well as the core. A run-time prefetch distance instead of a compile-time one cost
the narrow-row loop 8–10 % on Zen 3 and Apple.
[X26](experiments/X26-grid-copies.md), [X26b](experiments/X26b-padded-grid.md).

3.8 **Whether L2 is the core's own matters more than its size.** On the Xeon (32 KiB L1,
1 MiB L2 per core) and the EPYC (32 KiB, 512 KiB per core), whole-row chains beat a blocked
grid for rows up to half of L2: 0.74–0.78 of the grid's time at 4096 points (double), and
0.83–0.90 over the rows between L1 and L2 with the rule that followed (chain to L2/2, three
phases beyond L2), which took FFTN from 1.42 to 1.05 times FFTW at 4096 points on the Xeon. On the Mac (128 KiB L1,
16 MiB L2 shared by six cores) the chain at twice L1 took 1.67 times the grid's time, and
applying the x86 rule there cost up to 2.1 times. The Mac's caches are the larger, yet it
needs the blocked form earlier, so no rule in the sizes alone fits both: key cache
thresholds on the L2's sharing (Linux `shared_cpu_list`, macOS `cpusperl2`) as well.
[X12b](experiments/X12b-grid-thresholds.md).

3.9 **A blocked algorithm's copies can be instruction-bound, not memory-bound.** FFTN's
grid for rows of 46080–100000 points gathers and scatters column blocks four complex
doubles wide, so each copied row is one 64-byte vector. The copy loop spent about 33
instructions per row (row loop, two prefetches, a one-trip inner loop around one load and
one store), which made `block_copy` 40–52 % of all instructions and about 40 % of the
cycles on the Xeon. The whole transform ran at a higher IPC than FFTW's (2.6–3.0 against
1.6–1.8 on the Xeon, 4.0–4.8 against 2.7–3.0 on Zen 3) with no more TLB walks and, on the
Xeon, fewer L2 misses. Before blaming memory for a long transform, compare IPC and
instructions per byte moved: a high IPC with too many instructions points at the copy
loops' shape (rows per iteration, prefetch per line, block width).
[X25](experiments/X25-x86-diagnosis.md). **Corrected by X26:** cutting those instructions
by a quarter to a third left the Mac and the Xeon within the noise and made Zen 3's rows of
2^20 points up to 1.56 times slower. A block's rows lie a multiple of 4 KiB apart, so all
its source lines share one or two 8-way L1 sets; a faster loop keeps more of them in
flight, they evict the scratch block the next pass reads (twice the fills from beyond L2),
and the slow loop had been throttling that by accident. On the same machine removing the
loop's software prefetch of eight rows ahead, which adds eight more lines to those sets,
gave 0.92 of the time. A high IPC with many instructions says what the core executes, not
what limits it: test the instruction cut on the target before building on it, and for
copies over power-of-two strides count L1 fills and data-cache accesses as well.
[X26](experiments/X26-grid-copies.md). **Then X26b:** padding the gathered rows so that they
spread over all of L1's sets did not cut the L1 refills (EPYC +2–7 %, Xeon +0–10 %) nor the
time: rows concentrated in one or two sets evict only those sets while they stream
through, rows spread evenly turn over every set, the scratch block included. Set conflicts
of a streaming copy are not by themselves a cost; what they evict is.
[X26b](experiments/X26b-padded-grid.md).

3.10 **Halving the twiddle bytes trades loads for footprint, and the trade goes both
ways.** Storing x86 twiddles as (wr, wi) instead of the pre-arranged (wr, wr), (−wi, wi),
and building the registers with duplicating loads (`vmovddup`/`vmovsldup` from p and
p + 1, load ports only) and `vfmaddsub`, keeps every bit and halves the tables. It won where
tables overflow L1 (EPYC double 384–4096 points 0.95–0.97, float 0.80–0.94, rows of 16384
0.91, Bluestein's chirps at 10403 0.88; Xeon `avx512` similar) and lost on short rows and
small grouped passes (EPYC 64–192 points 1.03–1.04, 97 1.08, batches of 100 points 1.08;
Xeon 16–17 points 1.06–1.10): a duplicating load cannot be folded into the multiply as a
plain load can, and the load at p + 1 crosses a cache line in half the units (in all of
them on 64-byte registers). Measure footprint changes on short transforms too; a per-table
choice of storage would keep both sides. [X30](experiments/X30-x86-pass-loops.md).
**Then X30b:** the per-table choice kept the geomeans (EPYC x30 double 0.986, float 0.962
of aligned tables, compact broadcast tables everywhere and compact per-lane tables above
16 KiB) but not the short cases: float 256 × 64, whose tables stay pre-arranged, ran 1.04
in every variant, as did the variant with no compact table at all, and the Xeon's 16-point
leaf 1.05. So part of what looked like the storage's cost in X30 was the FMADDSUB form of
the product (or code placement), which every variant shared. Single-process counters also
misled: compact per-lane tables cost 1.06–1.13 of the cycles at 2048–4096 there and won
0.97–0.98 in fbench's rounds. Separate a representation change from the arithmetic it
drags along, and judge footprint changes by interleaved rounds, not single processes.
And an allocator is part of the kernel: tables at `std::vector`'s 16-byte offset made half
the twiddle loads cross a line; 64-byte alignment alone gave 0.97–0.99 on x86.
[X30b](experiments/X30b-twiddle-tables.md).

3.11 **A chain's cache trouble beyond L1 can be a power-of-two problem, not a size
problem.** On the Mac a whole-row chain lost to the blocked grid beyond L1 for powers of
two (1.6–1.8 times its time), but for a length whose power-of-two part is at most 1024
points it took 0.49–0.72 of the grid's time at every size tried, 144 KiB to 62.5 MiB, and
beat FFTW (46080 and 100000 points: 1.20–1.23 times FFTW with the grid, 0.74–0.76 with the
chain). The legs of a power-of-two pass lie a power of two apart and share L1 sets; an odd
factor in every leg distance spreads them. Rules tuned on powers of two (X12, X12b) carry
that bias: test mixed lengths before extending a threshold to them.
[X29](experiments/X29-real-footprint.md).

3.12 **Do not extrapolate a chain-or-blocked threshold past the sizes tried, and look at
every process's time, not only the median.** X12b kept the x86 chain to L2/2 because it
tried thresholds up to L2 only. On the EPYC 7763 (512 KiB L2, 32 MiB L3 per 8 cores) whole-
row chains beat the grid far beyond: lengths with a small power-of-two part took 0.64–0.85
of its time steadily up to 2 Mi points. Powers of two looked as good in two screens
(0.53–0.73) and then, in a third, ran 0.7 to 1.8 times the grid's time from process to
process at 1–2 MiB (leg distances a multiple of the L2's way size, which collide when the
pages lie contiguous): a median over few processes hid a bimodal cost. The Xeon 6240
(1 MiB L2, 0.7 MiB of L3 per CPU) behaved as X12b said for every length: the chain lost
1.1–1.6 beyond L2/2. Same instruction-set family, opposite answers: key the rule on the L3
per CPU. [X29](experiments/X29-real-footprint.md). **Explained by X32:** the slow mode was
the rules' radix-16 passes on transparent huge pages (3.14), not the chain or its size;
chains of radices ≤ 8 were steady and took 0.57–0.85 of the grid's time on the same EPYC
from 32768 to 2²² points. [X32](experiments/X32-long-pow2.md).

3.13 **Fuse a pass when it removes an array, not merely a pass.** Folding the c2r
pairing into the grid's first gather (each element reads its mirror from the input) made
out-of-place rows 6–8 % faster on the Mac, where it also removed the workspace array, and
in-place and three-phase rows 2–6 % slower, where the array stayed and the input was read
twice instead of once by a streaming in-place pass; on the EPYC even the out-of-place
rows lost (1.11). [X29](experiments/X29-real-footprint.md).

3.14 **When a cost changes from process to process, look at the page size before the
algorithm.** Linux's transparent huge pages ("always" on the EPYC nodes) back a buffer of
1–2 MiB with 2 MiB pages or not depending on where its mapping falls. On huge pages the
physical address bits that index L2 equal the virtual ones, so the 16 streams of a radix-16
pass, a power of two apart, share one 8-way L2 set: FFTN's chain of 131072 points ran 14.3
ns per point in 14 processes of 16 and 8.7 in the other 2, and 21–25 when every buffer was
forced onto huge pages, against 4.4–4.8 for radix-8 and radix-4 chains in every placement
(the grid 6.3). Record each buffer's backing (`/proc/self/smaps`, AnonHugePages) next to
its time, and test with `MADV_HUGEPAGE` and `MADV_NOHUGEPAGE` to separate the two modes.
Large radices at power-of-two strides were already known to conflict in L1 (3.1); huge
pages carry the conflict to L2. [X32](experiments/X32-long-pow2.md).

3.15 **Two streaming buffers a few cache lines apart modulo 4 KiB cost more than their
alignment.** A Stockham chain reads one buffer and writes the other; with the output 16 to
128 bytes past the scratch line or 48 to 128 before it modulo 4 KiB, its loads alias the
stores just issued (4K aliasing) and a radix-4 chain of 65536 points on Zen 3 ran 1.15–1.9
times as long as with the two half a page apart; an output merely not 64-byte aligned cost
5–8 %. Two large malloc'd buffers lie exactly so (each 16 bytes into a page). A library
that owns one of the two buffers can place it half a page from the other at execution; FFTN
tried that and fbench saw nothing (0.99–1.01), because fbench's buffers sit at random
multiples of 64 bytes, where the near-alias cases are rare: a benchmark that never places
buffers as malloc does cannot judge such a fix. [X32](experiments/X32-long-pow2.md).

3.16 **Where L3 is small and far, count the passes over memory before the cache fit of
each pass.** X12b's rule (three grid phases beyond L2, so that each phase's block stays
within L1) was set on powers of two. On the Xeon 6240 (1 MiB L2, 0.7 MiB of L3 per CPU)
rows with an odd part of 15 or more ran in 0.70–1.01 of the cycles as two phases, whose
blocks no longer fit L1, from 1.1 MiB to 64 MiB (100000 points 0.73, 10^6 0.70; L2
misses per point 1.3 → 0.8), while on the EPYC 7763 (512 KiB L2, 4 MiB of L3 per CPU) the
same two-phase plans took 1.13–1.67 of the cycles below 16 MiB. Rows near a power of two
(odd part 3–9) lost as two phases on both: their gathers' rows still collide in L1. One
blocking rule does not fit both kinds of uncore. [X34](experiments/X34-xeon-mixed.md).

3.17 **Before fusing away a streaming pass, check what the pass costs alone.** FFTN's
separate r2c pairing pass over the half spectrum looked like 10–30 % of a long real row
(time of the real row over the c2c row of half its length), but fusing it into the grid's
last phase (no pass, no extra sweep) recovered only 2–6 % on the M5 Max and the Xeon 6240
and nothing on the EPYC 7763: a streaming in-place pass over an array in L2 or L3 is cheap,
and the rest of the "real overhead" is the real row's own plan. The fused form's blocks
also had to stay within the phase's scratch budget and be at least a cache line wide, or
it lost 5–9 %. Meanwhile FFTW's r2c rows of 2²⁰ cost 0.76–0.86 of its own c2c of half
their length: its real plans are not a half-length c2c plus a pass.
[X33](experiments/X33-long-real.md).

3.18 **In a VM, "L3 per CPU" is the socket's L3 divided by the VM's CPUs.** A Granite
Rapids VM of 8 cores reports one 504 MiB L3 shared by its 8 CPUs, 63 MiB each, against about
4 MiB per physical core; a Zen 5 VM of 8 cores reports its CCX's 32 MiB, which is true.
Rules keyed on L3 per CPU (X29's mixed chains, X32's power-of-two chains, X34's two phases)
therefore gave Granite Rapids Zen's plans, and they lost: long powers of two ran at 1.07–1.69
times the grid's time, and rows with a small power-of-two part beyond L2 at 1.46–1.69 times
a two-phase grid's. Yet X34's two phases, the Xeon's plan, also lost there, by 1.03–1.46, on
the rows with a large power-of-two part. The machine fits neither neighbour. A cache key is
only as good as the topology the guest is shown, and crossovers that move by core want a
measured choice, not a guessed key. [X31b](experiments/X31b-cloud-followup.md).

3.19 **A cache size can stand for the core behind it only within one family.** The private
L2's size, unlike L3 per CPU, does not change with a VM's size or SMT, and on x86 it separates
the cores that want long-row chains (Zen 3 and Zen 5, 0.5–1 MiB) from Granite Rapids (2 MiB),
which wants grids: its long powers of two ran at 0.59–0.98 and 100000 at 0.67 of the chains'
time. Axion's Neoverse V2 has the same 2 MiB private L2 and lost 1.15–1.28 with Granite Rapids'
plans, up to 1.97 on single rows. Within Granite Rapids itself, two- and three-phase grids
traded places by factorisation (71680 lost 1.10 in three phases while 86016 and 98304 gained
0.92–0.93; c2r rows lost 1.15 in two phases while c2c rows gained). Key such a rule on
geometry and architecture together, and expect the fine crossovers to need a measured choice.
[X35](experiments/X35-gnr-profile.md).

3.20 **Where the kernel backs memory with transparent huge pages, a program's speed can flip
from process to process, and a timing-based choice inherits the flip.** On the EPYC (THP
"always"), r2c 65536 ran 140 µs on 4 KiB pages and 176–184 µs on huge pages with the rules'
radix-8 chain. The other chains changed less, so measured planning on scratch that happened
to be huge-page backed chose programs 1.18 times slower on the benchmark's buffers. Rader's
random gathers at 65537 were 1.75 times slower than Bluestein in one run and 1.41–1.49 times
faster in two others. Time candidates on both page kinds (madvise each way) and judge each by
the slower, or the "best" program is a property of one allocation. Judging by the slower
kind trades a few cases where the caller's memory suits the faster program (Xeon 262144,
1.07). [P1](experiments/P1-rules-vs-measure.md), [P1b](experiments/P1b-rader-65537.md),
[P2](experiments/P2-measure-budget.md).

3.21 **A cost model with one memory factor per cache level cannot rank blocked and unblocked
long-row structures.** Fitted per machine to 52 long rows under chain, two and three grid
phases, it chose within 5 % of the fastest on 100 % of rows on the EPYC, but on only 55–80 %
on the Mac and the Xeon, worse than hand thresholds. Its fit errors of 15–29 % were as large
as the gaps it had to rank. Which structure wins beyond L1 depends on the cache sets the
passes and copies hit (strides, block widths, the power-of-two part of the length), not on
the level alone. Fitting the thresholds themselves did no better with one sample per region.
[P3](experiments/P3-cost-model.md).

3.22 **A breadth-first chain that half fills a private L2 already misses it.** On the Xeon
6240 (1 MiB L2) FFTN's Stockham chains of 16384–32768 complex doubles (256–512 KiB) missed L2
0.7–1.7 times per point at an IPC near 1, with fewer instructions than FFTW (12 against 16
per point), and ran 1.23–1.51 times FFTW's time: input, output, a scratch line and four
passes' twiddles overfill L2. FFTW plans the same rows as recursive decimation in time
(16384 = 8 · 64 · 32), whose sub-transforms of 2048 points finish in L1, and misses L2
0.0–0.3 times per point. A two-phase grid, the blocked answer FFTN has, pays only from 512
KiB (32768: 0.89 of the chain); at 16384 its copies cost 30–50 instructions per point and it
lost in every split (1.02–1.69). So the gap between L2/4 and L2/2 is a missing strategy, a
depth-first chain, not a threshold. [X39](experiments/X39-xeon-long.md).

3.23 **A strided axis is a batch of adjacent columns: vectorise across them and read each
line once.** Running a plane's columns one at a time (copied element by element on wide
registers) took 1.5–7.6 times FFTW's time from 64² to 2048²; blocks of 16–32 adjacent columns
as the lanes of the column's chain, the block's first and last passes taking every row's
columns at once, took the planes to 0.45–0.61 of that time (FFTN ÷ FFTW 0.80–0.94) on an M5
Max, an EPYC 7763 and a Xeon 6240. Block by bytes per row (256 bytes, four lines where each
element is on its own page), not by fitting tiles in L1: where every column element has its
own page the widest blocks won (2048² 0.126 against 0.183), while tiles beyond L2 lost where
rows share pages. [X40](experiments/X40-planes.md).

3.24 **Count a pass's output legs against the L1's ways; its input legs mattered less.** A
column chain whose last pass wrote 16 legs a power of two apart into one 8-way L1 set made
128² and 256² planes 2.2 times slower on the M5 Max than the same columns ending in radix 8,
whereas a first pass reading 16 legs from one set cost nothing measurable. Choose the
radices of passes that write strided memory by the sets their legs hit (DESIGN.md 10,
rule 1). [X40](experiments/X40-planes.md).

3.25 **Vectorising across adjacent columns pays only for what a column alone costs.** The
same column blocks that took X40's c2c planes to half their time lost 1.04 over the real
planes on an M5 Max: there a column needs no copy (one complex double per NEON register) and
the real planes' complex rows, n/2 + 1 values long, lie an odd multiple of 16 bytes apart,
so a column alone crowds no L1 set; the blocks' tiles only added traffic. On x86 (two or
four complex values per register, each strided column copied) the blocks won 0.73–0.97 on
an EPYC 7763, except where their tiles left L2 while the column alone fit L1 (1024²: 1.05
with 4 columns). Ask what the column route pays (copies, set conflicts) before blocking.
[X41](experiments/X41-planes-2.md).

## 4. Kernels and algorithms

4.1 **A whole-transform codelet wins only where it fills the register.** One 32-point
codelet beat the planner's two-pass chain by 16–22 % in double on NEON (one complex per
register) and lost in float and on x86 (2–8 complex per register, where a single line uses
one lane). Radix-64 codelets spill on every target. [X8](experiments/X8-leaves.md).

4.2 **Bluestein beats a direct prime pass from small primes up.** Above 16, a chirp-z
convolution through smooth-length FFTs beat the O(q²) generic pass at every prime measured
(17: 2.5 times in double, 7 in float, on the M5 Max; 12–20 times on the Xeon 6240 and the
EPYC 7763, where a lone row fills one lane of 2–8 in a pass vectorised across copies);
primes from 97 up went from 12–5000 times FFTW's time to 0.64–1.28. Only where many copies
fill wide vectors (AVX-512 float, 64+ copies of 17) did the direct pass win, by up to 13 %. Store the chirp and kernel spectrum once per (prime, sign), not once per
copy: 889 MB of tables became under 1 KB at n = 17·2²⁰.
[X17](experiments/X17-bluestein-threshold.md). Both lost to better routes later: generated
codelets at 17, 19 and 23 (4.10) and Rader for primes with a smooth q − 1 (4.11).

4.3 **Real transforms: fuse the pairing into a pass, but only with a small radix and in
L1.** Reading reals as half-length complex and pairing bins k and h−k in the last pass
(r2c) or first pass (c2r) took 256–4096-point real transforms from 1.04–1.31 to 0.82–0.96
of FFTW's time on the Mac; fusion gains 10–15 points with radix 8 in L1 and loses with
radix 16 or beyond L1. The remaining in-L1 overhead (+39–43 % over half-length complex) is
the pairing arithmetic itself. [X6](experiments/X6-real-codelets.md).

4.4 **Fewer or more passes is a trade of memory traffic against compute.** More, cheaper
passes win on Apple's big caches (FFTL.jl); fewer, bigger passes win when data streams
through L2 (FFTW on the Xeon). Tails cost several units each in measured tables, so a
first radix that is a multiple of the register's complex count, and prefix products that
are too, are worth seeking. [X2](experiments/X2-radix-sets.md).

4.5 **Twiddles: load once per butterfly group, broadcast across copies; special-case
N/4 and N/8.** The prototype gained 2 % hoisting by hand what the compiler did not.

4.6 **Big-radix passes pay only where registers are wide and many.** With generated,
depth-ordered codelets, a radix-32 pass beat the best radix ≤ 16 chain only on
AVX-512 in float (32 registers of 8 complex): 512 as 32·16 at 0.82–0.85 of the planner's
time, 1024 as 32·32 at 0.92–0.94 (Xeon 6240); planner tables with radix 32 columns took
float chains on that machine to 0.977 of their time over the multiples of 32 up to 4096. In double on the same machine, on AVX2 (16
registers, EPYC 7763) and on NEON (M5 Max) they tied or lost; radix 64 spills (67–681
values per pass) and won only as a grid phase at 32–64 thousand points (a 2 % win as a
chain pass at 2048 vanished once the radix-16 codelets lost 12 of their 19 swaps, X19). Fewer passes is not enough: the pass must fit the
register file. And a cost table fitted on lengths up to 2048 extrapolates its big-radix
columns badly: at 4096 it chose 32·4·32, 2–11 % slower than the measured 8·16·32. [X3](experiments/X3-big-radices.md).
**Partly overturned by X30:** after X19's rotations, X1's split runs and X23's masked
tails, AVX2 double (16 registers, EPYC 7763) gains from a radix-32 pass where it replaces a
third pass: 256 as 8·32 took 0.90 of 2·16·8's time and 512 as 16·32 0.87–0.95 of 4·8·16's
(rows, batches of 64, real transforms of twice the length), although the radix-32 lane loop
makes 142 stack references per unit. One pass fewer outweighed the spills; 1024 as 16·64
did not (1.03 alone, 1.09 in batches of 256). Re-test a "loses" verdict on big radices
after the codelets change. [X30](experiments/X30-x86-pass-loops.md).

4.7 **For tiny transforms the bookkeeping is the transform: interpret once per call and put
the batch loop inside the kernel.** On the M5 Max an 8-point leaf costs 2.5 ns, while
walking FFTN's program records and copying the 144-byte argument struct for each
transform cost about 3.5 ns more, and the call's entry another 1 ns; copying the struct
alone is about 1.2 ns. Preparing the stages once per call, passing a leaf its addresses
as arguments (no struct copy at all) and running a batch of rows in one kernel call took
an 8-point transform from 7.7 to 4.3 ns at batch 1 and from 7.1 to 1.7 ns per transform
in a batch of 64, a 2-point one from 4.9 to 0.32 ns: rows of 2–16 points in batches of 64
went from 1.9–17 times NEON FFTW's time to 0.85–1.18 (fbench run 20261115-002244-eba3).
The loop inside the kernel matters beyond the call: a 32-point leaf took 20 ns per call
one row at a time but 15 ns per row in one call over 64 rows (constants and saved
registers set up once). Three pitfalls at this scale: on x86, passing a small struct of pointers by reference right
after writing it field by field stalls when the callee copies it with wider loads (no
store forwarding from two 8-byte stores to one 16-byte load): two such stalls were most of
the 8–11 ns between FFTN's API and a leaf on the Xeon, and passing the two pointers as
arguments took a 2-point row from 13.1 to 6.6 ns; a hot path that shares a function with
a cold one pays the cold one's frame (an on-stack array of argument copies cost 6 ns per
call on the Mac and 2–3 ns of a leaf row on the Xeon until the paths were split); and a
loop over rows can make the compiler keep every leg address as its own induction variable
(GCC 14 on AVX-512: 171 instructions per call of an 8-point float leaf against 153 without
the loop, 7–11 % slower for one row). Measure the levels separately (entry, interpretation,
kernel through its pointer, kernel in a loop): the harness of
[X20](experiments/X20-call-overhead.md) does this in a page of code.
A fresh copy of an argument struct made just before each call costs more than the copying
on x86: copying FFTN's 176-byte pass arguments per stage cost 3.7–6 ns per stage on Zen 3
(GCC 16) and about 3.5 on Cascade Lake, against 1.5–2.7 on the M5 Max; made before the
first kernel instead, half of it went, so much of it is the kernel's loads waiting on the
copy's stores. Passing the per-call pointers as arguments and the fixed fields by pointer
to the plan's own memory removed it: a 24-point row 47 → 31 ns on the EPYC, 60 points
97 → 80, and on the Mac 0.69–0.88 of the time for 24–120-point rows.
[X28](experiments/X28-short-rows.md).

4.8 **Batches of short rows: vectorise across rows only where the per-row code idles
lanes, and measure against the best row-by-row code.** Putting one row in each lane, with the
row's own codelets so every bit is unchanged, took 8–16-point rows to 0.13–0.59 of the
row-by-row time on AVX-512, AVX2 and the M5, but most of that was the row-by-row path's fixed
cost per row: once a batch of leaf rows became one kernel call (X20), the lanes lost at 8
points in double and on NEON (1.09–1.66) and kept their win only where many lanes would idle
(float on x86, 0.32–0.55). The transpose into lanes costs two passes over the block; where a
run of C rows is one vector access (C = 1, interleaved rows) no transpose is needed, and
building each vector element by element through a stack tile was slower than the transpose.
Read and write the rows in the order the row-by-row code does: walking them group by group
cost 1.3 times beyond L1. [X7](experiments/X7-lanes.md). Taking out the transposes does not change the answer for double on x86, where C is 2–4.
X24 built each register from C 128-bit row loads (`vinsertf128`, then `vinsertf64x4` on
AVX-512) and took it apart with 128-bit stores, so short rows needed no gather or scatter.
That took the lanes to 0.80–0.88 of their own time on AVX2 (geometric means; EPYC, Xeon),
but they were still 1.08–1.37 of the batched-leaf chain's time at 8 points. On AVX-512, rows with inner
passes ran up to 1.17 times slower than with the transposes (24–256 points, Xeon), so four
128-bit pieces per register only pay where there is no inner pass to share a transpose
over. Preparing the lanes' arguments once per call, as X20 did for rows, was 1–7 % slower
at the smallest batches (64 rows of 8–16 points) and up to 4 % faster elsewhere: a fixed-cost
change needs its smallest case in the catalogue. [X24](experiments/X24-lanes-x86.md).

4.9 **Keep a loop's rare path out of the loop, with constant indices.** Masked vector
accesses for the partial units of FFT passes (the last unit when the lane count C does not
divide the butterfly count) took GCC 14's float 7-smooth rows to 0.60 of the old time on
AVX-512 and 0.80 on AVX2 (Xeon 6240), but written inside the loop over units they first made
whole units slower, up to 1.15 of the old time at lengths with no partial unit at all.
Three mechanisms, each found by perf or disassembly and fixed without changing a bit: the
compilers hoisted the rare path's set-up (masks, a pointer per leg) into the function's
prologue (Clang 23: 80 to 350 instructions for a radix-8 pass called on two units) and
lost registers in the hot loop (GCC: 4 % more instructions per transform at 128 points);
a store loop bounded by the runtime count indexed the register array at run time, so the
compiler kept the array in memory for every unit (Clang: 1.16–1.46 of the old time); and a
count the compiler could not see was constant stopped a store loop from unrolling (Clang,
double 75 points: 1.6). Running whole units in the loop and the partial one after it, with
the count a constant inside the loop, removed all three. It did not remove a Clang-only
loss on odd double lengths (AVX2 75 points still 1.42): a code-generation change must be
measured with every compiler a user may build with, since two compilers can disagree by
40 % on the same source. Rejecting it for that would have thrown away GCC's gain, so it
was kept behind a build option that is on for GCC, the release compiler, and off for Clang;
with it off the objects are byte for byte the old ones, which settles that the switch
costs nothing. On a later main the same source still gained (GCC float 0.53 of the old
time on AVX-512, 0.72 on Zen 3) but two cases with no partial unit at all came out 3–5 %
slower (float 256, Xeon; double 72, EPYC, which had been 0.93 with the same source on the older
main): code around a hot loop moves its speed, so such losses need re-measuring after
unrelated changes. [X23](experiments/X23-masked-tails.md).

4.10 **Generate codelets for the small primes before reaching for a convolution.** At 17,
19 and 23, straight-line codelets in the symmetric-pair form (168–297 vector operations,
20–26 live registers, so they spill a little on AVX2's 16) took 0.15–0.42 of a Bluestein
stage's time on the M5 Max, 0.12–0.65 on the EPYC 7763 and 0.06–0.50 on the Xeon 6240 over
rows, composites (2431 = 11·13·17, 4913 = 17³), batches and real rows, and put FFTN at
0.28–0.54 of FFTW's time at 17–23 where it had been 1.6–2.0. FFTW has no codelet for these sizes either. The cost is
library size: each prime adds 6–11 % of a target's code (all of them +25 %), because every
pass kind and edge variant is instantiated per radix. [X27](experiments/X27-prime-codelets.md).

4.11 **Rader halves Bluestein's convolution, but its permutations are random accesses.**
For a prime q whose q − 1 has only codelet factors, Rader's cyclic convolution of q − 1
points replaces Bluestein's of M ≥ 2q − 1: 0.21–0.68 of the Bluestein stage's time from 29
to 40961 on the M5 Max (8191 and 16381, whose q − 1 has a factor 13, 0.21–0.45), and
0.4–0.8 from 97 up on the EPYC 7763 and the Xeon 6240. At 65537 it lost on the M5 Max
(1.2: the gather by powers of a primitive root and the scatter back each touch a megabyte
in random order) but won on both x86 machines (0.56–0.83). With wide vectors (C ≥ 4) a
single small prime lost (29–97 on AVX-512 float: 1.3–1.9), because the gather and scatter
of one row are scalar while Bluestein's chirp passes are not: the crossover moves with the
lane count, so the rule is per target. Two implementation
details decided whether it won at all: stepping g^m mod q with a data-dependent
subtraction loop (mispredicted branches) and running one row one broadcast register at a
time made it slower than Bluestein; an index table and plain scalar code for single rows
made it twice as fast. [X27](experiments/X27-prime-codelets.md).

4.12 **A short transform that fits the register file should never touch memory between
its passes.** For n = R1 M with M a multiple of the register's complex count C, a DIF
chain's grouped first pass leaves its outputs in registers, and the C × C transposes it
does before storing are exactly the second pass's input; held in registers instead of
stored, the two passes become one call of about n / C registers with no partial unit in
memory (the last block of copies is computed whole and stored with a partial run). On the
EPYC 7763 (AVX2, GCC 16) single rows of 20–48 points took 0.51–0.76 of the two-pass
chain's time in double and 0.39–0.78 in float, batches of 24 and 48 points 0.51–0.70: 24
points went from 0.99 to 0.53 of FFTW's time. On the Xeon 6240 (AVX-512, 32 registers) the
same reached 120 points: 20–120 at 0.42–0.69 of the chain's time, 60 from 1.04 to 0.63 of
FFTW's and 120 from 1.23 to 0.73. The code is small (+0.75 % of the x86 library for 37
lengths and precisions) because only the length table grows, not the pass kinds.
[X28b](experiments/X28b-short-leaves.md).

4.13 **At 8–23 points, a leaf's set-up for any stride is a fifth of its time.** A leaf that
computes its 2R leg offsets from run-time strides keeps them in general registers (and
spills them on x86); the same leaf with constant offsets, chosen for rows of stride 1, took
single rows of 8–23 points to 0.75–0.92 of their time on the EPYC 7763, 0.79–0.91 on the
Xeon 6240 and 0.80–0.87 on the M5 Max, bit for bit the same. The one rare path a kernel supports can cost more than the
arithmetic at this size: specialise the common case at plan time.
[X28b](experiments/X28b-short-leaves.md).

4.14 **A Bluestein stage is only as fast as the long-row and layout rules its convolution
inherits.** FFTW runs Bluestein for 131071, 999983 and 4093 too and was 1.3–2.6 times
faster: FFTN's convolution, a chain of M points built apart from the line rules, took the
in-cache cost table's radices at any size (radix-16 passes over 2¹⁸ points: 2.5 times the
long-row radices' time on the M5 Max), never the split layout x86 lines take, and the
cheapest length by the table, which is a power of two. On Apple powers of two from the L1's
size up ran 1.4–1.7 times as long per point as 7-smooth lengths just above (as did lengths
with a power-of-two part of 4096 or more), and on the EPYC 7763 the convolution's two
scratch lines, a power of two apart when M is one, made 4093–32749 take 1.4–5 times as
long as with a 7-smooth M; the Xeon 6240 showed neither. With the line rules applied and
the length's power-of-two part limited per machine, the Mac's large primes took 0.33–0.64
of their time (FFTN ÷ FFTW 1.4–2.6 → 0.70–0.90). Choose transform lengths inside other
algorithms with the same memory rules as the user's lengths, and avoid power-of-two
distances between buffers that are read together.
[X36](experiments/X36-large-primes.md).
4.15 **Read the competitor's printed plans for the radices you lack before tuning the ones
you have.** On the EPYC (Zen 3, AVX2) FFTW planned 160–3600 points with radix-20 passes (240
= 12 × 20, 2000 = 10 × 20 × 10) and FFTN, without a codelet of 20, needed a pass more;
re-running FFTN's exhaustive sequence search over every radix it had (2–16, 32, 64) gained
nothing at 80–160 and 512–2048 and 4–16 % at three lengths, while one generated radix-20
codelet gained 4–19 % at seven (0.868 of the time over the changed lengths, 240 from 1.13 to
0.93 of FFTW). The converse also held: FFTW's two-pass plans with a radix-64 first pass ran
1.2–1.3 times FFTN's three passes, because FFTN's radix-64 codelet holds 67 live values in
16 registers; a plan shape transfers only with codelets that keep it in registers.
[X38](experiments/X38-epyc-mid.md).

4.16 **A batch of short rows wants whole registers: a two-pass leaf beats a one-pass leaf
that fills one lane.** The x86 leaves of 8–16 points hold one complex double per register
(128-bit, X28), so a batch of them does a whole radix-16 codelet per row at a quarter
(AVX-512) or half (AVX2) of the machine's width. The same rows as pair leaves (4.12) on
whole registers, 16 = 4 × 4 on AVX-512 and 8 × 2 on AVX2, took 16 × 16 and 16 × 256 to
0.59–0.60 of their time on the Xeon 6240 (FFTN ÷ MKL 1.58–1.70 → 0.94–1.01, instructions
per 16-point row 138 → 50) and 0.75 on the EPYC 7763 (÷ FFTW 1.19–1.26 → 0.90–0.93), and
single rows of 16 to 0.67 and 0.83. Two pitfalls: where r1 < C (8 = 2 × 4, 12 = 3 × 4 on
AVX-512) the second pass stores partial runs, and an in-place batch whose next execution
reads them back ran 1.5–2.1 times as long (masked stores do not forward to the next
loads), so those lengths take AVX2's leaves (r1 ≥ C) in an AVX-512 plan instead; and the
leaves of another vector width need their twiddles in that width's storage, so a pair set
carries its lane count and packing. Lengths that no C divides (18, 22, 26, 30 on AVX-512)
are AVX2's pair leaves too: 0.40–0.54 of the two-pass chain's time.
[X37](experiments/X37-short-batches.md).

4.17 **Count the spills before blaming them: a codelet that overflows the register file
may already cost what a register-fitting one would.** FFTN's radix-32 and 64 codelets peak
at 35 and 67 live values on AVX2's 16 registers, and FFTW-shaped two-pass plans with them
lost to three radix-16 passes on the EPYC 7763; the obvious cause was spills. A two-level
form that stages R values once through a fenced local array (each level within 13–16 live
values) cut Clang's stack moves in the radix-64 lane pass by two thirds, but with GCC 16
it left instructions (± 3 %) and time unchanged (1.000 of the flat codelets over `avx2`'s
radix-32 plans; 1024 as 16 · 64 5.39 against 5.55 cycles per point): GCC already folds the
reloads into memory operands. The two-pass plans executed 7–13 % fewer instructions than
three passes and lost on IPC (3.0 against 3.5), with no FP pipe over 52 % busy. For an odd
radix the staging cost more than it saved (radix 25: 12 % more vector operations, 3–8 %
slower on both x86 machines). [X42](experiments/X42-codelets.md).

4.18 **Count a column phase's instructions against the rows' before tuning its radices.**
On the Xeon 6240 FFTN's rows of 16–64 points ran oneMKL's instructions and cycles, but its
planes' column phases ran about twice MKL's instructions per point: each lanes edge pass cost
as many instructions as the row's whole two-pass leaf, mostly scalar address arithmetic for
R legs on each side and loop state spilled to the stack, repeated for every run of C
columns, with the group's twiddles reloaded per run. No choice of column radices or block
width moved that (0.97–1.03 of the cycles); running a block's runs in an inner loop per
group did (64² 0.84, 128² 0.89, 512² 0.92 of the cycles), and so did removing a pass where
the row's pair leaf had left the columns two (32²: one radix-32 pass, 0.72 of the time on
the EPYC 7763). The per-call set-up is the next cost for the smallest planes: a radix-16
column leaf computes and spills 32 leg addresses for four runs at 16².
[X44](experiments/X44-small-planes.md).

4.19 **In-cache FFTs on Zen 3 stall on the FP register file; count the lines a chain
touches, and judge integer savings by time, not instruction counts.** On the EPYC 7763
(GCC 16, AVX2) FFTN's 512–2048-point rows did FFTW's FP work (FP-pipe uops within 3 %) in
14–18 % more cycles, and most of the extra cycles were dispatch stalls on the FP register
file (PMCx0AE unit 0x20: 26–37 % of cycles against FFTW's 21–32 %), which fills while
retirement waits on loads that miss L1. FFTN refilled L1 1.4–2.9 times as often, because a
Stockham chain out of place touches three lines (source, destination, scratch) where FFTW's
in-place passes on the output touch two. The last pass of a Stockham chain reads and writes
the same elements per unit, so it can run in place: two-pass chains then touch two lines,
and 512 points refilled L1 172 times instead of 509 and took 0.95 of the time (batches of
256–512 points 0.95–0.98; rows that already fit L1 unchanged, as was the Mac's tracking
tier). The converse warning: taking per-unit index arithmetic out of the lane-pass loop cut
GCC's radix-16 unit from 190 to 158 instructions, yet ran 1.02–1.17 times as long, because
GCC then kept the 32 leg offsets in stack slots and every leg load waited on an offset load.
Zen 3's counters need groups of at most five events with the NMI watchdog on (a group of
six never runs). [X47](experiments/X47-epyc-mid-2.md).

## 5. Threading

5.1 **Synchronisation costs set the smallest row worth splitting.** Spin barrier /
fork-join, ns: Mac libomp 368 / 485 at 6 threads (with an explicit block time; the default
is 14 671), Xeon libgomp 708 / 1364 at 6 and 1145 / 3165 at 18, EPYC node 688 / 1701 at 6
and 2513 / 9799 at 64. A pass of a short row takes 0.5–2 µs, so rows up to about 2k points
must not be split (prototype). FFTN's own claiming engine, which counts phases instead of
meeting at barriers, measured per phase hand-off (1 µs chunks, members pinned): 461 ns
among 6 cores of an M5 cluster, 1141 ns among the 8 of a Zen 3 CCX, 2846 ns among the 18 of
the Xeon's mesh, and 5.3 / 22 µs across 16 / 64 Zen 3 cores: no cheaper than the
prototype's barrier inside a group, and it grows about linearly with the members.
[T1](experiments/T1-dispatch-wait-costs.md).

5.2 **Keep a team inside a cache group.** On Zen 3 a team within one CCX (8 cores, shared
L3) scales one row from 2048 points; across CCXs it collapses. On the Xeon's mesh a single
row pays only from about 8k points (prototype). FFTN's re-measurement with SIMD kernels,
splitting a lone row whenever it may against never: on a CCX a 3072-point row gained 0.94
with 8 OpenMP threads but lost 1.68 behind a 64-thread executor (6144 gained at both); on
the M5 cluster 4096 lost 1.17 and 8192 gained 0.84; on the Xeon mesh c2c rows gained from
10240 points but real transforms of 16384 still lost 1.11–1.17, and a threshold of 20480
points, which those needed, made planes and few rows of 16384 points 1.5–2.4 times slower.
The smallest transform worth splitting is a time, not a size: about 10–20 µs of serial
work, roughly ten phase hand-offs plus one dispatch, which a single size threshold over
all kinds of transform fits only roughly. [T5](experiments/T5-split-thresholds.md).

5.3 **No hidden threads.** A library pool fighting the application's OpenMP team is the
failure mode of several libraries; run in the caller's threads through an executor.

5.4 **Bitwise independence of team size costs nothing and makes testing easy.** Every row
runs the same kernel whatever the split; tests compare threaded results with serial ones
bit for bit (FFTN's team tests, sizes 1–16).

5.5 **Dispatch cost grows with the width of the team, not just its existence.** Starting
a team and waiting for its end (one 64-point row per lane, hot), on 2 / 8 / 64 Zen 3
cores: FFTN's pool 1.7 / 5.6 / 26 µs, libgomp regions through an executor 1.7 / 3.3 /
35 µs; libgomp's last thread starts 8.3 µs after the fork at 64. An OpenMP region of
constant width pays the full width even when a few lanes have work. A scheduler that
models dispatch as one constant either inlines jobs a group runs 2–5 times faster (with
the full-width cost) or splits lone transforms that then lose to serial behind a wide
executor (with the group's): T1 needed a per-lane term. On the Xeon (18 cores) the pool
took 11 µs and the executor 8.6 µs; on the M5 (6 cores) 2.0 and 2.4 µs (the latter with a
block time, 5.8).
[T1](experiments/T1-dispatch-wait-costs.md).

5.6 **Waking a blocked thread costs more the longer it slept.** Notify-to-running latency
of a waiter blocked in `std::atomic::wait` (futex) after 100–300 µs / 1 ms: Zen 3 5.0 /
5.4 µs (8.5 across CCXs), Xeon 2.5–3.3 / 10–13 µs (deep idle states), M5 5.3–6.6 / 6.4 µs.
Spinning wakes in 0.1–0.3 µs and a yield loop in 0.2–1.1 µs. Shortening the spin budget to
the parked latency (9 µs on Zen 3, T1's rule) made back-to-back batches 5–11 % slower, so
spin budgets should cover the gap between consecutive calls, not just the wake cost.
[T1](experiments/T1-dispatch-wait-costs.md).

5.7 **A bound OpenMP runtime pins every thread created after it.** With
`OMP_PROC_BIND` set, libgomp binds the initial thread to one CPU, and `std::thread`s
created afterwards inherit the one-CPU mask: a thread pool started in such a process ran
all its threads on one core (32 ms per call at 8 lanes instead of 4.7 µs). Benchmarks that
mix runtimes in one binary silently measure this (the first T1 campaign's pool numbers in
its OpenMP binary). [T1](experiments/T1-dispatch-wait-costs.md).

5.8 **libomp on macOS does not spin by default.** On hybrid Apple processors LLVM's runtime
puts idle threads to sleep at once: a region through an executor cost 20–26 µs at 6
threads, against 1.6–2.4 µs with `KMP_BLOCKTIME=200`, and split rows up to 10240 points
were slower than serial, 16384 no faster (prototype: 14.7 µs; FFTN's T1 and T5 on the M5).

5.9 **Compare threaded runs against the competitor's better plan, serial or threaded.**
FFTW 3.3.11 planned with 8 threads (`fftw_plan_with_nthreads`, MEASURE) ran 8 × 6 and 5 × 7
planes 280–380 times slower than its serial plans on an EPYC 7763, and 16² to 64² 3–50
times slower: its planner does not fall back to one thread for tiny problems. A threaded
comparison that takes FFTW's threaded time alone flatters the other library on every small
case. [X41](experiments/X41-planes-2.md).

## 6. Toolchain and testing pitfalls

6.1 **Build with GCC as well as Clang before a remote round trip.** GCC warns where Clang
does not (`-Wsign-conversion` on `x += c ? 1 : 0`, `-Wshadow`, CTAD); a local GCC build
catches what otherwise fails a Linux gate an hour later.

6.2 **Build with `-ffp-contract=off` when results must be reproducible.** GCC 15
contracted `a*b+c` into FMAs differently in two inlined copies of one function, so threaded
and serial results differed in the last bits (prototype).

6.3 **Sanitizer builds of large generated kernels need memory and time, mostly for debug
information.** GCC's ASan on radix-64 codelets was killed for memory with 16 parallel
compiles and ran past 90 minutes with 8 (prototype). With line tables only (`-g1`,
`-gline-tables-only`), which is all a sanitizer report needs, one FFTN kernel unit under
ASan + UBSan fell from 23.4 to 5.6 GB with GCC 16 and from 9.1 to 4.7 GB with Clang 23
(`-g` made GCC overflow variable tracking and redo it per radix-64 pass); -O1 instead of
-O2 saved no memory. Size parallel compiles to the machine's memory, not its cores.

6.4 **GCC's ASan fake stack misplaces 64-byte locals in AVX-512 code.** GCC aligns a 64-byte
local array (an accessor's tile) to a cache line and so emits a 512-bit store to it as an
aligned `vmovapd`. In a function that realigns its stack to 64, GCC lays out the ASan frame
for a base 32 bytes off a 64-byte boundary, as it is on the real stack, but the
use-after-return fake frame (`__asan_stack_malloc_N`, on by default in GCC's runtime) is
64-byte aligned: the tile lands 32 bytes off, and the store faults (a general-protection
fault that ASan reports as a SEGV on an unknown address, "READ"). Xeon Gold 6240: with GCC
14.4 and 15.2 every test that runs `avx512` leaves or Bluestein lane passes failed when run
outside the gate, which turned the check off at run time; `detect_stack_use_after_return=0`,
or compiling with `--param=asan-use-after-return=0`, passes. GCC 16.2 and Clang 23 pass with
the check on (their runtimes, like GCC 14's, enable it by default on Linux), so the gate now
leaves it on. A lone tile in a small frame faults only with
GCC 13.3 (GCC PR 110027, fixed in 14); 14.4 and 15.2 still fault once the frame also holds a
24-byte escaping object and a zmm spill (fake frame of 288 bytes, no bias; GCC 16.2 asks for
320 and passes), so a passing toy is no evidence that a real kernel is safe (reproducer in
pull request 18's description). An earlier attempt set the compile-time parameter on one
unit through a source-file property and was recorded as changing no code (the reason was not
found); as a global compile option in sanitizer builds with GCC before 16 (CMakeLists.txt)
it removes the fake-stack path, and all tests pass under the runtime's defaults.

6.5 **Identical-code folding fools disassembly checks.** GCC's IPA-ICF and the Microsoft
linker's `/OPT:ICF` fold identical functions into jumps or shared addresses; a check for
non-inlined calls must accept those aliases, and an address-comparison test must link
without folding.

6.6 **Some clusters forbid what sanitizers need.** `kernel.yama.ptrace_scope=3` on compute
nodes stops LeakSanitizer even for a two-line program; disable leak checks there and cover
leaks elsewhere.

6.7 **Reading CPU features: use the OS.** On Windows on Arm, reading ID registers with
`mrs` traps; ask the OS (`IsProcessorFeaturePresent`, the registry).
[B2](experiments/B2-isa-regions-windows-clang.md).

6.8 **GCC's by-value vector ABI changes silently across targets.** Pass `-Werror=psabi`
so that a vector passed by value between differently-targeted functions is an error.
[B1](experiments/B1-isa-regions-gcc-linux.md).

6.9 **Copy builds by content, not by modification time.** `rsync -a` keeps source mtimes
and left stale remote builds (a crash, in the prototype); two jobs must never share a build
directory, and a Slurm `afterok` chain dies silently if the first job fails.

6.10 **Never pipe a gate into `tail` inside an `&&` chain**: it hides the exit status (this
let failures through twice in the prototype).

6.11 **A reviewer who has not seen your reasoning finds real bugs.** Independent second
reads of kernel changes found an unscaled single-pass phase (every element wrong for some
shapes) and missing fused-vs-unfused bit comparisons that the author's tests had missed;
an exact (`memcmp`) comparison of two paths that should agree is a cheap, strong test.

6.12 **oneMKL's `mkl-devel` wheel (2026.2.1) ships only versioned libraries.** It
installs `libmkl_core.so.3` and the like without the unversioned linker names, so a build
that looks for `-lmkl_core` under the installation fails; add the three symlinks
(`mkl_intel_lp64`, `mkl_sequential`, `mkl_core`) by hand. Likewise, copy an unpacked
autotools tree with `cp -a`: a plain `cp -r` makes the `.texi` sources newer than the shipped
manuals and `make` then wants `makeinfo` (FFTW 3.3.11). [X21](experiments/X21-phase1-cloud.md).

6.13 **Intel's icpx defaults to fast math without saying so.** Its default model
reassociates, drops signed zeros, uses approximate reciprocals and flushes denormals, but
defines no `__FAST_MATH__`, so header guards miss it; and an optimised `main()` it compiles
turns on FTZ and DAZ for the whole process. `-ffp-contract=off` alone left 55 of 1608 FFTN
tests failing (twiddles 1 ulp off). Pass `-fno-fast-math` before `-ffp-contract=off`
(`-fp-model=precise -ffp-contract=off` draws an override warning that `-Werror` makes an
error), to third-party targets too when their `main()` runs your tests.
[X22](experiments/X22-icpx.md).

6.14 **AVX2 code runs on Apple silicon under Rosetta 2 when asked.** An x86-64 build
(Apple Clang, `-arch x86_64`) executes AVX, AVX2 and FMA under Rosetta (seen on Darwin 27),
but CPUID reports none of them unless the process is started with
`ROSETTA_ADVERTISE_AVX=1` (`arch -x86_64 -e ROSETTA_ADVERTISE_AVX=1 ./test`), so runtime
dispatch picks the scalar path by default. With it, FFTN's whole x86-64 suite (1089 tests,
`avx2` forced in turn) ran on an M5 Max, guard-page faults included. AVX-512 is not
emulated; that still needs SDE or the hardware. Build the test oracles for x86-64 too (an
ExternalProject that ignores `CMAKE_OSX_ARCHITECTURES` links arm64 objects).
[X23](experiments/X23-masked-tails.md).

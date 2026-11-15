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

## 3. Memory and caches

3.1 **Buffers a power of two apart conflict in cache; offset them.** Staggering the
grid's second scratch buffer by seven cache lines made 2²² points about 8 % faster on the
Mac; padding rows of the intermediate array did nothing on top of it. The prototype found
the same with streams N/R apart at 4 KiB multiples: stagger work buffers by an odd number
of cache lines, avoid radix ≥ 16 at large power-of-two strides.
[X4](experiments/X4-padding.md).

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
FFTW's radix-32/64 three-pass plans win there. [X12](experiments/X12-beyond-l2.md).

3.6 **Misaligned wide accesses split cache lines.** With four complex doubles per AVX-512
vector, a stride ≡ 2 mod 4 makes every other access split a 64-byte line (prototype).

3.7 **Prefetchers differ by vendor.** Software next-row prefetch helped the Xeon by 25 % on
45 MB batches and cost Zen 3 and Apple a few percent (prototype).

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
[X25](experiments/X25-x86-diagnosis.md).

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
[X17](experiments/X17-bluestein-threshold.md).

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

6.4 **GCC's ASan fake stack breaks aligned AVX-512 spills.** With
`detect_stack_use_after_return`, 32-byte-aligned fake frames met 64-byte `vmovapd` spills and
segfaulted (169 tests); `ASAN_OPTIONS=detect_stack_use_after_return=0` fixed it (the
compile-time parameter did not).

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

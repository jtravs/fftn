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

## 2. The instruction set and the machine

2.1 **The data layout that wins depends on the ISA.** Interleaved complex numbers suit
NEON with FCMA, which multiplies and rotates complex values in one instruction; FFTN beats
NEON FFTW on smooth rows (0.75–1.0 for 128–100000 points) in that form. On x86 every
multiplication by ±i in interleaved form is a shuffle; the prototype's main x86 path used
split re/im instead. Split form doubles register use (radix R needs 2R vectors).
[X18](experiments/X18-x86-kernels.md), [X10](experiments/X10-pass-loops.md).

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

2.4 **Integer division is expensive enough to matter in bookkeeping.** A 64-bit `div` costs
40–90 cycles on Cascade Lake; four per chunk in the driver were about 37 % of a 64-point
transform there. Computing chunk boundaries without dividing in the common case made the
Xeon 15 % faster in geometric mean (64 points: 0.59 of the time) with identical bits. Look
for divisions and modulos in per-call and per-chunk paths, not only in kernels.
[X18](experiments/X18-x86-kernels.md).

2.5 **The compiler is rarely the explanation.** FFTN built with Clang 23 and GCC 14/16 was
within 1 % on x86 after the X18 fixes and within 7 % on NEON (two cases at 1.4 with GCC).
Test it early, because it is cheap, then look elsewhere.
[X18](experiments/X18-x86-kernels.md).

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
switching was 23 % slower. Inside L2 the pass count (L2 bandwidth) decides, which is why
FFTW's radix-32/64 three-pass plans win there. [X12](experiments/X12-beyond-l2.md).

3.6 **Misaligned wide accesses split cache lines.** With four complex doubles per AVX-512
vector, a stride ≡ 2 mod 4 makes every other access split a 64-byte line (prototype).

3.7 **Prefetchers differ by vendor.** Software next-row prefetch helped the Xeon by 25 % on
45 MB batches and cost Zen 3 and Apple a few percent (prototype).

## 4. Kernels and algorithms

4.1 **A whole-transform codelet wins only where it fills the register.** One 32-point
codelet beat the planner's two-pass chain by 16–22 % in double on NEON (one complex per
register) and lost in float and on x86 (2–8 complex per register, where a single line uses
one lane). Radix-64 codelets spill on every target. [X8](experiments/X8-leaves.md).

4.2 **Bluestein beats a direct prime pass from small primes up.** Above 16, a chirp-z
convolution through smooth-length FFTs beat the O(q²) generic pass at every prime measured
(17: 2.5 times in double, 7 in float); primes from 97 up went from 12–5000 times FFTW's
time to 0.64–1.28. Store the chirp and kernel spectrum once per (prime, sign), not once per
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

## 5. Threading

5.1 **Synchronisation costs set the smallest row worth splitting.** Spin barrier /
fork-join, ns: Mac libomp 368 / 485 at 6 threads (with an explicit block time; the default
is 14 671), Xeon libgomp 708 / 1364 at 6 and 1145 / 3165 at 18, EPYC node 688 / 1701 at 6
and 2513 / 9799 at 64. A pass of a short row takes 0.5–2 µs, so rows up to about 2k points
must not be split (prototype; FFTN's T1 measures its own).

5.2 **Keep a team inside a cache group.** On Zen 3 a team within one CCX (8 cores, shared
L3) scales one row from 2048 points; across CCXs it collapses. On the Xeon's mesh a single
row pays only from about 8k points (prototype).

5.3 **No hidden threads.** A library pool fighting the application's OpenMP team is the
failure mode of several libraries; run in the caller's threads through an executor.

5.4 **Bitwise independence of team size costs nothing and makes testing easy.** Every row
runs the same kernel whatever the split; tests compare threaded results with serial ones
bit for bit (FFTN's team tests, sizes 1–16).

## 6. Toolchain and testing pitfalls

6.1 **Build with GCC as well as Clang before a remote round trip.** GCC warns where Clang
does not (`-Wsign-conversion` on `x += c ? 1 : 0`, `-Wshadow`, CTAD); a local GCC build
catches what otherwise fails a Linux gate an hour later.

6.2 **Build with `-ffp-contract=off` when results must be reproducible.** GCC 15
contracted `a*b+c` into FMAs differently in two inlined copies of one function, so threaded
and serial results differed in the last bits (prototype).

6.3 **Sanitizer builds of large generated kernels need memory and time.** GCC's ASan on
radix-64 codelets was killed for memory with 16 parallel compiles and ran past 90 minutes
with 8; give such jobs 96–160 GB and hours, or exclude the largest kernels from
instrumentation.

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

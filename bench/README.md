# fbench

FFTN's benchmark rig. It answers "is B faster than A?" with a stated confidence, checks
every library's output against an independent reference while it times it, and stores
every result for later. The design and its reasons are in
[DESIGN.md section 16](../dev/DESIGN.md#16-benchmark-rig-fbench); this page is the manual.

**Licence.** The FFTW and KFR backends (`fbench-fftw`, `fbench-kfr`) link GPL libraries and
the MKL backend links Intel's proprietary oneMKL. Benchmark binaries are for our own
measurements and are never distributed
([AGENTS.md section 10](../AGENTS.md#10-third-party-code-and-licensing)). No code of FFTW,
KFR, benchFFT or the prototype's rig is copied here.

## Quick start

```bash
# Configure once with an FFTW installation (never a default search path: Homebrew's FFTW
# has no SIMD on arm64; on Apple silicon use a NEON build).
export FFTN_BENCH_FFTW_ROOT=/path/to/fftw-3.3.11
cmake --preset check-release && cmake --build --preset check-release

uv run --frozen python bench/fbench.py smoke --build build/check-release   # correctness
uv run --frozen python bench/fbench.py list --tier quick                   # the cases
uv run --frozen python bench/fbench.py run A=fftw B=fftw+PLANNER=PATIENT   # an A/B
uv run --frozen python bench/fbench.py show latest                         # its report
```

Without `FFTN_BENCH_FFTW_ROOT` the FFTW backend is not built, `smoke` says so, and the
gate stays green. The variable may also be given to CMake (`-DFFTN_BENCH_FFTW_ROOT=...`).
The other competitors are configured the same way; see
[Competitor backends](#competitor-backends).

## What is in this directory

| Path | What it is |
|---|---|
| `fbench.py`, `fbenchlib/` | the driver (Python 3.14, standard library only, run through uv) |
| `harness/` | the C++ harness shared by every backend: cases, buffers, timing, the check |
| `harness/backend.hpp` | the C interface a backend implements |
| `backends/<library>.cpp` | one backend per library, built as `<build>/bench/fbench-<library>` (see "Competitor backends") |
| `catalogues/*.toml` | case catalogues (`standard`, `threads`) |
| `t1/` | the harness of experiment T1 and its driver (see "The T1 harness") |
| `tests/` | the driver's self-tests (CTest `bench.selftest`) |
| `x2/` | experiment X2's radix-sequence search, a campaign of fbench runs (`x2/search.py --help`) |
| [`../scripts/remote/`](../scripts/remote/fbench_remote.py) | the Slurm runner (see "Remote runs") |

## Arms

An arm is `label=library[:build][+OPTION=value...]`. The first arm is the baseline A; every
other arm is compared with it. The label may be omitted (it defaults to the library).

- `build` is a build directory (default `--build`, itself `build/check-release`) or a git
  revision. A revision is exported with `git archive` into `bench-out/builds/<commit>-<key>/`
  and built there in Release; nothing from the working tree leaks into it.
- Options are passed to the backend, except the ones the harness or driver handles:

| Option | Applies to | Meaning |
|---|---|---|
| `PLANNER=ESTIMATE\|MEASURE\|PATIENT\|EXHAUSTIVE` | fftw | planning effort (default MEASURE) |
| `WISDOM=run\|none\|<file>` | fftw | wisdom: fresh per run and shared by that run's processes with the same binary and planner (default), none at all, or imported read-only from a file |
| `API=dft\|interleaved` | accelerate | vDSP's split-complex DFT with conversions (default) or its interleaved DFT (see below) |
| `DELAY=<fraction>` | any (test only) | spin for that fraction of every measured batch: a seeded slowdown that proves the statistics |
| `PERTURB=<eps>` | any (test only) | multiply every output value by 1 ± eps before the check: proves the correctness guard |

`DELAY` and `PERTURB` are benchmark switches, not library switches: they never reach a
library, so [`dev/KNOBS.toml`](../dev/KNOBS.toml) does not list them. A backend refuses
options it does not know. FFTN's own options are listed under [FFTN options](#fftn-options).

## `run`: how an A/B works

1. Every arm's binary is asked to `describe` itself: library, version, instruction set,
   compiler, build type and commit. fbench refuses a binary that loaded a shared library from
   outside the installation it was built against (a file reached through the installation's
   directory counts as inside it, even when that directory links elsewhere), a sanitizer or
   Debug build, and two arms that are meant to differ but report identical configurations
   (`--allow-identical`).
2. An identical copy of A, labelled `A~`, joins the run (`--no-control` drops it).
3. **Rounds.** A round launches every arm once, as a separate process, in an order from a
   Williams design (each arm in each position and after each other arm equally often). All
   arms of a round get the same case order and the same random page offsets of their
   buffers (multiples of 64 bytes), so a 4K-aliasing accident cannot decide a result.
4. **Inside a process**, each case is planned (plan time recorded separately), checked
   against the reference, calibrated to batches of `--batch-time` seconds, warmed up for
   `--warmup` seconds, and timed over `--batches` batches. The process reports every batch,
   their median and their minimum.
5. **Load.** Before and after each round fbench reads the load average and the CPU used by
   other processes (Linux `/proc/stat` over a quarter second; macOS `ps`) and stores them
   with the round. Load does not invalidate a round: the Williams order, the repeated
   process rounds and the A/A control absorb it, so on a busy machine the noise band widens
   and the verdicts follow it. A round above the quiet thresholds (load average above half
   the logical CPUs, other processes above a tenth of them and at least one core;
   `--max-load` and `--max-busy` change them) is marked "loaded", and the report counts
   such rounds. The one guard is against overload: while other processes use more than
   half the logical CPUs, a round waits, for at most `--overload-wait` seconds (default
   60), then runs anyway and is marked overloaded. With `--strict-quiet`, for runs that
   want it, a round above the thresholds is invalid instead and is repeated after
   `--busy-wait` seconds. On macOS the harness asks for the user-interactive QoS class,
   which keeps it on the performance cores.
6. **Timeouts.** A process that runs longer than `--process-timeout` seconds (default an
   hour) is stopped. The cases it finished are kept; the case it was on is stored as timed
   out and the ones it did not reach as skipped, and both are left out of that round. The
   run goes on; a case lost in every round gets the verdict `timed out`. A crash or a
   refused option still stops the run. An FFTW process stopped this way exports no wisdom,
   so the next round plans its cases again.
7. **Stopping.** After `--min-rounds` valid rounds, fbench analyses after every round and
   stops when every target case (`--target`, default all) is resolved, or at
   `--max-rounds` valid rounds or `--budget` seconds (defaults 6, 80 and an hour). On a
   shared machine more rounds are cheaper than waiting for a quiet or exclusive one: a run
   on a busy machine simply takes more rounds to resolve its cases.
8. The report is printed and the run is stored. The exit status is 1 if any output was
   wrong.

Cold mode (`--mode cold`) rotates through enough buffer sets to exceed the pool
(`--pool-bytes`, default the larger of four times the largest reported cache and 256 MiB;
Apple's system-level cache is not reported). In-place and c2r cases have their input
restored outside the timer before unnormalised growth could overflow.

## FFTN options

The FFTN backend executes serially unless an arm asks for threads. Its options combine
freely, for example `fftn+ISA=avx2+THREADS=8` or `fftn_omp+THREADS=8+CALL=team`.
`fbench-fftn_omp`, built when the compiler has OpenMP (and not under ThreadSanitizer), is
the same backend linked with the compiler's OpenMP runtime and nothing else that brings
one; its arms are written `fftn_omp`.

| Option | Backends | Meaning |
|---|---|---|
| `ISA=<name>` | fftn, fftn_omp | force every plan to one instruction set (`options::force_isa`; `scalar`, `avx2`, `avx512`, `neon`, `neon_fcma`, … as `fftn::isa_name` writes them; default `best`), so that one build compares targets: `A=fftn+ISA=avx2 B=fftn+ISA=avx512`. A target the build or processor lacks is refused; `describe` reports the plans' instruction set (`isa`) and `isa_forced` |
| `THREADS=N` | fftn, fftn_omp | execute through an executor of N lanes: `fftn::thread_pool` (fftn) or `fftn::omp::executor` (fftn_omp), created once per process |
| `GROUP_SIZE=G` | fftn, fftn_omp (with `THREADS`) | lanes per cache group the executor reports; default its own, `fftn::cache_group_size(N)`; 0 for one group |
| `CALL=executor\|team` | fftn_omp (with `THREADS`) | the executor overload (default), or a parallel region of N threads each calling the team overload with `fftn::omp::this_team()` |
| `CHUNK_POINTS=P` | fftn, fftn_omp | points per chunk of a stage (the plan's chunking) |
| `NS_PER_WORK`, `DISPATCH_NS`, `INLINE_MULTIPLE`, `SYNC_NS`, `MIN_POINTS_PER_MEMBER`, `SPLIT_MIN_POINTS`, `SPLIT_MIN_POINTS_MESH`, `MESH_GROUP_LANES`, `REMAINDER_SPLIT_MIN_POINTS`, `SPIN_NS`, `YIELD_NS`, `CLAIM_DIVISOR` | fftn, fftn_omp | the machine profile's field of that name (`src/threading/schedule.hpp`); a size threshold may be `never` |
| `RADICES=r1.r2...` | fftn, fftn_omp | the chain's radices in pass order (`build_settings::radices`), for c2c rows whose length they multiply to (see below) |
| `STRATEGY=rules\|chain\|grid` | fftn, fftn_omp | the strategy of c2c rows (`build_settings::strategy`): the planner's choice (default), or the chain or the grid ([DESIGN 8.5](../dev/DESIGN.md)) forced where the row can be one; every case's `note` says which was built. Experiment [X12](../dev/experiments/X12-beyond-l2.md) compares them |
| `GRID_FACTORS=n1.n2[.n3]` | fftn, fftn_omp | the grid's factors in phase order (`build_settings::grid_factors`), for rows they multiply to; other rows plan as without it |
| `GRID_MIN_BYTES`, `GRID_THREE_BYTES`, `GRID_BLOCK_BYTES`, `GRID_BLOCK`, `GRID_FUSE_BYTES` | fftn, fftn_omp | the planner's grid fields of the machine profile (`src/threading/schedule.hpp`): a row of more bytes than the first is a grid, of more than the second a grid of three phases; blocks widen while one fits the third; the fourth forces the block width; up to the fifth, phases with small radices read and write rows in their end passes (`never` disables the first two and makes the fifth unlimited) |

The backend builds and executes plans through the functions the public `plan` forwards
every call to, so an arm with none of these options times exactly the public plan. `ISA`
is the public `options::force_isa`. The profile and chunking options reach the library
only through the private plan core's build settings, as the tests' do: the library has no
switch for them, so `dev/KNOBS.toml` lists none. They exist for the threading experiments
([T1](../dev/experiments/T1-dispatch-wait-costs.md),
[T5](../dev/experiments/T5-split-thresholds.md)); describe reports the executor, its width
and grouping, the OpenMP runtime and its placement and waiting variables, and every tuning
option given.

`RADICES` forces FFTN's chain to a radix sequence, for example `B=fftn+RADICES=4.8.8` for
length 256, or `B=fftn+ISA=avx2+RADICES=4.8.8` for that sequence on one target: it sets
`build_settings::radices` (an internal test knob in `src/core/plan_core.hpp`, not public
API, so not in `dev/KNOBS.toml`). Every case of the run must be a c2c row (either
precision) whose length the radices multiply to; any other case is an error, and so is a
radix that no pass of the target takes. `describe` (and the run record) shows the forced
sequence; every FFTN case's `note` in the store (with or without `RADICES`)
names the radices its plan used. Experiment X2 drives it with
[`x2/search.py`](x2/search.py) ([dev/experiments/X2-radix-sets.md](../dev/experiments/X2-radix-sets.md));
its CTest `bench.x2_selftest` forces every candidate of a few lengths and checks the
refusals.

```bash
F="uv run --frozen python bench/fbench.py"
# Serial against 8 lanes of the pool and of OpenMP, on the threads catalogue.
$F run S=fftn P8=fftn+THREADS=8 O8=fftn_omp+THREADS=8 --catalogue threads
# A split-threshold sweep at 16 pinned OpenMP threads.
OMP_PLACES=cores OMP_PROC_BIND=close $F run A=fftn_omp+THREADS=16 \
  N=fftn_omp+THREADS=16+SPLIT_MIN_POINTS=never Y=fftn_omp+THREADS=16+SPLIT_MIN_POINTS=1+SYNC_NS=0 \
  --catalogue threads --filter 'rows-long|^c2c-[0-9]+-b1-'
```

On macOS new threads do not inherit the harness's user-interactive QoS class, so the
backend raises every lane of its executor to it at the start of each process.

## Reading a verdict

For each case and arm B the report shows:

- **ratio**: the median over rounds of B's time ÷ A's time in the same round (below 1 is
  faster), from each process's median batch (`--stat min` uses the minimum);
- **95 % interval**: a percentile bootstrap over rounds of that median;
- **verdict**: `slower` if the whole interval lies above 1 + band, `faster` if it lies
  below 1 − band, otherwise `no evidence`. `WRONG OUTPUT` means B's output failed the check
  in some round (`BASELINE WRONG` for A); `unsupported` means the library cannot do the case;
  `timed out` means its process hit `--process-timeout` on it (or before reaching it) in
  every round, and a note such as "2 rounds lost to timeouts" means some rounds were;
- A's and B's time per point (ns), B's mflops (5 n log₂ n / t for complex, 2.5 n log₂ n / t
  for real, the benchFFT convention) and B's worst relative RMS error.

The **noise band** at the top is the 95th percentile over cases of |log(A~ ÷ A)|, the
identical-arm control, and at least `--min-band` (1 %). It is the effect size an identical
arm reaches in 95 % of cases of this run: a claim must clear it as well as its own interval.
State it next to every effect you report, with the load line above it (median and maximum
load average and other processes' CPU use, and how many rounds ran under load): a claim is
judged against the band of the same run, whatever the load was. The **weighted geomean**
line is the catalogue-weighted geometric mean of the ratios, with a bootstrap over whole
rounds.

"No evidence" is not "equal": a case whose interval lies inside ±band is resolved as no
difference worth claiming; one whose interval is wider than the band is unresolved, and
more rounds would help.

## The correctness check

Each process executes every case once on deterministic data and compares the output with a
direct DFT evaluated in double-double arithmetic (about 32 digits) at sampled outputs of
the first and last transform of a batch, every output when `--check-budget` allows, and
checks Parseval's identity over whole rows. The reference shares no code with any library.
It reports relative RMS and maximum errors; the case passes when both are within
10 ε (1 + log₂ n) (8 times that for the maximum), with ε the epsilon of the case's
precision. A single-precision input is the double-precision one rounded to float, and the
reference transforms the rounded values. Accuracy is stored with every timing.

## `smoke`

`smoke --build DIR --strict` runs every backend in `DIR/bench` through its own `selftest`
(the reference, the check, and that `describe` names the real library file rather than the
backend executable) and through the smoke tier, correctness only, then checks that the
guard flags a perturbed output. It is the gate's `bench` stage and the CTest test
`bench.smoke`. An unsupported case fails when its kind is in the backend's `supports` list
(and, for a single-precision case, `f32` is in its `precisions`) and is only counted
otherwise. A
kind in the backend's `partial` list (supported for some lengths or ranks only, as with
Accelerate) may be unsupported case by case, but must pass at least one case of the
standard smoke tier (other selections need not contain lengths it can do). A library with
several code paths behind an option is smoke-tested once per path (`SMOKE_OPTIONS` in
`fbenchlib/cli.py`): FFTN serially, on three lanes at the default profile, and on three
lanes with a profile that splits everything it can into small chunks (with OpenMP also
through the team overload), so that the threaded paths meet the reference check.

## Catalogues

A catalogue is a TOML file of case groups; each group is the product of its lists:

```toml
[catalogue]
name = "example"

[[group]]
name = "rows"
kind = ["c2c", "r2c", "c2r"]   # default c2c
dims = [64, 1000, "64x48"]       # lengths, or "n0xn1[xn2]" for planes and volumes
batch = [1, 256]                 # default 1
place = ["out", "in"]            # default out
direction = "fwd"                # c2c only: fwd or bwd
precision = ["f64", "f32"]       # default f64
weight = 1.0                     # share in the weighted geomean (default 1)
tiers = ["quick", "tracking"]    # every case is also in "full"
```

Case IDs are `kind-dims-bBATCH-op|ip[-bwd][-f32]`, for example `r2c-64x48-b256-ip` or
`c2c-1024-b1-op-f32`. A case listed by several groups is one case with the union of their
tiers (its weight must agree).
A backend takes single-precision cases only when its `describe` lists `"f32"` under
`precisions` (FFTN does; the competitors do not yet); for the others the harness reports
them unsupported, and `smoke` does not count that as a failure.
Layouts follow FFTW: rows and batches contiguous, r2c output n/2+1 complex values in the
last dimension, in-place real arrays padded to 2(n/2+1).

The standard catalogue has tiers `smoke` (seconds, correctness, every kind, placement and
rank), `quick` (an A/B in a few minutes), `tracking` (a fixed set for history) and `full`.
The `threads` catalogue holds the shapes that exercise threading's rules (single long rows,
few rows, leftover rows after whole rounds, many rows, small jobs, planes and volumes), for
threaded arms; its groups are named after the rule they exercise.
`--catalogue` takes a name under `catalogues/` or any path, so client catalogues can live
anywhere.

## The store

Every run writes `bench-out/runs/<run-id>.jsonl` (`bench-out/` is git-ignored): a `run`
record (arms with their `describe`, binary and commit; host fingerprint: CPU model, core
counts, OS and caches, never the hostname; catalogue path and hash; settings), one `round`
record per attempt (order, offsets, load before and after, loaded or overloaded, the
seconds it waited, arms whose process timed out, valid or why not), one `timing`
record per round, arm and case (every batch, median, minimum, plan time, accuracy, ns per
point, mflops, and the backend's note on what it planned, if any) and a final `summary`.
`show <run-id>` (or a unique prefix, or `latest`) recomputes the report from the records. Documents cite run IDs, never copied numbers.

The store may be copied between machines, so it holds no absolute paths: the checkout is
written `<repo>`, the directory holding `bench-out` `<base>`, a library installation
`<fftw-root>` (`<fftw-root-2>`, ... for further ones) and the home directory `~`. The run
record also names the Python version (the checkout pins 3.14 in `.python-version`).

## Competitor backends

Every competitor is built the way its best users build it (AGENTS.md section 6) and only
when configured, by a CMake or environment variable, never from a default search path. Each
backend's source starts with what it times besides the library, and `describe` reports the
library's version and the instruction set it selected.

| Backend | Platforms | Configure with | Licence |
|---|---|---|---|
| `fftw` | all | `FFTN_BENCH_FFTW_ROOT=<install>` | GPL-2.0-or-later |
| `pocketfft` | all | `FFTN_BENCH_POCKETFFT=ON` (fetched) | BSD-3-Clause |
| `kfr` | all | `FFTN_BENCH_KFR_ROOT=<install>` | GPL-2.0-or-later or commercial |
| `accelerate` | macOS | `FFTN_BENCH_ACCELERATE=ON` | system |
| `mkl` | Linux x86-64 | `FFTN_BENCH_MKL_ROOT=<oneMKL root>` | Intel Simplified Software License |
| `fftl` | macOS, Linux | `FFTN_BENCH_FFTL_ROOT=<libfftl directory>` | MIT |

Each configured backend is smoke-tested by `bench.smoke` and has its own CTest
`bench.harness_selftest_<library>`. With none configured, only the FFTN backend is built.

**FFTW 3.3.11**: an installation built with the SIMD and FMA options for the machine (NEON
on Apple silicon, whose `fftw_version` ends in `-neon`; SSE2 to AVX-512 and FMA on x86-64),
never Homebrew's (no SIMD on arm64). MEASURE by default, wisdom fresh per run; `describe`
reports the SIMD extensions FFTW was built with.

**pocketfft** (the C++ header-only version, `pocketfft_hdronly.h` of
[mreineck/pocketfft](https://github.com/mreineck/pocketfft), branch `cpp`, commit
`4d865cb`): fetched by CMake pinned by SHA-256; offline, set
`FETCHCONTENT_SOURCE_DIR_FBENCH_POCKETFFT` to a directory holding the header. It is compiled
into the backend as SciPy uses it: the high-level `c2c`/`r2c`/`c2r` over byte-strided arrays,
with a plan cache of 16 (`POCKETFFT_CACHE_SIZE`), one thread, FMA contraction allowed
(`-ffp-contract=fast`), and on x86-64 `-march=native` (`FFTN_BENCH_POCKETFFT_ARCH_FLAGS`
overrides; the binary is run where it is built). `fbench_create` fetches each axis's plan
from the cache, so twiddle computation is plan time.

**KFR 7.2.0**, from KFR's own CMake build with Clang, installed:

```bash
curl -LO https://github.com/kfrlib/kfr/archive/refs/tags/7.2.0.tar.gz
shasum -a 256 7.2.0.tar.gz   # 4f1603d5b6e609c25d25cff58fe7154a371583fd6d379615b342e44223be0763
tar xzf 7.2.0.tar.gz
cmake -S kfr-7.2.0 -B kfr-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
  -DCMAKE_INSTALL_PREFIX=$HOME/buildtools/kfr-7.2.0 \
  -DENABLE_TESTS=OFF -DKFR_ENABLE_DFT=ON -DKFR_ENABLE_AUDIO=OFF -DKFR_ENABLE_IO=OFF
cmake --build kfr-build && cmake --install kfr-build
export FFTN_BENCH_KFR_ROOT=$HOME/buildtools/kfr-7.2.0
```

On x86-64 KFR's run-time dispatch is on by default (`KFR_ENABLE_MULTIARCH`, targets sse2,
sse41, avx, avx2, avx512): the configure log says "Runtime dispatch is enabled" and
`describe` lists the installed targets under `archs`. AArch64 has one target, neon64. The
plans are KFR's `dft_plan`, `dft_plan_real` (CCs packing, FFTW's n/2+1 layout),
`dft_plan_md` and `dft_plan_md_real`, with the scratch buffer allocated at plan time; a
batch is a loop over transforms. An in-place real plane or volume is copied to a contiguous
real array (KFR's multidimensional real transforms take one), and that copy is timed.

**Apple Accelerate** (vDSP): rows only, of the lengths vDSP implements (planes, volumes and
other lengths are unsupported; `describe` lists the kinds as `partial`). `API=dft`
(default) uses Apple's recommended split-complex `vDSP_DFT_zop`/`zrop` with `vDSP_ctozD` and
`vDSP_ztocD` around each row, conversions timed (lengths 2^k and 3, 5 or 15 times 2^k).
`API=interleaved` uses `vDSP_DFT_Interleaved_ExecuteD` on the caller's interleaved data with
no conversion, but on macOS 27 it refuses every complex length above 4096 and some below
(1536, 3072), so it covers few realistic shapes and is not the default. Real transforms
convert vDSP's packed format (2 X[0] and 2 X[n/2] in the first value, 2 X[k] after it) to
FFTW's inside the timed call. Rows are executed one by one; vDSP has no batched DFT.

**Intel oneMKL** (Linux x86-64; compiled against the headers of 2025.2 and 2026.2): the root
of an install (oneAPI's `mkl/<version>`, or a `mkl-devel` environment with `include/` and
`lib/`). The backend links the link-line advisor's one-thread choice, `mkl_intel_lp64`,
`mkl_sequential` and `mkl_core`, dynamically, with an rpath to the root; do not source
`setvars.sh` for runs (it would put another oneMKL first on the library path, which the
loaded-library check refuses). One DFTI descriptor per case, for its one direction: the
batch, the layout through `DFTI_INPUT_`/`OUTPUT_STRIDES` and `_DISTANCE` (which every
release has; 2025.2 and 2026.1 lack the newer `DFTI_FWD_`/`BWD_STRIDES`), and
`DFTI_COMPLEX_COMPLEX` storage, with no scaling. `describe` reports the code path oneMKL
chose (`isa`, from `MKLVersion::Processor`) and `MKL_ENABLE_INSTRUCTIONS` and `MKL_CBWR`,
which must be unset for a fair run.

**FFTL.jl** through libfftl, its C library: FFTL.jl compiled ahead of time with juliac into a
self-contained shared library (the Julia runtime is embedded; nothing of Julia is needed at
run time) that implements FFTW's C interface. Building it needs Julia 1.12 or newer (with
juliaup, `juliaup add 1.13`) and takes about five minutes:

```bash
git clone https://github.com/jtravs/FFTL.jl && cd FFTL.jl && git checkout 53171e8
cd libfftl && ./build.sh        # JULIA_CPU_TARGET: apple-m1 on macOS, x86-64-v3 on x86-64
export FFTN_BENCH_FFTL_ROOT=$PWD
```

Its instruction set is fixed by `JULIA_CPU_TARGET` at build time and cannot be read from the
library: set `FFTN_BENCH_FFTL_CPU_TARGET` if the build used another target than
`build.sh`'s default, so that `describe` says so (on a Xeon, `JULIA_CPU_TARGET=cascadelake`
builds for AVX-512). Planner flags and wisdom do nothing in FFTL; plans use one thread. The
backend is not built in sanitizer builds: the embedded runtime handles SIGSEGV for its
garbage collector, which the sanitizers' own handlers break.

## Adding a backend

1. Write `backends/<library>.cpp` implementing the six functions of `harness/backend.hpp`
   (`backends/common.hpp` has the shared helpers).
   Plan in `fbench_create` on the buffers given (their alignment is the one every execute
   will use), refuse unknown options in `fbench_begin`, and report in `fbench_describe` the
   version, the instruction set actually selected, `supports` (the kinds it claims for
   every shape) and, if any, `partial` (kinds it supports for some lengths or ranks);
   for a shared library also `loaded_from` and `expected_root`. `loaded_from` must name the
   file that defines the library: on Linux, the address of a symbol taken in the
   executable points into the executable (PLT entry or copy relocation), so resolve a
   function with `dlsym(RTLD_NEXT, ...)` first, as the FFTW backend does.
2. In `CMakeLists.txt`, find the library only where the user points
   (`FFTN_BENCH_<LIB>_ROOT`), call `fbench_add_backend(<library> <link targets>)`, and skip
   it with a status message when it is not configured.
3. Convert conventions inside the backend (packed formats, scaling) so the check stays
   meaningful, and run `smoke`.

One library per executable: libraries with the FFTW API share symbol names, and two OpenMP
runtimes in one process crash or crawl.

## The T1 harness

`bench/t1/` holds the harness of experiment
[T1](../dev/experiments/T1-dispatch-wait-costs.md): the fixed costs of team executions,
which fbench cannot isolate. It is built as `<build>/bench/t1_costs` and, where the
compiler has OpenMP (not under ThreadSanitizer), `t1_costs_omp`, which adds
`fftn::omp::executor` and the threads of a parallel region. Probes: `work` (serial time per
unit of the cost model's work), `bulk` (an executor's dispatch: time until every lane runs,
and from the last task's end to the return), `team` (a minimal team execution through the
executor overload, and through the team overload with every member already running),
`phase` (the claiming engine on a synthetic task of 64 phases: time per phase hand-off),
`claim` (one compare-exchange claim under contention) and `wake` (notify-to-running latency
of a spinning, yielding, blocked waiter and of the engine's policy after each waiting
time). Each prints one JSON line per probe, team size and variant, with the median and
spread over repetitions; `--pin` pins the harness's own threads one per allowed CPU
(Linux), the pool's threads are never pinned, OpenMP's follow `OMP_PLACES`.

```bash
T="uv run --frozen python bench/t1/t1.py"
$T run --build build/check-release --rounds 10 -- --threads 2,4,8,16,32,64 --pin
$T show latest
# What the scheduler decides for each catalogue case at 8, 16 and 64 lanes in groups of 8,
# on a machine of 128 CPUs, under any tuning option of fbench's FFTN backend:
$T schedule --catalogue threads --tier quick -- --threads 8,16,64 --group 8 --cpus 128 \
  SPLIT_MIN_POINTS=4096
```

`run` launches each binary once per round in rotating order, checks before each round that
the machine is quiet (as fbench does; `--no-quiet-check` skips it), stores every process's
output with the host fingerprint and the placement variables in
`bench-out/t1/runs/<run-id>.jsonl`, and prints the median over rounds of each in-process
median with the range over rounds. The CTest tests `bench.t1_smoke` and `bench.t1_schedule`
run both commands in their short form; they check that every probe reports, not what it
measures.

## Remote runs

`scripts/remote/fbench_remote.py` runs fbench on a Slurm machine in one job, on an exclusive
node unless the profile says `exclusive = false` (a shared machine whose other jobs must keep
running; ask for a few cores with `-c`). Describe each machine in an untracked file, `~/.config/fftn/remote.toml` (or
`$FFTN_REMOTE_CONFIG`):

```toml
[profile.cluster]
ssh = "user@login-node"                 # or an ~/.ssh/config alias
workdir = "/scratch/user/fftn-bench"    # each job gets its own directory below this
partition = "compute"
fftw_root = "/path/to/fftw-3.3.11"      # built with its best SIMD and FMA options
setup = ["source /path/to/toolchain.sh"]  # compiler, CMake and uv on the compute node
sbatch = ["--time=02:00:00"]             # further #SBATCH options
cxx = "g++-14"                           # optional
exclusive = false                        # optional: share the node (default: whole node)
```

`FFTN_REMOTE_SSH`, `FFTN_REMOTE_WORKDIR`, `FFTN_REMOTE_PARTITION`, `FFTN_REMOTE_FFTW_ROOT` and
`FFTN_REMOTE_SETUP` override the file. uv must be able to provide Python 3.14 on the compute
node (install it once on shared storage if the nodes have no network).

```bash
R="uv run --frozen python scripts/remote/fbench_remote.py"
$R submit cluster A=fftw B=fftn:main -- --tier quick   # arms' builds must be commits
$R log <job>      # check the first job's log as soon as it starts
$R status <job>
$R fetch <job>    # copies the run into bench-out/runs/; then fbench show <run-id>
```

`submit` refuses a dirty working tree, packs every commit an arm needs with `git archive`
(by content: `rsync -a` keeps mtimes and has left stale builds), unpacks each into its own
directory in the job's directory, builds each into its own build directory, and records
the commit of every arm in `bench-out/remote/<job>/manifest.json` and in the run itself.
`--dry-run` writes the job files without contacting the machine. `--campaign SCRIPT` runs a
campaign driver of the checkout in the job instead of one `fbench run` (a driver that runs
many, such as experiment X2's): the arms then only name the builds it needs, the driver is
called as `SCRIPT --out-dir <job>/bench-out --build <job>/build-<head> <arguments after -->`,
and `fetch` also copies its state from `bench-out/campaigns/`. `--time` replaces the
profile's Slurm time limit for one job. `--after <jobid>` queues a
job behind another with `afterok`: if the first job fails, the second never starts and
Slurm reports it only in `squeue` (DependencyNeverSatisfied), so always read the first
job's log before waiting on a chain.

## Limits of v0

- Contiguous rows, planes and volumes. Strided axes come with the phases that need them.
  Single precision is timed for FFTN only, and threaded arms exist for FFTN only
  (competitors' own threading and a team of serial plans, DESIGN.md 16.1, come with
  phase 3's exit).
- Cold mode rotates data buffers, not the libraries' plan tables (FFTW shares twiddle
  tables between plans, so they cannot be rotated from outside).
- Two FFTN plan options in one process (same-process A/B, DESIGN.md 16.3) waits for FFTN's
  plan options.

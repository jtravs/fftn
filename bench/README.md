# fbench

FFTN's benchmark rig. It answers "is B faster than A?" with a stated confidence, checks
every library's output against an independent reference while it times it, and stores
every result for later. The design and its reasons are in
[DESIGN.md section 16](../dev/DESIGN.md#16-benchmark-rig-fbench); this page is the manual.

**Licence.** The FFTW backend (`fbench-fftw`) links FFTW, which is GPL. Binaries that link
it are for our own measurements and are never distributed
([AGENTS.md section 10](../AGENTS.md#10-third-party-code-and-licensing)). No code of FFTW,
benchFFT or the prototype's rig is copied here.

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

## What is in this directory

| Path | What it is |
|---|---|
| `fbench.py`, `fbenchlib/` | the driver (Python 3.14, standard library only, run through uv) |
| `harness/` | the C++ harness shared by every backend: cases, buffers, timing, the check |
| `harness/backend.hpp` | the C interface a backend implements |
| `backends/<library>.cpp` | one backend per library, built as `<build>/bench/fbench-<library>` |
| `catalogues/*.toml` | case catalogues |
| `tests/` | the driver's self-tests (CTest `bench.selftest`) |
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
| `DELAY=<fraction>` | any (test only) | spin for that fraction of every measured batch: a seeded slowdown that proves the statistics |
| `PERTURB=<eps>` | any (test only) | multiply every output value by 1 ± eps before the check: proves the correctness guard |

`DELAY` and `PERTURB` are benchmark switches, not library switches: they never reach a
library, so [`dev/KNOBS.toml`](../dev/KNOBS.toml) does not list them. A backend refuses
options it does not know.

## `run`: how an A/B works

1. Every arm's binary is asked to `describe` itself: library, version, instruction set,
   compiler, build type and commit. fbench refuses a binary that loaded a shared library from
   outside the installation it was built against, a sanitizer or Debug build, and two arms
   that are meant to differ but report identical configurations (`--allow-identical`).
2. An identical copy of A, labelled `A~`, joins the run (`--no-control` drops it).
3. **Rounds.** A round launches every arm once, as a separate process, in an order from a
   Williams design (each arm in each position and after each other arm equally often). All
   arms of a round get the same case order and the same random page offsets of their
   buffers (multiples of 64 bytes), so a 4K-aliasing accident cannot decide a result.
4. **Inside a process**, each case is planned (plan time recorded separately), checked
   against the reference, calibrated to batches of `--batch-time` seconds, warmed up for
   `--warmup` seconds, and timed over `--batches` batches. The process reports every batch,
   their median and their minimum.
5. **Quiet machine.** Before and after each round fbench reads the load average and the CPU
   used by other processes (Linux `/proc/stat` over a quarter second; macOS `ps`). A round on
   a busy machine is invalid and is repeated after `--busy-wait` seconds. Defaults: load
   average at most half the logical CPUs, other processes at most a tenth of them (at least
   one core); `--max-load` and `--max-busy` change them. On macOS the harness asks for the
   user-interactive QoS class, which keeps it on the performance cores.
6. **Stopping.** After `--min-rounds` valid rounds, fbench analyses after every round and
   stops when every target case (`--target`, default all) is resolved, or at
   `--max-rounds` valid rounds or `--budget` seconds.
7. The report is printed and the run is stored. The exit status is 1 if any output was
   wrong.

Cold mode (`--mode cold`) rotates through enough buffer sets to exceed the pool
(`--pool-bytes`, default the larger of four times the largest reported cache and 256 MiB;
Apple's system-level cache is not reported). In-place and c2r cases have their input
restored outside the timer before unnormalised growth could overflow.

## Reading a verdict

For each case and arm B the report shows:

- **ratio**: the median over rounds of B's time ÷ A's time in the same round (below 1 is
  faster), from each process's median batch (`--stat min` uses the minimum);
- **95 % interval**: a percentile bootstrap over rounds of that median;
- **verdict**: `slower` if the whole interval lies above 1 + band, `faster` if it lies
  below 1 − band, otherwise `no evidence`. `WRONG OUTPUT` means B's output failed the check
  in some round (`BASELINE WRONG` for A); `unsupported` means the library cannot do the case;
- A's and B's time per point (ns), B's mflops (5 n log₂ n / t for complex, 2.5 n log₂ n / t
  for real, the benchFFT convention) and B's worst relative RMS error.

The **noise band** at the top is the 95th percentile over cases of |log(A~ ÷ A)|, the
identical-arm control, and at least `--min-band` (1 %). It is the effect size an identical
arm reaches in 95 % of cases of this run: a claim must clear it as well as its own interval.
State it next to every effect you report. The **weighted geomean** line is the catalogue-
weighted geometric mean of the ratios, with a bootstrap over whole rounds.

"No evidence" is not "equal": a case whose interval lies inside ±band is resolved as no
difference worth claiming; one whose interval is wider than the band is unresolved, and
more rounds would help.

## The correctness check

Each process executes every case once on deterministic data and compares the output with a
direct DFT evaluated in double-double arithmetic (about 32 digits) at sampled outputs of
the first and last transform of a batch, every output when `--check-budget` allows, and
checks Parseval's identity over whole rows. The reference shares no code with any library.
It reports relative RMS and maximum errors; the case passes when both are within
10 ε (1 + log₂ n) (8 times that for the maximum), with ε the double-precision epsilon.
Accuracy is stored with every timing.

## `smoke`

`smoke --build DIR --strict` runs every backend in `DIR/bench` through its own `selftest`
(the reference, the check, and that `describe` names the real library file rather than the
backend executable) and through the smoke tier, correctness only, then checks that the
guard flags a perturbed output. It is the gate's `bench` stage and the CTest test
`bench.smoke`. An unsupported case fails when its kind is in the backend's `supports` list
and is only counted otherwise (the FFTN backend supports nothing until it has a plan).

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
weight = 1.0                     # share in the weighted geomean (default 1)
tiers = ["quick", "tracking"]    # every case is also in "full"
```

Case IDs are `kind-dims-bBATCH-op|ip[-bwd]`, for example `r2c-64x48-b256-ip`. A case listed
by several groups is one case with the union of their tiers (its weight must agree).
Layouts follow FFTW: rows and batches contiguous, r2c output n/2+1 complex values in the
last dimension, in-place real arrays padded to 2(n/2+1).

The standard catalogue has tiers `smoke` (seconds, correctness, every kind, placement and
rank), `quick` (an A/B in a few minutes), `tracking` (a fixed set for history) and `full`.
`--catalogue` takes a name under `catalogues/` or any path, so client catalogues can live
anywhere.

## The store

Every run writes `bench-out/runs/<run-id>.jsonl` (`bench-out/` is git-ignored): a `run`
record (arms with their `describe`, binary and commit; host fingerprint: CPU model, core
counts, OS and caches, never the hostname; catalogue path and hash; settings), one `round`
record per attempt (order, offsets, load before and after, valid or why not), one `timing`
record per round, arm and case (every batch, median, minimum, plan time, accuracy, ns per
point, mflops) and a final `summary`. `show <run-id>` (or a unique prefix, or `latest`)
recomputes the report from the records. Documents cite run IDs, never copied numbers.

The store may be copied between machines, so it holds no absolute paths: the checkout is
written `<repo>`, the directory holding `bench-out` `<base>`, a library installation
`<fftw-root>` (`<fftw-root-2>`, ... for further ones) and the home directory `~`. The run
record also names the Python version (the checkout pins 3.14 in `.python-version`).

## Adding a backend

1. Write `backends/<library>.cpp` implementing the six functions of `harness/backend.hpp`.
   Plan in `fbench_create` on the buffers given (their alignment is the one every execute
   will use), refuse unknown options in `fbench_begin`, and report in `fbench_describe` the
   version, the instruction set actually selected and `supports` (the kinds it claims);
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

## Remote runs

`scripts/remote/fbench_remote.py` runs fbench on a Slurm machine in one job on an exclusive
node. Describe each machine in an untracked file, `~/.config/fftn/remote.toml` (or
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
`--dry-run` writes the job files without contacting the machine. `--after <jobid>` queues a
job behind another with `afterok`: if the first job fails, the second never starts and
Slurm reports it only in `squeue` (DependencyNeverSatisfied), so always read the first
job's log before waiting on a chain.

## Limits of v0

- One thread, double precision, contiguous rows, planes and volumes. Strided axes, thread
  catalogues and `float` come with the phases that need them.
- Cold mode rotates data buffers, not the libraries' plan tables (FFTW shares twiddle
  tables between plans, so they cannot be rotated from outside).
- Two FFTN plan options in one process (same-process A/B, DESIGN.md 16.3) waits for FFTN's
  plan options.
- The FFTN backend reports every case unsupported until task 0.5's plan exists; what is left
  to wire is listed at the top of `backends/fftn.cpp`.

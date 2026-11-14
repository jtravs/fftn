# AGENTS.md

Rules for everyone who changes FFTN, people and coding agents alike. Read
[dev/DESIGN.md](dev/DESIGN.md) for what we are building and why, and
[dev/PLAN.md](dev/PLAN.md) for what to do next. Tested code and build configuration are
authoritative: when a document disagrees with them, correct the document in the same change.

## 1. Product in one paragraph

FFTN is a C++20 FFT library for CPUs: batched 1-D rows, 2-D planes and 3-D volumes; complex
and real transforms; any length; fused pointwise work; threading on the caller's threads.
The goal is the fastest open-source single-thread performance on Apple M-series, x86-64
(AVX2, AVX-512) and AArch64 Linux, and threading with no pathological cases. The public API
is `include/fftn/*.hpp`; everything under `src/` is private. FFTN is a compiled static
library with no dependencies. Conventions are FFTW's (signs, no normalisation by default, r2c writes
n/2+1 values). A shape is never silently computed by a slower route without the plan
saying so (`plan::describe()`).

FFTN is unreleased. Until the first tagged release the public API may change in any way: a
choice that turns out wrong is fixed, not kept for compatibility. API changes still need the
maintainer's agreement (section 7).

## 2. Sources of truth

One fact lives in one place. Do not restate it elsewhere; link to it.

| Fact | Where it lives |
|---|---|
| Goals, architecture, contracts, decisions and their reasons | `dev/DESIGN.md` (decision log in its last section) |
| Phases, tasks, who is doing what, status | `dev/PLAN.md` ("Status" is updated in every working session) |
| Experiments: hypothesis, protocol, result, verdict | `dev/experiments/<ID>-<slug>.md`, one file each, indexed in `dev/PLAN.md` |
| Experimental switches | `dev/KNOBS.toml` (owner and expiry date; the gate enforces it) |
| Benchmark numbers | the fbench results store, cited by run ID; never copied into documents by hand |
| Public API (signatures, preconditions, guarantees) | comments in the public headers |
| Conventions and usage | `docs/*.md`, with examples quoted from compiled files in `examples/` |

`dev/DESIGN.md` describes the design as it is intended now. When work changes the design,
edit the affected section and add a line to the decision log. Do not append history to the
sections themselves.

## 3. The gate

```bash
scripts/check.sh                                   # the merge gate
FFTN_CHECK_STAGES=all scripts/check.sh             # every stage
FFTN_CHECK_STAGES="format build" scripts/check.sh  # some of them
```

Stages are listed at the top of `scripts/check.sh`. While FFTN is an unreleased experiment
the bar for merging is that the code runs and is correct: the merge gate (`spdx gen build
sanitize bench`: Release tests and ASan + UBSan tests under every kernel variant the CPU
has) green on the machine where the change was made, plus `build` with one GCC version on
Linux x86-64. Style, lint, documentation completeness, coverage, ThreadSanitizer (except
for threading changes, which run `tsan` too), the second GCC version and Windows run with
`all` in CI on `main` and in a periodic full gate; their failures are fixed in batches and
do not block merges. Second readers are for kernels, the generator and the threading
engine. In addition:

- A kernel, vocabulary or generator change for an instruction set the machine lacks also
  needs `build` and `sanitize` on a machine that has it (see section 9). Until emulated runs
  exist in CI, that means AVX2 and AVX-512 on the Xeon or the EPYC cluster, and
  NEON+FCMA on the Mac.
- Report the compiler, the instruction sets exercised, the build type and every stage that
  was skipped or did not apply. "Green" without that list is not a report.
- Never disable or delete a test, widen a tolerance after seeing a failure, lower a warning
  level or hide a diagnostic to pass the gate. If a test is wrong, fix the test and say why
  in the commit.
- A `NOLINT` names its check and gives its reason on the same line.
- Never pipe the gate into `tail` or `head` inside an `&&` chain: it masks the exit status.

## 4. C++ rules

- **Standard.** C++20, extensions off. Must compile cleanly with GCC 14 and newer, Clang 16
  and newer, Apple Clang, and clang on the MSVC ABI. MSVC's own compiler is not a target.
  Warnings: `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow`, as errors
  in the gate. Third-party code is `SYSTEM`; its warnings are not ours.
- **Dependencies.** The library has none beyond the standard library. Tests may use
  GoogleTest, FFTW, Boost.Multiprecision and Kokkos (for the optional `fftn/kokkos.hpp`);
  benchmarks may use any competitor. Adding a dependency to the
  library needs the maintainer's approval and a decision-log entry.
- **Ownership and types.** RAII only: no owning raw pointers, no `new`/`delete`. Sizes and
  indices are `std::size_t` or `std::ptrdiff_t` (strides are signed); convert with the
  helpers in `detail/`, not scattered casts.
- **The execute contract.** A core execute call does not allocate, lock, do I/O or throw
  (only the convenience layer's cached plans and workspaces may allocate, on first use). All
  validation, allocation and choice happen at plan construction, which reports misuse with
  `std::invalid_argument` naming the offending argument. Plans are immutable after
  construction and safe to execute concurrently on different data. Scratch is per
  execution context, never hidden mutable state in a `const` plan.
- **Instruction sets.** No translation unit is compiled with a global ISA flag
  (`-march`, `-mavx2`, `/arch`). ISA-specific code lives only in the per-ISA regions set
  up by `src/isa/` (target attributes), inside a per-ISA namespace. Inside such a region
  use only the vector vocabulary and built-in types: no standard-library templates, no
  lambdas, no vector types across a function boundary that is not force-inlined, and
  never the first inclusion of a header. Kernel text is written once in `.inl` files and
  included per region; it is not a template over a vocabulary type. Dispatch happens
  once, at plan construction; never per call.
- **Floating point.** No fast-math, no `-ffinite-math-only`, no flush-to-zero, no
  approximate reciprocals. Fused multiply-adds are written explicitly in the vocabulary;
  the build sets `-ffp-contract=off` so the compiler adds none. Public headers refuse
  `__FAST_MATH__`.
- **Determinism.** For a given plan, the result is bitwise identical however the work is
  shared out: serial, any team size, any executor, any schedule. Tests hold this exactly.
  What may change bits is listed in `dev/DESIGN.md` ("Determinism") and nowhere else.
- **Generated code.** Everything under `src/gen/` (codelets and their tables) is written by the
  generator in `gen/` and committed. Never edit them by hand: change the generator and
  regenerate. The `gen` stage fails if the tree and the generator disagree.
- **No dead code.** A path that loses is deleted, not parked behind a switch; git keeps it.
  Experimental switches follow section 6.
- **Naming and layout.** `snake_case` for everything except template parameters
  (`PascalCase`) and macros (`FFTN_UPPER`). Macros only for ISA regions, force-inline and
  feature detection. One concept per header.
- **Comments and documentation.**
  - Every public declaration has a comment giving what it does, its preconditions and
    its guarantees. The public headers are the API reference; the gate checks that
    nothing public is undocumented.
  - Internal code, C++ and Python alike, gets no boilerplate docstrings. It gets a
    comment wherever something is subtle or was learned the hard way, stating the reason
    and citing the experiment or finding. Comments say why, not what.
  - A change to the public API or to a convention updates `docs/` and the examples in
    the same commit. Examples in documents are quoted from files under `examples/` that
    the build compiles and runs; never paste untested code into a document.
- **Python.** The generator and all tools target Python 3.14 or newer and run in the
  project's `uv` environment. Tools use the standard library only; the generator may
  also use `mpmath`.

## 5. Correctness and tests

- **Oracles, in order.** (1) Analytic identities: impulses, single bins, real tones,
  Parseval, shift, linearity, round trips. (2) A direct DFT evaluated in quadruple
  precision with Boost.Multiprecision (`long double` and `__float128` are not available
  on every platform, so they are not the reference).
  (3) FFTW as an independent implementation. The generator additionally verifies every
  codelet against an exact DFT before emitting it.
- **Tolerances** are computed from machine epsilon, the length and the algorithm's error
  bound before the test runs. They are tight enough that a wrong twiddle or a lost FMA
  fails.
- **Both precisions.** Everything is written generically over the real type. `float` is
  instantiated and tested alongside `double` from the first kernels on, at least on the
  scalar target and one SIMD target per architecture.
- **Every path is exercised.** Each kernel variant, instruction set, pass kind and planner
  strategy the build contains is forced in turn by the parameterised suites. A new path
  comes with its forcing parameter. Coverage of hand-written code is measured by the gate
  and may not fall.
- **Contracts are tested, not assumed:** untouched memory outside the described rows
  (canaries), any alignment, in-place and out-of-place, strides and distances, input
  preservation, bitwise equality of threaded and serial results, plan reuse across threads.
- New behaviour needs a test. A bug fix needs a regression test that fails without the fix.

## 6. Experiments and performance work

Performance changes are experiments until measured. The protocol:

1. **Write the experiment file first** (`dev/experiments/<ID>-<slug>.md`, from
   `dev/experiments/TEMPLATE.md`): hypothesis, the single change, target cases, primary
   metric, decision rule, machines. Add it to the index in `dev/PLAN.md`.
2. **One change per experiment.** Every arm is a commit; never benchmark a dirty tree.
3. **Measure with fbench A/B** (interleaved arms, process rounds, paired confidence
   intervals, an identical-arm control for the noise floor). Compare arms only within one
   session on one machine. State the noise floor next to every claimed effect.
4. **Check for collateral damage:** run the tracking tier and look at cases outside the
   target set before concluding.
5. **Record the verdict in the experiment file**, negative results included, with run IDs,
   host, compiler, instruction set, thread mode and load. A result that is not written
   down did not happen.
6. **Then act:** a win becomes the default and the old path is deleted; a loss is deleted.
   A switch kept for further study is registered in `dev/KNOBS.toml` with an owner and an
   expiry date at most 30 days ahead. The gate fails on expired or unregistered switches.

Further rules:

- Never add `-march=native` to anything distributable. Benchmarks of FFTN use the same
  build a user would get.
- Competitors are built the way their best users would build them (FFTW with its SIMD and
  FMA options at MEASURE and PATIENT, fresh wisdom per campaign; the NEON build on Apple).
  A comparison against a handicapped competitor is a bug in the benchmark.
- The Mac and the other machines are shared. fbench runs interleaved, with repeated rounds
  and an A/A control, on any machine, and records the load of every round; a claim is
  judged against the noise band of the same run. Prefer an exclusive Slurm node when one is
  available, and say where a run was made.
- A default that changes needs an experiment file. So does a planner rule.

## 7. Working in this repository

- **Picking up work.** Read `dev/PLAN.md` "Status", then the phase you are in. Claim a task
  by writing your session's short name and the date beside it, in its own small commit when
  several agents are active. Release the claim when you stop, with a handoff note under
  "Status": what is done, what is half done, what you would do next.
- **One task, one branch.** Parallel agents work in separate git worktrees on separate
  branches. Merge only gate-green work.
- **Sub-agents.** Use them for independent reading, review and parallel implementation
  tasks with disjoint files. Brief them with the goal, the files, what is already ruled out
  and the acceptance test. Verify what they report before building on it.
- **Review.** Kernel, generator, threading and public-API changes get a second reader
  (an agent with fresh context or the maintainer) who reads the diff and the evidence.
  Changes to the public API, the conventions, the determinism contract or the threading
  contract need the maintainer's agreement before implementation.
- **Ask.** When a decision is the maintainer's (API shape, licence, scope, spending money
  on cloud machines), ask rather than guess; meanwhile do everything that does not depend
  on the answer.

## 8. Git

- Small commits, one logical change each, imperative subject with a scope
  (`gen: fold pending rotations into addj`). The body states what was measured and which
  gate stages ran. Never mix a default change with a refactor.
- Stage explicit paths. Read the files, tests and callers before editing; preserve work
  that is not yours.
- Commit, push, merge and tag only when asked. Never run a destructive or history-rewriting
  command (`git reset --hard`, `git checkout .`, `git clean`, `git commit --amend`,
  `git rebase`, `git push --force`).
- Never commit build trees, fetched dependencies, benchmark output or anything from the
  ignored directories.
- Code is Apache-2.0. Every source file (C++, CMake, Python, shell, TOML) starts with the
  SPDX header, below any shebang:

  ```
  SPDX-FileCopyrightText: 2026 John C. Travers and contributors
  SPDX-License-Identifier: Apache-2.0
  ```

## 9. Machines and shared resources

- Remote runs go through the runner scripts under `scripts/` once they exist; they ship
  pinned dependencies, copy by content (never `rsync -a`, which keeps mtimes and leaves
  stale builds) and give every job its own build directory.
- Slurm: exclusive nodes for timing; check the first job's log before waiting on a chain
  (`afterok` chains die silently).
- Cloud machines cost money: get agreement first, keep a ledger of every rental in the
  experiment file, and destroy the instance when the run ends.
- Two OpenMP runtimes in one process crash or crawl. Benchmark binaries link one library
  each for this reason.

## 10. Third-party code and licensing

- FFTN is Apache-2.0. **Do not copy code from any third-party source** into the tree
  without the maintainer's agreement, and never from a copyleft source (FFTW, KFR, ducc0,
  AOCL-FFTW): that would change FFTN's licence.
- Reading any library for ideas is fine and encouraged. Algorithms and techniques are
  not copyrightable; code is. So: implement from the design, the published papers and
  your own understanding, not with a copyleft library's source open beside you; do not
  port a file function by function or reuse its identifiers; cite the idea's source in
  `dev/DESIGN.md`.
- If copying would clearly be the better engineering choice, stop and raise it: it is a
  licence decision for the maintainer, recorded in the decision log.
- Test and benchmark binaries that link GPL libraries are for our own use and are never
  distributed.
- Competitor libraries are linked only by benchmark and test binaries, never by the
  library.

# B2: ISA attribute regions with clang on the MSVC ABI (Windows)

- **Status:** done (the runs are the CI's Windows jobs)
- **Owner:** agent-ci (task 0.8), 2026-11-13
- **Phase / task:** 0.3, run by 0.8

## Question

Do the ISA attribute regions of DESIGN.md section 6.4 hold on Windows with clang on the
MSVC ABI (clang-cl, and clang++ with the MSVC standard library), on x64 and on Arm64; and
how can a Windows on Arm program tell that the processor has FCMA and pointer
authentication? The first answer decides whether section 6.4 needs a Windows exception;
the second decides whether `neon_fcma` can be dispatched on Windows on Arm, where it is
reported absent today (DESIGN.md 14.2).

## Hypothesis

The regions hold: clang-cl and clang++ share Clang's front end and code generator, which
B1 and the proof of concept of section 14.1 showed sound on Linux and macOS, and the
rules that guard against the linker (internal linkage of region text, one exported table
per target) do not depend on the object format. What may break is tooling, not the
design: `tools/asm_check.py` has never read COFF objects or Microsoft-mangled names on a
real build, and CMake sets no objdump for the MSVC front end (fixed in task 0.8 by
looking for `llvm-objdump` beside the compiler).

For detection: Windows documents no `IsProcessorFeaturePresent` flag for FCMA or pointer
authentication, but it copies the AArch64 ID registers into the registry
(`HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\<n>`, values `CP <hex>`, the name
encoding op0, op1, CRn, CRm and op2). `CP 4031` would then be ID_AA64ISAR1_EL1, whose
FCMA field (bits 19:16) and APA/API fields (bits 7:4, 11:8) answer the question, with
ID_AA64ISAR2_EL1 (`CP 4032`, APA3 in bits 15:12) for QARMA3 implementations. A direct
`MRS` of the ID register from user mode is expected to trap on Windows (Linux emulates
it; Windows is not known to).

## Change under test

None: a feasibility check of the mechanism on a new toolchain and operating system, as in
B1. Subject: FFTN's own regions (`src/isa`), vocabularies and tests at the commit CI runs,
and a stand-alone probe, `.github/b2/woa_features.cpp`.

## Method

- Machines: GitHub-hosted runners. `windows-2025` (x64; the processor is whatever Azure
  assigns, recorded by `fftn_platform_dump` in the log) and `windows-11-arm` (Arm64,
  Windows 11; Azure Cobalt 100, a Neoverse N2, which has FCMA and pointer
  authentication). No other Windows machine is available.
- Compilers: LLVM as installed on the images (20.1 at the time of writing), with the
  Visual Studio 2022 or 2026 libraries, through `.github/scripts/msvc-env.ps1`.
  Configurations: x64 clang-cl, x64 clang++ (GNU driver, MSVC target and STL), Arm64
  clang-cl. Release (`check-release`), warnings as errors.
- Checks, per configuration (the CI's `windows` jobs of `.github/workflows/ci.yml`):
  1. Build: the library, the tests and fbench compile with FFTN's warnings as errors.
  2. Regions at run time: `fftn_isa_tests` (vocabulary and kernels bit for bit against
     the reference, per target) pass under every target the runner can run: scalar and
     avx2 on x64; scalar on Arm64 while FCMA is reported absent.
  3. One definition rule and link: the all-targets link test passes, and `asm_check`
     finds every target's code only in its own object (`<target>.cpp.obj`).
  4. By-value vectors across a region boundary: Clang rejects them on every target (B1);
     clang-cl and clang++ use the same front end. Covered by check 1 compiling FFTN; no
     separate probe is run.
  5. Disassembly: `isa.asm_check` passes on `fftn.lib` and the probe objects (COFF,
     Microsoft-mangled names), and every `isa.asm_check_finds_*` test passes, which
     shows that the checker detects each kind of fault in COFF objects rather than
     silently seeing nothing.
  6. The test suites: every CTest test passes (analytic, reference, support, platform,
     API, fbench self-test and smoke).
  7. Detection on Windows on Arm (`windows-11-arm` only): the probe prints every
     `IsProcessorFeaturePresent` index that reports 1, the `PF_ARM_*` names defined by
     the SDK's `winnt.h`, every `CP` registry value with the FCMA and pointer
     authentication fields decoded, whether `MRS ID_AA64ISAR1_EL1` works, and the ground
     truth: whether an FCMLA and an XPACI instruction execute.
- Decision rule, written before running:
  - The regions **hold** on Windows if checks 1, 2, 3, 5 and 6 pass in all three
    configurations. A failure caused by tooling (CMake, `asm_check`'s parsing of COFF or
    Microsoft names, a POSIX assumption in a test) is fixed in the tool and the check is
    repeated; it does not count against the design. A wrong result, a target's
    instruction in baseline code, or one target's copy of a function reached from
    another reopens DESIGN.md 6.4 for Windows.
  - Detection: if `winnt.h` defines a documented `PF_ARM_*` flag for FCMA and one for
    pointer authentication, and they agree with the ground truth, FFTN uses them.
    Otherwise, if the registry fields agree with the ground truth, FFTN reads FCMA and
    pointer authentication from `CP 4031` and `CP 4032` of processor 0, and reports them
    absent when the values are missing. Otherwise both stay reported absent. The
    implementation is a follow-up change to `src/platform/cpu_features.cpp` with a test
    over recorded registry values; one runner type is evidence for the mechanism, not
    for every Arm64 Windows machine, and the decoding follows the Arm architecture
    reference rather than that one machine.

## Results

CI runs of task 0.8 on GitHub-hosted runners: 43487560137, 43495592375 and 43498297281
(workflow `ci`, jobs `windows x64 clang-cl`, `windows x64 clang++`, `windows arm64
clang-cl`), then the run of the commit that records these results, the first with every
Windows job green. Compilers: LLVM 20.1.8 on `windows-2025`, 22.1.8 on `windows-11-arm`,
each with Visual Studio 2026 (MSVC 14.51) and Windows SDK 10.0.26100; Release, warnings
as errors; CMake 4.4, Ninja.

| Check | x64 clang-cl | x64 clang++ | Arm64 clang-cl |
|---|---|---|---|
| 1. Build with FFTN's warnings as errors | passes after one fix (below) | passes after one fix | passes after one fix |
| 2. Kernels bit for bit, every runnable target | scalar, avx2 | scalar, avx2 | scalar; neon_fcma once detected (run 43498297281) |
| 3. All-targets link test; each target's code in its own object | passes | passes | passes once the test's executable is linked with `/OPT:NOICF` |
| 5. `isa.asm_check` on `fftn.lib` (COFF, Microsoft names) and every `asm_check_finds_*` | passes after the lambda pattern fix | same | same, `finds_pauth` included |
| 6. Every CTest test | passes after the fbench fixes | same | same |

Failures met on the way, all in tooling or tests, none in the regions:

- CMake gave the GNU-driver `clang.exe` MSVC-style flags when only the C++ compiler was
  named (GoogleTest's `project()` enables C): fixed by naming the C compiler (`CC`).
- `std::getenv` is deprecated by the Microsoft C runtime (a `-Werror` failure with
  clang++): environment variables are read through `detail::environment_variable`
  (`_dupenv_s` on Windows).
- `asm_check` reported the seeded lambda call correctly, as `??R<lambda_0>@...`; the
  test's expected pattern knew only Itanium names.
- The linker's identical-code folding (`/OPT:ICF`, default in optimised links) gave
  neon_fcma an entry point equal to another function's on Arm64. Folding merges only
  identical machine code, so it cannot make one target run another's instructions; the
  link test, which compares addresses, is linked without it.
- fbench assumed POSIX paths and executables (no `.exe`, `/` separators, `#!` scripts in
  its own tests).

Detection on Windows on Arm (Azure Cobalt 100, `ProcessorNameString` "Cobalt 100", MIDR
`CP 4000` = 0x410fd490, a Neoverse N2):

| Method | FCMA | Pointer authentication |
|---|---|---|
| `IsProcessorFeaturePresent`: SDK 10.0.26100 defines `PF_ARM_*` flags up to 88, none for either; the runner reports 2, 12, 18, 19, 23-25, 27, 29-31, 34, 43-47, 49, 51, 52, 55-57, 64-68 (JSCVT 44 and LRCPC 45 among them) | no flag | no flag |
| Registry `CP 4031` (ID_AA64ISAR1_EL1) = 0x0010111101211052; `CP 4032` absent | FCMA = 1 | APA = 5 (QARMA5), API = 0 |
| `MRS ID_AA64ISAR1_EL1` in user mode | faults (illegal instruction) | faults |
| Ground truth: the instruction executes | FCMLA executes | XPACI executes |
| `__try`/`__except` around a call to the instruction | catches the MRS fault | (instruction runs) |

The registry holds 17 `CP` values: ID_AA64PFR0/1, DFR0/1, AFR0/1, ISAR0/1 and MMFR0/1/2,
MIDR_EL1, CTR_EL0 and four EL1 system registers; there is no ISAR2, so a QARMA3 implementation would show its pointer
authentication nowhere Windows exposes. The probe's first run put the instructions inline
in `__try` bodies and died of the first fault with nothing printed; with each
instruction in a function of its own, `__except` caught the fault. That fits Clang
catching faults raised in the calls a `__try` makes and not in the instructions of its
own body (which would need asynchronous exceptions), but one run does not prove it.

## Verdict

**The regions hold** on Windows with clang on the MSVC ABI, x64 and Arm64, both drivers:
checks 1, 2, 3, 5 and 6 pass in all three configurations, and every failure on the way
was tooling (CMake's C compiler, the C runtime's deprecations, a test's name pattern,
the linker's code folding in an address test, fbench's POSIX assumptions). DESIGN.md 6.4
needs no Windows exception.

**Detection:** there is no documented flag, MRS faults, and the registry fields agree with
the ground truth, so by the decision rule FFTN reads FCMA and pointer authentication from
`CP 4031` and `CP 4032` of processor 0 and reports them absent when the values are
missing. Done in `src/platform/cpu_features.cpp` (`decode_windows_arm64`, tested over the
recorded value and each field; DESIGN.md 14.2 and the decision log updated), so
`neon_fcma` is now chosen and tested on Windows on Arm. One runner type is the evidence;
the decoding follows the Arm ARM. A processor with QARMA3 only would be reported without
pointer authentication, since Windows does not copy ID_AA64ISAR2_EL1 there; that loses
`neon_fcma` on such a machine but never runs it where it cannot run.

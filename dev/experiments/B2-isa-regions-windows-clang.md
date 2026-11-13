# B2: ISA attribute regions with clang on the MSVC ABI (Windows)

- **Status:** planned (written before its runs; the runs are the CI's Windows jobs)
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

Pending: the CI runs of task 0.8 (run IDs and job links to be recorded here).

## Verdict

Pending.

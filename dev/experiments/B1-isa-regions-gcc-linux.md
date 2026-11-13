# B1: ISA attribute regions with GCC and Clang on Linux x86-64

- **Status:** done
- **Owner:** lead, 2026-11-13
- **Phase / task:** 0.3

## Question

Do the ISA attribute regions of DESIGN.md section 6.4 hold on Linux x86-64 with GCC 14
and 15 (and current GCC and Clang)? If they do not, the dispatch mechanism (and with it
sections 6.4 and 14.2) must change before any kernel is written against it.

## Hypothesis

They hold, with the same rules as on AArch64 and with Clang (DESIGN.md section 14.1).
Clang on x86 refuses to pass a vector by value to a function outside its region; GCC is
expected only to warn (`-Wpsabi`). The open risk is that GCC accepts such a call at -O0
and miscompiles it silently, which would make the region rules mandatory and require a
lint.

## Change under test

None: this is a feasibility check of the mechanism, not an A/B comparison. Subject: the
proof of concept of the header-only study (a radix-4 transform in scalar, SSE2, AVX2 and
AVX-512 regions, two translation units, a function table chosen once), together with the
study's small single-file probes of each rule. Once task 0.3 lands, the same checks are
repeated on FFTN's own regions (`tests/isa`).

## Method

- Machine: Intel Xeon Gold 6240 (AVX-512 F/DQ/BW/VL, AVX2, FMA), Ubuntu 24.04, GNU ld.
- Compilers: GCC 13.3 (system; below the floor, for reference), GCC 14.4, GCC 15.2,
  GCC 16.2, Clang 18, 21 and 23.
- Checks, per compiler:
  1. Probes of each region rule: inlining across region boundaries at -O0 and -O2,
     lambdas, `std::` templates used with vectors inside a region, function
     multi-versioning traps, AVX-512 target strings, template instantiation after the
     region closes.
  2. A vector passed by value into a baseline template, compiled and run at -O0, -O1 and
     -O2: compile error, warning, or silently wrong result.
  3. One definition rule: a shared inline helper compiled once inside an AVX-512 region
     (and, for contrast, with a per-file `-mavx512f`), linked in both orders; count
     AVX-512 instructions reaching baseline code.
  4. The proof of concept at -O0, -O2, -O3 with LTO, and with ASan and UBSan; every
     instruction set forced in turn against the reference.
  5. Assembly per region: no calls to vocabulary functions; no VEX or EVEX encodings
     outside the AVX2 and AVX-512 namespaces; zmm only in the AVX-512 namespace.
- Decision rule, written before running: the regions **hold** if, for GCC 14 and newer
  and Clang 18 and newer, check 4 passes at every level, check 5 is clean, and check 3
  shows no AVX-512 instructions in baseline code for the region form. If check 2 gives a
  silently wrong result for any supported compiler, the region rules are kept but made
  mandatory by a lint (`-Werror=psabi` with GCC, and a check that kernel text calls no
  `std::` function with a vector argument). Any other failure reopens DESIGN.md
  section 6.4.

## Results

Run 2026-11-13 on the machine above (Linux 6.17, GNU ld 2.42), two Slurm jobs (the second
added GCC 14.4 and 15.2 once they were built natively for this system). Raw logs are kept
with the campaign, outside the repository.

| Check | GCC 13.3 | GCC 14.4 | GCC 15.2 | GCC 16.2 | Clang 18 / 21 / 23 |
|---|---|---|---|---|---|
| 1. Rule probes | as on AArch64 (below) | same | same | same | as on AArch64 |
| 2. Vector by value into a baseline template | compiles with `-Wpsabi` warnings only; **segfaults** at -O0, -O1, -O2 | same | same | same | compile error |
| 3. ODR, region form | 0 AVX-512 instructions in baseline code, both link orders | 0 | 0 | 0 | 0 |
| 3. ODR, per-file flag or header first included in a region | AVX-512 copy kept when its object links first | same | same | AVX-512 copy kept in **both** link orders | AVX-512 copy kept when its object links first |
| 4. Proof of concept, 4 instruction sets each, -O0, -O2, ASan+UBSan | all correct | all correct | all correct | all correct | all correct |
| 4. Same at -O3 with LTO | correct | correct | correct | correct | Clang 18 correct; Clang 21 and 23 not run (their LTO needs the LLVM gold plugin or lld, absent from that install) |
| 5. Assembly per namespace | no vocabulary calls; zmm only in avx512; no VEX in scalar or SSE2 | same | same | same | same |

Rule probes (check 1): Clang rejects a vector argument or return across a region boundary,
and an always-inline function whose target is wider than its caller; GCC rejects the
latter and only warns (`-Wpsabi`) on the former. A lambda in a region is not inlined at -O0
by GCC and does not inherit the target. Every GCC warns `-Wignored-attributes` when a
vector type is a template argument (56 such warnings in the proof of concept, harmless
there, fatal under `-Werror`). Errors in the proof of concept: none; its accuracy matched
the reference (maximum relative error 1.05e-15 scalar, 8.95e-16 to 9.99e-16 AVX2 and
AVX-512) in every configuration, and forcing each instruction set selected it.

## Verdict

**The regions hold** on Linux x86-64 for GCC 14 and newer and Clang 18 and newer: checks
3 (region form), 4 and 5 pass everywhere. Check 2 gives a silently wrong program on every
GCC, so, by the decision rule, the region rules stay as designed and are made mandatory:

- the library is compiled with `-Werror=psabi` under GCC, always (not only in the gate);
- vector types are never template arguments in region code (GCC's
  `-Wignored-attributes`, an error under the gate's `-Werror`);
- the disassembly test of task 0.3 enforces check 5 on FFTN's own regions.

Task 0.3 implements these and repeats checks 4 and 5 on FFTN's own regions with GCC 14 and
15 on the same machine. DESIGN.md section 14.1 is updated. B2 (Windows) is still open.

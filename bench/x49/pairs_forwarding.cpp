// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The screen of experiment X49 (dev/experiments/X49-tiny-calls.md): does an AArch64 core
// forward a recent store to a load as quickly when either access is a pair (LDP, STP) as
// when both are single 128-bit accesses? Each variant is a 2-point butterfly over `rows`
// rows of 32 bytes, in place, called through a pointer as fbench calls a plan; every call
// loads what the previous one stored. On the M5 Max single accesses run 0.75 ns per call at
// one row and any pair 3.3-3.8 ns.
//
//   x49_pairs_forwarding [rows (3)] [byte offset of the rows (0)] [calls (20000000)]
//
// One line per variant and repetition: rows, offset, variant, ns per call.

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ratio>

namespace {

using kernel = void (*)(const void*, void*, long, long, unsigned long);

// x0 src, x1 dst, x2 and x3 the bytes between rows, x4 the rows. The loads and stores are
// written out because a compiler pairs adjacent single accesses itself.
// NOLINTBEGIN(portability-no-assembler): the instructions are what the screen measures
[[gnu::naked, gnu::noinline]] void ldp_stp(const void* /*src*/, void* /*dst*/, long /*src_step*/,
                                           long /*dst_step*/, unsigned long /*rows*/) {
  asm volatile("cbz x4, 2f\n"
               "1:\n"
               "ldp q0, q1, [x0]\n"
               "fadd v2.2d, v0.2d, v1.2d\n"
               "fsub v0.2d, v0.2d, v1.2d\n"
               "stp q2, q0, [x1]\n"
               "add x0, x0, x2\n"
               "add x1, x1, x3\n"
               "subs x4, x4, #1\n"
               "b.ne 1b\n"
               "2: ret\n");
}

[[gnu::naked, gnu::noinline]] void ldr_str(const void* /*src*/, void* /*dst*/, long /*src_step*/,
                                           long /*dst_step*/, unsigned long /*rows*/) {
  asm volatile("cbz x4, 2f\n"
               "1:\n"
               "ldr q0, [x0]\n"
               "ldr q1, [x0, #16]\n"
               "fadd v2.2d, v0.2d, v1.2d\n"
               "fsub v0.2d, v0.2d, v1.2d\n"
               "str q2, [x1]\n"
               "str q0, [x1, #16]\n"
               "add x0, x0, x2\n"
               "add x1, x1, x3\n"
               "subs x4, x4, #1\n"
               "b.ne 1b\n"
               "2: ret\n");
}

[[gnu::naked, gnu::noinline]] void ldp_str(const void* /*src*/, void* /*dst*/, long /*src_step*/,
                                           long /*dst_step*/, unsigned long /*rows*/) {
  asm volatile("cbz x4, 2f\n"
               "1:\n"
               "ldp q0, q1, [x0]\n"
               "fadd v2.2d, v0.2d, v1.2d\n"
               "fsub v0.2d, v0.2d, v1.2d\n"
               "str q2, [x1]\n"
               "str q0, [x1, #16]\n"
               "add x0, x0, x2\n"
               "add x1, x1, x3\n"
               "subs x4, x4, #1\n"
               "b.ne 1b\n"
               "2: ret\n");
}

[[gnu::naked, gnu::noinline]] void ldr_stp(const void* /*src*/, void* /*dst*/, long /*src_step*/,
                                           long /*dst_step*/, unsigned long /*rows*/) {
  asm volatile("cbz x4, 2f\n"
               "1:\n"
               "ldr q0, [x0]\n"
               "ldr q1, [x0, #16]\n"
               "fadd v2.2d, v0.2d, v1.2d\n"
               "fsub v0.2d, v0.2d, v1.2d\n"
               "stp q2, q0, [x1]\n"
               "add x0, x0, x2\n"
               "add x1, x1, x3\n"
               "subs x4, x4, #1\n"
               "b.ne 1b\n"
               "2: ret\n");
}
// NOLINTEND(portability-no-assembler)

} // namespace

int main(int argc, char* const* argv) {
  const unsigned long rows = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 3;
  const long offset = argc > 2 ? std::strtol(argv[2], nullptr, 10) : 0;
  const long calls = argc > 3 ? std::strtol(argv[3], nullptr, 10) : 20000000;
  if (rows == 0 || rows > 64 || offset < 0 || offset > 4096 || offset % 16 != 0 || calls < 1) {
    std::fprintf(stderr, "rows 1-64, offset 0-4096 in steps of 16, calls >= 1\n");
    return 2;
  }
  alignas(4096) static std::array<double, 1024> buffer;
  double* rows_at = buffer.data() + offset / 8;
  const std::array<kernel, 4> kernels{ldp_stp, ldr_str, ldp_str, ldr_stp};
  const std::array<const char*, 4> names{"ldp/stp", "ldr/str", "ldp/str", "ldr/stp"};
  for (int rep = 0; rep < 3; ++rep) {
    for (std::size_t k = 0; k < kernels.size(); ++k) {
      for (unsigned long i = 0; i < 4 * rows; ++i) {
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound): rows and offset are checked above
        rows_at[i] = 1e-3 * static_cast<double>(i);
      }
      kernel volatile f = kernels[k]; // called through a pointer, as fbench calls a plan
      const auto t0 = std::chrono::steady_clock::now();
      for (long i = 0; i < calls; ++i) {
        f(rows_at, rows_at, 32, 32, rows);
      }
      const auto t1 = std::chrono::steady_clock::now();
      std::printf("%lu %ld %s %.3f\n", rows, offset, names[k],
                  std::chrono::duration<double, std::nano>(t1 - t0).count() /
                      static_cast<double>(calls));
    }
  }
  return 0;
}

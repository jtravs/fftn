// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The rows of a target's codelet table (passes/entries.inl) that one row unit defines,
// included inside the unit's region after the kernel text (isa/kernel_text.inl). The unit
// names itself with FFTN_ISA_ROWS before it includes anything. No include guard.
//
// Why a target is several units (DESIGN.md 6.4): one unit holding every pass of a target
// needed 9-16 GB to compile under the sanitizers, and the CI runners ran out of memory. The
// passes of the codelet rows are nearly all of a target's code, and their cost grows fast
// with the radix. Clang 23 with ASan + UBSan (neon_fcma, Apple M5 Max), one row alone, both
// precisions: radices 2 to 12 together about 75 s, 13 to 16 55 s, 32 39 s and 1.2 GB, 64
// 135 s and 2.9 GB; the main unit (isa/table.inl) 2 s. So radix 64 has a unit per precision,
// and the units below take about the same time each.
//
// Every row is defined by exactly one unit: the main unit's table refers to all of them, so
// a row left out fails to link.

template <class T, std::size_t I>
constinit const radix_passes codelet_row<T, I>::passes = radix_entries<T, I>::value;

// NOLINTBEGIN(bugprone-macro-parentheses): a template argument list, not an expression.
#define FFTN_ISA_ROW(T, I) template struct codelet_row<T, I>;
#define FFTN_ISA_ROWS_OF(I) FFTN_ISA_ROW(double, I) FFTN_ISA_ROW(float, I)
// NOLINTEND(bugprone-macro-parentheses)

static_assert(gen::radix_count == 17 && gen::radices[10] == 12 && gen::radices[14] == 16 &&
                  gen::radices[15] == 32 && gen::radices[16] == 64,
              "the row units below split gen::radices by radix");

#if FFTN_ISA_ROWS == 1 // radices 2 to 12
FFTN_ISA_ROWS_OF(0)
FFTN_ISA_ROWS_OF(1)
FFTN_ISA_ROWS_OF(2)
FFTN_ISA_ROWS_OF(3)
FFTN_ISA_ROWS_OF(4)
FFTN_ISA_ROWS_OF(5)
FFTN_ISA_ROWS_OF(6)
FFTN_ISA_ROWS_OF(7)
FFTN_ISA_ROWS_OF(8)
FFTN_ISA_ROWS_OF(9)
FFTN_ISA_ROWS_OF(10)
#elif FFTN_ISA_ROWS == 2 // radices 13 to 16
FFTN_ISA_ROWS_OF(11)
FFTN_ISA_ROWS_OF(12)
FFTN_ISA_ROWS_OF(13)
FFTN_ISA_ROWS_OF(14)
#elif FFTN_ISA_ROWS == 3 // radix 32
FFTN_ISA_ROWS_OF(15)
#elif FFTN_ISA_ROWS == 4 // radix 64, double
FFTN_ISA_ROW(double, 16)
#elif FFTN_ISA_ROWS == 5 // radix 64, float
FFTN_ISA_ROW(float, 16)
#else
#error "rows.inl: FFTN_ISA_ROWS names no row unit"
#endif

#undef FFTN_ISA_ROWS_OF
#undef FFTN_ISA_ROW

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# One translation unit per instruction set (DESIGN.md 6.4). Each is compiled on every
# architecture, with no ISA flag; its body is guarded by the architecture it targets, so
# universal and cross builds need no CMake logic. Kernel units are always optimised.

set(fftn_isa_units scalar.cpp avx2.cpp neon_fcma.cpp)
target_sources(fftn PRIVATE ${fftn_isa_units})
fftn_always_optimise(${fftn_isa_units})

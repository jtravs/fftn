// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The kernel text every unit of a target compiles (DESIGN.md 6.4), included inside the
// target's region after its vocabulary: twiddle packing, the generated codelets and the
// passes. All of it is templates and force-inlined functions with internal linkage, so a
// unit pays only to parse what it does not instantiate; what each unit instantiates is
// chosen after this (isa/rows.inl, isa/table.inl). No include guard.

#include "isa/kernels.inl"

#include "gen/complex_codelets.inl"

#include "gen/split_codelets.inl"

#include "passes/passes.inl"

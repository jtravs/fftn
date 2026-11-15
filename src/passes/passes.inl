// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Every pass kernel (DESIGN.md 8.2), included inside each target's region after its
// vocabulary, kernels.inl and the generated codelets, and before table.inl. One text for
// every target: what differs between targets is the vocabulary (C complex numbers per
// register, the twiddle storage form), never the pass. No include guard.

#include "passes/access.inl"

#include "passes/stockham.inl"

#include "passes/split.inl"

#include "passes/generic.inl"

#include "passes/bluestein.inl"

#include "passes/real.inl"

#include "passes/grid.inl"

#include "passes/lanes.inl"

#include "passes/entries.inl"

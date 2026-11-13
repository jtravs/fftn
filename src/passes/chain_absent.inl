// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// For targets without the chain strategy's passes yet: every entry null, so plan
// construction falls back to a target that has them (src/core/plan_core.cpp). In phase 0
// the passes exist on the scalar target only; phase 1 brings them to the SIMD targets.
// Included inside a region, before table.inl. No include guard.

namespace {

template <class T> inline constexpr chain_kernels<T> chain_entries{};

} // namespace

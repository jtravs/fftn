// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The out-of-line parts of the public API that are not the plan core.

#include "fftn/isa.hpp"
#include "fftn/workspace.hpp"
#include "isa/target.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace fftn::inline v0 {

// The public enumeration mirrors the internal one value for value, so a plan converts its
// target with a cast (plan_core.cpp).
static_assert(static_cast<unsigned char>(isa::scalar) ==
              static_cast<unsigned char>(detail::target::scalar));
static_assert(static_cast<unsigned char>(isa::x86_v2) ==
              static_cast<unsigned char>(detail::target::x86_v2));
static_assert(static_cast<unsigned char>(isa::avx) ==
              static_cast<unsigned char>(detail::target::avx));
static_assert(static_cast<unsigned char>(isa::avx2) ==
              static_cast<unsigned char>(detail::target::avx2));
static_assert(static_cast<unsigned char>(isa::avx512) ==
              static_cast<unsigned char>(detail::target::avx512));
static_assert(static_cast<unsigned char>(isa::neon) ==
              static_cast<unsigned char>(detail::target::neon));
static_assert(static_cast<unsigned char>(isa::neon_fcma) ==
              static_cast<unsigned char>(detail::target::neon_fcma));

const char* isa_name(isa target) noexcept {
  // target_name returns views of string literals, so data() is null-terminated.
  return detail::target_name(static_cast<detail::target>(target)).data();
}

workspace::workspace(std::size_t bytes)
    // NOLINTNEXTLINE(modernize-avoid-c-arrays): an owned array of uninitialised bytes
    : owned_(bytes == 0 ? nullptr : std::make_unique_for_overwrite<std::byte[]>(bytes)),
      memory_(owned_.get(), bytes) {}

} // namespace fftn::inline v0

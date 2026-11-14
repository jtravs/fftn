// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "isa/arch.hpp"
#include "isa/codelet_table.hpp"
#include "isa/target.hpp"

namespace fftn::inline v0::detail {

const codelet_table* find_codelets(target t) noexcept {
  switch (t) {
  case target::scalar:
    return &scalar::codelet_entries;
#if FFTN_ARCH_X86_64
  case target::avx2:
    return &avx2::codelet_entries;
#endif
#if FFTN_ARCH_AARCH64
  case target::neon:
    return &neon::codelet_entries;
  case target::neon_fcma:
    return &neon_fcma::codelet_entries;
#endif
  default:
    return nullptr;
  }
}

} // namespace fftn::inline v0::detail

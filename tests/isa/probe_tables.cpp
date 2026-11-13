// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "isa/arch.hpp"
#include "isa/probe_table.hpp"
#include "isa/target.hpp"

namespace fftn::inline v0::detail {

const probe_table* find_probes(target t) noexcept {
  switch (t) {
  case target::scalar:
    return &scalar::probes;
#if FFTN_ARCH_X86_64
  case target::avx2:
    return &avx2::probes;
#endif
#if FFTN_ARCH_AARCH64
  case target::neon_fcma:
    return &neon_fcma::probes;
#endif
  default:
    return nullptr;
  }
}

} // namespace fftn::inline v0::detail

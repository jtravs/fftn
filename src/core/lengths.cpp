// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Task 2.5: the public length queries over the planner's cost tables (planner/rules.hpp,
// next_fast_length and relative_row_cost).

#include "fftn/lengths.hpp"

#include "core/plan_core.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "planner/rules.hpp"
#include "platform/topology.hpp"

#include <cstddef>
#include <stdexcept>

namespace fftn::inline v0 {

std::size_t next_fast_length(std::size_t n, length_kind kind) {
  if (kind != length_kind::complex && kind != length_kind::real) {
    throw std::invalid_argument(
        "fftn::next_fast_length: kind is not an fftn::length_kind enumerator");
  }
  if (n > detail::max_length_query) {
    throw std::invalid_argument("fftn::next_fast_length: n is above 2^40, the longest length "
                                "the length queries accept");
  }
  return detail::next_fast_length(n, kind == length_kind::real);
}

namespace detail {
namespace {

// Complex numbers per register of a target in double: from its kernels where the build has
// them, else from its register width (the tables are per C, so a target this build lacks
// is still priced).
std::size_t complex_lanes(target t, bool single) {
  if (const kernel_table* k = find_kernels(t)) {
    return single ? of<float>(*k).complex_lanes : of<double>(*k).complex_lanes;
  }
  std::size_t lanes = 1;
  switch (t) {
  case target::avx:
  case target::avx2:
    lanes = 2;
    break;
  case target::avx512:
    lanes = 4;
    break;
  case target::scalar:
  case target::x86_v2:
  case target::neon:
  case target::neon_fcma:
    break;
  }
  return single && t != target::scalar ? 2 * lanes : lanes;
}

} // namespace

double relative_cost(std::size_t n, bool single, isa which) {
  if (static_cast<unsigned>(which) > static_cast<unsigned>(isa::best)) {
    throw std::invalid_argument("fftn::relative_cost: target is not an fftn::isa enumerator");
  }
  if (n > max_length_query) {
    throw std::invalid_argument("fftn::relative_cost: n is above 2^40, the longest length the "
                                "length queries accept");
  }
  const target t = which == isa::best ? plan_target_of(options{})
                                      : static_cast<target>(static_cast<unsigned char>(which));
  // This machine's estimate: its planning vendor, as its plans take it (DESIGN.md 13.5).
  return relative_row_cost(n, profile_of(t, complex_lanes(t, single), system_cpu_vendor()));
}

} // namespace detail
} // namespace fftn::inline v0

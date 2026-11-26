// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The library's own access to a wisdom's entries (fftn/wisdom.hpp befriends it): the plan
// core's lookup (core/wisdom.cpp) and the C API's fftn_wisdom_add (c_api/c_api.cpp), which
// holds a plan core rather than a typed plan.
#pragma once

#include "fftn/detail/core.hpp"
#include "fftn/wisdom.hpp"

#include <string>

namespace fftn::inline v0::detail {

struct wisdom_access {
  static const std::string* find(const wisdom& w, const std::string& place) {
    const auto it = w.index_.find(place);
    return it == w.index_.end() ? nullptr : &w.entries_[it->second];
  }
  // wisdom::add(plan) for the plan's core.
  static void add(wisdom& w, const plan_core& core) { w.add_entry(wisdom::entry_of(core)); }
};

} // namespace fftn::inline v0::detail

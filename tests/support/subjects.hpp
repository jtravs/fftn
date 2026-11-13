// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The subjects the suites run against, by name, with the lengths each is tested at.
// Task 0.5 adds FFTN's plans here.

#pragma once

#include "support/fftw_subject.hpp"
#include "support/lengths.hpp"
#include "support/naive_dft.hpp"
#include "support/transform_under_test.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace fftn::test {

inline constexpr std::array<std::string_view, 2> subject_names{"naive", "fftw"};

// The subject of that name, or nothing when this build does not have it (FFTW off).
template <class T> std::optional<subject<T>> make_subject(std::string_view name) {
  if (name == "naive") {
    return naive_dft<T>().as_subject();
  }
  if (name == "fftw") {
    return fftw_subject<T>();
  }
  return std::nullopt;
}

// O(n^2) subjects get the short list.
inline std::vector<std::size_t> subject_lengths(std::string_view name) {
  return name == "naive" ? naive_lengths() : fast_lengths();
}

} // namespace fftn::test

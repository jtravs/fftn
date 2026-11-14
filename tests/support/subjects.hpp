// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The subjects the suites run against, by name, with the lengths each is tested at: the
// naive DFT and FFTW, which check the suites themselves, and FFTN, which they test, once
// per instruction set the machine runs ("fftn_scalar", "fftn_neon_fcma", "fftn_avx2").

#pragma once

#include "support/fftn_subject.hpp"
#include "support/fftw_subject.hpp"
#include "support/lengths.hpp"
#include "support/naive_dft.hpp"
#include "support/transform_under_test.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::test {

inline std::vector<std::string> subject_names() {
  std::vector<std::string> names{"naive", "fftw"};
  for (const fftn::isa t : runnable_isas()) {
    names.push_back(std::string("fftn_") + fftn::isa_name(t));
  }
  return names;
}

// The subject of that name, or nothing when this build does not have it (FFTW off).
template <class T> std::optional<subject<T>> make_subject(std::string_view name) {
  if (name == "naive") {
    return naive_dft<T>().as_subject();
  }
  if (name == "fftw") {
    return fftw_subject<T>();
  }
  for (const fftn::isa t : runnable_isas()) {
    if (name == std::string("fftn_") + fftn::isa_name(t)) {
      return fftn_subject<T>(t);
    }
  }
  return std::nullopt;
}

// O(n^2) subjects get the short list.
inline std::vector<std::size_t> subject_lengths(std::string_view name) {
  return name == "naive" ? naive_lengths() : fast_lengths();
}

} // namespace fftn::test

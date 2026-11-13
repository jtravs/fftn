// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Small helpers shared by the competitor backends: option parsing, messages, the file that
// defines a shared library's code, and the instruction set a header-only library was
// compiled for. Each backend is its own executable, so these are header-only.
#pragma once

#include "harness/backend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace fbench::backends {

// The arm's backend options, "KEY=VALUE KEY=VALUE", in order.
inline std::vector<std::pair<std::string, std::string>> parse_options(std::string_view text) {
  std::vector<std::pair<std::string, std::string>> out;
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t end = std::min(text.find(' ', at), text.size());
    const std::string_view token = text.substr(at, end - at);
    at = end + 1;
    if (token.empty()) {
      continue;
    }
    const std::size_t eq = token.find('=');
    out.emplace_back(token.substr(0, eq),
                     eq == std::string_view::npos ? "" : token.substr(eq + 1));
  }
  return out;
}

inline void say(char* message, std::size_t size, const std::string& text) {
  std::snprintf(message, size, "%s", text.c_str());
}

// The file that defines `symbol` in this process, or empty when it is linked statically or
// not found. dlsym with RTLD_NEXT skips the executable, whose PLT entries and copy
// relocations would otherwise be found first on Linux (see the FFTW backend).
inline std::string defining_file(const char* symbol) {
#ifdef _WIN32
  (void)symbol;
  return "";
#else
  const void* const address = dlsym(RTLD_NEXT, symbol);
  Dl_info info{};
  if (address == nullptr || dladdr(address, &info) == 0 || info.dli_fname == nullptr) {
    return "";
  }
  return info.dli_fname;
#endif
}

// The widest vector extension this translation unit was compiled for: what a header-only
// library included here can use.
constexpr const char* compiled_isa() {
#if defined(__AVX512F__)
  return "avx512f";
#elif defined(__AVX2__) && defined(__FMA__)
  return "avx2+fma";
#elif defined(__AVX__)
  return "avx";
#elif defined(__SSE2__)
  return "sse2";
#elif defined(__ARM_NEON)
  return "neon";
#else
  return "scalar";
#endif
}

// Element offsets of the transforms of a case: transform b of the batch starts at
// b * dist elements; inside one transform, dimension d (slowest first) has the stride
// below, in elements, with the last dimension embedded in `last` (its padded length).
inline std::vector<std::int64_t> element_strides(const fbench_case& c, std::int64_t last) {
  std::vector<std::int64_t> strides(static_cast<std::size_t>(c.rank));
  std::int64_t s = 1;
  for (int d = c.rank - 1; d >= 0; --d) {
    strides[static_cast<std::size_t>(d)] = s;
    s *= d == c.rank - 1 ? last : c.dims[d];
  }
  return strides;
}

} // namespace fbench::backends

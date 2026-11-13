// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Runs a subject's call on a layout and watches the memory contracts while it does.
//
// Every run places each buffer against a guard page (canary.hpp), fills every byte that the
// layout does not describe with canary values, fills the described output elements with
// NaN (so an output that is never written cannot pass), and afterwards verifies that the
// input buffer is byte-for-byte unchanged (out of place), that no undescribed byte changed
// and that the slack beside the guard pages is intact. Violations are returned with the
// outputs; the checks report them as failures.

#pragma once

#include "support/canary.hpp"
#include "support/input_snapshot.hpp"
#include "support/transform_under_test.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fftn::test {

// Where element 0 of each side lies relative to a 64-byte boundary, in bytes (a multiple of
// the element's alignment), and on which side of each buffer the guard page is.
struct run_options {
  std::size_t in_offset = 0;
  std::size_t out_offset = 0;
  guard_side guard = guard_side::above;
};

template <class Out> struct run_result {
  std::vector<Out> out; // the logical outputs, row after row
  std::vector<std::string> violations;
};

// Element offsets of a side's logical elements, in order (row by row, x fastest).
inline std::vector<std::ptrdiff_t> row_offsets(std::size_t len, std::size_t count,
                                               std::ptrdiff_t stride, std::ptrdiff_t dist) {
  std::vector<std::ptrdiff_t> v;
  v.reserve(len * count);
  for (std::size_t r = 0; r < count; ++r) {
    for (std::size_t e = 0; e < len; ++e) {
      v.push_back(static_cast<std::ptrdiff_t>(r) * dist + static_cast<std::ptrdiff_t>(e) * stride);
    }
  }
  return v;
}

inline std::vector<std::ptrdiff_t> plane_offsets(std::size_t ny, std::size_t nx, std::size_t count,
                                                 std::ptrdiff_t y_stride, std::ptrdiff_t x_stride,
                                                 std::ptrdiff_t dist) {
  std::vector<std::ptrdiff_t> v;
  v.reserve(ny * nx * count);
  for (std::size_t p = 0; p < count; ++p) {
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t x = 0; x < nx; ++x) {
        v.push_back(static_cast<std::ptrdiff_t>(p) * dist +
                    static_cast<std::ptrdiff_t>(y) * y_stride +
                    static_cast<std::ptrdiff_t>(x) * x_stride);
      }
    }
  }
  return v;
}

namespace detail {

struct byte_range {
  std::ptrdiff_t lo = 0; // relative to element 0
  std::ptrdiff_t hi = 0; // one past the last byte
};

inline byte_range bytes_of(std::span<const std::ptrdiff_t> offsets, std::size_t size) {
  if (offsets.empty()) {
    return {.lo = 0, .hi = 0};
  }
  const auto [lo, hi] = std::ranges::minmax(offsets);
  const auto s = static_cast<std::ptrdiff_t>(size);
  return {.lo = lo * s, .hi = (hi + 1) * s};
}

inline byte_range unite(byte_range a, byte_range b) {
  return {.lo = std::min(a.lo, b.lo), .hi = std::max(a.hi, b.hi)};
}

// A guarded buffer whose element 0 (at byte -range.lo from its start) lies at `offset`
// modulo 64.
inline guarded_buffer buffer_for(byte_range range, std::size_t offset, guard_side side) {
  constexpr std::ptrdiff_t align = 64;
  const std::ptrdiff_t start =
      ((static_cast<std::ptrdiff_t>(offset) + range.lo) % align + align) % align;
  return {static_cast<std::size_t>(range.hi - range.lo), static_cast<std::size_t>(align),
          static_cast<std::size_t>(start), side};
}

inline void allow_elements(canary_bytes& canary, std::span<const std::ptrdiff_t> offsets,
                           std::size_t size, std::ptrdiff_t lo) {
  for (const std::ptrdiff_t off : offsets) {
    canary.allow(static_cast<std::size_t>(off * static_cast<std::ptrdiff_t>(size) - lo), size);
  }
}

inline void fill_nan(std::byte* base, std::span<const std::ptrdiff_t> offsets, std::size_t size) {
  for (const std::ptrdiff_t off : offsets) {
    std::memset(base + off * static_cast<std::ptrdiff_t>(size), 0xFF, size);
  }
}

inline std::string where(std::string_view what, std::size_t first, std::size_t count) {
  std::ostringstream s;
  s << what << ": " << count << " byte(s), the first at byte " << first << " of the buffer";
  return s.str();
}

} // namespace detail

// Runs `call(in, out)` with the logical `input` placed at the input offsets, and gathers
// the outputs from the output offsets. For in-place runs both sides share one buffer and
// element 0 of each side is at the same address.
template <class In, class Out, class Call>
run_result<Out> execute(const Call& call, std::span<const std::ptrdiff_t> in_offsets,
                        std::span<const std::ptrdiff_t> out_offsets, placement place,
                        std::span<const In> input, const run_options& options = {}) {
  if (input.size() != in_offsets.size()) {
    throw std::invalid_argument("execute: input size does not match the layout");
  }
  if (options.in_offset % alignof(In) != 0 || options.out_offset % alignof(Out) != 0) {
    throw std::invalid_argument("execute: offset below the element alignment");
  }
  run_result<Out> result;
  const detail::byte_range in_range = detail::bytes_of(in_offsets, sizeof(In));
  const detail::byte_range out_range = detail::bytes_of(out_offsets, sizeof(Out));

  if (place == placement::in_place) {
    const detail::byte_range range = detail::unite(in_range, out_range);
    const guarded_buffer buf = detail::buffer_for(range, options.in_offset, options.guard);
    std::byte* base = buf.data() - range.lo;
    canary_bytes canary({buf.data(), buf.size()});
    detail::allow_elements(canary, in_offsets, sizeof(In), range.lo);
    detail::allow_elements(canary, out_offsets, sizeof(Out), range.lo);
    canary.arm();
    detail::fill_nan(base, out_offsets, sizeof(Out));
    for (std::size_t i = 0; i < input.size(); ++i) {
      std::memcpy(base + in_offsets[i] * static_cast<std::ptrdiff_t>(sizeof(In)), &input[i],
                  sizeof(In));
    }
    call(reinterpret_cast<const In*>(base), reinterpret_cast<Out*>(base));
    if (const std::size_t bad = canary.violations(); bad != 0) {
      result.violations.push_back(
          detail::where("bytes outside the rows were written", canary.first_violation(), bad));
    }
    if (!buf.surroundings_intact()) {
      result.violations.emplace_back("bytes beside the guard pages were written");
    }
    result.out.resize(out_offsets.size());
    for (std::size_t i = 0; i < out_offsets.size(); ++i) {
      std::memcpy(&result.out[i], base + out_offsets[i] * static_cast<std::ptrdiff_t>(sizeof(Out)),
                  sizeof(Out));
    }
    return result;
  }

  const guarded_buffer in_buf = detail::buffer_for(in_range, options.in_offset, options.guard);
  const guarded_buffer out_buf = detail::buffer_for(out_range, options.out_offset, options.guard);
  std::byte* in_base = in_buf.data() - in_range.lo;
  std::byte* out_base = out_buf.data() - out_range.lo;
  canary_bytes in_canary({in_buf.data(), in_buf.size()});
  detail::allow_elements(in_canary, in_offsets, sizeof(In), in_range.lo);
  in_canary.arm();
  for (std::size_t i = 0; i < input.size(); ++i) {
    std::memcpy(in_base + in_offsets[i] * static_cast<std::ptrdiff_t>(sizeof(In)), &input[i],
                sizeof(In));
  }
  canary_bytes out_canary({out_buf.data(), out_buf.size()});
  detail::allow_elements(out_canary, out_offsets, sizeof(Out), out_range.lo);
  out_canary.arm();
  detail::fill_nan(out_base, out_offsets, sizeof(Out));
  const input_snapshot snapshot({in_buf.data(), in_buf.size()});

  call(reinterpret_cast<const In*>(in_base), reinterpret_cast<Out*>(out_base));

  if (!snapshot.unchanged()) {
    result.violations.push_back(
        detail::where("the input buffer changed", snapshot.first_change(), 1));
  }
  if (const std::size_t bad = out_canary.violations(); bad != 0) {
    result.violations.push_back(detail::where("bytes outside the output rows were written",
                                              out_canary.first_violation(), bad));
  }
  if (!in_buf.surroundings_intact() || !out_buf.surroundings_intact()) {
    result.violations.emplace_back("bytes beside the guard pages were written");
  }
  result.out.resize(out_offsets.size());
  for (std::size_t i = 0; i < out_offsets.size(); ++i) {
    std::memcpy(&result.out[i],
                out_base + out_offsets[i] * static_cast<std::ptrdiff_t>(sizeof(Out)), sizeof(Out));
  }
  return result;
}

template <class T>
run_result<std::complex<T>> run_c2c(const subject<T>& s, direction d, const rows& r,
                                    std::span<const std::complex<T>> input,
                                    const run_options& options = {}) {
  const auto call = s.c2c(d, r);
  const auto in = row_offsets(r.n, r.count, r.in_stride, r.in_dist);
  const auto out = row_offsets(r.n, r.count, r.out_stride, r.out_dist);
  return execute<std::complex<T>, std::complex<T>>(call, in, out, r.place, input, options);
}

template <class T>
run_result<std::complex<T>> run_r2c(const subject<T>& s, const rows& r, std::span<const T> input,
                                    const run_options& options = {}) {
  const auto call = s.r2c(r);
  const auto in = row_offsets(r.n, r.count, r.in_stride, r.in_dist);
  const auto out = row_offsets(r.n / 2 + 1, r.count, r.out_stride, r.out_dist);
  return execute<T, std::complex<T>>(call, in, out, r.place, input, options);
}

template <class T>
run_result<T> run_c2r(const subject<T>& s, const rows& r, std::span<const std::complex<T>> input,
                      const run_options& options = {}) {
  const auto call = s.c2r(r);
  const auto in = row_offsets(r.n / 2 + 1, r.count, r.in_stride, r.in_dist);
  const auto out = row_offsets(r.n, r.count, r.out_stride, r.out_dist);
  return execute<std::complex<T>, T>(call, in, out, r.place, input, options);
}

template <class T>
run_result<std::complex<T>> run_c2c_planes(const subject<T>& s, direction d, const planes& p,
                                           std::span<const std::complex<T>> input,
                                           const run_options& options = {}) {
  const auto call = s.c2c_planes(d, p);
  const auto in = plane_offsets(p.ny, p.nx, p.count, p.in_y_stride, p.in_x_stride, p.in_dist);
  const auto out = plane_offsets(p.ny, p.nx, p.count, p.out_y_stride, p.out_x_stride, p.out_dist);
  return execute<std::complex<T>, std::complex<T>>(call, in, out, p.place, input, options);
}

} // namespace fftn::test

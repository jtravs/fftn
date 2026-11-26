// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/callbacks.hpp"

#include "core/edges.hpp"
#include "core/program.hpp"
#include "fftn/callback.hpp"
#include "fftn/detail/core.hpp"
#include "passes/pass_args.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

constexpr std::size_t scratch_alignment = 128;

// The callback's own type, recovered from the erased one it was converted from (view_of in
// fftn/detail/core.hpp), so that it is called through its own type.
template <class E>
void invoke_as(erased_block_fn fn, void* user, const block_info& info, std::byte* data) noexcept {
  reinterpret_cast<block_fn<E>>(fn)(user, info, reinterpret_cast<E*>(data));
}

// Measured planning's stand-in (callback_env::stand_in).
void invoke_nothing(erased_block_fn /*fn*/, void* /*user*/, const block_info& /*info*/,
                    std::byte* /*data*/) noexcept {}

} // namespace

std::byte* callback_area(std::byte* slot, std::size_t offset, std::size_t bytes) noexcept {
  if (slot == nullptr || bytes == 0) {
    return nullptr;
  }
  const auto at = reinterpret_cast<std::uintptr_t>(slot + offset);
  return slot + offset + (scratch_alignment - at % scratch_alignment) % scratch_alignment;
}

void callback_guard(const pass_args* /*args*/, const void* /*src*/, void* /*dst*/,
                    std::size_t /*first*/, std::size_t /*last*/) {
  contract_failure("fftn: a plan with block callbacks was executed through an overload without "
                   "exec_args");
}

std::size_t callback_chunks(std::size_t count, std::size_t lines, std::size_t element_bytes,
                            std::size_t block_bytes) noexcept {
  if (count == 0) {
    return 1;
  }
  const std::size_t per_element = std::max<std::size_t>(lines, 1) * element_bytes;
  // The elements of every line a block may hold, at least one.
  const std::size_t fit =
      std::max<std::size_t>(block_bytes / std::max<std::size_t>(per_element, 1), 1);
  return std::min<std::size_t>((count + fit - 1) / fit, std::numeric_limits<std::uint32_t>::max());
}

stage callback_stage(precision prec, bool real, const side_callback& call, bool output,
                     std::size_t count, std::size_t logical, slot where, std::ptrdiff_t stride,
                     std::ptrdiff_t offset, std::size_t block_bytes,
                     std::vector<std::unique_ptr<callback_site>>& sites, std::size_t lines,
                     std::ptrdiff_t line_dist) {
  const std::size_t real_bytes = prec == precision::f64 ? sizeof(double) : sizeof(float);
  auto site = std::make_unique<callback_site>();
  site->fn = call.fn;
  if (prec == precision::f64) {
    site->invoke = real ? &invoke_as<double> : &invoke_as<std::complex<double>>;
  } else {
    site->invoke = real ? &invoke_as<float> : &invoke_as<std::complex<float>>;
  }
  site->user = output ? 1 : 0;
  site->logical = logical;
  site->element_bytes = real ? real_bytes : 2 * real_bytes;
  site->scratch_bytes = call.scratch_bytes;
  stage st;
  st.run = &callback_guard;
  st.src = where;
  st.dst = where;
  st.units = count;
  st.chunks =
      static_cast<std::uint32_t>(callback_chunks(count, lines, site->element_bytes, block_bytes));
  st.moving_factors = moving_callback;
  st.args.src_stride = stride;
  st.args.dst_stride = stride;
  st.args.dst_dist = line_dist;
  // Two's complement of a negative offset (a run along a reversed axis), as the edge passes
  // keep theirs (core/edges.cpp).
  st.args.dst_cols = static_cast<std::size_t>(offset);
  // Where the stage's edge record finds its site (mark_cursors, plan_core.cpp).
  st.args.roots = site.get();
  st.name = std::string(output ? "callback out: " : "callback in: ") + mode_name(call.mode) +
            " on logical " + std::to_string(logical) + " to " + std::to_string(logical + count);
  sites.push_back(std::move(site));
  return st;
}

void run_callback_stage(const stage& st, const callback_site& site, std::byte* data,
                        const callback_env& env, std::size_t lines, std::ptrdiff_t line_stride,
                        std::size_t first, std::size_t last) noexcept {
  std::byte* line = data + static_cast<std::ptrdiff_t>(st.args.dst_cols);
  const std::ptrdiff_t stride = st.args.dst_stride;
  const auto element = static_cast<std::ptrdiff_t>(site.element_bytes);
  // Rows of at most three axes (shape.hpp); origin's entries other than the element axis are 0
  // for a row of one axis, the only plans with callbacks so far.
  std::array<std::size_t, 3> origin{};
  block_info info{
      .transform = env.transform,
      .origin = std::span<const std::size_t>(origin.data(), site.rank),
      .element_axis = site.axis,
      .line_axis = along_transforms,
      .length = 0,
      .lines = lines,
      .element_stride = stride,
      .line_stride = lines > 1 ? line_stride : 0,
      .group = 1,
      .group_stride = 0,
      .copy = 0,
      .member = env.member,
      .scratch = env.scratch != nullptr ? std::span<std::byte>(env.scratch, site.scratch_bytes)
                                        : std::span<std::byte>(),
  };
  void* user = env.user[site.user];
  // A stand-in calls this instead (planning): the same indirect call, doing nothing.
  const auto invoke = env.stand_in ? &invoke_nothing : site.invoke;
  for (std::size_t c = first; c < last; ++c) {
    const std::size_t begin = chunk_start(st.units, st.chunks, c);
    const std::size_t end = chunk_start(st.units, st.chunks, c + 1);
    if (begin == end) {
      continue;
    }
    origin[site.axis] = site.logical + begin;
    info.length = end - begin;
    invoke(site.fn, user, info, line + static_cast<std::ptrdiff_t>(begin) * stride * element);
  }
}

const char* mode_name(callback_mode m) noexcept {
  switch (m) {
  case callback_mode::produce:
    return "produce";
  case callback_mode::consume:
    return "consume";
  case callback_mode::modify:
    break;
  }
  return "modify";
}

std::string callbacks_text(const side_callback& in, const side_callback& out,
                           const callback_place& in_place, const callback_place& out_place) {
  std::string t = "callbacks:";
  const auto side = [&](const char* name, const side_callback& c, const callback_place& at) {
    if (!c.any()) {
      return;
    }
    if (t.back() != ':') {
      t += ";";
    }
    t += std::string(" the ") + name + "'s (" + mode_name(c.mode) + ") " + at.where + ", " +
         at.block;
    if (at.sweep) {
      t += ", one more sweep over each row (longer than a block)";
    }
    if (c.scratch_bytes != 0) {
      t += ", " + std::to_string(c.scratch_bytes) + " bytes of scratch per member";
    }
  };
  side("input", in, in_place);
  side("output", out, out_place);
  return t;
}

callback_place callback_blocks(std::string where, std::size_t count, std::size_t lines,
                               std::size_t element_bytes, std::size_t block_bytes,
                               const char* what) {
  const std::size_t chunks = callback_chunks(count, lines, element_bytes, block_bytes);
  const std::size_t longest = (count + chunks - 1) / chunks;
  callback_place p{.where = std::move(where), .block = {}, .sweep = chunks > 1};
  if (lines > 1) {
    p.block = "blocks of up to " + std::to_string(lines) + " " + what + " together, " +
              (chunks == 1 ? "whole" : "up to " + std::to_string(longest) + " elements of each");
  } else {
    p.block = chunks == 1 ? "blocks of one whole row"
                          : "blocks of up to " + std::to_string(longest) + " elements of one row";
  }
  return p;
}

} // namespace fftn::inline v0::detail

// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/runs.hpp"

#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

[[noreturn]] void reject(const std::string& argument, const std::string& why) {
  throw std::invalid_argument("fftn::plan: " + argument + " " + why);
}

std::string run_name(std::size_t i) { return "shape.runs[" + std::to_string(i) + "]"; }

// The length of an axis's logical line on a side: n, or n/2 + 1 for the halved axis on the
// complex side of a real transform.
std::size_t side_length(const shape_view& s, std::size_t d, bool output) {
  const std::size_t n = s.axes[d].n;
  const bool complex_half = d + 1 == s.axes.size() && ((output && s.kind == transform_kind::r2c) ||
                                                       (!output && s.kind == transform_kind::c2r));
  return complex_half ? n / 2 + 1 : n;
}

} // namespace

const std::vector<line_run>& side_runs::of(std::size_t d) const { return axes.at(d); }

std::size_t side_runs::stored_extent(std::size_t d, std::size_t whole) const {
  if (!any() || axes.at(d).empty()) {
    return whole;
  }
  std::size_t end = 0;
  for (const line_run& r : axes[d]) {
    end = std::max(end, r.stored + r.count);
  }
  return end;
}

std::size_t side_runs::positions(std::size_t d) const {
  std::size_t n = 0;
  for (const line_run& r : axes.at(d)) {
    n += r.count;
  }
  return n;
}

run_plan resolve_runs(const shape_view& s, const options& opts) {
  run_plan p;
  if (s.runs.empty()) {
    return p;
  }
  if (opts.in_place) {
    reject("shape.runs", "are not taken by an in-place plan yet (task 5.1d builds them out of "
                         "place)");
  }
  const std::size_t r = s.axes.size();
  // The runs' indices per axis and side, in the shape's order.
  std::vector<std::vector<std::size_t>> listed(2 * r);
  for (std::size_t i = 0; i < s.runs.size(); ++i) {
    const stored_run& run = s.runs[i];
    if (run.axis >= r) {
      reject(run_name(i) + ".axis",
             "is " + std::to_string(run.axis) + "; the shape has " + std::to_string(r) + " axes");
    }
    if (run.where != side::in && run.where != side::out) {
      reject(run_name(i) + ".where", "is not a side");
    }
    if (run.count == 0) {
      reject(run_name(i) + ".count", "is 0; a run holds at least one element");
    }
    const bool output = run.where == side::out;
    const std::size_t length = side_length(s, run.axis, output);
    if (run.logical >= length || run.count > length - run.logical) {
      reject(run_name(i) + ".logical",
             "and count reach past the line: logical positions " + std::to_string(run.logical) +
                 " to " + std::to_string(run.logical) + " + " + std::to_string(run.count) +
                 " of an axis whose line has " + std::to_string(length) + " on that side");
    }
    std::size_t end = 0;
    if (__builtin_add_overflow(run.stored, run.count, &end) ||
        end > static_cast<std::size_t>(PTRDIFF_MAX)) {
      reject(run_name(i) + ".stored", "and count reach past the addressable range");
    }
    listed[2 * run.axis + (output ? 1 : 0)].push_back(i);
  }
  for (std::size_t d = 0; d < r; ++d) {
    for (const bool output : {false, true}) {
      std::vector<std::size_t>& ids = listed[2 * d + (output ? 1 : 0)];
      if (ids.empty()) {
        continue;
      }
      // Neither the logical nor the stored positions of two runs may coincide: each logical
      // position has one stored place, and each stored element one logical position.
      const auto overlapping = [&](auto key, const char* field) {
        std::ranges::sort(ids, [&](std::size_t a, std::size_t b) { return key(a) < key(b); });
        for (std::size_t k = 1; k < ids.size(); ++k) {
          if (key(ids[k - 1]) + s.runs[ids[k - 1]].count > key(ids[k])) {
            reject(run_name(ids[k]) + field, "overlaps " + run_name(ids[k - 1]) +
                                                 " (runs of one axis and side are disjoint)");
          }
        }
      };
      overlapping([&](std::size_t i) { return s.runs[i].stored; }, ".stored");
      overlapping([&](std::size_t i) { return s.runs[i].logical; }, ".logical");
      side_runs& held = output ? p.out : p.in;
      held.axes.resize(r);
      for (const std::size_t i : ids) {
        held.axes[d].push_back(
            {.logical = s.runs[i].logical, .stored = s.runs[i].stored, .count = s.runs[i].count});
      }
    }
  }
  return p;
}

std::string runs_key(const shape_view& s) {
  if (s.runs.empty()) {
    return {};
  }
  // In the shape's order: the same runs listed in another order are the same layout, but a key
  // need only be exact, not canonical.
  std::ostringstream o;
  o << " runs";
  for (const stored_run& run : s.runs) {
    o << ' ' << run.axis << (run.where == side::in ? 'i' : 'o') << run.logical << '/' << run.stored
      << '/' << run.count;
  }
  return o.str();
}

std::string runs_text(const run_plan& runs, bool accumulate, const char* in_to, std::size_t passes,
                      run_pruning in, run_pruning out) {
  const auto axes_text = [](const side_runs& held) {
    std::string t;
    for (std::size_t d = 0; d < held.axes.size(); ++d) {
      if (held.axes[d].empty()) {
        continue;
      }
      const std::size_t n = held.axes[d].size();
      t += (t.empty() ? "axis " : ", axis ") + std::to_string(d) + " in " + std::to_string(n) +
           (n == 1 ? " run of " : " runs of ") + std::to_string(held.positions(d)) +
           (held.positions(d) == 1 ? " position" : " positions");
    }
    return t;
  };
  std::string text = "stored runs:";
  if (runs.in.any()) {
    text += std::string(" the input's ") + axes_text(runs.in);
    switch (in) {
    case run_pruning::direct:
      text += ", read where they are stored by the first pass, pruned to the line's first half";
      break;
    case run_pruning::half:
      text += std::string(", read with zeros elsewhere in the line's first half into ") + in_to +
              " in a separate pass, the first pass pruned to that half";
      break;
    case run_pruning::none:
      text += std::string(", read with zeros elsewhere into ") + in_to + " in a separate pass";
      break;
    }
  }
  if (runs.out.any()) {
    text += std::string(runs.in.any() ? ";" : "") + " the output's " + axes_text(runs.out) +
            ", the rest not written, ";
    switch (out) {
    case run_pruning::direct:
      text += "written where they are stored by the last pass, pruned to the line's first half";
      break;
    case run_pruning::half:
      text += std::string(accumulate ? "accumulated" : "stored") +
              " from the workspace block in a separate pass, the last pass pruned to the line's "
              "first half";
      break;
    case run_pruning::none:
      text += std::string(accumulate ? "accumulated" : "stored") +
              " from the workspace block in a separate pass";
      break;
    }
  } else if (accumulate) {
    text += "; the output accumulated from the workspace block in a separate pass";
  }
  const bool whole = in == run_pruning::none && out == run_pruning::none;
  text += std::string(whole ? "; the whole transform computed between them" : "") + "; they cost " +
          std::to_string(passes) + (passes == 1 ? " separate pass" : " separate passes");
  return text;
}

} // namespace fftn::inline v0::detail

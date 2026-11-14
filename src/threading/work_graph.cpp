// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "threading/work_graph.hpp"

#include "core/program.hpp"
#include "passes/pass_args.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>

namespace fftn::inline v0::detail {
namespace {

// The cost model's unit: a point through a pass. A codelet pass (grouped, lane, leaf) moves
// the line's n points whatever its unit (a block of C butterflies or copies, or the line:
// src/strategies/chain.hpp), so its work is n; a generic pass of radix r sums r terms per
// point, n r; an edge pass (radix 1 in its arguments: copies, the real edges) touches about
// two points per unit.
double stage_work(const stage& st) {
  const std::size_t r = st.args.radix;
  if (r <= 1) {
    return 2.0 * static_cast<double>(st.units);
  }
  const auto n = static_cast<double>(st.args.n);
  return r > max_codelet_radix ? n * static_cast<double>(r) : n;
}

double records_work(const program& p, std::size_t begin, std::size_t end) {
  double work = 0;
  for (std::size_t i = begin; i < end; ++i) {
    if (const stage* st = std::get_if<stage>(&p.records[i])) {
      work += stage_work(*st);
      continue;
    }
    const loop& l = std::get<loop>(p.records[i]);
    work += static_cast<double>(l.count) * records_work(p, i + 1, l.body_end);
    i = l.body_end - 1;
  }
  return work;
}

// The longest line among the stages of [begin, end): the line length n every stage records.
std::size_t line_length(const program& p, std::size_t begin, std::size_t end) {
  std::size_t n = 1;
  for (std::size_t i = begin; i < end; ++i) {
    if (const stage* st = std::get_if<stage>(&p.records[i])) {
      n = std::max(n, st->args.n);
    }
  }
  return n;
}

} // namespace

work_graph build_work_graph(const program& p, std::size_t points, std::size_t chunk_points) {
  work_graph g;
  g.tasks = p.transforms;
  g.points = points;
  const std::size_t end = p.records.size();
  for (std::size_t i = p.batch_loops; i < end;) {
    graph_phase ph;
    ph.record = static_cast<std::uint32_t>(i);
    ph.first = g.chunks;
    if (const stage* st = std::get_if<stage>(&p.records[i])) {
      ph.units = st->units;
      ph.chunks = st->chunks;
      ph.work = stage_work(*st);
      ++i;
    } else {
      // The loops over the lines of an axis are perfectly nested (plan_core.cpp, add_phase):
      // each opens right after the one before and all end together.
      const std::uint32_t nest_end = std::get<loop>(p.records[i]).body_end;
      std::size_t lines = 1;
      std::size_t j = i;
      while (j < nest_end) {
        const loop* l = std::get_if<loop>(&p.records[j]);
        if (l == nullptr) {
          break;
        }
        assert(l->body_end == nest_end && "fftn: the loops over lines are perfectly nested");
        lines *= l->count;
        ++j;
      }
      ph.nest = static_cast<std::uint32_t>(j - i);
      ph.body_end = nest_end;
      ph.units = lines;
      const std::size_t per_chunk =
          std::max<std::size_t>(1, chunk_points / line_length(p, j, nest_end));
      const std::size_t chunks = (lines + per_chunk - 1) / per_chunk;
      ph.chunks = static_cast<std::uint32_t>(
          std::min<std::size_t>(chunks, std::numeric_limits<std::uint32_t>::max()));
      ph.work = static_cast<double>(lines) * records_work(p, j, nest_end);
      i = nest_end;
    }
    g.chunks += ph.chunks;
    g.widest = std::max(g.widest, ph.chunks);
    g.work += ph.work;
    g.phases.push_back(ph);
  }
  return g;
}

} // namespace fftn::inline v0::detail

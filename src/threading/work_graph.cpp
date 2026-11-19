// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "threading/work_graph.hpp"

#include "core/program.hpp"
#include "passes/pass_args.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>

namespace fftn::inline v0::detail {
namespace {

// The cost model's unit: a point through a pass. A codelet pass (grouped, lane, leaf) moves
// the line's n points whatever its unit (a block of C butterflies or copies, or the line:
// src/strategies/chain.hpp), so its work is n; a generic pass of radix r sums r terms per
// point, n r; an edge pass (radix 1 in its arguments: copies, the real edges, a Bluestein
// stage's chirp passes) touches about two points per unit (an element or a pair; a block
// of C elements for the chirp passes, so their work is C times larger than counted).
double stage_work(const stage& st) {
  const std::size_t r = st.args.radix;
  if (r <= 1) {
    return 2.0 * static_cast<double>(st.units);
  }
  const auto n = static_cast<double>(st.args.n);
  return has_codelet(r) ? n : n * static_cast<double>(r);
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

// The points of one line of a nest whose body is [begin, end): the longest line its stages
// record (a lanes pass records its block's points), or where the body holds loops of its own
// (a volume's plane, core/plan_core.cpp add_volume_planes) the most points one of them covers.
// The work of records [begin, end) of a lanes program: its gather, scatter and pairing
// (radix 1) move a block's points.
double lane_records_work(const program& p, std::size_t begin, std::size_t end) {
  double work = 0;
  for (std::size_t i = begin; i < end; ++i) {
    if (const stage* st = std::get_if<stage>(&p.records[i])) {
      work += st->args.radix > 1 ? stage_work(*st) : static_cast<double>(st->args.n * st->args.s);
      continue;
    }
    const loop& l = std::get<loop>(p.records[i]);
    work += static_cast<double>(l.count) * lane_records_work(p, i + 1, l.body_end);
    i = l.body_end - 1;
  }
  return work;
}

std::size_t line_points(const program& p, std::size_t begin, std::size_t end) {
  std::size_t n = 1;
  for (std::size_t i = begin; i < end; ++i) {
    if (const stage* st = std::get_if<stage>(&p.records[i])) {
      n = std::max(n, st->args.n);
      continue;
    }
    const loop& l = std::get<loop>(p.records[i]);
    n = std::max(n, l.count * line_points(p, i + 1, l.body_end));
    i = l.body_end - 1;
  }
  return n;
}

// The loops that open at record i (a nest, as build_work_graph describes it).
struct nest_shape {
  std::uint32_t depth = 0;
  std::uint32_t end = 0;
  std::size_t lines = 1;
};

nest_shape nest_at(const program& p, std::size_t i) {
  nest_shape s;
  s.end = std::get<loop>(p.records[i]).body_end;
  std::size_t j = i;
  while (j < s.end) {
    const loop* l = std::get_if<loop>(&p.records[j]);
    if (l == nullptr || l->body_end != s.end) {
      break;
    }
    s.lines *= l->count;
    ++j;
  }
  s.depth = static_cast<std::uint32_t>(j - i);
  return s;
}

} // namespace

work_graph build_work_graph(const program& p, std::size_t points, std::size_t chunk_points) {
  work_graph g;
  g.tasks = p.transforms;
  g.points = points;
  const std::size_t end = p.records.size();
  if (p.lanes != 0) {
    // A lanes program's body runs blocks of transforms (program.hpp), so a transform has no
    // stage of its own to share: one phase of one chunk, never split, holding the body's
    // work per transform. The gather, scatter and pairing (radix 1) move a block's points.
    graph_phase ph;
    ph.record = static_cast<std::uint32_t>(p.batch_loops);
    ph.units = 1;
    ph.work = lane_records_work(p, p.batch_loops, end) / static_cast<double>(p.lanes);
    g.chunks = 1;
    g.work = ph.work;
    g.phases.push_back(ph);
    return g;
  }
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
      // each opens right after the one before and all end together. A volume's loop over
      // its planes (add_volume_planes) holds a plane's phases, loops that end before it: the
      // nest is that one loop, and its lines are whole planes.
      const nest_shape first = nest_at(p, i);
      ph.nest = first.depth;
      ph.body_end = first.end;
      ph.fair_runs = std::get<loop>(p.records[i]).shares_lines;
      ph.units = first.lines;
      ph.work = static_cast<double>(first.lines) * records_work(p, i + first.depth, first.end);
      i = first.end;
      // A nest that joins this one (loop::joins): its lines follow, in the same phase.
      if (i < end && std::holds_alternative<loop>(p.records[i]) &&
          std::get<loop>(p.records[i]).joins) {
        const nest_shape tail = nest_at(p, i);
        ph.head = ph.units;
        ph.tail_record = static_cast<std::uint32_t>(i);
        ph.tail_nest = tail.depth;
        ph.tail_body_end = tail.end;
        ph.units += tail.lines;
        ph.work += static_cast<double>(tail.lines) * records_work(p, i + tail.depth, tail.end);
        i = tail.end;
      }
      const std::size_t per_chunk =
          std::max<std::size_t>(1, chunk_points / line_points(p, ph.record + ph.nest, ph.body_end));
      const std::size_t chunks = (ph.units + per_chunk - 1) / per_chunk;
      ph.chunk_points = per_chunk * line_points(p, ph.record + ph.nest, ph.body_end);
      ph.chunks = static_cast<std::uint32_t>(
          std::min<std::size_t>(chunks, std::numeric_limits<std::uint32_t>::max()));
    }
    g.chunks += ph.chunks;
    g.widest = std::max(g.widest, ph.chunks);
    g.work += ph.work;
    g.phases.push_back(ph);
  }
  return g;
}

} // namespace fftn::inline v0::detail

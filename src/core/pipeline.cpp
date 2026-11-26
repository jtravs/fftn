// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/pipeline.hpp"

#include "core/driver.hpp"
#include "core/edge_driver.hpp"
#include "core/edges.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "core/runs.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "isa/kernel_table.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"
#include "strategies/chain.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

[[noreturn]] void refuse(const std::string& field, const std::string& why) {
  throw std::invalid_argument("fftn::pipeline: " + field + " " + why);
}

void replace_all(std::string& s, std::string_view from, std::string_view to) {
  for (std::size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) {
    s.replace(at, from.size(), to);
  }
}

// Runs f, the plan's own validation of the line's layout, the edges or the options, with its
// messages naming the description's fields instead of a plan's: the line is a c2c shape of one
// axis (shape.axes[0]) under the description's batch dimensions (shape.batches).
template <class F> auto renaming(const F& f) {
  try {
    return f();
  } catch (const std::invalid_argument& e) {
    std::string m = e.what();
    replace_all(m, "fftn::plan: ", "fftn::pipeline: ");
    replace_all(m, "shape.axes[0]", "desc.line");
    replace_all(m, "shape.batches", "desc.batches");
    replace_all(m, "shape.runs", "desc.runs");
    replace_all(m, "shape (", "desc (");
    replace_all(m, "fftn::pipeline: shape ", "fftn::pipeline: desc ");
    replace_all(m, "the shape has", "the line has");
    throw std::invalid_argument(m);
  }
}

// The tags the chains are built with in line_edges' factor pointers for the fused factors: their
// addresses only, which nothing reads; mark_cursors finds the stages built from them.
const std::array<unsigned char, 3> pipeline_tags{};
const void* const prologue_tag = pipeline_tags.data();
const void* const middle_tag = pipeline_tags.data() + 1;
const void* const epilogue_tag = pipeline_tags.data() + 2;

// Where the one product of the scales goes: a separate pass, or a free edge of a chain (a
// fan-out's on the forward chain's last pass, forward_out: once per line, not per copy).
enum class scale_at : std::uint8_t { none, pass, forward_in, forward_out, inverse_in, inverse_out };

// The vocabulary as validated (resolve_edge_side), before its placement: the three operations,
// the one product of the scales, and the line's runs.
struct vocab {
  edge_side pro;
  edge_side mid;
  edge_side epi;
  double scale = 1.0;
  run_plan runs;
};

struct placed {
  edge_side pro;
  edge_side mid;
  edge_side epi;
  double scale = 1.0;
  scale_at at = scale_at::none;
};

// Each operation's place: a complex factor of stride 1 along the line (or on a line of one point)
// fuses into its chain's pass where the pair has such a pass (`sides`: the prologue's and the
// epilogue's, the <C, C, C> chain pair's; `middle`: the middle's, which the depth-first pair has
// too); anything else, and accumulation, is a separate pass on the line. The real chain pair fuses
// nothing: its sides take real factors, which no pass fuses, and a middle fused into the pairing
// passes is an arm of experiment X67 (PLAN 5.2c). The scale goes to a separate pass where there is
// one (it costs nothing there), else to the inverse chain's output, else (the epilogue's factor
// fused) to its input. The sides' runs go with them (task 5.2d).
placed place(const vocab& v, std::size_t n, bool sides, bool middle) {
  // A stride of 0 along a line of more than one point broadcasts one value, which the fused
  // passes (one value per element) cannot read (side_of gives 0 for a line of one point, where
  // the stride reads nothing more).
  const auto fusable = [n](const edge_side& s, bool fuse) {
    return fuse && s.kind == factor_kind::complex &&
           (s.axis_stride == 1 || (s.axis_stride == 0 && n == 1));
  };
  const double scale = v.scale;
  placed p{.pro = v.pro, .mid = v.mid, .epi = v.epi, .scale = scale};
  p.pro.runs = v.runs.in;
  p.epi.runs = v.runs.out;
  for (edge_side* s : {&p.pro, &p.mid, &p.epi}) {
    s->fused = fusable(*s, s == &p.mid ? middle : sides);
    s->separate = s->kind != factor_kind::none && !s->fused;
  }
  p.epi.fused = p.epi.fused && !p.epi.accumulate;
  p.epi.separate = p.epi.accumulate || (p.epi.kind != factor_kind::none && !p.epi.fused);
  if (scale != 1.0) {
    edge_side* pass = nullptr;
    for (edge_side* s : {&p.epi, &p.mid, &p.pro}) {
      if (pass == nullptr && s->separate) {
        pass = s;
      }
    }
    if (pass != nullptr) {
      pass->scale = scale;
      p.at = scale_at::pass;
    } else if (!p.epi.fused) {
      p.at = scale_at::inverse_out;
    } else if (!p.pro.fused) {
      p.at = scale_at::forward_in;
    } else {
      p.at = scale_at::inverse_in;
    }
  }
  return p;
}

// A fan-out's placement (task 5.2e): place's, but the scale on a line's own pass (the prologue's,
// or the forward chain's last, which takes no middle with fan-out), so that it is applied once per
// line rather than once per copy. middle: whether the middle's factor may fuse (into the inverse's
// first pass, once per copy).
placed place_fanout(const vocab& v, std::size_t n, bool sides, bool middle) {
  vocab unscaled = v;
  unscaled.scale = 1.0;
  placed p = place(unscaled, n, sides, middle);
  p.scale = v.scale;
  if (v.scale != 1.0) {
    if (p.pro.separate) {
      p.pro.scale = v.scale;
      p.at = scale_at::pass;
    } else {
      p.at = scale_at::forward_out;
    }
  }
  return p;
}

// What a fan-out adds to the edge driver's records (task 5.2e): the copies loop's record and the
// bytes it moves the middle cursor per copy, and the stages' offsets into the block
// (edge_record::block_src, block_dst).
struct fanout_marks {
  struct offset {
    std::size_t record = 0;
    std::ptrdiff_t src = 0;
    std::ptrdiff_t dst = 0;
  };
  std::size_t loop = SIZE_MAX;
  std::ptrdiff_t mid_step = 0;
  std::vector<offset> offsets;
};

// The edge driver's records (core/edge_driver.hpp): the stages built with a tag, and the edge
// passes (which edge_pass_stage marks), keep their kernels there and carry guards; the batch
// loops move the three cursors by the factors' batch strides (one loop per batch dimension, in
// the description's order).
std::vector<edge_record> mark_cursors(program& p, const placed& pl, const fanout_marks& fan) {
  std::vector<edge_record> e(p.records.size());
  for (std::size_t i = 0; i < p.records.size(); ++i) {
    stage* st = std::get_if<stage>(&p.records[i]);
    if (st == nullptr) {
      continue;
    }
    if (st->args.in_factor == prologue_tag) {
      st->args.in_factor = nullptr;
      st->moving_factors |= moving_in_factor;
    } else if (st->args.in_factor == middle_tag) {
      // A fan-out's copy k: its kernel in the inverse chain's first pass (task 5.2e).
      st->args.in_factor = nullptr;
      st->moving_factors |= moving_mid_in_factor;
    }
    if (st->args.out_factor == middle_tag) {
      st->args.out_factor = nullptr;
      st->moving_factors |= moving_mid_out_factor;
    } else if (st->args.out_factor == epilogue_tag) {
      st->args.out_factor = nullptr;
      st->moving_factors |= moving_out_factor;
    }
    if (st->moving_factors != 0) {
      e[i].run = st->run;
      e[i].leaf = st->leaf;
      st->run = st->run != nullptr ? &factor_guard_pass : nullptr;
      st->leaf = st->leaf != nullptr ? &factor_guard_leaf : nullptr;
    }
  }
  const auto step = [](const edge_side& s, std::size_t b) -> std::ptrdiff_t {
    return s.kind != factor_kind::none ? s.batch_steps.at(b) : 0;
  };
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    e[b].factor_step = {step(pl.pro, b), step(pl.epi, b), step(pl.mid, b)};
  }
  // The copies loop moves the middle cursor only: the epilogue's factor is every copy's.
  if (fan.loop != SIZE_MAX) {
    e[fan.loop].factor_step = {0, 0, fan.mid_step};
  }
  for (const fanout_marks::offset& o : fan.offsets) {
    e[o.record].block_src = o.src;
    e[o.record].block_dst = o.dst;
  }
  return e;
}

std::string kind_text(factor_kind k) { return k == factor_kind::real ? "real" : "complex"; }

// describe()'s line for one operation of the vocabulary.
std::string operation_text(const char* name, const edge_side& s, const char* fused_where,
                           const char* separate_where) {
  std::string t;
  if (s.kind != factor_kind::none) {
    t = "times a " + kind_text(s.kind) + " factor";
  }
  if (s.accumulate) {
    t += t.empty() ? "accumulated" : ", accumulated";
  }
  if (t.empty()) {
    return std::string(name) + ": none";
  }
  return std::string(name) + ": " + t + " " + (s.fused ? fused_where : separate_where);
}

// The middle's separate pass, in place on the spectrum of `bins` complex values in the workspace
// block (or, the depth-first pair's, in scratch a), its factor through the middle's cursor; or,
// with fan-out, from the spectrum (`where`) to where copy k's inverse starts (`to`), a copy where
// the middle has no factor.
stage middle_stage(precision prec, const edge_side& mid, std::size_t bins, std::size_t chunk,
                   slot where = slot::block, slot to = slot::block) {
  stage m = edge_pass_stage(prec, mid, true, bins, where, 1, to, 1, chunk, false, mid.axis_stride);
  if (mid.kind != factor_kind::none) {
    m.moving_factors = moving_mid_out_factor;
  }
  m.name.replace(0, std::string_view("edge out").size(), "edge middle");
  return m;
}

// The copies loop of a fan-out (task 5.2e): desc.fanout iterations over `body`, copy k's inverse,
// moving the output by fanout_out_dist (out_bytes per element) and, through its edge record, the
// middle cursor by fanout_factor_dist; the block not at all, so that every copy reads the line's
// spectrum where the forward chain left it. serial: the copies work in the task's block
// (loop::serial_lines).
void add_copies(program& p, const pipeline_view& d, const edge_side& mid, std::ptrdiff_t out_bytes,
                bool serial, std::vector<stage>& body, fanout_marks& fan) {
  fan.loop = p.records.size();
  loop lp{.kind = loop_kind::lines, .count = d.fanout};
  lp.step[1] = d.fanout_out_dist * out_bytes;
  lp.serial_lines = serial;
  p.records.emplace_back(lp);
  for (stage& st : body) {
    p.records.emplace_back(std::move(st));
  }
  std::get<loop>(p.records[fan.loop]).body_end = static_cast<std::uint32_t>(p.records.size());
  fan.mid_step = mid.kind != factor_kind::none
                     ? d.fanout_factor_dist * static_cast<std::ptrdiff_t>(mid.factor_bytes)
                     : 0;
}

// The copies of a chain pair whose inverse works in the workspace block (an epilogue pass,
// accumulation or output runs): the spectrum stays in the block's first region and each copy's
// line moves to its second, `second` bytes on, where the inverse and the store stage take it. The
// body's first stage reads the spectrum (the middle's pass or the inverse chain's first pass);
// every other reference to the block is to the second region.
void second_region(const program& p, const fanout_marks& fan, std::ptrdiff_t second,
                   std::vector<fanout_marks::offset>& offsets) {
  const std::uint32_t end = std::get<loop>(p.records[fan.loop]).body_end;
  for (std::size_t i = fan.loop + 1; i < end; ++i) {
    const stage& st = std::get<stage>(p.records[i]);
    const bool first = i == fan.loop + 1;
    if (first && st.src != slot::block) {
      throw std::logic_error("fftn: internal: a fan-out's copy that does not read the spectrum");
    }
    offsets.push_back({.record = i,
                       .src = st.src == slot::block && !first ? second : 0,
                       .dst = st.dst == slot::block ? second : 0});
  }
}

// The batch loops of a pipeline's program, the slots in and out moved by the layout's distances
// (elements of elem_bytes each); the block never moves.
void add_batch_loops(program& p, const pipeline_view& d, const row_layout& l,
                     std::ptrdiff_t elem_bytes) {
  p.transforms = 1;
  for (std::size_t b = 0; b < d.batches.size(); ++b) {
    p.records.emplace_back(loop{
        .kind = loop_kind::batch,
        .count = d.batches[b].count,
        .step = {l.in_dist[b] * elem_bytes, l.out_dist[b] * elem_bytes, 0},
    });
    p.transforms *= d.batches[b].count;
  }
  p.batch_loops = d.batches.size();
}

// The line's stages after the batch loops, which then end with them.
void end_program(program& p, std::vector<stage>& line) {
  for (stage& st : line) {
    p.records.emplace_back(std::move(st));
  }
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    std::get<loop>(p.records[b]).body_end = static_cast<std::uint32_t>(p.records.size());
  }
}

// How a pair takes its line's runs (core/runs.hpp): the input's and the output's pruning.
struct run_prunings {
  run_pruning in = run_pruning::none;
  run_pruning out = run_pruning::none;
};

// Whether a side's runs lie within the line's first half (n/2 elements), where the chain's
// pruned first or last pass (task 5.1e) reads or writes them.
bool within_half(const std::vector<line_run>& runs, std::size_t n) {
  return std::ranges::all_of(runs, [n](const line_run& r) { return r.logical + r.count <= n / 2; });
}

// Whether the side is one run of exactly the line's first half, stored with stride 1: the pruned
// pass then reads or writes it where it is stored.
bool half_stored(const std::vector<line_run>& runs, std::size_t n, const endpoint& stored) {
  return runs.size() == 1 && runs[0].logical == 0 && 2 * runs[0].count == n && stored.stride == 1;
}

// The depth-first pair (DESIGN.md 12.2, task 5.2d): batch loops, then the stages of one line,
//
//   [read stage into scratch b] -> forward depth-first chain (its first pass from the input or
//   b, its sub-rows on their blocks of the workspace block, its top pass into scratch a times
//   the middle's factor) -> [middle pass on a] -> backward depth-first chain (from a, its
//   blocks in the block, its last top pass to the output, or in place) -> [store stage]
//
// built by chain_builder::depth_pair, which also builds Bluestein's depth-first convolution (X50:
// for long lines, whose sub-rows then run in cache with loops moving the block, as a row of the
// line's length would). One input run from logical 0 is read by the forward first pass, zeros
// beyond; one output run from a logical start that C divides is written by the last top pass,
// without accumulation; any other runs, a prologue or epilogue factor and accumulation go through
// the read and store stages (run stages or edge passes), which sweep the line once more each. The
// middle's factor of stride 1 fuses into the forward top pass, through the middle cursor.
//
// With fan-out (task 5.2e) the forward top pass writes the spectrum to the block's second region
// (n elements on, an offset of its edge record), where it stays, and the copies loop runs, for
// copy k, the middle's pass from there into scratch a (copy k's factor, or a copy: the depth-first
// first pass takes no factor) and the backward chain from a, its sub-rows on the block's first
// region, to copy k's output or the store stage. The copies share that first region, so a team
// runs them one after another (loop::serial_lines), sharing the forward chain's phases only.
template <class T>
void build_depth_pair(plan_core& c, const pipeline_view& d, const row_layout& l, const placed& pl,
                      const build_settings& settings, chain_builder<T>& chain,
                      const std::vector<std::size_t>& tops, run_prunings& pr, fanout_marks& fan) {
  program& p = c.prog;
  const std::size_t n = d.line.n;
  const std::size_t lanes = of<T>(*find_kernels(c.isa_target)).complex_lanes;
  const std::size_t chunk = settings.chunk_points;
  const auto complex_bytes = static_cast<std::ptrdiff_t>(2 * sizeof(T));
  add_batch_loops(p, d, l, complex_bytes);
  typename chain_builder<T>::depth_pair_form form;
  form.kernel = pl.mid.fused ? middle_tag : nullptr;
  form.out = pl.at == scale_at::inverse_out ? edge_op::scale : edge_op::none;
  form.scale = pl.scale;
  std::vector<stage> read;
  std::vector<stage> store;
  slot src = slot::in;
  slot dst = slot::out;
  if (pl.pro.runs.any()) {
    const std::vector<line_run>& runs = pl.pro.runs.of(0);
    if (runs.size() == 1 && runs[0].logical == 0) {
      form.in_count = runs[0].count;
      form.in_offset = runs[0].stored;
      pr.in = run_pruning::direct;
    } else {
      run_line_stages(c.prec, pl.pro, false, n, runs, {.src = slot::in, .dst = slot::b}, chunk,
                      false, read);
      src = slot::b;
    }
  } else if (pl.pro.separate) {
    read.push_back(edge_pass_stage(c.prec, pl.pro, false, n, slot::in, 1, slot::b, 1, chunk, false,
                                   pl.pro.axis_stride));
    src = slot::b;
  }
  if (pl.epi.runs.any()) {
    const std::vector<line_run>& runs = pl.epi.runs.of(0);
    if (runs.size() == 1 && runs[0].logical % lanes == 0 && !pl.epi.accumulate) {
      form.out_logical = runs[0].logical;
      form.out_count = runs[0].count;
      form.out_offset = runs[0].stored;
      pr.out = run_pruning::direct;
    } else {
      run_line_stages(c.prec, pl.epi, true, n, runs, {.src = slot::block, .dst = slot::out}, chunk,
                      false, store);
      dst = slot::block;
    }
  } else if (pl.epi.separate) {
    store.push_back(edge_pass_stage(c.prec, pl.epi, true, n, slot::block, 1, slot::out, 1, chunk,
                                    false, pl.epi.axis_stride));
    dst = slot::block;
  }
  typename chain_builder<T>::bluestein_depth_stages parts =
      chain.depth_pair(n, tops, src, dst, form);
  const bool fanned = d.fanout > 1;
  if (fanned) {
    // The forward top pass from the sub-rows' blocks to the spectrum's region (block_dst below).
    stage& top = parts.forward.top.back();
    top.dst = slot::block;
    replace_all(top.name, "to the output", "to the workspace block's second region");
  }
  const std::size_t subs = checked_product(tops);
  const auto stages = [&p](std::vector<stage>& from) {
    for (stage& st : from) {
      p.records.emplace_back(std::move(st));
    }
  };
  // A lines loop of `count` iterations moving the block by `step` elements: within one line, a
  // phase whose lines a team's members run on their own scratch, the owner's block shared.
  const auto looped = [&](std::size_t count, std::size_t step, std::vector<stage>& body) {
    const std::size_t first = p.records.size();
    loop lp{.kind = loop_kind::lines, .count = count};
    lp.step[2] = static_cast<std::ptrdiff_t>(step) * complex_bytes;
    p.records.emplace_back(lp);
    stages(body);
    std::get<loop>(p.records[first]).body_end = static_cast<std::uint32_t>(p.records.size());
  };
  stages(read);
  const auto second = static_cast<std::ptrdiff_t>(n) * complex_bytes;
  for (typename chain_builder<T>::depth_stages* half : {&parts.forward, &parts.backward}) {
    if (half == &parts.backward && fanned) {
      // The copies loop, opened here (add_copies' loop, its body the records up to the store
      // stages' end): copy k's middle from the spectrum into scratch a, then the backward chain.
      fan.loop = p.records.size();
      loop lp{.kind = loop_kind::lines, .count = d.fanout};
      lp.step[1] = d.fanout_out_dist * complex_bytes;
      lp.serial_lines = true;
      p.records.emplace_back(lp);
      fan.mid_step = pl.mid.kind != factor_kind::none
                         ? d.fanout_factor_dist * static_cast<std::ptrdiff_t>(pl.mid.factor_bytes)
                         : 0;
      fan.offsets.push_back({.record = p.records.size(), .src = second});
      p.records.emplace_back(middle_stage(c.prec, pl.mid, n, chunk, slot::block, slot::a));
    }
    stages(half->first);
    looped(subs, n / subs, half->sub);
    if (!half->mid.empty()) {
      looped(tops[0], n / tops[0], half->mid);
    }
    if (half == &parts.forward && fanned) {
      fan.offsets.push_back({.record = p.records.size(), .dst = second});
    }
    stages(half->top);
    if (half == &parts.forward && pl.mid.separate && !fanned) {
      p.records.emplace_back(middle_stage(c.prec, pl.mid, n, chunk, slot::a, slot::a));
    }
  }
  stages(store);
  if (fanned) {
    std::get<loop>(p.records[fan.loop]).body_end = static_cast<std::uint32_t>(p.records.size());
  }
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    std::get<loop>(p.records[b]).body_end = static_cast<std::uint32_t>(p.records.size());
  }
  c.block_elements = fanned ? 2 * n : n;
  c.depth_tops = tops;
  lay_out_workspace(c, parts.scratch, 2 * sizeof(T), false, 0);
  c.axis_notes.push_back("line (n = " + std::to_string(n) + "): " + chain.depth_pair_text(n, tops));
}

// The line's stages (the forward chain's) after the batch loops, then the copies loop over `body`
// (add_copies), which end the batch loops; with `second` (not 0) the copies work in the block's
// second region, that many bytes on (second_region).
void end_with_copies(program& p, const pipeline_view& d, const edge_side& mid,
                     std::ptrdiff_t out_bytes, std::ptrdiff_t second, std::vector<stage>& line,
                     std::vector<stage>& body, fanout_marks& fan) {
  for (stage& st : line) {
    p.records.emplace_back(std::move(st));
  }
  add_copies(p, d, mid, out_bytes, second != 0, body, fan);
  if (second != 0) {
    second_region(p, fan, second, fan.offsets);
  }
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    std::get<loop>(p.records[b]).body_end = static_cast<std::uint32_t>(p.records.size());
  }
}

// The copies of a <C, C, C> chain pair (DESIGN.md 12.2, task 5.2e), after the forward chain into
// the workspace block (`line`), whose spectrum stays there. Where each copy writes its output
// itself (an epilogue fused into the inverse chain's last pass or none, no accumulation, and the
// output whole or one run of the line's first half that the pruned last pass writes where it is
// stored), copy k runs the inverse chain from the block to copy k's output, its kernel fused into
// the chain's first pass (or the middle's pass from the block into scratch b, where it cannot
// fuse): the copies only read the block, so a team shares them as lines on each member's own
// scratch. Otherwise (an epilogue pass, accumulation, other output runs) copy k's line goes to the
// block's second region, where its inverse ends and its store stage reads it, and a team runs a
// line's copies one after another (loop::serial_lines).
template <class T>
void build_complex_copies(plan_core& c, const pipeline_view& d, const row_layout& l,
                          const build_settings& settings, chain_builder<T>& chain,
                          const placed& pl, run_prunings& pr, std::vector<stage>& line,
                          std::size_t scratch, fanout_marks& fan) {
  program& p = c.prog;
  const std::size_t n = d.line.n;
  const std::size_t chunk = settings.chunk_points;
  const auto complex_bytes = static_cast<std::ptrdiff_t>(2 * sizeof(T));
  const endpoint home{.where = slot::block, .stride = 1};
  const endpoint stored_out{.where = slot::out, .stride = l.out_stride};
  std::vector<stage> body;
  line_edges inverse;
  if (pl.mid.fused) {
    inverse.in = edge_op::factor;
    inverse.in_factor = middle_tag;
  }
  const bool direct = !pl.epi.accumulate && !pl.epi.separate &&
                      (!pl.epi.runs.any() ||
                       (within_half(pl.epi.runs.of(0), n) && chain.head_last(n) &&
                        half_stored(pl.epi.runs.of(0), n, stored_out)));
  if (direct) {
    endpoint from = home;
    if (pl.mid.separate) {
      body.push_back(middle_stage(c.prec, pl.mid, n, chunk, slot::block, slot::b));
      from = {.where = slot::b, .stride = 1};
      scratch = std::max(scratch, n);
    }
    if (pl.epi.runs.any()) {
      inverse.out_head = true;
      inverse.out_offset = pl.epi.runs.of(0)[0].stored * 2 * sizeof(T);
      pr.out = run_pruning::direct;
    } else if (pl.epi.fused) {
      inverse.out = edge_op::factor;
      inverse.out_factor = epilogue_tag;
    }
    scratch = std::max(scratch, chain.c2c_line(n, +1, from, stored_out, false, inverse, body));
    end_with_copies(p, d, pl.mid, complex_bytes, 0, line, body, fan);
    c.block_elements = n;
  } else {
    if (pl.mid.separate) {
      body.push_back(middle_stage(c.prec, pl.mid, n, chunk));
    }
    if (pl.epi.runs.any()) {
      // As build_pair's output runs, from the second region (a pruned last pass writes its half).
      const std::vector<line_run>& runs = pl.epi.runs.of(0);
      inverse.out_head =
          inverse.out != edge_op::factor && within_half(runs, n) && chain.head_last(n);
      scratch = std::max(scratch, chain.c2c_line(n, +1, home, home, true, inverse, body));
      run_line_stages(c.prec, pl.epi, true, n, runs,
                      {.src = slot::block, .dst = slot::out, .dst_stride = l.out_stride}, chunk,
                      false, body);
      pr.out = inverse.out_head ? run_pruning::half : run_pruning::none;
    } else {
      scratch = std::max(scratch, chain.c2c_line(n, +1, home, home, true, inverse, body));
      body.push_back(edge_pass_stage(c.prec, pl.epi, true, n, slot::block, 1, slot::out,
                                     l.out_stride, chunk, false, pl.epi.axis_stride));
    }
    end_with_copies(p, d, pl.mid, complex_bytes, static_cast<std::ptrdiff_t>(n) * complex_bytes,
                    line, body, fan);
    c.block_elements = 2 * n;
  }
  lay_out_workspace(c, scratch, 2 * sizeof(T), false, 0);
  const std::string radices = chain.radix_text(n);
  c.axis_notes.push_back("line (n = " + std::to_string(n) + "): forward " + radices +
                         ", then backward " + radices + " per copy (shared twiddle tables)");
}

// The pair a <C, C, C> line takes and its stages: the depth-first pair where it is forced
// (convolution_depth_tops, for contiguous lines only: its passes read and write stride 1; the
// rules' choice waits for X67), otherwise the chain pair (below). Returns whether it is the
// depth-first pair.
template <class T>
bool build_pair(plan_core& c, const pipeline_view& d, const row_layout& l, const vocab& v,
                const build_settings& settings, placed& pl, run_prunings& pr, fanout_marks& fan) {
  const target_profile profile =
      plan_profile(c, of<T>(*find_kernels(c.isa_target)).complex_lanes, settings);
  chain_builder<T> chain = plan_chain<T>(c, profile, settings);
  program& p = c.prog;
  const std::size_t n = d.line.n;
  if (l.in_stride == 1 && l.out_stride == 1) {
    if (const std::vector<std::size_t> tops =
            convolution_depth_tops<T>(n, settings, chain, c.isa_target);
        !tops.empty()) {
      // With fan-out the middle is a pass per copy (build_depth_pair).
      pl = place(v, n, false, d.fanout == 1);
      build_depth_pair<T>(c, d, l, pl, settings, chain, tops, pr, fan);
      return true;
    }
  }
  const bool fanned = d.fanout > 1;
  pl = fanned ? place_fanout(v, n, true, true) : place(v, n, true, true);
  add_batch_loops(p, d, l, static_cast<std::ptrdiff_t>(2 * sizeof(T)));

  std::vector<stage> line;
  std::size_t scratch = 0;
  const std::size_t chunk = settings.chunk_points;
  const endpoint home{.where = slot::block, .stride = 1};
  line_edges forward;
  forward.scale = pl.scale;
  if (pl.pro.fused) {
    forward.in = edge_op::factor;
    forward.in_factor = prologue_tag;
  } else if (pl.at == scale_at::forward_in) {
    forward.in = edge_op::scale;
  }
  if (pl.mid.fused && !fanned) {
    forward.out = edge_op::factor;
    forward.out_factor = middle_tag;
  } else if (pl.at == scale_at::forward_out) {
    forward.out = edge_op::scale;
  }
  // The forward transform reads the input, or the prologue's pass (or the input's runs, with
  // zeros elsewhere) has put the line in the block.
  endpoint forward_src{.where = slot::in, .stride = l.in_stride};
  bool forward_aliased = false;
  const endpoint stored_out{.where = slot::out, .stride = l.out_stride};
  if (pl.pro.runs.any()) {
    // Pruning as a plan's c2c row (task 5.1e): runs within the line's first half and a first
    // pass of an even codelet radix read the half only, where the one run of the half is
    // stored (stride 1: no read stage), else through the block's half.
    const std::vector<line_run>& runs = pl.pro.runs.of(0);
    forward.in_head = forward.in == edge_op::none && within_half(runs, n) && chain.head_first(n);
    if (forward.in_head && half_stored(runs, n, forward_src)) {
      forward.in_offset = runs[0].stored * 2 * sizeof(T);
      pr.in = run_pruning::direct;
    } else {
      run_line_stages(c.prec, pl.pro, false, forward.in_head ? n / 2 : n, runs,
                      {.src = slot::in, .src_stride = l.in_stride}, chunk, false, line);
      forward_src = home;
      forward_aliased = true;
      pr.in = forward.in_head ? run_pruning::half : run_pruning::none;
    }
  } else if (pl.pro.separate) {
    line.push_back(edge_pass_stage(c.prec, pl.pro, false, n, slot::in, l.in_stride, slot::block, 1,
                                   chunk, false, pl.pro.axis_stride));
    forward_src = home;
    forward_aliased = true;
  }
  scratch =
      std::max(scratch, chain.c2c_line(n, -1, forward_src, home, forward_aliased, forward, line));
  if (fanned) {
    build_complex_copies<T>(c, d, l, settings, chain, pl, pr, line, scratch, fan);
    return false;
  }
  if (pl.mid.separate) {
    line.push_back(middle_stage(c.prec, pl.mid, n, chunk));
  }
  line_edges inverse;
  inverse.scale = pl.scale;
  if (pl.at == scale_at::inverse_in) {
    inverse.in = edge_op::scale;
  }
  if (pl.epi.runs.any()) {
    // The output's runs: the inverse chain ends in the block, then the store stage writes (or
    // accumulates) only the runs; or its pruned last pass writes the one run of the line's
    // first half where it is stored (no store stage), as a plan's c2c row.
    const std::vector<line_run>& runs = pl.epi.runs.of(0);
    if (pl.at == scale_at::inverse_out) {
      inverse.out = edge_op::scale;
    }
    inverse.out_head = inverse.out != edge_op::factor && within_half(runs, n) && chain.head_last(n);
    if (inverse.out_head && !pl.epi.accumulate && half_stored(runs, n, stored_out)) {
      inverse.out_offset = runs[0].stored * 2 * sizeof(T);
      scratch = std::max(scratch, chain.c2c_line(n, +1, home, stored_out, false, inverse, line));
      pr.out = run_pruning::direct;
    } else {
      scratch = std::max(scratch, chain.c2c_line(n, +1, home, home, true, inverse, line));
      run_line_stages(c.prec, pl.epi, true, n, runs,
                      {.src = slot::block, .dst = slot::out, .dst_stride = l.out_stride}, chunk,
                      false, line);
      pr.out = inverse.out_head ? run_pruning::half : run_pruning::none;
    }
  } else if (pl.epi.separate) {
    scratch = std::max(scratch, chain.c2c_line(n, +1, home, home, true, inverse, line));
    line.push_back(edge_pass_stage(c.prec, pl.epi, true, n, slot::block, 1, slot::out, l.out_stride,
                                   chunk, false, pl.epi.axis_stride));
  } else {
    if (pl.epi.fused) {
      inverse.out = edge_op::factor;
      inverse.out_factor = epilogue_tag;
    } else if (pl.at == scale_at::inverse_out) {
      inverse.out = edge_op::scale;
    }
    scratch =
        std::max(scratch, chain.c2c_line(n, +1, home, {.where = slot::out, .stride = l.out_stride},
                                         false, inverse, line));
  }
  end_program(p, line);
  c.block_elements = n;
  lay_out_workspace(c, scratch, 2 * sizeof(T), false, 0);
  const std::string radices = chain.radix_text(n);
  c.axis_notes.push_back("line (n = " + std::to_string(n) + "): forward " + radices +
                         ", then backward " + radices + " (shared twiddle tables)");
  return false;
}

// The real chain pair (<T, C, T>, task 5.2c): batch loops, then the stages of one line,
//
//   [prologue pass into the block] -> r2c chain into the block -> [middle pass on its n/2 + 1
//   bins] -> c2r chain out of the block (or back into it) -> [epilogue pass from the block]
//
// built by chain_builder::r2c_line and c2r_line, which take each line's real route (the pairing
// fused into the half-length chain's last or first pass, a leaf, a separate pairing pass, odd n by
// the full-length chain, a strided line through scratch). The block holds the n reals the
// prologue's pass writes (n/2 complex values at most) and then the spectrum in place; the c2r
// chain reads it there, writing it only on its way to an epilogue pass. Every factor is a separate
// pass on the line in cache (place, fuse false); the scales' product goes to such a pass, else to
// the c2r chain's output.
//
// With fan-out (task 5.2e) the spectrum stays in the block and the scales' product goes to the
// prologue's pass or the r2c chain's output (place_fanout). Where each copy writes its output itself
// (no epilogue factor, accumulation or output runs), copy k is the middle's pass from the block
// into scratch b (copy k's factor) and the c2r chain from there to copy k's output (or the c2r
// chain from the block, without a middle factor), which only read the block: a team shares the
// copies. Otherwise copy k's line goes to the block's second region, as for the <C, C, C> pair
// (second_region), and a team runs a line's copies one after another.
template <class T>
void build_real_pair(plan_core& c, const pipeline_view& d, const row_layout& l, const vocab& v,
                     const build_settings& settings, placed& pl, fanout_marks& fan) {
  const bool fanned = d.fanout > 1;
  pl = fanned ? place_fanout(v, d.line.n, false, false) : place(v, d.line.n, false, false);
  const target_profile profile =
      plan_profile(c, of<T>(*find_kernels(c.isa_target)).complex_lanes, settings);
  chain_builder<T> chain = plan_chain<T>(c, profile, settings);
  program& p = c.prog;
  const std::size_t n = d.line.n;
  const std::size_t bins = n / 2 + 1;
  add_batch_loops(p, d, l, static_cast<std::ptrdiff_t>(sizeof(T)));

  std::vector<stage> line;
  std::size_t scratch = 0;
  const std::size_t chunk = settings.chunk_points;
  endpoint forward_src{.where = slot::in, .stride = l.in_stride};
  bool forward_aliased = false;
  if (pl.pro.runs.any()) {
    // The input's runs (task 5.2d), read into the block with zeros elsewhere: real lines take no
    // pruned passes yet (DESIGN.md 12.1).
    run_line_stages(c.prec, pl.pro, false, n, pl.pro.runs.of(0),
                    {.src = slot::in, .src_stride = l.in_stride}, chunk, true, line);
    forward_src = {.where = slot::block, .stride = 1};
    forward_aliased = true;
  } else if (pl.pro.separate) {
    line.push_back(edge_pass_stage(c.prec, pl.pro, false, n, slot::in, l.in_stride, slot::block, 1,
                                   chunk, true, pl.pro.axis_stride));
    forward_src = {.where = slot::block, .stride = 1};
    forward_aliased = true;
  }
  const endpoint home{.where = slot::block, .stride = 1};
  scratch = std::max(scratch, chain.r2c_line(n, forward_src, home, forward_aliased,
                                             pl.at == scale_at::forward_out ? pl.scale : 1.0, line));
  const auto real_bytes = static_cast<std::ptrdiff_t>(sizeof(T));
  const auto complex_bytes = static_cast<std::ptrdiff_t>(2 * sizeof(T));
  if (fanned && !pl.epi.separate && !pl.epi.runs.any()) {
    std::vector<stage> body;
    endpoint from = home;
    if (pl.mid.separate) {
      body.push_back(middle_stage(c.prec, pl.mid, bins, chunk, slot::block, slot::b));
      from = {.where = slot::b, .stride = 1};
      scratch = std::max(scratch, bins);
    }
    scratch = std::max(
        scratch, chain.c2r_line(n, from, {.where = slot::out, .stride = l.out_stride}, 1.0, body));
    end_with_copies(p, d, pl.mid, real_bytes, 0, line, body, fan);
    c.block_elements = bins;
  } else if (fanned) {
    std::vector<stage> body;
    if (pl.mid.separate) {
      body.push_back(middle_stage(c.prec, pl.mid, bins, chunk));
    }
    scratch = std::max(scratch, chain.c2r_line(n, home, home, 1.0, body));
    if (pl.epi.runs.any()) {
      run_line_stages(c.prec, pl.epi, true, n, pl.epi.runs.of(0),
                      {.src = slot::block, .dst = slot::out, .dst_stride = l.out_stride}, chunk, true,
                      body);
    } else {
      body.push_back(edge_pass_stage(c.prec, pl.epi, true, n, slot::block, 1, slot::out,
                                     l.out_stride, chunk, true, pl.epi.axis_stride));
    }
    end_with_copies(p, d, pl.mid, real_bytes, static_cast<std::ptrdiff_t>(bins) * complex_bytes,
                    line, body, fan);
    c.block_elements = 2 * bins;
  }
  if (fanned) {
    lay_out_workspace(c, scratch, 2 * sizeof(T), false, 0);
    const bool strided_out = !pl.epi.separate && !pl.epi.runs.any() && l.out_stride != 1;
    c.axis_notes.push_back("line (n = " + std::to_string(n) + "): forward " +
                           chain.real_text(n, true, !pl.pro.separate && !pl.pro.runs.any() &&
                                                        l.in_stride != 1) +
                           ", then backward " + chain.real_text(n, false, strided_out) +
                           " per copy");
    return;
  }
  if (pl.mid.separate) {
    line.push_back(middle_stage(c.prec, pl.mid, bins, chunk));
  }
  if (pl.epi.runs.any()) {
    // The output's runs: the c2r chain ends in the block, the store stage writes the runs.
    const double scale = pl.at == scale_at::inverse_out ? pl.scale : 1.0;
    scratch = std::max(scratch, chain.c2r_line(n, home, home, scale, line));
    run_line_stages(c.prec, pl.epi, true, n, pl.epi.runs.of(0),
                    {.src = slot::block, .dst = slot::out, .dst_stride = l.out_stride}, chunk, true,
                    line);
  } else if (pl.epi.separate) {
    scratch = std::max(scratch, chain.c2r_line(n, home, home, 1.0, line));
    line.push_back(edge_pass_stage(c.prec, pl.epi, true, n, slot::block, 1, slot::out, l.out_stride,
                                   chunk, true, pl.epi.axis_stride));
  } else {
    const double scale = pl.at == scale_at::inverse_out ? pl.scale : 1.0;
    scratch =
        std::max(scratch, chain.c2r_line(n, home, {.where = slot::out, .stride = l.out_stride},
                                         scale, line));
  }
  end_program(p, line);
  c.block_elements = bins;
  lay_out_workspace(c, scratch, 2 * sizeof(T), false, 0);
  const bool strided_in = !pl.pro.separate && !pl.pro.runs.any() && l.in_stride != 1;
  const bool strided_out = !pl.epi.separate && !pl.epi.runs.any() && l.out_stride != 1;
  c.axis_notes.push_back("line (n = " + std::to_string(n) + "): forward " +
                         chain.real_text(n, true, strided_in) + ", then backward " +
                         chain.real_text(n, false, strided_out));
}

// A fan-out's copies (task 5.2e; DESIGN.md 18.2, proposal 5): copy k writes at fanout_out_dist k
// output elements on, which must overlap neither another copy nor another line, judged as a plan
// judges its batch (sorted magnitudes, each beyond the reach of the smaller), the copies one more
// dimension of the output; and every span in bytes, the output's and the middle factor's over
// the copies, fits std::ptrdiff_t. Returns the bytes the copies add to the output's reach on
// each side.
std::pair<std::ptrdiff_t, std::ptrdiff_t> check_copies(const pipeline_view& d, const row_layout& l,
                                                       const run_plan& runs,
                                                       std::size_t out_bytes,
                                                       std::size_t factor_bytes) {
  constexpr auto limit = static_cast<std::size_t>(PTRDIFF_MAX);
  const std::size_t copies = d.fanout - 1;
  const auto magnitude = [](std::ptrdiff_t v) {
    return v < 0 ? static_cast<std::size_t>(-(v + 1)) + 1 : static_cast<std::size_t>(v);
  };
  const auto fits = [&](std::size_t dist, std::size_t bytes) {
    return dist == 0 || (copies <= limit / dist && copies * dist <= limit / bytes);
  };
  const std::size_t out_dist = magnitude(d.fanout_out_dist);
  if (!fits(out_dist, out_bytes) ||
      copies * out_dist * out_bytes >
          limit - static_cast<std::size_t>(l.out_hi - l.out_lo)) {
    refuse("desc.fanout_out_dist", "times desc.fanout - 1 makes the output's span overflow");
  }
  if (factor_bytes != 0 && !fits(magnitude(d.fanout_factor_dist), factor_bytes)) {
    refuse("desc.fanout_factor_dist",
           "times desc.fanout - 1 makes the middle factor's span overflow");
  }
  std::vector<std::pair<std::size_t, std::size_t>> dims; // |distance|, extent
  std::size_t reach = d.line.n;
  if (runs.out.any()) {
    reach = 0;
    for (const line_run& r : runs.out.of(0)) {
      reach = std::max(reach, r.stored + r.count);
    }
  }
  if (reach > 1) {
    dims.emplace_back(magnitude(l.out_stride), reach);
  }
  for (std::size_t b = 0; b < d.batches.size(); ++b) {
    if (d.batches[b].count > 1) {
      dims.emplace_back(magnitude(l.out_dist[b]), d.batches[b].count);
    }
  }
  dims.emplace_back(out_dist, d.fanout);
  std::ranges::sort(dims);
  std::size_t covered = 0;
  for (const auto& [dist, extent] : dims) {
    if (dist <= covered) {
      refuse("desc.fanout_out_dist", "makes copies' output elements overlap each other or other "
                                     "lines' (DESIGN.md 18.2, proposal 5: overlapping copies are "
                                     "refused)");
    }
    covered += (extent - 1) * dist;
  }
  const auto span = static_cast<std::ptrdiff_t>(copies * out_dist * out_bytes);
  return d.fanout_out_dist < 0 ? std::pair{-span, std::ptrdiff_t{0}}
                               : std::pair{std::ptrdiff_t{0}, span};
}

// describe()'s line for the runs of a depth-first pair: one run per side read by the forward
// first pass or written by the last top pass (run_pruning::direct), others through scratch b or
// from the workspace block.
std::string depth_runs_text(const run_plan& runs, bool accumulate, const run_prunings& pr,
                            std::size_t passes) {
  const auto count = [](const side_runs& r) {
    const std::size_t k = r.of(0).size();
    return std::to_string(k) + (k == 1 ? " run of " : " runs of ") +
           std::to_string(r.positions(0)) + (r.positions(0) == 1 ? " position" : " positions");
  };
  std::string text = "stored runs:";
  if (runs.in.any()) {
    text += " the input's " + count(runs.in) +
            (pr.in == run_pruning::direct
                 ? ", read where it is stored by the forward chain's first pass, zeros beyond"
                 : ", read with zeros elsewhere into scratch in a separate pass");
  }
  if (runs.out.any()) {
    text += std::string(runs.in.any() ? ";" : "") + " the output's " + count(runs.out) +
            ", the rest not written, " +
            (pr.out == run_pruning::direct
                 ? "written where it is stored by the backward chain's last top pass"
                 : std::string(accumulate ? "accumulated" : "stored") +
                       " from the workspace block in a separate pass");
  }
  return text + "; they cost " + std::to_string(passes) +
         (passes == 1 ? " separate pass" : " separate passes") + " over the line";
}

} // namespace

std::shared_ptr<const plan_core> make_pipeline_core(const pipeline_view& d, const options& opts,
                                                    const build_settings& settings) {
  switch (d.kind) {
  case pipeline_kind::complex_pair:
  case pipeline_kind::real_convolution:
    break;
  case pipeline_kind::real_middle:
    refuse("desc", "is a <C, T, C> pipeline, which is not built yet (task 5.3)");
  }
  // <T, C, T>: an r2c then a c2r transform, whatever `first` says (pipeline_desc).
  const bool real = d.kind == pipeline_kind::real_convolution;
  if (!real && d.first != direction::forward && d.first != direction::backward) {
    refuse("desc.first", "must be direction::forward or direction::backward");
  }
  if (!real && d.first == direction::backward) {
    refuse("desc.first", "is direction::backward: a pipeline whose inverse transform runs first "
                         "(a spectral pair) is not built yet (task 5.3)");
  }
  if (d.fanout == 0) {
    refuse("desc.fanout", "is 0; one inverse transform per forward is 1");
  }
  if (d.fanout > 1 && opts.in_place) {
    refuse("desc.fanout", "is above 1 on an in-place pipeline, which has one inverse transform per "
                          "forward (DESIGN.md 18.2, proposal 6)");
  }
  if (!std::isfinite(opts.measure_seconds) || opts.measure_seconds < 0) {
    refuse("options.measure_seconds", "must be finite and not negative");
  }
  const shape_view line{
      .kind = transform_kind::c2c,
      .prec = d.prec,
      .dir = direction::forward,
      .axes = std::span<const axis>(&d.line, 1),
      .batches = d.batches,
  };
  // Stored runs of the line (task 5.2d), validated as a plan's (core/runs.hpp) on the line's c2c
  // shape (n logical positions on each side, for <T, C, T> too): their stored extents set the
  // default distances and the reach.
  run_plan runs;
  if (!d.runs.empty()) {
    if (opts.in_place) {
      refuse("desc.runs", "are not taken by an in-place pipeline yet (task 5.2d builds them out "
                          "of place)");
    }
    shape_view with_runs = line;
    with_runs.runs = d.runs;
    runs = renaming([&] { return resolve_runs(with_runs, options{}); });
    // DESIGN.md 18.2, proposal 8: as plans refuse runs beside an edge's factor (task 5.1d).
    for (const auto& [e, name] : {std::pair{&d.prologue, "desc.prologue.mul.kind"},
                                  std::pair{&d.epilogue, "desc.epilogue.mul.kind"}}) {
      if (e->kind != factor_kind::none) {
        refuse(name, "is a factor, which a pipeline with stored runs does not take yet (DESIGN.md "
                     "18.2, proposal 8)");
      }
    }
  }
  const row_layout layout =
      renaming([&] { return resolve_row_layout(line, opts.in_place, real, runs); });

  // The vocabulary, judged as a plan's edges are: on the line's c2c shape, or for <T, C, T> the
  // prologue on the input of its r2c shape and the epilogue on the output of its c2r shape (real
  // sides, which take real factors only), and the middle on the r2c shape's output, the n/2 + 1
  // bins.
  if (d.prologue.accumulate) {
    refuse("desc.prologue.accumulate", "is for the epilogue only");
  }
  // Block callbacks (fftn/callback.hpp) reach pipelines in task 5.3d, with the middle's; until
  // then an edge's callback is refused rather than dropped.
  if (d.prologue.call != nullptr) {
    refuse("desc.prologue.call.fn",
           "is a block callback: a pipeline's edges take none yet (task 5.3d)");
  }
  if (d.epilogue.call != nullptr) {
    refuse("desc.epilogue.call.fn",
           "is a block callback: a pipeline's edges take none yet (task 5.3d)");
  }
  shape_view r2c_line = line;
  r2c_line.kind = transform_kind::r2c;
  shape_view c2r_line = line;
  c2r_line.kind = transform_kind::c2r;
  const vocab v = renaming([&] {
    check_edge_scale(opts.scale, d.prec, "options.scale");
    edge_view middle = d.middle;
    middle.accumulate = false;
    edge_side pro = resolve_edge_side(real ? r2c_line : line, d.prologue, false, "desc.prologue");
    edge_side mid = resolve_edge_side(real ? r2c_line : line, middle, true, "desc.middle");
    edge_side epi = resolve_edge_side(real ? c2r_line : line, d.epilogue, true, "desc.epilogue");
    if (opts.scale != 1.0 && d.epilogue.scale != 1.0) {
      refuse("desc.epilogue.scale", "and options.scale are both given; options.scale is a "
                                    "shortcut for desc.epilogue.scale, so give one of them");
    }
    const double scale = opts.scale * d.prologue.scale * d.middle.scale * d.epilogue.scale;
    if (!scale_fits_precision(scale, d.prec)) {
      refuse("desc.prologue.scale", "times desc.middle.scale, desc.epilogue.scale and "
                                    "options.scale is not finite or outside the range of the "
                                    "pipeline's precision");
    }
    if (epi.accumulate && opts.in_place) {
      refuse("desc.epilogue.accumulate", "is refused on an in-place pipeline, whose output's "
                                         "prior values are its input");
    }
    return vocab{.pro = std::move(pro),
                 .mid = std::move(mid),
                 .epi = std::move(epi),
                 .scale = scale,
                 .runs = runs};
  });

  auto c = std::make_shared<plan_core>();
  c->kind = transform_kind::c2c;
  c->prec = d.prec;
  c->dir = direction::forward;
  c->in_place = opts.in_place;
  c->lengths = {d.line.n};
  for (const batch& b : d.batches) {
    c->counts.push_back(b.count);
  }
  c->in_lo = layout.in_lo;
  c->in_hi = layout.in_hi;
  c->out_lo = layout.out_lo;
  c->out_hi = layout.out_hi;
  const bool fanned = d.fanout > 1;
  if (fanned) {
    const std::size_t real_bytes = d.prec == precision::f64 ? sizeof(double) : sizeof(float);
    const auto [lo, hi] = check_copies(d, layout, runs, real ? real_bytes : 2 * real_bytes,
                                       v.mid.kind != factor_kind::none ? v.mid.factor_bytes : 0);
    c->out_lo += lo;
    c->out_hi += hi;
  }
  renaming([&] {
    c->isa_target = plan_target_of(opts);
    return 0;
  });
  placed pl;
  run_prunings pr;
  fanout_marks fan;
  bool depth = false;
  if (real) {
    if (d.prec == precision::f64) {
      build_real_pair<double>(*c, d, layout, v, settings, pl, fan);
    } else {
      build_real_pair<float>(*c, d, layout, v, settings, pl, fan);
    }
  } else if (d.prec == precision::f64) {
    depth = build_pair<double>(*c, d, layout, v, settings, pl, pr, fan);
  } else {
    depth = build_pair<float>(*c, d, layout, v, settings, pl, pr, fan);
  }
  c->edge_records = mark_cursors(c->prog, pl, fan);
  c->factor_args =
      static_cast<std::uint8_t>((pl.pro.kind != factor_kind::none ? moving_in_factor : 0) |
                                (pl.epi.kind != factor_kind::none ? moving_out_factor : 0) |
                                (pl.mid.kind != factor_kind::none ? moving_mid_out_factor : 0));
  // A fan-out runs on the edge driver even without factors: its copies loop and its offsets into
  // the block are the edge driver's records (task 5.2e).
  c->needs_args = c->factor_args != 0 || fanned;
  const bool fan_serial = fanned && std::get<loop>(c->prog.records[fan.loop]).serial_lines;
  // Batched forms from the start (X65): the flat body, or the calls of a longer line unrolled, on
  // either driver.
  mark_flat(c->prog, settings.unroll_calls);
  const auto complex_bytes = static_cast<std::ptrdiff_t>(
      d.prec == precision::f64 ? 2 * sizeof(double) : 2 * sizeof(float));
  if (c->needs_args) {
    c->edge_batched =
        make_edge_forms(c->prog, c->edge_records, settings.unroll_calls, 0, 0, complex_bytes);
  }
  c->graph = build_work_graph(c->prog, d.line.n, settings.chunk_points);
  c->profile = settings.profile;
  c->runs_inline = runs_inline(c->graph, c->profile);
  c->built = depth ? strategy_choice::depth : strategy_choice::chain;
  c->strategy_name = real    ? "the real chain pair"
                     : depth ? "the depth-first pair"
                             : "the chain pair";

  // describe().
  const std::size_t lines = c->prog.transforms;
  c->pipeline_title = std::string("fftn pipeline: ") +
                      (real ? "<t, c, t> r2c then c2r, " : "<c, c, c> forward then backward, ") +
                      (d.prec == precision::f64 ? "double" : "float") +
                      ", n = " + std::to_string(d.line.n) + ", " + std::to_string(lines) +
                      (lines == 1 ? " line" : " lines") +
                      (opts.in_place ? ", in place" : ", out of place");
  c->strategy_note =
      real ? "the real chain pair: each line through the r2c chain into the workspace block (its "
             "scratch-resident home, which the batch loops never move), the middle there on its "
             "n/2 + 1 bins, then the c2r chain out of it (DESIGN.md 12.2)"
      : depth
          ? "the depth-first pair: each line through the forward depth-first chain (its sub-rows "
            "finished one at a time on their blocks of the workspace block), its top pass into "
            "scratch with the middle, then the backward depth-first chain from there (DESIGN.md "
            "12.2)"
          : "the chain pair: each line through the forward chain into the workspace block (its "
            "scratch-resident home, which the batch loops never move), the middle there, then "
            "the backward chain out of it (DESIGN.md 12.2)";
  const auto stages = std::ranges::count_if(
      c->prog.records, [](const record& r) { return std::holds_alternative<stage>(r); });
  const std::size_t calls = c->needs_args ? c->edge_batched.calls.size() : c->prog.calls.size();
  if (c->prog.flat) {
    c->axis_notes.push_back("form: flat, the line's " + std::to_string(stages) +
                            " stages per line with the slots moved by the batch loops");
  } else if (calls != 0) {
    c->axis_notes.push_back("form: unrolled, " + std::to_string(calls) + " kernel calls per line");
  } else {
    c->axis_notes.push_back("form: walked, the program's records per line");
  }
  // A side with runs has their read or store stage instead (runs_text below counts them).
  const std::size_t passes = (pl.pro.separate && !runs.in.any() ? 1U : 0U) +
                             (pl.mid.separate ? 1U : 0U) +
                             (pl.epi.separate && !runs.out.any() ? 1U : 0U);
  // With fan-out the middle and the epilogue are once per copy, the middle's pass from the
  // spectrum to where the copy's inverse starts.
  const char* fan_middle = fan_serial ? "in a separate pass per copy into the workspace block's "
                                        "second region"
                                      : "in a separate pass per copy into scratch";
  std::string ops =
      operation_text("prologue", pl.pro, "in the forward chain's first pass",
                     depth ? "in a separate pass into scratch"
                           : "in a separate pass into the workspace block") +
      "\n" +
      operation_text("middle", pl.mid,
                     fanned  ? "in the backward chain's first pass, copy k's factor"
                     : depth ? "in the forward chain's last top pass"
                             : "in the forward chain's last pass",
                     fanned && depth ? "in a separate pass per copy into scratch"
                     : fanned        ? fan_middle
                     : real ? "in a separate pass on the line's n/2 + 1 bins in the workspace block"
                     : depth ? "in a separate pass on the line in scratch"
                             : "in a separate pass on the line in the workspace block") +
      "\n" +
      operation_text("epilogue", pl.epi,
                     fanned ? "in the backward chain's last pass, per copy"
                            : "in the backward chain's last pass",
                     fan_serial && !depth
                         ? "in a separate pass per copy from the workspace block's second region"
                     : fanned ? "in a separate pass per copy from the workspace block"
                              : "in a separate pass from the workspace block");
  if (pl.scale != 1.0) {
    const char* where = "a separate pass";
    switch (pl.at) {
    case scale_at::forward_in:
      where = "the forward chain's first pass";
      break;
    case scale_at::forward_out:
      where = "the forward chain's last pass";
      break;
    case scale_at::inverse_in:
      where = "the backward chain's first pass";
      break;
    case scale_at::inverse_out:
      where = depth ? "the backward chain's last top pass" : "the backward chain's last pass";
      break;
    case scale_at::none:
    case scale_at::pass:
      break;
    }
    ops += std::string("\nscales: one product, applied once in ") + where;
  }
  // The depth-first pair's line is beyond the cache its sub-rows run in: a separate pass sweeps it.
  const char* sweep =
      depth ? " over the line, each a sweep of it" : " on the line in cache, no sweep of memory";
  ops += "\nthe vocabulary costs " +
         (passes == 0 ? std::string("no separate pass")
                      : std::to_string(passes) +
                            (passes == 1 ? " separate pass" : " separate passes") + sweep);
  if (runs.any()) {
    const std::size_t run_passes = (runs.in.any() && pr.in != run_pruning::direct ? 1U : 0U) +
                                   (runs.out.any() && pr.out != run_pruning::direct ? 1U : 0U);
    ops += "\n" + (depth ? depth_runs_text(runs, pl.epi.accumulate, pr, run_passes)
                         : runs_text(runs, pl.epi.accumulate, "the workspace block", run_passes,
                                     pr.in, pr.out) +
                               " on the line in cache");
  }
  if (fanned) {
    // Task 5.2e: where the copies run, and whether a team shares them.
    ops += "\nfan-out: " + std::to_string(d.fanout) +
           " copies of the backward chain per line, from its spectrum, which stays in the "
           "workspace block" +
           (depth ? "'s second region" : "") + "; copy k's output " +
           std::to_string(d.fanout_out_dist) + " k elements on (fanout_out_dist)" +
           (pl.mid.kind != factor_kind::none
                ? ", its middle factor " + std::to_string(d.fanout_factor_dist) +
                      " k elements on (fanout_factor_dist)"
                : std::string()) +
           (fan_serial ? std::string("; each copy works in the workspace block") +
                             (depth ? "'s first region" : "'s second region") +
                             ", so a team runs a line's copies one after another"
                       : "; a team shares a line's copies, each on a member's own scratch");
  }
  c->edges_note = ops;
  if (opts.effort != planning::rules || opts.use != nullptr) {
    c->planning_note = "by the rules: options.effort and options.use are ignored for pipelines "
                       "until task 5.2g";
  }
  return c;
}

std::shared_ptr<const plan_core> make_pipeline_core(const pipeline_view& desc,
                                                    const options& opts) {
  return make_pipeline_core(desc, opts, build_settings{});
}

} // namespace fftn::inline v0::detail

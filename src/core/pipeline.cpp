// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/pipeline.hpp"

#include "core/driver.hpp"
#include "core/edge_driver.hpp"
#include "core/edges.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
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
template <class F> auto renaming(F&& f) {
  try {
    return f();
  } catch (const std::invalid_argument& e) {
    std::string m = e.what();
    replace_all(m, "fftn::plan: ", "fftn::pipeline: ");
    replace_all(m, "shape.axes[0]", "desc.line");
    replace_all(m, "shape.batches", "desc.batches");
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

// Where the one product of the scales goes: a separate pass, or a free edge of a chain.
enum class scale_at : std::uint8_t { none, pass, forward_in, inverse_in, inverse_out };

struct placed {
  edge_side pro;
  edge_side mid;
  edge_side epi;
  double scale = 1.0;
  scale_at at = scale_at::none;
};

// Each operation's place: a complex factor of stride 1 along the line (or on a line of one point)
// fuses into its chain's pass; anything else, and accumulation, is an edge pass on the line in the
// workspace block. The scale goes to a separate pass where there is one (it costs nothing
// there), else to the inverse chain's output, else (the epilogue's factor fused) to its input.
placed place(edge_side pro, edge_side mid, edge_side epi, double scale, std::size_t n) {
  // A stride of 0 along a line of more than one point broadcasts one value, which the fused
  // passes (one value per element) cannot read (side_of gives 0 for a line of one point, where
  // the stride reads nothing more).
  const auto fusable = [n](const edge_side& s) {
    return s.kind == factor_kind::complex && (s.axis_stride == 1 || (s.axis_stride == 0 && n == 1));
  };
  placed p{.pro = std::move(pro), .mid = std::move(mid), .epi = std::move(epi), .scale = scale};
  for (edge_side* s : {&p.pro, &p.mid, &p.epi}) {
    s->fused = fusable(*s);
    s->separate = s->kind != factor_kind::none && !s->fused;
  }
  p.epi.fused = p.epi.fused && !p.epi.accumulate;
  p.epi.separate = p.epi.accumulate || (p.epi.kind != factor_kind::none && !p.epi.fused);
  if (scale != 1.0) {
    if (p.epi.separate) {
      p.epi.scale = scale;
      p.at = scale_at::pass;
    } else if (p.mid.separate) {
      p.mid.scale = scale;
      p.at = scale_at::pass;
    } else if (p.pro.separate) {
      p.pro.scale = scale;
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

// The edge driver's records (core/edge_driver.hpp): the stages built with a tag, and the edge
// passes (which edge_pass_stage marks), keep their kernels there and carry guards; the batch
// loops move the three cursors by the factors' batch strides (one loop per batch dimension, in
// the description's order).
std::vector<edge_record> mark_cursors(program& p, const placed& pl) {
  std::vector<edge_record> e(p.records.size());
  for (std::size_t i = 0; i < p.records.size(); ++i) {
    stage* st = std::get_if<stage>(&p.records[i]);
    if (st == nullptr) {
      continue;
    }
    if (st->args.in_factor == prologue_tag) {
      st->args.in_factor = nullptr;
      st->moving_factors |= moving_in_factor;
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

template <class T>
void build_pair(plan_core& c, const pipeline_view& d, const row_layout& l, const placed& pl,
                const build_settings& settings) {
  const target_profile profile =
      plan_profile(c, of<T>(*find_kernels(c.isa_target)).complex_lanes, settings);
  chain_builder<T> chain = plan_chain<T>(c, profile, settings);
  program& p = c.prog;
  const std::size_t n = d.line.n;
  const auto cb = static_cast<std::ptrdiff_t>(2 * sizeof(T));
  p.transforms = 1;
  for (std::size_t b = 0; b < d.batches.size(); ++b) {
    p.records.emplace_back(loop{
        .kind = loop_kind::batch,
        .count = d.batches[b].count,
        .step = {l.in_dist[b] * cb, l.out_dist[b] * cb, 0},
    });
    p.transforms *= d.batches[b].count;
  }
  p.batch_loops = d.batches.size();

  std::vector<stage> line;
  std::size_t scratch = 0;
  const std::size_t chunk = settings.chunk_points;
  // The forward transform reads the input, or the prologue's pass has put the line in the block.
  endpoint forward_src{.where = slot::in, .stride = l.in_stride};
  bool forward_aliased = false;
  if (pl.pro.separate) {
    line.push_back(edge_pass_stage(c.prec, pl.pro, false, n, slot::in, l.in_stride, slot::block, 1,
                                   chunk, false, pl.pro.axis_stride));
    forward_src = {.where = slot::block, .stride = 1};
    forward_aliased = true;
  }
  line_edges forward;
  forward.scale = pl.scale;
  if (pl.pro.fused) {
    forward.in = edge_op::factor;
    forward.in_factor = prologue_tag;
  } else if (pl.at == scale_at::forward_in) {
    forward.in = edge_op::scale;
  }
  if (pl.mid.fused) {
    forward.out = edge_op::factor;
    forward.out_factor = middle_tag;
  }
  const endpoint home{.where = slot::block, .stride = 1};
  scratch =
      std::max(scratch, chain.c2c_line(n, -1, forward_src, home, forward_aliased, forward, line));
  if (pl.mid.separate) {
    stage m = edge_pass_stage(c.prec, pl.mid, true, n, slot::block, 1, slot::block, 1, chunk, false,
                              pl.mid.axis_stride);
    m.moving_factors = moving_mid_out_factor;
    m.name.replace(0, std::string_view("edge out").size(), "edge middle");
    line.push_back(std::move(m));
  }
  line_edges inverse;
  inverse.scale = pl.scale;
  if (pl.at == scale_at::inverse_in) {
    inverse.in = edge_op::scale;
  }
  if (pl.epi.separate) {
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
  for (stage& st : line) {
    p.records.emplace_back(std::move(st));
  }
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    std::get<loop>(p.records[b]).body_end = static_cast<std::uint32_t>(p.records.size());
  }
  c.block_elements = n;
  lay_out_workspace(c, scratch, 2 * sizeof(T), false);
  const std::string radices = chain.radix_text(n);
  c.axis_notes.push_back("line (n = " + std::to_string(n) + "): forward " + radices +
                         ", then backward " + radices + " (shared twiddle tables)");
}

} // namespace

std::shared_ptr<const plan_core> make_pipeline_core(const pipeline_view& d, const options& opts,
                                                    const build_settings& settings) {
  switch (d.kind) {
  case pipeline_kind::complex_pair:
    break;
  case pipeline_kind::real_convolution:
    refuse("desc", "is a <T, C, T> pipeline (a real convolution), which is not built yet "
                   "(task 5.2c)");
  case pipeline_kind::real_middle:
    refuse("desc", "is a <C, T, C> pipeline, which is not built yet (task 5.3)");
  }
  if (d.first != direction::forward && d.first != direction::backward) {
    refuse("desc.first", "must be direction::forward or direction::backward");
  }
  if (d.first == direction::backward) {
    refuse("desc.first", "is direction::backward: a pipeline whose inverse transform runs first "
                         "(a spectral pair) is not built yet (task 5.3)");
  }
  if (d.fanout == 0) {
    refuse("desc.fanout", "is 0; one inverse transform per forward is 1");
  }
  if (d.fanout > 1) {
    refuse("desc.fanout", "is above 1: fan-out is not built yet (task 5.2e)");
  }
  if (!d.runs.empty()) {
    refuse("desc.runs",
           "are given: stored runs on a pipeline's line are not built yet (task 5.2d)");
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
  const row_layout layout = renaming([&] { return resolve_row_layout(line, opts.in_place); });

  // The vocabulary, judged as a plan's edges are, on the line's c2c shape.
  if (d.prologue.accumulate) {
    refuse("desc.prologue.accumulate", "is for the epilogue only");
  }
  const placed pl = renaming([&] {
    check_edge_scale(opts.scale, d.prec, "options.scale");
    edge_view middle = d.middle;
    middle.accumulate = false;
    edge_side pro = resolve_edge_side(line, d.prologue, false, "desc.prologue");
    edge_side mid = resolve_edge_side(line, middle, true, "desc.middle");
    edge_side epi = resolve_edge_side(line, d.epilogue, true, "desc.epilogue");
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
    return place(std::move(pro), std::move(mid), std::move(epi), scale, d.line.n);
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
  renaming([&] {
    c->isa_target = plan_target_of(opts);
    return 0;
  });
  if (d.prec == precision::f64) {
    build_pair<double>(*c, d, layout, pl, settings);
  } else {
    build_pair<float>(*c, d, layout, pl, settings);
  }
  c->edge_records = mark_cursors(c->prog, pl);
  c->factor_args =
      static_cast<std::uint8_t>((pl.pro.kind != factor_kind::none ? moving_in_factor : 0) |
                                (pl.epi.kind != factor_kind::none ? moving_out_factor : 0) |
                                (pl.mid.kind != factor_kind::none ? moving_mid_out_factor : 0));
  c->needs_args = c->factor_args != 0;
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
  c->built = strategy_choice::chain;
  c->strategy_name = "the chain pair";

  // describe().
  const std::size_t lines = c->prog.transforms;
  c->pipeline_title = std::string("fftn pipeline: <c, c, c> forward then backward, ") +
                      (d.prec == precision::f64 ? "double" : "float") +
                      ", n = " + std::to_string(d.line.n) + ", " + std::to_string(lines) +
                      (lines == 1 ? " line" : " lines") +
                      (opts.in_place ? ", in place" : ", out of place");
  c->strategy_note = "the chain pair: each line through the forward chain into the workspace "
                     "block (its scratch-resident home, which the batch loops never move), the "
                     "middle there, then the backward chain out of it (DESIGN.md 12.2)";
  std::size_t stages = 0;
  for (const record& r : c->prog.records) {
    stages += std::holds_alternative<stage>(r) ? 1 : 0;
  }
  const std::size_t calls = c->needs_args ? c->edge_batched.calls.size() : c->prog.calls.size();
  if (c->prog.flat) {
    c->axis_notes.push_back("form: flat, the line's " + std::to_string(stages) +
                            " stages per line with the slots moved by the batch loops");
  } else if (calls != 0) {
    c->axis_notes.push_back("form: unrolled, " + std::to_string(calls) + " kernel calls per line");
  } else {
    c->axis_notes.push_back("form: walked, the program's records per line");
  }
  std::size_t passes = 0;
  for (const edge_side* s : {&pl.pro, &pl.mid, &pl.epi}) {
    passes += s->separate ? 1 : 0;
  }
  std::string ops = operation_text("prologue", pl.pro, "in the forward chain's first pass",
                                   "in a separate pass into the workspace block") +
                    "\n" +
                    operation_text("middle", pl.mid, "in the forward chain's last pass",
                                   "in a separate pass on the line in the workspace block") +
                    "\n" +
                    operation_text("epilogue", pl.epi, "in the backward chain's last pass",
                                   "in a separate pass from the workspace block");
  if (pl.scale != 1.0) {
    const char* where = "a separate pass";
    switch (pl.at) {
    case scale_at::forward_in:
      where = "the forward chain's first pass";
      break;
    case scale_at::inverse_in:
      where = "the backward chain's first pass";
      break;
    case scale_at::inverse_out:
      where = "the backward chain's last pass";
      break;
    case scale_at::none:
    case scale_at::pass:
      break;
    }
    ops += std::string("\nscales: one product, applied once in ") + where;
  }
  ops += "\nthe vocabulary costs " +
         (passes == 0
              ? std::string("no separate pass")
              : std::to_string(passes) + (passes == 1 ? " separate pass" : " separate passes") +
                    " on the line in cache, no sweep of memory");
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

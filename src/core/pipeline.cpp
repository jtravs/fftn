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

// Where the one product of the scales goes: a separate pass, or a free edge of a chain.
enum class scale_at : std::uint8_t { none, pass, forward_in, inverse_in, inverse_out };

struct placed {
  edge_side pro;
  edge_side mid;
  edge_side epi;
  double scale = 1.0;
  scale_at at = scale_at::none;
};

// Each operation's place: with `fuse` (the <C, C, C> chain pair), a complex factor of stride 1
// along the line (or on a line of one point) fuses into its chain's pass; anything else, and
// accumulation, is an edge pass on the line in the workspace block. The real chain pair (`fuse`
// false) fuses nothing: its sides take real factors, which no pass fuses, and a middle fused into
// the pairing passes is an arm of experiment X67 (PLAN 5.2c). The scale goes to a separate pass
// where there is one (it costs nothing there), else to the inverse chain's output, else (the
// epilogue's factor fused) to its input.
placed place(edge_side pro, edge_side mid, edge_side epi, double scale, std::size_t n, bool fuse) {
  // A stride of 0 along a line of more than one point broadcasts one value, which the fused
  // passes (one value per element) cannot read (side_of gives 0 for a line of one point, where
  // the stride reads nothing more).
  const auto fusable = [n, fuse](const edge_side& s) {
    return fuse && s.kind == factor_kind::complex &&
           (s.axis_stride == 1 || (s.axis_stride == 0 && n == 1));
  };
  placed p{.pro = std::move(pro), .mid = std::move(mid), .epi = std::move(epi), .scale = scale};
  for (edge_side* s : {&p.pro, &p.mid, &p.epi}) {
    s->fused = fusable(*s);
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

// The middle's separate pass, in place on the spectrum of `bins` complex values in the workspace
// block, its factor through the middle's cursor.
stage middle_stage(precision prec, const edge_side& mid, std::size_t bins, std::size_t chunk) {
  stage m = edge_pass_stage(prec, mid, true, bins, slot::block, 1, slot::block, 1, chunk, false,
                            mid.axis_stride);
  m.moving_factors = moving_mid_out_factor;
  m.name.replace(0, std::string_view("edge out").size(), "edge middle");
  return m;
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
  return std::ranges::all_of(runs,
                             [n](const line_run& r) { return r.logical + r.count <= n / 2; });
}

// Whether the side is one run of exactly the line's first half, stored with stride 1: the pruned
// pass then reads or writes it where it is stored.
bool half_stored(const std::vector<line_run>& runs, std::size_t n, const endpoint& stored) {
  return runs.size() == 1 && runs[0].logical == 0 && 2 * runs[0].count == n && stored.stride == 1;
}

template <class T>
void build_pair(plan_core& c, const pipeline_view& d, const row_layout& l, const placed& pl,
                const build_settings& settings, run_prunings& pr) {
  const target_profile profile =
      plan_profile(c, of<T>(*find_kernels(c.isa_target)).complex_lanes, settings);
  chain_builder<T> chain = plan_chain<T>(c, profile, settings);
  program& p = c.prog;
  const std::size_t n = d.line.n;
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
  if (pl.mid.fused) {
    forward.out = edge_op::factor;
    forward.out_factor = middle_tag;
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
    inverse.out_head =
        inverse.out != edge_op::factor && within_half(runs, n) && chain.head_last(n);
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
  lay_out_workspace(c, scratch, 2 * sizeof(T), false);
  const std::string radices = chain.radix_text(n);
  c.axis_notes.push_back("line (n = " + std::to_string(n) + "): forward " + radices +
                         ", then backward " + radices + " (shared twiddle tables)");
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
template <class T>
void build_real_pair(plan_core& c, const pipeline_view& d, const row_layout& l, const placed& pl,
                     const build_settings& settings) {
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
  scratch = std::max(scratch, chain.r2c_line(n, forward_src, home, forward_aliased, 1.0, line));
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
  lay_out_workspace(c, scratch, 2 * sizeof(T), false);
  const bool strided_in = !pl.pro.separate && !pl.pro.runs.any() && l.in_stride != 1;
  const bool strided_out = !pl.epi.separate && !pl.epi.runs.any() && l.out_stride != 1;
  c.axis_notes.push_back("line (n = " + std::to_string(n) + "): forward " +
                         chain.real_text(n, true, strided_in) + ", then backward " +
                         chain.real_text(n, false, strided_out));
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
  if (d.fanout > 1) {
    refuse("desc.fanout", "is above 1: fan-out is not built yet (task 5.2e)");
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
  shape_view r2c_line = line;
  r2c_line.kind = transform_kind::r2c;
  shape_view c2r_line = line;
  c2r_line.kind = transform_kind::c2r;
  placed pl = renaming([&] {
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
    return place(std::move(pro), std::move(mid), std::move(epi), scale, d.line.n, !real);
  });
  pl.pro.runs = runs.in;
  pl.epi.runs = runs.out;

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
  run_prunings pr;
  if (real) {
    if (d.prec == precision::f64) {
      build_real_pair<double>(*c, d, layout, pl, settings);
    } else {
      build_real_pair<float>(*c, d, layout, pl, settings);
    }
  } else if (d.prec == precision::f64) {
    build_pair<double>(*c, d, layout, pl, settings, pr);
  } else {
    build_pair<float>(*c, d, layout, pl, settings, pr);
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
  c->strategy_name = real ? "the real chain pair" : "the chain pair";

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
  std::string ops = operation_text("prologue", pl.pro, "in the forward chain's first pass",
                                   "in a separate pass into the workspace block") +
                    "\n" +
                    operation_text("middle", pl.mid, "in the forward chain's last pass",
                                   real ? "in a separate pass on the line's n/2 + 1 bins in the "
                                          "workspace block"
                                        : "in a separate pass on the line in the workspace block") +
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
  if (runs.any()) {
    const std::size_t run_passes =
        (runs.in.any() && pr.in != run_pruning::direct ? 1U : 0U) +
        (runs.out.any() && pr.out != run_pruning::direct ? 1U : 0U);
    ops += "\n" + runs_text(runs, pl.epi.accumulate, "the workspace block", run_passes, pr.in,
                            pr.out) +
           " on the line in cache";
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

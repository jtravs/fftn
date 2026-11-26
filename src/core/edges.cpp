// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/edges.hpp"

#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/options.hpp"
#include "passes/pass_args.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

[[noreturn]] void reject(const std::string& argument, const std::string& why) {
  throw std::invalid_argument("fftn::plan: " + argument + " " + why);
}

// Whether a scale can be applied in the plan's precision: finite, and in float 0 or within
// float's range (converting a double beyond it to float is undefined, so the range is
// compared, not converted).
bool scale_fits(double scale, precision prec) {
  if (!std::isfinite(scale)) {
    return false;
  }
  if (prec == precision::f32) {
    const double m = std::abs(scale);
    return m <= static_cast<double>(std::numeric_limits<float>::max()) &&
           (scale == 0.0 || m >= static_cast<double>(std::numeric_limits<float>::denorm_min()));
  }
  return true;
}

void check_scale(double scale, precision prec, const std::string& name) {
  if (!std::isfinite(scale)) {
    reject(name, "must be finite");
  }
  if (!scale_fits(scale, prec)) {
    reject(name, "is outside the range of float, the plan's precision (0, or a magnitude from "
                 "the smallest positive float to the largest)");
  }
}

// Whether the side is real: the input of r2c, the output of c2r.
bool real_side(const shape_view& s, bool output) {
  return output ? s.kind == transform_kind::c2r : s.kind == transform_kind::r2c;
}

// The lengths of the side's logical coordinates along each axis: for c2c the axes' lengths;
// on the complex side of a real transform the last axis has n/2 + 1 bins.
std::size_t side_length(const shape_view& s, std::size_t d, bool output) {
  const std::size_t n = s.axes[d].n;
  const bool complex_half = d + 1 == s.axes.size() && ((output && s.kind == transform_kind::r2c) ||
                                                       (!output && s.kind == transform_kind::c2r));
  return complex_half ? n / 2 + 1 : n;
}

// Validates one edge and returns it as the plan applies it (fused and separate set later).
// name: the edge's field in messages.
edge_side side_of(const shape_view& s, const edge_view& e, bool output, const std::string& name) {
  check_scale(e.scale, s.prec, name + ".scale");
  if (e.accumulate && !output) {
    reject(name + ".accumulate", "is for the output edge only");
  }
  edge_side side;
  side.accumulate = e.accumulate;
  // The callback (fftn/callback.hpp, task 5.3a). A null fn is no callback, whose other fields
  // keep their defaults (DESIGN.md 18.2, "Callbacks" 4).
  const std::string call = name + ".call";
  if (e.call == nullptr) {
    if (e.call_mode != callback_mode::modify) {
      reject(call + ".mode", "is set without " + call +
                                 ".fn; a null fn is no callback, and its other fields keep "
                                 "their defaults");
    }
    if (e.call_scratch != 0) {
      reject(call + ".scratch_bytes", "is set without " + call +
                                          ".fn; a null fn is no callback, and its other fields "
                                          "keep their defaults");
    }
  } else {
    if (e.call_mode != callback_mode::modify && e.call_mode != callback_mode::produce &&
        e.call_mode != callback_mode::consume) {
      reject(call + ".mode", "is not a callback_mode");
    }
    if (!output && e.call_mode == callback_mode::consume) {
      reject(call + ".mode", "is consume on the input edge, which takes modify or produce");
    }
    if (output && e.call_mode == callback_mode::produce) {
      reject(call + ".mode", "is produce on the output edge, which takes modify or consume");
    }
    if (output && e.call_mode == callback_mode::consume && e.accumulate) {
      reject(call + ".mode", "is consume beside " + name +
                                 ".accumulate: a consumed block is not stored, so nothing "
                                 "could be accumulated");
    }
    // Bounded so that a workspace's size, with this much per lane, is computed without overflow.
    if (e.call_scratch > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 4) {
      reject(call + ".scratch_bytes", "is too large to address");
    }
    if (s.kind != transform_kind::c2c || s.axes.size() != 1) {
      reject(call + ".fn", "is a callback on a real or multi-axis plan, which takes none yet "
                           "(task 5.3a built them for c2c rows; tasks 5.3b and 5.3c add the "
                           "others)");
    }
    side.call = {.fn = e.call, .mode = e.call_mode, .scratch_bytes = e.call_scratch};
  }
  if (e.kind != factor_kind::none && e.kind != factor_kind::real &&
      e.kind != factor_kind::complex) {
    reject(name + ".mul.kind", "is not a factor_kind");
  }
  side.kind = e.kind;
  if (e.kind == factor_kind::none) {
    return side;
  }
  if (e.kind == factor_kind::complex && real_side(s, output)) {
    reject(name + ".mul.kind", std::string("is complex on the real ") +
                                   (output ? "output" : "input") +
                                   " of a real transform; a real side takes a real factor");
  }
  if (e.axis_strides.size() != s.axes.size()) {
    reject(name + ".mul.axis_strides", "has " + std::to_string(e.axis_strides.size()) +
                                           " strides; the shape has " +
                                           std::to_string(s.axes.size()) + " axes");
  }
  // No batch strides broadcast along every batch dimension, as strides of 0 do.
  const bool broadcast = e.batch_strides.empty();
  if (!broadcast && e.batch_strides.size() != s.batches.size()) {
    reject(name + ".mul.batch_strides",
           "has " + std::to_string(e.batch_strides.size()) + " strides; the shape has " +
               std::to_string(s.batches.size()) +
               " batch dimensions (give one stride for each, or none to broadcast)");
  }
  const auto batch_stride = [&](std::size_t b) -> std::ptrdiff_t {
    return broadcast ? 0 : e.batch_strides[b];
  };
  const std::size_t real_bytes = s.prec == precision::f64 ? sizeof(double) : sizeof(float);
  side.factor_bytes = e.kind == factor_kind::complex ? 2 * real_bytes : real_bytes;
  // The array's span in bytes, sum |stride| (extent - 1) element bytes, must fit ptrdiff_t, so
  // that every offset the cursors form does. Strides of a dimension of one point are ignored.
  const auto span_limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  std::size_t span = 0;
  const auto add_reach = [&](std::ptrdiff_t stride, std::size_t extent, const std::string& what) {
    if (extent <= 1) {
      return;
    }
    // |stride| without overflow: -PTRDIFF_MIN does not fit.
    const std::size_t magnitude =
        stride < 0 ? 0 - static_cast<std::size_t>(stride) : static_cast<std::size_t>(stride);
    std::size_t reach = 0;
    if (__builtin_mul_overflow(magnitude, extent - 1, &reach) ||
        __builtin_mul_overflow(reach, side.factor_bytes, &reach) ||
        __builtin_add_overflow(span, reach, &span) || span > span_limit) {
      reject(what, "makes the factor array too large to address");
    }
  };
  for (std::size_t d = 0; d < s.axes.size(); ++d) {
    add_reach(e.axis_strides[d], side_length(s, d, output),
              name + ".mul.axis_strides[" + std::to_string(d) + "]");
  }
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    add_reach(batch_stride(b), s.batches[b].count,
              name + ".mul.batch_strides[" + std::to_string(b) + "]");
  }
  // One axis: the line's stride, 0 where the line has one point (the fused rows, 5.1a).
  side.axis_stride = s.axes.size() == 1 && s.axes[0].n > 1 ? e.axis_strides[0] : 0;
  for (std::size_t d = 0; d < s.axes.size(); ++d) {
    side.axis_strides.push_back(side_length(s, d, output) > 1 ? e.axis_strides[d] : 0);
  }
  const auto bytes = static_cast<std::ptrdiff_t>(side.factor_bytes);
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    side.batch_steps.push_back(s.batches[b].count > 1 ? batch_stride(b) * bytes : 0);
  }
  return side;
}

// The edge pass (edges.hpp, edge_pass_stage) over elements [first, last) of each of the
// stage's lines of complex values (or, Real, of real values: the real side of a real
// transform): x times the factor, times the scale, stored, added to the destination
// (edge_mode::accumulate) or added to a prior value (edge_mode::add_prior: x is read at the
// destination, the prior value at the source). Baseline code, outside every ISA region; X65 is
// to say where a vectorised or fused form pays. Written out rather than with std::complex's
// operator*, whose NaN recovery (C Annex G) is not the passes' arithmetic.
//
// The stage's lines (pass_args::rows, set by the driver for each block of a lanes program;
// 0 elsewhere, meaning one) are src_dist and dst_dist elements apart, the factor's
// dst_qstride bytes apart; src_cols and dst_cols are bytes added to the source and the
// destination (a region of the workspace block past the strategy's own).
template <class T, factor_kind Kind, bool Output, edge_mode Mode, bool Real>
void edge_pass(const pass_args* a, const void* src, void* dst, std::size_t first,
               std::size_t last) {
  constexpr std::ptrdiff_t width = Real ? 1 : 2; // reals per element
  const std::size_t rows = a->rows == 0 ? 1 : a->rows;
  const auto scale = static_cast<T>(a->scale);
  // pass_args::src_qstride: the factor's stride along the line, in its elements.
  const std::ptrdiff_t fs = a->src_qstride;
  for (std::size_t row = 0; row < rows; ++row) {
    const auto q = static_cast<std::ptrdiff_t>(row);
    // The offsets may be negative (a stored run along a reversed axis, run_line_stages): their
    // two's complement in the unsigned fields.
    const T* x = reinterpret_cast<const T*>(static_cast<const std::byte*>(src) +
                                            static_cast<std::ptrdiff_t>(a->src_cols)) +
                 q * a->src_dist * width;
    T* y = reinterpret_cast<T*>(static_cast<std::byte*>(dst) +
                                static_cast<std::ptrdiff_t>(a->dst_cols)) +
           q * a->dst_dist * width;
    const T* f = nullptr;
    if constexpr (Kind != factor_kind::none) {
      f = reinterpret_cast<const T*>(
          static_cast<const std::byte*>(Output ? a->out_factor : a->in_factor) +
          q * a->dst_qstride);
    }
    for (std::size_t k = first; k < last; ++k) {
      const auto i = static_cast<std::ptrdiff_t>(k);
      // add_prior reads the value to multiply at the destination.
      const T* xi = Mode == edge_mode::add_prior ? y + width * i * a->dst_stride
                                                 : x + width * i * a->src_stride;
      T re = xi[0];
      T im{};
      if constexpr (!Real) {
        im = xi[1];
      }
      if constexpr (Kind == factor_kind::real) {
        const T w = f[i * fs];
        re = re * w;
        im = im * w;
      } else if constexpr (Kind == factor_kind::complex) {
        // A real side takes a real factor only (side_of refuses a complex one).
        static_assert(!Real);
        const T wr = f[2 * i * fs];
        const T wi = f[2 * i * fs + 1];
        const T r = re * wr - im * wi;
        im = re * wi + im * wr;
        re = r;
      }
      re = re * scale;
      im = im * scale;
      T* yi = y + width * i * a->dst_stride;
      if constexpr (Mode == edge_mode::accumulate) {
        re = yi[0] + re;
        if constexpr (!Real) {
          im = yi[1] + im;
        }
      } else if constexpr (Mode == edge_mode::add_prior) {
        const T* pi = x + width * i * a->src_stride;
        re = pi[0] + re;
        if constexpr (!Real) {
          im = pi[1] + im;
        }
      }
      yi[0] = re;
      if constexpr (!Real) {
        yi[1] = im;
      }
    }
  }
}

// Zeros over elements [first, last) of a line at dst plus dst_cols bytes (signed, as above),
// dst_stride elements apart: the logical positions outside an input's stored runs.
template <class T, bool Real>
void zero_pass(const pass_args* a, const void* /*src*/, void* dst, std::size_t first,
               std::size_t last) {
  constexpr std::ptrdiff_t width = Real ? 1 : 2;
  T* y =
      reinterpret_cast<T*>(static_cast<std::byte*>(dst) + static_cast<std::ptrdiff_t>(a->dst_cols));
  for (std::size_t k = first; k < last; ++k) {
    T* yi = y + width * static_cast<std::ptrdiff_t>(k) * a->dst_stride;
    yi[0] = T{};
    if constexpr (!Real) {
      yi[1] = T{};
    }
  }
}

template <class T, bool Output, edge_mode Mode> pass_fn edge_pass_of(factor_kind kind, bool real) {
  if (real) {
    if (kind == factor_kind::complex) {
      // side_of refuses a complex factor on a real side first.
      throw std::logic_error("fftn: internal: a complex factor on a real side");
    }
    return kind == factor_kind::real ? &edge_pass<T, factor_kind::real, Output, Mode, true>
                                     : &edge_pass<T, factor_kind::none, Output, Mode, true>;
  }
  switch (kind) {
  case factor_kind::real:
    return &edge_pass<T, factor_kind::real, Output, Mode, false>;
  case factor_kind::complex:
    return &edge_pass<T, factor_kind::complex, Output, Mode, false>;
  case factor_kind::none:
    break;
  }
  return &edge_pass<T, factor_kind::none, Output, Mode, false>;
}

template <class T> pass_fn edge_pass_for(factor_kind kind, bool output, edge_mode mode, bool real) {
  if (!output) {
    return edge_pass_of<T, false, edge_mode::store>(kind, real);
  }
  switch (mode) {
  case edge_mode::accumulate:
    return edge_pass_of<T, true, edge_mode::accumulate>(kind, real);
  case edge_mode::add_prior:
    return edge_pass_of<T, true, edge_mode::add_prior>(kind, real);
  case edge_mode::store:
    break;
  }
  return edge_pass_of<T, true, edge_mode::store>(kind, real);
}

const char* kind_word(factor_kind k) {
  switch (k) {
  case factor_kind::real:
    return "real";
  case factor_kind::complex:
    return "complex";
  case factor_kind::none:
    break;
  }
  return "none";
}

std::string exact(double v) {
  std::array<char, 64> text{};
  std::snprintf(text.data(), text.size(), "%a", v);
  return text.data();
}

} // namespace

edge_side resolve_edge_side(const shape_view& s, const edge_view& e, bool output,
                            const std::string& name) {
  return side_of(s, e, output, name);
}

bool scale_fits_precision(double scale, precision prec) { return scale_fits(scale, prec); }

void check_edge_scale(double scale, precision prec, const std::string& name) {
  check_scale(scale, prec, name);
}

resolved_edges resolve_edges(const shape_view& s, const options& opts, const edges_view& e,
                             bool fuse) {
  resolved_edges r{.opts = opts, .plan = {}};
  edge_plan& p = r.plan;
  // options::scale first, as a plan without edges judges it: the product below may move it
  // into a separate pass, where the plan's own check would not see it.
  check_scale(opts.scale, s.prec, "options.scale");
  p.in = side_of(s, e.in, false, "edges.in");
  p.out = side_of(s, e.out, true, "edges.out");
  if (opts.scale != 1.0 && e.out.scale != 1.0) {
    reject("edges.out.scale", "and options.scale are both given; options.scale is a shortcut for "
                              "edges.out.scale, so give one of them");
  }
  // One product for every scale, each factor valid by now; options::scale and edges.out.scale
  // are not both other than 1, so the product involves edges.in.scale when it fails.
  const double scale = opts.scale * e.in.scale * e.out.scale;
  if (!scale_fits(scale, s.prec)) {
    reject("edges.in.scale", "times edges.out.scale and options.scale is not finite or outside "
                             "the range of the plan's precision");
  }
  r.opts.scale = scale;
  if (p.out.call.mode == callback_mode::consume && p.out.call.any() && opts.in_place) {
    reject("edges.out.call.mode", "is consume on an in-place plan, whose output is its input's "
                                  "memory; give the plan out of place");
  }
  p.any = p.in.kind != factor_kind::none || p.out.kind != factor_kind::none || p.out.accumulate ||
          any_callback(p);
  if (!p.any) {
    return r;
  }
  if (p.out.accumulate && opts.in_place) {
    reject("edges.out.accumulate", "is refused on an in-place plan, whose output's prior values "
                                   "are its input (the maintainer, 2026-11-25)");
  }
  // On c2c rows a complex factor of stride 1 along the line fuses into the line's edge pass;
  // anything else, and every factor of a multi-axis or real plan, goes through a separate pass
  // (an edge phase, plan_core.cpp), as accumulation does.
  const bool rows = fuse && s.kind == transform_kind::c2c && s.axes.size() == 1;
  const auto fusable = [rows](const edge_side& side) {
    return rows && side.kind == factor_kind::complex &&
           (side.axis_stride == 1 || side.axis_stride == 0);
  };
  // A factor whose stride along a line of more than one point is 0 broadcasts one value along
  // it, which the fused passes (one value per element) cannot read: a separate pass.
  const auto broadcast_along = [&](const edge_side& side) {
    return side.axis_stride == 0 && s.axes[0].n > 1;
  };
  p.in.fused = fusable(p.in) && !broadcast_along(p.in);
  p.in.separate = p.in.kind != factor_kind::none && !p.in.fused;
  p.out.fused = fusable(p.out) && !broadcast_along(p.out) && !p.out.accumulate;
  p.out.separate = p.out.accumulate || (p.out.kind != factor_kind::none && !p.out.fused);
  r.opts.scale = 1.0;
  if (scale != 1.0) {
    if (p.out.separate) {
      p.out.scale = scale;
    } else if (p.in.separate) {
      p.in.scale = scale;
    } else if (!p.out.fused) {
      r.opts.scale = scale; // on the output edge, as options::scale alone
    } else if (!p.in.fused) {
      r.opts.scale = scale;
      p.scale_on_input = true;
    } else {
      // Both factors fused: a pass applies one operation per edge, so the output's factor
      // moves to a separate pass, which takes the scale too.
      p.out.fused = false;
      p.out.separate = true;
      p.out.scale = scale;
    }
  }
  return r;
}

stage edge_pass_stage(precision prec, const edge_side& side, bool output, std::size_t n, slot src,
                      std::ptrdiff_t src_stride, slot dst, std::ptrdiff_t dst_stride,
                      std::size_t chunk_points, bool real, std::ptrdiff_t factor_stride) {
  return edge_pass_stage(prec, side, output, n, src, src_stride, dst, dst_stride, chunk_points,
                         real, factor_stride,
                         side.accumulate ? edge_mode::accumulate : edge_mode::store, edge_rows{});
}

stage edge_copy_stage(precision prec, std::size_t n, slot src, std::ptrdiff_t src_stride, slot dst,
                      std::ptrdiff_t dst_stride, std::size_t chunk_points, bool real,
                      const edge_rows& rows) {
  stage st = edge_pass_stage(prec, edge_side{}, true, n, src, src_stride, dst, dst_stride,
                             chunk_points, real, 0, edge_mode::store, rows);
  st.name = "edge out: the output's prior values saved";
  return st;
}

void run_zero_line(precision prec, std::size_t length, const run_line& line,
                   std::size_t chunk_points, bool real, std::vector<stage>& stages) {
  stage st = edge_pass_stage(prec, edge_side{}, false, length, line.src, line.src_stride, line.dst,
                             line.dst_stride, chunk_points, real, 0, edge_mode::store, edge_rows{});
  st.run = prec == precision::f64 ? (real ? &zero_pass<double, true> : &zero_pass<double, false>)
                                  : (real ? &zero_pass<float, true> : &zero_pass<float, false>);
  st.args.dst_cols = static_cast<std::size_t>(line.dst_offset);
  st.name = "runs in: zeros (no stored run on another axis)";
  stages.push_back(std::move(st));
}

void run_line_stages(precision prec, const edge_side& side, bool output, std::size_t length,
                     const std::vector<line_run>& runs, const run_line& line,
                     std::size_t chunk_points, bool real, std::vector<stage>& stages) {
  const std::size_t real_bytes = prec == precision::f64 ? sizeof(double) : sizeof(float);
  const auto bytes = static_cast<std::ptrdiff_t>(real ? real_bytes : 2 * real_bytes);
  const auto at = [&](std::size_t pos, std::ptrdiff_t stride, std::ptrdiff_t offset) {
    return static_cast<std::size_t>(offset + static_cast<std::ptrdiff_t>(pos) * stride * bytes);
  };
  // A piece of the line: count elements from logical position `logical`, stored at `stored`
  // (the stored side being the input's source or the output's destination).
  const auto copy = [&](std::size_t logical, std::size_t stored, std::size_t count) {
    stage st = edge_pass_stage(prec, side, output, count, line.src, line.src_stride, line.dst,
                               line.dst_stride, chunk_points, real, 0,
                               output && side.accumulate ? edge_mode::accumulate : edge_mode::store,
                               edge_rows{});
    st.args.src_cols = at(output ? logical : stored, line.src_stride, line.src_offset);
    st.args.dst_cols = at(output ? stored : logical, line.dst_stride, line.dst_offset);
    st.name = std::string(output ? "runs out" : "runs in") + ": logical " +
              std::to_string(logical) + " to " + std::to_string(logical + count) + " at stored " +
              std::to_string(stored) + (output && side.accumulate ? ", accumulated" : "");
    stages.push_back(std::move(st));
  };
  const auto zeros = [&](std::size_t logical, std::size_t count) {
    run_line piece = line;
    piece.dst_offset = static_cast<std::ptrdiff_t>(at(logical, line.dst_stride, line.dst_offset));
    run_zero_line(prec, count, piece, chunk_points, real, stages);
    stages.back().name = "runs in: zeros at logical " + std::to_string(logical) + " to " +
                         std::to_string(logical + count);
  };
  if (runs.empty()) {
    copy(0, 0, length);
    return;
  }
  // The runs are sorted by logical position (resolve_runs); the input's gaps are zeros, the
  // output's are not written.
  std::size_t next = 0;
  for (const line_run& r : runs) {
    if (!output && r.logical > next) {
      zeros(next, r.logical - next);
    }
    copy(r.logical, r.stored, r.count);
    next = r.logical + r.count;
  }
  if (!output && next < length) {
    zeros(next, length - next);
  }
}

stage edge_pass_stage(precision prec, const edge_side& side, bool output, std::size_t n, slot src,
                      std::ptrdiff_t src_stride, slot dst, std::ptrdiff_t dst_stride,
                      std::size_t chunk_points, bool real, std::ptrdiff_t factor_stride,
                      edge_mode mode, const edge_rows& rows) {
  stage st;
  st.run = prec == precision::f64 ? edge_pass_for<double>(side.kind, output, mode, real)
                                  : edge_pass_for<float>(side.kind, output, mode, real);
  st.args.src_stride = src_stride;
  st.args.dst_stride = dst_stride;
  st.args.scale = side.scale;
  st.args.src_qstride = factor_stride;
  st.args.src_dist = rows.src_dist;
  st.args.dst_dist = rows.dst_dist;
  st.args.dst_qstride = rows.factor_step;
  st.args.src_cols = rows.src_offset;
  st.args.dst_cols = rows.dst_offset;
  st.src = src;
  st.dst = dst;
  st.units = n;
  const std::size_t chunks = std::clamp<std::size_t>(n / std::max<std::size_t>(chunk_points, 1), 1,
                                                     std::numeric_limits<std::uint32_t>::max());
  st.chunks = static_cast<std::uint32_t>(chunks);
  if (side.kind != factor_kind::none) {
    st.moving_factors = output ? moving_out_factor : moving_in_factor;
  }
  std::string name = output ? "edge out" : "edge in";
  if (side.kind != factor_kind::none) {
    name += std::string(" times a ") + kind_word(side.kind) + " factor (stride " +
            std::to_string(factor_stride) + ")";
  }
  if (side.scale != 1.0) {
    name += ", scaled";
  }
  if (mode == edge_mode::accumulate) {
    name += ", accumulated";
  } else if (mode == edge_mode::add_prior) {
    name += ", added to the saved prior values";
  }
  st.name = name;
  return st;
}

void contract_failure(const char* what) noexcept {
  std::fputs(what, stderr);
  std::fputs("\n", stderr);
  std::abort();
}

std::size_t edge_passes(const program& p) noexcept {
  std::size_t passes = 0;
  for (const record& r : p.records) {
    if (const stage* st = std::get_if<stage>(&r);
        st != nullptr && (st->name.starts_with("edge in") || st->name.starts_with("edge out"))) {
      ++passes;
    }
  }
  return passes;
}

std::string forced_text(const std::string& plain, const std::string& built,
                        const std::string& fallback, const char* what) {
  return "not the strategy the rules would choose: " + plain + " without " + what + ", " + built +
         " with them" + (fallback.empty() ? "" : ": " + fallback);
}

std::string edges_text(const edge_plan& p, edge_layout layout, std::size_t passes) {
  const auto describe_side = [layout](const edge_side& side, bool output) {
    std::string t;
    if (side.kind != factor_kind::none) {
      t += std::string("times a ") + kind_word(side.kind) + " factor";
    }
    if (side.accumulate) {
      t += t.empty() ? "accumulated" : ", accumulated";
    }
    if (t.empty()) {
      return std::string();
    }
    if (side.fused) {
      t += output ? " in the last pass" : " in the first pass";
    } else if (side.separate && layout == edge_layout::row_chain) {
      t += output ? " in a separate pass from the workspace block"
                  : " in a separate pass into the workspace block";
    } else if (side.separate && layout == edge_layout::row_strategy) {
      t += output ? " in a separate pass after the row's strategy"
                  : " in a separate pass before the row's strategy";
      // Accumulation into a strategy that writes the output (edge_mode::add_prior).
      if (output && side.accumulate) {
        t += ", the output's prior values saved in a pass before it";
      }
    } else if (side.separate) {
      t += output ? " in a separate phase after the transform"
                  : " in a separate phase before the transform";
    }
    return std::string(output ? "the output " : "the input ") + t;
  };
  std::string in = describe_side(p.in, false);
  std::string out = describe_side(p.out, true);
  std::string both = in.empty() ? out : (out.empty() ? in : in + "; " + out);
  if (both.empty()) {
    both = "block callbacks only"; // their own line follows (callbacks_text)
  }
  // The scales alone cost nothing; a fused factor costs no pass of its own.
  const std::string cost =
      passes == 0 ? "no separate pass"
                  : std::to_string(passes) + (passes == 1 ? " separate pass" : " separate passes");
  return "edges: " + both + "; they cost " + cost;
}

std::pair<std::ptrdiff_t, std::ptrdiff_t> factor_span(const shape_view& s, const edge_view& e,
                                                      bool output) {
  if (e.kind == factor_kind::none) {
    return {0, 0};
  }
  std::ptrdiff_t lo = 0;
  std::ptrdiff_t hi = 0;
  // resolve_edges has bounded the span, so nothing here overflows.
  const auto reach = [&](std::ptrdiff_t stride, std::size_t extent) {
    if (extent > 1) {
      const std::ptrdiff_t last = stride * static_cast<std::ptrdiff_t>(extent - 1);
      lo += std::min<std::ptrdiff_t>(last, 0);
      hi += std::max<std::ptrdiff_t>(last, 0);
    }
  };
  for (std::size_t d = 0; d < s.axes.size(); ++d) {
    reach(e.axis_strides[d], side_length(s, d, output));
  }
  for (std::size_t b = 0; b < s.batches.size() && !e.batch_strides.empty(); ++b) {
    reach(e.batch_strides[b], s.batches[b].count);
  }
  return {lo, hi};
}

std::string edges_key(const edges_view& e, double options_scale) {
  const auto any = [](const edge_view& v) {
    return v.kind != factor_kind::none || v.accumulate || v.call != nullptr;
  };
  if (!any(e.in) && !any(e.out)) {
    return {};
  }
  std::ostringstream o;
  const auto side = [&](const char* name, const edge_view& v) {
    o << ' ' << name << " scale " << exact(v.scale) << ' ' << kind_word(v.kind);
    if (v.kind != factor_kind::none) {
      for (const std::ptrdiff_t st : v.axis_strides) {
        o << ' ' << st;
      }
      o << " /";
      for (const std::ptrdiff_t st : v.batch_strides) {
        o << ' ' << st;
      }
    }
    o << (v.accumulate ? " accumulate" : "");
    // A callback by its mode and scratch, not its function (DESIGN.md 12.3, "Planning").
    if (v.call != nullptr) {
      o << " call " << mode_name(v.call_mode) << ' ' << v.call_scratch;
    }
  };
  // options::scale too: the key's own scale is the part the plan fuses, which a separate pass
  // can leave at 1 whatever options::scale is.
  o << " edges options scale " << exact(options_scale);
  side("in", e.in);
  side("out", e.out);
  return o.str();
}

} // namespace fftn::inline v0::detail

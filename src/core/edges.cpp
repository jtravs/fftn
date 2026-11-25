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
edge_side side_of(const shape_view& s, const edge_view& e, bool output) {
  const std::string name = output ? "edges.out" : "edges.in";
  check_scale(e.scale, s.prec, name + ".scale");
  if (e.accumulate && !output) {
    reject(name + ".accumulate", "is for the output edge only");
  }
  edge_side side;
  side.accumulate = e.accumulate;
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

// The edge pass (edges.hpp, edge_pass_stage) over elements [first, last) of a line of complex
// values (or, Real, of real values: the real side of a real transform): x times the factor,
// times the scale, stored or added. Baseline code, outside every ISA region; X65 is to say
// where a vectorised or fused form pays. Written out rather than with std::complex's
// operator*, whose NaN recovery (C Annex G) is not the passes' arithmetic.
template <class T, factor_kind Kind, bool Output, bool Accumulate, bool Real>
void edge_pass(const pass_args* a, const void* src, void* dst, std::size_t first,
               std::size_t last) {
  const auto* x = static_cast<const T*>(src);
  auto* y = static_cast<T*>(dst);
  const auto scale = static_cast<T>(a->scale);
  const auto* f = static_cast<const T*>(Output ? a->out_factor : a->in_factor);
  // pass_args::src_qstride: the factor's stride along the line, in its elements.
  const std::ptrdiff_t fs = a->src_qstride;
  if constexpr (Real) {
    // A real side takes a real factor only (side_of refuses a complex one).
    static_assert(Kind != factor_kind::complex);
    for (std::size_t k = first; k < last; ++k) {
      const auto i = static_cast<std::ptrdiff_t>(k);
      T v = x[i * a->src_stride];
      if constexpr (Kind == factor_kind::real) {
        v = v * f[i * fs];
      }
      v = v * scale;
      T* yi = y + i * a->dst_stride;
      if constexpr (Accumulate) {
        *yi = *yi + v;
      } else {
        *yi = v;
      }
    }
    return;
  }
  for (std::size_t k = first; k < last; ++k) {
    const auto i = static_cast<std::ptrdiff_t>(k);
    const T* xi = x + 2 * i * a->src_stride;
    T re = xi[0];
    T im = xi[1];
    if constexpr (Kind == factor_kind::real) {
      const T w = f[i * fs];
      re = re * w;
      im = im * w;
    } else if constexpr (Kind == factor_kind::complex) {
      const T wr = f[2 * i * fs];
      const T wi = f[2 * i * fs + 1];
      const T r = re * wr - im * wi;
      im = re * wi + im * wr;
      re = r;
    }
    re = re * scale;
    im = im * scale;
    T* yi = y + 2 * i * a->dst_stride;
    if constexpr (Accumulate) {
      yi[0] = yi[0] + re;
      yi[1] = yi[1] + im;
    } else {
      yi[0] = re;
      yi[1] = im;
    }
  }
}

template <class T, bool Output, bool Accumulate> pass_fn edge_pass_of(factor_kind kind, bool real) {
  if (real) {
    return kind == factor_kind::real ? &edge_pass<T, factor_kind::real, Output, Accumulate, true>
                                     : &edge_pass<T, factor_kind::none, Output, Accumulate, true>;
  }
  switch (kind) {
  case factor_kind::real:
    return &edge_pass<T, factor_kind::real, Output, Accumulate, false>;
  case factor_kind::complex:
    return &edge_pass<T, factor_kind::complex, Output, Accumulate, false>;
  case factor_kind::none:
    break;
  }
  return &edge_pass<T, factor_kind::none, Output, Accumulate, false>;
}

template <class T>
pass_fn edge_pass_for(factor_kind kind, bool output, bool accumulate, bool real) {
  if (!output) {
    return edge_pass_of<T, false, false>(kind, real);
  }
  return accumulate ? edge_pass_of<T, true, true>(kind, real)
                    : edge_pass_of<T, true, false>(kind, real);
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

resolved_edges resolve_edges(const shape_view& s, const options& opts, const edges_view& e) {
  resolved_edges r{.opts = opts, .plan = {}};
  edge_plan& p = r.plan;
  // options::scale first, as a plan without edges judges it: the product below may move it
  // into a separate pass, where the plan's own check would not see it.
  check_scale(opts.scale, s.prec, "options.scale");
  p.in = side_of(s, e.in, false);
  p.out = side_of(s, e.out, true);
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
  p.any = p.in.kind != factor_kind::none || p.out.kind != factor_kind::none || p.out.accumulate;
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
  const bool rows = s.kind == transform_kind::c2c && s.axes.size() == 1;
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
  stage st;
  st.run = prec == precision::f64 ? edge_pass_for<double>(side.kind, output, side.accumulate, real)
                                  : edge_pass_for<float>(side.kind, output, side.accumulate, real);
  st.args.src_stride = src_stride;
  st.args.dst_stride = dst_stride;
  st.args.scale = side.scale;
  st.args.src_qstride = factor_stride;
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
  if (side.accumulate) {
    name += ", accumulated";
  }
  st.name = name;
  return st;
}

void contract_failure(const char* what) noexcept {
  std::fputs(what, stderr);
  std::fputs("\n", stderr);
  std::abort();
}

std::string edges_text(const edge_plan& p, bool rows) {
  const auto describe_side = [rows](const edge_side& side, bool output) {
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
    } else if (side.separate && rows) {
      t += output ? " in a separate pass from the workspace block"
                  : " in a separate pass into the workspace block";
    } else if (side.separate) {
      t += output ? " in a separate phase after the transform"
                  : " in a separate phase before the transform";
    }
    return std::string(output ? "the output " : "the input ") + t;
  };
  std::string in = describe_side(p.in, false);
  std::string out = describe_side(p.out, true);
  std::string both = in.empty() ? out : (out.empty() ? in : in + "; " + out);
  if (rows) {
    return "edges: " + both +
           "; the chain, by the rules, whatever the strategy and effort would otherwise be "
           "(factors and accumulation reach no other strategy, nor measured planning, yet)";
  }
  return "edges: " + both +
         "; by the rules whatever the effort, without the lanes over adjacent transforms or "
         "a real row's grid, lanes or depth-first chain (not built with factors or accumulation "
         "yet)";
}

std::string edges_key(const edges_view& e, double options_scale) {
  const auto any = [](const edge_view& v) { return v.kind != factor_kind::none || v.accumulate; };
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
  };
  // options::scale too: the key's own scale is the part the plan fuses, which a separate pass
  // can leave at 1 whatever options::scale is.
  o << " edges options scale " << exact(options_scale);
  side("in", e.in);
  side("out", e.out);
  return o.str();
}

} // namespace fftn::inline v0::detail

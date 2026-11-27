// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The free-space stage's arms (fftn_stage.hpp; experiment X68). Everything here is written as an
// application writes it: public FFTN objects only, and the application's own loops and callbacks
// compiled as the user's (cloned for AVX2 and AVX-512 on x86-64 Linux, as the edge loops of the
// FFTN backend are).

#include "backends/fftn_stage.hpp"

#include <fftn/fftn.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace fbench::stage {
namespace {

using cd = std::complex<double>;

#if defined(__x86_64__) && defined(__linux__)
#define FBENCH_STAGE_CLONES [[gnu::target_clones("default", "avx2", "avx512f")]]
#else
#define FBENCH_STAGE_CLONES
#endif

// ---- The case ---------------------------------------------------------------------------

struct geometry {
  std::size_t ny = 0;
  std::size_t nx = 0;
  std::size_t nw = 0;     // stored frequencies
  std::size_t groups = 1; // components
  std::size_t nt = 0;     // the time line's points
  std::size_t first = 0;  // the band's first bin (real lines)
  bool real = false;
  [[nodiscard]] std::size_t pixels() const { return ny * nx; }
  [[nodiscard]] std::size_t comp() const { return pixels() * nw; } // elements of a component
  [[nodiscard]] std::size_t bins() const { return real ? nt / 2 + 1 : nt; }
};

// The signed frequency of index i of n.
double freq(std::size_t i, std::size_t n) {
  return i < (n + 1) / 2 ? static_cast<double>(i) : static_cast<double>(i) - static_cast<double>(n);
}

// The arithmetic's constants and tables, read at run time: the stage is the identity in exact
// arithmetic (cubic: a = 1, b = 0; the propagator multiplies by w (cos^2 phi + sin^2 phi) +
// i z sin phi with w = 1, z = 0, after computing cos phi and sin phi; the finish by norm times a
// window of ones times pr + i pi = 1), while every element pays the work. phi = dz sqrt(K(w)^2 -
// kx^2 - ky^2), up to about 50 radians, so that cos and sin take their ordinary paths. With
// `nonlinear` (STAGE_SELFCHECK only) the work is not the identity: a cubic term, a propagator that
// depends on the coordinates and a window that depends on the bin, so that arms whose coordinates
// or order differ give different outputs.
struct constants {
  double a = 1.0;
  double b = 0.0;
  double w = 1.0;
  double z = 0.0;
  double dz = 1.0;
  double pr = 1.0;
  double pi = 0.0;
  double norm = 1.0;
  std::vector<double> u2, v2, k2, window; // per x, per y, per stored bin, per stored bin

  constants(const geometry& g, bool nonlinear)
      : norm(1.0 /
             (static_cast<double>(g.ny) * static_cast<double>(g.nx) * static_cast<double>(g.nt))),
        u2(g.nx), v2(g.ny), k2(g.nw), window(g.nw, 1.0) {
    constexpr double kb = 1600.0;
    constexpr double k0 = 40.0;
    constexpr double k1 = 20.0;
    for (std::size_t x = 0; x < g.nx; ++x) {
      const double u = freq(x, g.nx) / static_cast<double>(g.nx);
      u2[x] = kb * u * u;
    }
    for (std::size_t y = 0; y < g.ny; ++y) {
      const double v = freq(y, g.ny) / static_cast<double>(g.ny);
      v2[y] = kb * v * v;
    }
    for (std::size_t j = 0; j < g.nw; ++j) {
      const double f = g.real ? static_cast<double>(g.first + j) : freq(j, g.nt);
      const double k = k0 + k1 * f / static_cast<double>(g.nt);
      k2[j] = k * k;
    }
    if (nonlinear) {
      b = 0.1 * norm;
      z = 1.0;
      pi = 0.5;
      for (std::size_t j = 0; j < g.nw; ++j) {
        window[j] = 1.0 + 0.5 * std::sin(static_cast<double>(j));
      }
    }
  }
};

// The work per element. e: one complex value (two doubles).
[[gnu::always_inline]] inline void propagate(double* e, const constants& k, std::size_t y,
                                             std::size_t x, std::size_t w) {
  const double q = k.k2[w] - k.u2[x] - k.v2[y];
  const double phi = k.dz * std::sqrt(q > 0.0 ? q : 0.0);
  const double c = std::cos(phi);
  const double s = std::sin(phi);
  const double fr = k.w * (c * c + s * s);
  const double fi = k.z * s;
  const double vr = e[0];
  const double vi = e[1];
  e[0] = vr * fr - vi * fi;
  e[1] = vr * fi + vi * fr;
}

[[gnu::always_inline]] inline void finish(double* e, const constants& k, std::size_t w) {
  const double f = k.norm * k.window[w];
  const double vr = e[0] * f;
  const double vi = e[1] * f;
  e[0] = vr * k.pr - vi * k.pi;
  e[1] = vr * k.pi + vi * k.pr;
}

// The nonlinear term on a time line of n values `es` apart (complex values for an envelope, reals
// otherwise), and its coupled form on two components' lines.
template <bool Complex>
[[gnu::always_inline]] inline void cubic(double* __restrict d, std::ptrdiff_t es, std::size_t n,
                                         const constants& k) {
  const double a = k.a;
  const double b = k.b;
  for (std::size_t i = 0; i < n; ++i) {
    double* e = d + static_cast<std::ptrdiff_t>(i) * es * (Complex ? 2 : 1);
    if constexpr (Complex) {
      const double m = a + b * (e[0] * e[0] + e[1] * e[1]);
      e[0] *= m;
      e[1] *= m;
    } else {
      e[0] *= a + b * e[0] * e[0];
    }
  }
}

template <bool Complex>
[[gnu::always_inline]] inline void coupled(double* __restrict d0, double* __restrict d1,
                                           std::ptrdiff_t es, std::size_t n, const constants& k) {
  const double a = k.a;
  const double b = k.b;
  for (std::size_t i = 0; i < n; ++i) {
    const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(i) * es * (Complex ? 2 : 1);
    double* e0 = d0 + at;
    double* e1 = d1 + at;
    if constexpr (Complex) {
      const double m = a + b * (e0[0] * e0[0] + e0[1] * e0[1] + e1[0] * e1[0] + e1[1] * e1[1]);
      e0[0] *= m;
      e0[1] *= m;
      e1[0] *= m;
      e1[1] *= m;
    } else {
      const double m = a + b * (e0[0] * e0[0] + e1[0] * e1[0]);
      e0[0] *= m;
      e1[0] *= m;
    }
  }
}

// ---- Callbacks (O, Of, G) -----------------------------------------------------------------

// Coordinates (y, x, w, c) of an element, and how an object's index space maps onto them: the
// roles of its transformed axes and of its batch dimensions (slowest first).
enum role : unsigned char { ry, rx, rw, rc };
using coords = std::array<std::size_t, 4>;

struct coord_map {
  std::array<role, 2> axis{};
  std::size_t axes = 0;
  std::array<role, 4> brole{};
  std::array<std::size_t, 4> bcount{};
  std::size_t batches = 0;

  void batch_coords(std::size_t t, coords& c) const {
    for (std::size_t i = batches; i > 0; --i) {
      c[brole[i - 1]] = t % bcount[i - 1];
      t /= bcount[i - 1];
    }
  }
  // The next transform's batch coordinates.
  void next(coords& c) const {
    for (std::size_t i = batches; i > 0; --i) {
      if (++c[brole[i - 1]] < bcount[i - 1]) {
        return;
      }
      c[brole[i - 1]] = 0;
    }
  }
};

struct cb_user {
  const constants* k = nullptr;
  coord_map map;
};

// Calls f(element, coords) on every element of a block, in memory order: where its lines lie
// closer together than its elements (a plane's columns with the batch adjacent, line_stride 1),
// along the lines in the inner loop (X66's advice in docs/fusion.md).
template <class F>
[[gnu::always_inline]] inline void walk(const coord_map& m, const fftn::block_info& info,
                                        double* data, F&& f) {
  const bool along = info.line_axis == fftn::along_transforms;
  coords base{};
  for (std::size_t d = 0; d < m.axes && d < info.origin.size(); ++d) {
    base[m.axis[d]] = info.origin[d];
  }
  m.batch_coords(info.transform, base);
  const role er = m.axis[info.element_axis];
  const std::ptrdiff_t es = 2 * info.element_stride;
  const std::ptrdiff_t ls = 2 * info.line_stride;
  auto advance = [&](coords& c) {
    if (along) {
      m.next(c);
    } else {
      ++c[m.axis[info.line_axis]];
    }
  };
  if (info.lines > 1 && std::abs(ls) < std::abs(es)) {
    for (std::size_t i = 0; i < info.length; ++i) {
      coords c = base;
      c[er] += i;
      double* p = data + static_cast<std::ptrdiff_t>(i) * es;
      for (std::size_t l = 0; l < info.lines; ++l) {
        f(p + static_cast<std::ptrdiff_t>(l) * ls, c);
        advance(c);
      }
    }
  } else {
    coords line = base;
    for (std::size_t l = 0; l < info.lines; ++l) {
      coords c = line;
      double* p = data + static_cast<std::ptrdiff_t>(l) * ls;
      for (std::size_t i = 0; i < info.length; ++i) {
        c[er] = line[er] + i;
        f(p + static_cast<std::ptrdiff_t>(i) * es, c);
      }
      advance(line);
    }
  }
}

FBENCH_STAGE_CLONES void prologue_cb(void* user, const fftn::block_info& info, cd* data) noexcept {
  const auto& u = *static_cast<const cb_user*>(user);
  const constants& k = *u.k;
  walk(u.map, info, reinterpret_cast<double*>(data),
       [&k](double* e, const coords& c) { propagate(e, k, c[ry], c[rx], c[rw]); });
}

FBENCH_STAGE_CLONES void finish_cb(void* user, const fftn::block_info& info, cd* data) noexcept {
  const auto& u = *static_cast<const cb_user*>(user);
  const constants& k = *u.k;
  walk(u.map, info, reinterpret_cast<double*>(data),
       [&k](double* e, const coords& c) { finish(e, k, c[rw]); });
}

// The pencils' middle: the whole time line of each line (each group's lines together).
template <bool Complex>
[[gnu::always_inline]] inline void middle_body(void* user, const fftn::block_info& info,
                                               double* data) {
  const auto& k = *static_cast<const constants*>(user);
  const std::ptrdiff_t width = Complex ? 2 : 1;
  for (std::size_t l = 0; l < info.lines; ++l) {
    double* line = data + static_cast<std::ptrdiff_t>(l) * info.line_stride * width;
    if (info.group == 2) {
      coupled<Complex>(line, line + info.group_stride * width, info.element_stride, info.length, k);
    } else {
      for (std::size_t m = 0; m < info.group; ++m) {
        cubic<Complex>(line + static_cast<std::ptrdiff_t>(m) * info.group_stride * width,
                       info.element_stride, info.length, k);
      }
    }
  }
}

FBENCH_STAGE_CLONES void middle_complex_cb(void* user, const fftn::block_info& info,
                                           cd* data) noexcept {
  middle_body<true>(user, info, reinterpret_cast<double*>(data));
}

FBENCH_STAGE_CLONES void middle_real_cb(void* user, const fftn::block_info& info,
                                        double* data) noexcept {
  middle_body<false>(user, info, data);
}

// ---- The application's loops (T, C) ---------------------------------------------------------

// The field F is (c, p, w), p = y nx + x; T's scratch S is (c, w, p), the client application's
// plane-major order. The crossings transpose in tiles of TP pixels by TW bins, as a user who cares
// writes them: each tile's rows read whole, its columns written in runs of TP values.
constexpr std::size_t TP = 8;
constexpr std::size_t TW = 64;

std::size_t tiles(std::size_t n, std::size_t t) { return (n + t - 1) / t; }

// Tiles [first, first + count) of the G x ceil(P / TP) pixel tiles: F -> S with the propagator.
FBENCH_STAGE_CLONES void cross_in(const geometry& g, const constants& k, const double* F, double* S,
                                  std::size_t first, std::size_t count) {
  const std::size_t P = g.pixels();
  const std::size_t pt = tiles(P, TP);
  for (std::size_t t = first; t < first + count; ++t) {
    const std::size_t c = t / pt;
    const std::size_t p0 = (t % pt) * TP;
    const std::size_t p1 = std::min(P, p0 + TP);
    for (std::size_t w0 = 0; w0 < g.nw; w0 += TW) {
      const std::size_t w1 = std::min(g.nw, w0 + TW);
      for (std::size_t p = p0; p < p1; ++p) {
        const std::size_t y = p / g.nx;
        const std::size_t x = p % g.nx;
        const double* src = F + 2 * ((c * P + p) * g.nw);
        for (std::size_t w = w0; w < w1; ++w) {
          double e[2] = {src[2 * w], src[2 * w + 1]};
          propagate(e, k, y, x, w);
          double* dst = S + 2 * ((c * g.nw + w) * P + p);
          dst[0] = e[0];
          dst[1] = e[1];
        }
      }
    }
  }
}

// The same tiles: S -> F with the finish.
FBENCH_STAGE_CLONES void cross_out(const geometry& g, const constants& k, const double* S,
                                   double* F, std::size_t first, std::size_t count) {
  const std::size_t P = g.pixels();
  const std::size_t pt = tiles(P, TP);
  for (std::size_t t = first; t < first + count; ++t) {
    const std::size_t c = t / pt;
    const std::size_t p0 = (t % pt) * TP;
    const std::size_t p1 = std::min(P, p0 + TP);
    for (std::size_t w0 = 0; w0 < g.nw; w0 += TW) {
      const std::size_t w1 = std::min(g.nw, w0 + TW);
      for (std::size_t p = p0; p < p1; ++p) {
        double* dst = F + 2 * ((c * P + p) * g.nw);
        for (std::size_t w = w0; w < w1; ++w) {
          const double* src = S + 2 * ((c * g.nw + w) * P + p);
          double e[2] = {src[0], src[1]};
          finish(e, k, w);
          dst[2 * w] = e[0];
          dst[2 * w + 1] = e[1];
        }
      }
    }
  }
}

// A node chunk's pack, over its tiles [first, first + count) of G x ceil(cnt / TP) row tiles: the
// spectrum rows (c, r, bin) of the chunk's pixels p0 + r, the band from S, the gaps zeroed.
FBENCH_STAGE_CLONES void pack(const geometry& g, const double* S, double* spec, std::size_t p0,
                              std::size_t cnt, std::size_t first, std::size_t count) {
  const std::size_t P = g.pixels();
  const std::size_t nb = g.bins();
  const std::size_t rt = tiles(cnt, TP);
  const std::size_t lo = g.real ? g.first : 0;
  for (std::size_t t = first; t < first + count; ++t) {
    const std::size_t c = t / rt;
    const std::size_t r0 = (t % rt) * TP;
    const std::size_t r1 = std::min(cnt, r0 + TP);
    for (std::size_t r = r0; r < r1; ++r) {
      double* row = spec + 2 * ((c * cnt + r) * nb);
      if (lo > 0) {
        std::memset(row, 0, 2 * lo * sizeof(double));
      }
      if (lo + g.nw < nb) {
        std::memset(row + 2 * (lo + g.nw), 0, 2 * (nb - lo - g.nw) * sizeof(double));
      }
    }
    for (std::size_t w0 = 0; w0 < g.nw; w0 += TW) {
      const std::size_t w1 = std::min(g.nw, w0 + TW);
      for (std::size_t r = r0; r < r1; ++r) {
        double* row = spec + 2 * ((c * cnt + r) * nb + lo);
        for (std::size_t w = w0; w < w1; ++w) {
          const double* src = S + 2 * ((c * g.nw + w) * P + p0 + r);
          row[2 * w] = src[0];
          row[2 * w + 1] = src[1];
        }
      }
    }
  }
}

// The crop back: S's band from the spectrum rows.
FBENCH_STAGE_CLONES void crop(const geometry& g, const double* spec, double* S, std::size_t p0,
                              std::size_t cnt, std::size_t first, std::size_t count) {
  const std::size_t P = g.pixels();
  const std::size_t nb = g.bins();
  const std::size_t rt = tiles(cnt, TP);
  const std::size_t lo = g.real ? g.first : 0;
  for (std::size_t t = first; t < first + count; ++t) {
    const std::size_t c = t / rt;
    const std::size_t r0 = (t % rt) * TP;
    const std::size_t r1 = std::min(cnt, r0 + TP);
    for (std::size_t w0 = 0; w0 < g.nw; w0 += TW) {
      const std::size_t w1 = std::min(g.nw, w0 + TW);
      for (std::size_t r = r0; r < r1; ++r) {
        const double* row = spec + 2 * ((c * cnt + r) * nb + lo);
        for (std::size_t w = w0; w < w1; ++w) {
          double* dst = S + 2 * ((c * g.nw + w) * P + p0 + r);
          dst[0] = row[2 * w];
          dst[1] = row[2 * w + 1];
        }
      }
    }
  }
}

// The nonlinear term on the chunk's time rows [first, first + count) (of cnt per component).
FBENCH_STAGE_CLONES void response(const geometry& g, const constants& k, double* rows,
                                  std::size_t cnt, std::size_t first, std::size_t count) {
  const std::ptrdiff_t width = g.real ? 1 : 2;
  const auto row = [&](std::size_t c, std::size_t r) {
    return rows + static_cast<std::ptrdiff_t>((c * cnt + r) * g.nt) * width;
  };
  for (std::size_t r = first; r < first + count; ++r) {
    if (g.groups == 2) {
      if (g.real) {
        coupled<false>(row(0, r), row(1, r), 1, g.nt, k);
      } else {
        coupled<true>(row(0, r), row(1, r), 1, g.nt, k);
      }
    } else if (g.real) {
      cubic<false>(row(0, r), 1, g.nt, k);
    } else {
      cubic<true>(row(0, r), 1, g.nt, k);
    }
  }
}

// One sweep of the field in place (arm C): each value times a, which is one.
FBENCH_STAGE_CLONES void sweep(const constants& k, double* F, std::size_t first,
                               std::size_t count) {
  const double a = k.a;
  for (std::size_t i = 2 * first; i < 2 * (first + count); ++i) {
    F[i] *= a;
  }
}

// ---- Shared pieces ------------------------------------------------------------------------

// Memory of n complex values on a 64-byte boundary, zeroed.
class buffer {
public:
  buffer() = default;
  explicit buffer(std::size_t n) : n_(n) {
    if (n == 0) {
      return;
    }
    const std::size_t bytes = (n * sizeof(cd) + 63) / 64 * 64;
    void* p = std::aligned_alloc(64, bytes);
    if (p == nullptr) {
      throw std::bad_alloc();
    }
    std::memset(p, 0, bytes);
    p_.reset(static_cast<cd*>(p));
  }
  [[nodiscard]] cd* data() const { return p_.get(); }
  [[nodiscard]] std::size_t size() const { return n_; }

private:
  struct release {
    void operator()(cd* p) const { std::free(p); }
  };
  std::unique_ptr<cd, release> p_;
  std::size_t n_ = 0;
};

// Runs f(first, count) over [0, items) in contiguous pieces, one per lane of the executor (the
// application sharing its loops out, as it shares the transforms), or whole when serial.
template <class F> void share(const config& cfg, std::size_t items, const F& f) {
  const std::size_t lanes = cfg.exec ? static_cast<std::size_t>(cfg.threads) : 1;
  const auto tasks = static_cast<int>(std::min(items, lanes));
  if (!cfg.exec || tasks < 2) {
    f(std::size_t{0}, items);
    return;
  }
  struct job {
    const F* f;
    std::size_t items;
    std::size_t tasks;
  };
  job j{&f, items, static_cast<std::size_t>(tasks)};
  cfg.exec->bulk(
      tasks,
      [](void* arg, int index) {
        const auto& jb = *static_cast<const job*>(arg);
        const auto i = static_cast<std::size_t>(index);
        const std::size_t a = jb.items * i / jb.tasks;
        const std::size_t b = jb.items * (i + 1) / jb.tasks;
        (*jb.f)(a, b - a);
      },
      &j);
}

// An object executed on the session's executor, or serially.
template <class Object, class In, class Out>
void run_on(const config& cfg, const Object& o, const In* in, Out* out, fftn::workspace& ws,
            const fftn::exec_args& args) {
  if (cfg.exec) {
    o.execute(in, out, *cfg.exec, ws, args);
  } else {
    o.execute(in, out, ws, args);
  }
}

std::string one_line(const std::string& text) {
  std::string out;
  for (const char ch : text) {
    out += ch == '\n' ? std::string("; ") : std::string(1, ch);
  }
  return out;
}

fftn::options opts(const config& cfg, bool in_place) {
  return fftn::options{.in_place = in_place, .force_isa = cfg.isa};
}

// The (y, x) planes of the field in its own order: axes y and x, the batch the components, then w
// adjacent (stride 1).
fftn::shape<cd, cd> field_planes(const geometry& g, fftn::direction dir) {
  fftn::shape<cd, cd> s;
  s.dir = dir;
  const auto nw = static_cast<std::ptrdiff_t>(g.nw);
  const auto row = static_cast<std::ptrdiff_t>(g.nx) * nw;
  s.axes.push_back(fftn::axis{.n = g.ny, .in_stride = row, .out_stride = row});
  s.axes.push_back(fftn::axis{.n = g.nx, .in_stride = nw, .out_stride = nw});
  if (g.groups > 1) {
    const auto v = static_cast<std::ptrdiff_t>(g.comp());
    s.batches.push_back(fftn::batch{.count = g.groups, .in_dist = v, .out_dist = v});
  }
  s.batches.push_back(fftn::batch{.count = g.nw, .in_dist = 1, .out_dist = 1});
  return s;
}

coord_map planes_map(const geometry& g) {
  coord_map m;
  m.axis = {ry, rx};
  m.axes = 2;
  if (g.groups > 1) {
    m.brole[m.batches] = rc;
    m.bcount[m.batches++] = g.groups;
  }
  m.brole[m.batches] = rw;
  m.bcount[m.batches++] = g.nw;
  return m;
}

// The pencils' spectral pair, inverse first, over `pixels` pencils `nw` apart from where the
// arrays start (a whole field or a y slab), the components a group. The envelope's is <C, C, C>
// in place; the real line's <C, T, C> with the band as runs, out of place (runs are refused in
// place, task 5.2d).
template <bool Real>
using pair_type =
    std::conditional_t<Real, fftn::pipeline<cd, double, cd>, fftn::pipeline<cd, cd, cd>>;

template <bool Real>
pair_type<Real> make_pair(const geometry& g, std::size_t pixels, const config& cfg) {
  using Mid = std::conditional_t<Real, double, cd>;
  fftn::pipeline_desc<cd, Mid, cd> d;
  d.first = fftn::direction::backward;
  d.line = fftn::axis{.n = g.nt};
  const auto nw = static_cast<std::ptrdiff_t>(g.nw);
  if (g.groups > 1) {
    const auto v = static_cast<std::ptrdiff_t>(g.comp());
    d.batches.push_back(fftn::batch{.count = g.groups, .in_dist = v, .out_dist = v});
    d.group = 0;
  }
  d.batches.push_back(fftn::batch{.count = pixels, .in_dist = nw, .out_dist = nw});
  if constexpr (Real) {
    d.runs = {
        fftn::stored_run{
            .axis = 0, .where = fftn::side::in, .logical = g.first, .stored = 0, .count = g.nw},
        fftn::stored_run{
            .axis = 0, .where = fftn::side::out, .logical = g.first, .stored = 0, .count = g.nw},
    };
    d.middle.call = {.fn = &middle_real_cb};
  } else {
    d.middle.call = {.fn = &middle_complex_cb};
  }
  return pair_type<Real>(d, opts(cfg, !Real));
}

// ---- T: today's route ----------------------------------------------------------------------

// The node chunk (pixels per chunk of rows) of the client application's catalogue for its three
// sizes; elsewhere a chunk whose spectrum rows hold about 48 MiB, at least three chunks.
std::size_t node_chunk(const geometry& g) {
  const std::size_t P = g.pixels();
  if (P == 384 * 384 && g.nt == 750) {
    return 4469;
  }
  if (P == 192 * 192 && g.nt == 384) {
    return 7373;
  }
  if (g.nt == 1728 && g.real) {
    return 1261;
  }
  const std::size_t budget = (std::size_t{48} << 20) / (sizeof(cd) * g.bins() * g.groups);
  return std::clamp<std::size_t>(budget, 1, std::max<std::size_t>(1, (P + 2) / 3));
}

template <bool Real> class today final : public arm {
  using time_t_ = std::conditional_t<Real, double, cd>;
  using inverse_rows = fftn::plan<cd, time_t_>;
  using forward_rows = fftn::plan<time_t_, cd>;

public:
  today(const geometry& g, const config& cfg)
      : g_(g), k_(g, cfg.nonlinear), cfg_(cfg), chunk_(std::min(node_chunk(g), g.pixels())),
        planes_inv_(
            fftn::planes_c2c<double>(g.ny, g.nx, g.groups * g.nw, fftn::direction::backward),
            opts(cfg, true)),
        planes_fwd_(fftn::planes_c2c<double>(g.ny, g.nx, g.groups * g.nw, fftn::direction::forward),
                    opts(cfg, true)),
        inv_(inverse_shape(chunk_), opts(cfg, false)),
        fwd_(forward_shape(chunk_), opts(cfg, false)), scratch_(g.groups * g.comp()),
        spec_(g.groups * chunk_ * g.bins()),
        time_(Real ? (g.groups * chunk_ * g.nt + 1) / 2 : g.groups * chunk_ * g.nt) {
    const std::size_t tail = g.pixels() % chunk_;
    if (tail != 0) {
      inv_tail_.emplace(inverse_shape(tail), opts(cfg, false));
      fwd_tail_.emplace(forward_shape(tail), opts(cfg, false));
    }
    const int lanes = cfg.threads;
    std::size_t bytes =
        std::max({planes_inv_.workspace_bytes(lanes), planes_fwd_.workspace_bytes(lanes),
                  inv_.workspace_bytes(lanes), fwd_.workspace_bytes(lanes)});
    if (inv_tail_) {
      bytes =
          std::max({bytes, inv_tail_->workspace_bytes(lanes), fwd_tail_->workspace_bytes(lanes)});
    }
    ws_ = fftn::workspace(bytes);
  }

  void run(void* field) const override {
    auto* F = static_cast<cd*>(field);
    auto* Fd = reinterpret_cast<double*>(F);
    auto* S = scratch_.data();
    auto* Sd = reinterpret_cast<double*>(S);
    auto* spec = spec_.data();
    auto* specd = reinterpret_cast<double*>(spec);
    auto* tbuf = reinterpret_cast<time_t_*>(time_.data());
    auto* timed = reinterpret_cast<double*>(time_.data());
    const std::size_t P = g_.pixels();
    // 1. The stage input: the propagator, crossing into the scratch's plane-major order.
    share(cfg_, g_.groups * tiles(P, TP),
          [&](std::size_t first, std::size_t count) { cross_in(g_, k_, Fd, Sd, first, count); });
    // 2. The inverse planes, in place on the scratch.
    run_on(cfg_, planes_inv_, S, S, ws_, {});
    // 3. The rows, a node chunk at a time: pack, to time, the response, to frequency, crop.
    for (std::size_t p0 = 0; p0 < P; p0 += chunk_) {
      const std::size_t cnt = std::min(chunk_, P - p0);
      const bool whole = cnt == chunk_;
      share(cfg_, g_.groups * tiles(cnt, TP), [&](std::size_t first, std::size_t count) {
        pack(g_, Sd, specd, p0, cnt, first, count);
      });
      run_on(cfg_, whole ? inv_ : *inv_tail_, spec, tbuf, ws_, {});
      share(cfg_, cnt, [&](std::size_t first, std::size_t count) {
        response(g_, k_, timed, cnt, first, count);
      });
      run_on(cfg_, whole ? fwd_ : *fwd_tail_, tbuf, spec, ws_, {});
      share(cfg_, g_.groups * tiles(cnt, TP), [&](std::size_t first, std::size_t count) {
        crop(g_, specd, Sd, p0, cnt, first, count);
      });
    }
    // 4. The forward planes, in place on the scratch.
    run_on(cfg_, planes_fwd_, S, S, ws_, {});
    // 5. The finish, crossing back into the field's order.
    share(cfg_, g_.groups * tiles(P, TP),
          [&](std::size_t first, std::size_t count) { cross_out(g_, k_, Sd, Fd, first, count); });
  }

  [[nodiscard]] std::string note() const override {
    return "T (today's route): 9 sweeps (input crossing, inverse planes, pack, rows to time, "
           "response, rows to frequency, crop, forward planes, finish crossing; planes counted "
           "once); node chunk " +
           std::to_string(chunk_) + " pixels; planes: " + one_line(planes_inv_.describe()) +
           "; rows to time: " + one_line(inv_.describe());
  }

private:
  [[nodiscard]] fftn::shape<cd, time_t_> inverse_shape(std::size_t cnt) const {
    if constexpr (Real) {
      return fftn::rows_c2r<double>(g_.nt, g_.groups * cnt);
    } else {
      return fftn::rows_c2c<double>(g_.nt, g_.groups * cnt, fftn::direction::backward);
    }
  }
  [[nodiscard]] fftn::shape<time_t_, cd> forward_shape(std::size_t cnt) const {
    if constexpr (Real) {
      return fftn::rows_r2c<double>(g_.nt, g_.groups * cnt);
    } else {
      return fftn::rows_c2c<double>(g_.nt, g_.groups * cnt, fftn::direction::forward);
    }
  }

  geometry g_;
  constants k_;
  config cfg_;
  std::size_t chunk_;
  fftn::c2c_plan<double> planes_inv_;
  fftn::c2c_plan<double> planes_fwd_;
  inverse_rows inv_;
  forward_rows fwd_;
  std::optional<inverse_rows> inv_tail_;
  std::optional<forward_rows> fwd_tail_;
  buffer scratch_;
  buffer spec_;
  buffer time_;
  mutable fftn::workspace ws_;
};

// ---- O and Of: the three objects -------------------------------------------------------------

template <bool Real> class objects final : public arm {
public:
  objects(const geometry& g, const config& cfg, bool factor)
      : g_(g), k_(g, cfg.nonlinear), cfg_(cfg), factor_(factor),
        planes_inv_(field_planes(g, fftn::direction::backward), opts(cfg, true),
                    inverse_edges(g, factor)),
        pair_(make_pair<Real>(g, g.pixels(), cfg)),
        planes_fwd_(field_planes(g, fftn::direction::forward), opts(cfg, !Real),
                    fftn::plan_edges<cd, cd>{.out = {.call = {.fn = &finish_cb}}}),
        second_(Real ? g.groups * g.comp() : 0) {
    user_.k = &k_;
    user_.map = planes_map(g);
    if (factor_) {
      // The propagator's factors, precomputed once (one per (y, x, w), shared by the components).
      factors_ = buffer(g.comp());
      auto* f = reinterpret_cast<double*>(factors_.data());
      for (std::size_t p = 0; p < g.pixels(); ++p) {
        for (std::size_t w = 0; w < g.nw; ++w) {
          double* e = f + 2 * (p * g.nw + w);
          e[0] = 1.0;
          e[1] = 0.0;
          propagate(e, k_, p / g.nx, p % g.nx, w);
        }
      }
    }
    const int lanes = cfg.threads;
    ws_ =
        fftn::workspace(std::max({planes_inv_.workspace_bytes(lanes), pair_.workspace_bytes(lanes),
                                  planes_fwd_.workspace_bytes(lanes)}));
  }

  void run(void* field) const override {
    auto* F = static_cast<cd*>(field);
    fftn::exec_args first{};
    if (factor_) {
      first.in_factor = factors_.data();
    } else {
      first.in_user = &user_;
    }
    run_on(cfg_, planes_inv_, F, F, ws_, first);
    const fftn::exec_args middle{.mid_user = const_cast<constants*>(&k_)};
    const fftn::exec_args last{.out_user = &user_};
    if constexpr (Real) {
      cd* B = second_.data();
      run_on(cfg_, pair_, F, B, ws_, middle);
      run_on(cfg_, planes_fwd_, B, F, ws_, last);
    } else {
      run_on(cfg_, pair_, F, F, ws_, middle);
      run_on(cfg_, planes_fwd_, F, F, ws_, last);
    }
  }

  [[nodiscard]] std::string note() const override {
    return std::string(factor_ ? "Of" : "O") +
           " (three objects): inverse planes: " + one_line(planes_inv_.describe()) +
           "; pencils: " + one_line(pair_.describe()) +
           "; forward planes: " + one_line(planes_fwd_.describe());
  }

private:
  static fftn::plan_edges<cd, cd> inverse_edges(const geometry& g, bool factor) {
    fftn::plan_edges<cd, cd> e;
    if (factor) {
      const auto nw = static_cast<std::ptrdiff_t>(g.nw);
      e.in.mul.kind = fftn::factor_kind::complex;
      e.in.mul.axis_strides = {static_cast<std::ptrdiff_t>(g.nx) * nw, nw};
      e.in.mul.batch_strides =
          g.groups > 1 ? std::vector<std::ptrdiff_t>{0, 1} : std::vector<std::ptrdiff_t>{1};
    } else {
      e.in.call = {.fn = &prologue_cb};
    }
    return e;
  }

  geometry g_;
  constants k_;
  config cfg_;
  bool factor_;
  fftn::c2c_plan<double> planes_inv_;
  pair_type<Real> pair_;
  fftn::c2c_plan<double> planes_fwd_;
  buffer second_;  // the real line's pencils' output
  buffer factors_; // Of
  mutable cb_user user_;
  mutable fftn::workspace ws_;
};

// ---- G: the (x, w) slab grouping ----------------------------------------------------------------

template <bool Real> class slabs final : public arm {
public:
  slabs(const geometry& g, const config& cfg)
      : g_(g), k_(g, cfg.nonlinear), cfg_(cfg),
        y_inv_(y_shape(g, fftn::direction::backward), opts(cfg, true),
               fftn::plan_edges<cd, cd>{.in = {.call = {.fn = &prologue_cb}}}),
        y_fwd_(y_shape(g, fftn::direction::forward), opts(cfg, true),
               fftn::plan_edges<cd, cd>{.out = {.call = {.fn = &finish_cb}}}),
        x_inv_(x_shape(g, fftn::direction::backward), opts(serial(cfg), true)),
        x_fwd_(x_shape(g, fftn::direction::forward), opts(serial(cfg), !Real)),
        pair_(slab_pair(g, serial(cfg))) {
    y_user_.k = &k_;
    y_user_.map.axis = {ry, rx};
    y_user_.map.axes = 1;
    if (g.groups > 1) {
      y_user_.map.brole[y_user_.map.batches] = rc;
      y_user_.map.bcount[y_user_.map.batches++] = g.groups;
    }
    y_user_.map.brole[y_user_.map.batches] = rx;
    y_user_.map.bcount[y_user_.map.batches++] = g.nx;
    y_user_.map.brole[y_user_.map.batches] = rw;
    y_user_.map.bcount[y_user_.map.batches++] = g.nw;
    const int lanes = cfg.threads;
    ws_ = fftn::workspace(std::max(y_inv_.workspace_bytes(lanes), y_fwd_.workspace_bytes(lanes)));
    const std::size_t member_bytes =
        std::max({x_inv_.workspace_bytes(1), x_fwd_.workspace_bytes(1), pair_.workspace_bytes(1)});
    const auto members = static_cast<std::size_t>(cfg.exec ? cfg.threads : 1);
    for (std::size_t m = 0; m < members; ++m) {
      member_ws_.emplace_back(member_bytes);
      if constexpr (Real) {
        slab_.emplace_back(g.groups * g.nx * g.nw);
      }
    }
  }

  void run(void* field) const override {
    auto* F = static_cast<cd*>(field);
    run_on(cfg_, y_inv_, F, F, ws_, fftn::exec_args{.in_user = &y_user_});
    // Each member runs whole slabs, its share dealt as transforms_of deals (contiguous ranges in
    // rank order differing by at most one).
    const auto members = static_cast<std::size_t>(cfg_.exec ? cfg_.threads : 1);
    if (!cfg_.exec || members < 2) {
      slab_range(F, 0, 0, g_.ny);
    } else {
      struct job {
        const slabs* self;
        cd* field;
        std::size_t members;
      };
      job j{this, F, members};
      cfg_.exec->bulk(
          static_cast<int>(members),
          [](void* arg, int index) {
            const auto& jb = *static_cast<const job*>(arg);
            const auto r = static_cast<std::size_t>(index);
            const std::size_t ny = jb.self->g_.ny;
            const std::size_t base = ny / jb.members;
            const std::size_t extra = ny % jb.members;
            const std::size_t first = r * base + std::min(r, extra);
            jb.self->slab_range(jb.field, r, first, base + (r < extra ? 1 : 0));
          },
          &j);
    }
    run_on(cfg_, y_fwd_, F, F, ws_, fftn::exec_args{.out_user = &y_user_});
  }

  [[nodiscard]] std::string note() const override {
    return "G (slabs): y planes: " + one_line(y_inv_.describe()) +
           "; per slab x: " + one_line(x_inv_.describe()) +
           "; pencils: " + one_line(pair_.describe());
  }

private:
  static config serial(const config& cfg) {
    config c = cfg;
    c.exec.reset();
    c.threads = 1;
    return c;
  }

  // Axis y of the field, the batch the components, x and w.
  static fftn::shape<cd, cd> y_shape(const geometry& g, fftn::direction dir) {
    fftn::shape<cd, cd> s;
    s.dir = dir;
    const auto nw = static_cast<std::ptrdiff_t>(g.nw);
    const auto row = static_cast<std::ptrdiff_t>(g.nx) * nw;
    s.axes.push_back(fftn::axis{.n = g.ny, .in_stride = row, .out_stride = row});
    if (g.groups > 1) {
      const auto v = static_cast<std::ptrdiff_t>(g.comp());
      s.batches.push_back(fftn::batch{.count = g.groups, .in_dist = v, .out_dist = v});
    }
    s.batches.push_back(fftn::batch{.count = g.nx, .in_dist = nw, .out_dist = nw});
    s.batches.push_back(fftn::batch{.count = g.nw, .in_dist = 1, .out_dist = 1});
    return s;
  }

  // Axis x of one y slab, the batch the components and w. Real lines: the forward x plan reads
  // the member's slab buffer (components nx nw apart) and writes the field's slab.
  static fftn::shape<cd, cd> x_shape(const geometry& g, fftn::direction dir) {
    fftn::shape<cd, cd> s;
    s.dir = dir;
    const auto nw = static_cast<std::ptrdiff_t>(g.nw);
    s.axes.push_back(fftn::axis{.n = g.nx, .in_stride = nw, .out_stride = nw});
    if (g.groups > 1) {
      const auto v = static_cast<std::ptrdiff_t>(g.comp());
      const auto slab = static_cast<std::ptrdiff_t>(g.nx) * nw;
      const bool from_slab = Real && dir == fftn::direction::forward;
      s.batches.push_back(
          fftn::batch{.count = g.groups, .in_dist = from_slab ? slab : v, .out_dist = v});
    }
    s.batches.push_back(fftn::batch{.count = g.nw, .in_dist = 1, .out_dist = 1});
    return s;
  }

  // The real line's pencils on a slab: out of place into the member's slab buffer, whose
  // components lie nx nw apart; the pair's batch says so.
  void slab_range(cd* F, std::size_t member, std::size_t first, std::size_t count) const {
    fftn::workspace& ws = member_ws_[member];
    const std::size_t slab = g_.nx * g_.nw;
    const fftn::exec_args middle{.mid_user = const_cast<constants*>(&k_)};
    for (std::size_t y = first; y < first + count; ++y) {
      cd* s = F + y * slab;
      x_inv_.execute(s, s, ws);
      if constexpr (Real) {
        cd* b = slab_[member].data();
        pair_.execute(s, b, ws, middle);
        x_fwd_.execute(b, s, ws);
      } else {
        pair_.execute(s, s, ws, middle);
        x_fwd_.execute(s, s, ws);
      }
    }
  }

  geometry g_;
  constants k_;
  config cfg_;
  fftn::c2c_plan<double> y_inv_;
  fftn::c2c_plan<double> y_fwd_;
  fftn::c2c_plan<double> x_inv_;
  fftn::c2c_plan<double> x_fwd_;
  pair_type<Real> pair_;
  std::vector<buffer> slab_; // the real line's pencils' output, one slab per member
  mutable std::vector<fftn::workspace> member_ws_;
  mutable cb_user y_user_;
  mutable fftn::workspace ws_;

  // The pencils of one y slab: the envelope's in place; the real line's out of place into a
  // member's slab buffer, whose components lie nx nw apart.
  static pair_type<Real> slab_pair(const geometry& g, const config& cfg) {
    if constexpr (!Real) {
      return make_pair<false>(g, g.nx, cfg);
    } else {
      fftn::pipeline_desc<cd, double, cd> d;
      d.first = fftn::direction::backward;
      d.line = fftn::axis{.n = g.nt};
      const auto nw = static_cast<std::ptrdiff_t>(g.nw);
      if (g.groups > 1) {
        const auto v = static_cast<std::ptrdiff_t>(g.comp());
        const auto slab = static_cast<std::ptrdiff_t>(g.nx) * nw;
        d.batches.push_back(fftn::batch{.count = g.groups, .in_dist = v, .out_dist = slab});
        d.group = 0;
      }
      d.batches.push_back(fftn::batch{.count = g.nx, .in_dist = nw, .out_dist = nw});
      d.runs = {
          fftn::stored_run{
              .axis = 0, .where = fftn::side::in, .logical = g.first, .stored = 0, .count = g.nw},
          fftn::stored_run{
              .axis = 0, .where = fftn::side::out, .logical = g.first, .stored = 0, .count = g.nw},
      };
      d.middle.call = {.fn = &middle_real_cb};
      return pair_type<Real>(d, opts(cfg, false));
    }
  }
};

// ---- C: one sweep ---------------------------------------------------------------------------

class copy final : public arm {
public:
  copy(const geometry& g, const config& cfg) : g_(g), k_(g, cfg.nonlinear), cfg_(cfg) {}
  void run(void* field) const override {
    auto* F = static_cast<double*>(field);
    // Pieces of whole cache lines' worth of complex values.
    const std::size_t n = g_.groups * g_.comp();
    share(cfg_, (n + 3) / 4, [&](std::size_t first, std::size_t count) {
      const std::size_t a = first * 4;
      sweep(k_, F, a, std::min(n, (first + count) * 4) - a);
    });
  }
  [[nodiscard]] std::string note() const override {
    return "C (one in-place sweep of the field, the bandwidth reference)";
  }

private:
  geometry g_;
  constants k_;
  config cfg_;
};

std::optional<std::size_t> parse_count(std::string_view text) {
  std::size_t v = 0;
  const auto res = std::from_chars(text.data(), text.data() + text.size(), v);
  if (text.empty() || res.ec != std::errc{} || res.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return v;
}

} // namespace

std::optional<arm_kind> parse_arm(std::string_view text) {
  if (text == "T") {
    return arm_kind::today;
  }
  if (text == "O") {
    return arm_kind::objects;
  }
  if (text == "Of") {
    return arm_kind::objects_factor;
  }
  if (text == "G") {
    return arm_kind::slabs;
  }
  if (text == "C") {
    return arm_kind::copy;
  }
  return std::nullopt;
}

std::optional<std::vector<time_line>> parse_time(std::string_view text) {
  std::vector<time_line> out;
  std::size_t at = 0;
  while (at <= text.size()) {
    const std::size_t end = std::min(text.find(',', at), text.size());
    const std::string_view item = text.substr(at, end - at);
    at = end + 1;
    std::array<std::size_t, 3> v{};
    std::size_t from = 0;
    for (std::size_t i = 0; i < 3; ++i) {
      const std::size_t dot = i < 2 ? item.find('.', from) : item.size();
      if (dot == std::string_view::npos) {
        return std::nullopt;
      }
      const std::optional<std::size_t> n = parse_count(item.substr(from, dot - from));
      if (!n) {
        return std::nullopt;
      }
      v[i] = *n;
      from = dot + 1;
    }
    const time_line t{.stored = v[0], .nt = v[1], .first = v[2]};
    if (t.stored == 0 || t.nt % 2 != 0 || t.first < 1 || t.first + t.stored > t.nt / 2) {
      return std::nullopt;
    }
    out.push_back(t);
  }
  return out;
}

namespace {
std::unique_ptr<arm> make_arm(const geometry& g, const config& cfg) {
  switch (cfg.arm) {
  case arm_kind::today:
    if (g.real) {
      return std::make_unique<today<true>>(g, cfg);
    }
    return std::make_unique<today<false>>(g, cfg);
  case arm_kind::objects:
  case arm_kind::objects_factor: {
    const bool factor = cfg.arm == arm_kind::objects_factor;
    if (g.real) {
      return std::make_unique<objects<true>>(g, cfg, factor);
    }
    return std::make_unique<objects<false>>(g, cfg, factor);
  }
  case arm_kind::slabs:
    if (g.real) {
      return std::make_unique<slabs<true>>(g, cfg);
    }
    return std::make_unique<slabs<false>>(g, cfg);
  case arm_kind::copy:
    return std::make_unique<copy>(g, cfg);
  }
  return nullptr;
}

// STAGE_SELFCHECK: the arm and T with the nonlinear work on one field of deterministic values,
// compared: their relative rms difference (about the rounding error when they compute the same).
double selfcheck(const geometry& g, const config& cfg) {
  config nl = cfg;
  nl.nonlinear = true;
  config ref = nl;
  ref.arm = arm_kind::today;
  const std::unique_ptr<arm> a = make_arm(g, nl);
  const std::unique_ptr<arm> t = make_arm(g, ref);
  const std::size_t n = g.groups * g.comp();
  std::vector<cd> x(n);
  std::uint64_t r = 0x9e3779b97f4a7c15ULL;
  for (cd& v : x) {
    r = r * 6364136223846793005ULL + 1442695040888963407ULL;
    const double re = static_cast<double>(r >> 11) * 0x1.0p-53 - 0.5;
    r = r * 6364136223846793005ULL + 1442695040888963407ULL;
    const double im = static_cast<double>(r >> 11) * 0x1.0p-53 - 0.5;
    v = cd(re, im);
  }
  std::vector<cd> y = x;
  a->run(x.data());
  t->run(y.data());
  double d2 = 0.0;
  double r2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    d2 += std::norm(x[i] - y[i]);
    r2 += std::norm(y[i]);
  }
  return std::sqrt(d2 / r2);
}
} // namespace

std::unique_ptr<arm> make(const fbench_case& c, const config& cfg, std::string& why) {
  if (c.kind != fbench_c2c || c.rank != 3 || c.in_place == 0 || c.precision != fbench_f64 ||
      c.interleave != 1 || c.outer != 1 || c.batch < 1 || c.batch > 2) {
    why = "STAGE takes c2c fields of rank 3 (ny x nx x stored frequencies), in place, in double, "
          "1 or 2 components (batch)";
    return nullptr;
  }
  geometry g;
  g.ny = static_cast<std::size_t>(c.dims[0]);
  g.nx = static_cast<std::size_t>(c.dims[1]);
  g.nw = static_cast<std::size_t>(c.dims[2]);
  g.groups = static_cast<std::size_t>(c.batch);
  g.nt = g.nw;
  for (const time_line& t : cfg.real_lines) {
    if (t.stored == g.nw) {
      g.real = true;
      g.nt = t.nt;
      g.first = t.first;
    }
  }
  if (cfg.selfcheck && cfg.arm != arm_kind::copy) {
    const double e = selfcheck(g, cfg);
    if (!(e <= 1e-12)) {
      why = "STAGE_SELFCHECK: the arm differs from T by " + std::to_string(e) +
            " (relative rms) with the nonlinear work";
      throw std::runtime_error(why);
    }
  }
  return make_arm(g, cfg);
}

} // namespace fbench::stage

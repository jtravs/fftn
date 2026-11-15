// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The harness of experiment X20 (dev/experiments/X20-call-overhead.md): where the time of a
// tiny serial execute goes. For each case (c2c rows, forward, out of place) it times, in
// ns per transform, the same work entered at four levels:
//
//   public  what plan::execute calls, execute_all (X28);
//   api     execute_core alone (argument checks, the workspace's team word and alignment,
//           then the driver);
//   driver  run_transforms with the slot addresses prepared once (the program's
//           interpretation: batch loops, records, the stage's pass_args copy, chunk ends);
//   kernel  each stage's kernel called directly per transform through its function
//           pointer, with pass_args prepared once and only src and dst set per call (the
//           floor of any program that calls kernels one transform at a time);
//   copy    the same with each pass stage's pass_args copied per transform, as the driver
//           does (X28: kernel - copy is what passing a pass_fn its addresses as arguments
//           would save; driver - copy the rest of the interpretation);
//   once    a leaf body's kernel called once per transform with nothing else: only the
//           loop around the indirect call.
//
//   x20_overhead [--n 2,4,8,16,32,64] [--batch 1,64] [--prec f64,f32] [--reps 15]
//                [--ms 20] [--isa avx2]
//
// One line per case and level: n batch prec level median_ns min_ns. The cases run in
// one process; fbench measures FFTN against FFTW, this harness only splits FFTN's time.

#include "core/driver.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "passes/pass_args.hpp"

#include <fftn/detail/core.hpp>
#include <fftn/fftn.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <ratio>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
namespace d = fftn::detail;

std::vector<std::size_t> parse_list(std::string_view s) {
  std::vector<std::size_t> v;
  std::size_t at = 0;
  while (at <= s.size()) {
    const std::size_t comma = std::min(s.find(',', at), s.size());
    v.push_back(std::strtoull(std::string(s.substr(at, comma - at)).c_str(), nullptr, 10));
    at = comma + 1;
  }
  return v;
}

template <class F> double time_ns(F&& f, std::size_t transforms, int reps, double ms) {
  // Calibrate calls per sample to about `ms` milliseconds.
  std::size_t calls = 1;
  for (;;) {
    const auto t0 = clock_type::now();
    for (std::size_t i = 0; i < calls; ++i) {
      f();
    }
    const double el = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
    if (el > ms * 0.5) {
      calls =
          std::max<std::size_t>(1, static_cast<std::size_t>(static_cast<double>(calls) * ms / el));
      break;
    }
    calls *= 2;
  }
  std::vector<double> s;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = clock_type::now();
    for (std::size_t i = 0; i < calls; ++i) {
      f();
    }
    const double el = std::chrono::duration<double, std::nano>(clock_type::now() - t0).count();
    s.push_back(el / static_cast<double>(calls * transforms));
  }
  std::ranges::sort(s);
  return s[s.size() / 2];
}

const void* slot_src(d::slot s, const d::slot_addresses& at) {
  switch (s) {
  case d::slot::in:
    return at.in;
  case d::slot::out:
    return at.out;
  case d::slot::a:
    return at.a;
  case d::slot::b:
    return at.b;
  case d::slot::block:
    break;
  }
  return at.block;
}

void* slot_dst(d::slot s, const d::slot_addresses& at) {
  switch (s) {
  case d::slot::in:
  case d::slot::out:
    return at.out;
  case d::slot::a:
    return at.a;
  case d::slot::b:
    return at.b;
  case d::slot::block:
    break;
  }
  return at.block;
}

template <class T>
void run_case(std::size_t n, std::size_t batch, int reps, double ms, fftn::isa isa) {
  using C = std::complex<T>;
  const auto shape = fftn::rows_c2c<T>(n, batch, fftn::direction::forward);
  const std::shared_ptr<const d::plan_core> core =
      d::make_plan_core(d::view_of(shape), fftn::options{.force_isa = isa});
  fftn::workspace ws(core->bytes);
  std::vector<C> in(n * batch);
  std::vector<C> out(n * batch);
  for (std::size_t i = 0; i < in.size(); ++i) {
    in[i] = C(static_cast<T>(i % 7) - T(3), static_cast<T>(i % 5) - T(2));
  }
  const char* prec = sizeof(T) == 8 ? "f64" : "f32";
  const auto report = [&](const char* level, double ns) {
    std::printf("%zu %zu %s %s %.3f\n", n, batch, prec, level, ns);
  };

  report("public", time_ns(
                       [&] {
                         d::execute_all(*core, in.data(), out.data(), ws);
                       },
                       batch, reps, ms));
  report("api", time_ns([&] { d::execute_core(*core, 0, batch, in.data(), out.data(), ws); }, batch,
                        reps, ms));

  d::slot_addresses at{
      .in = reinterpret_cast<const std::byte*>(in.data()),
      .out = reinterpret_cast<std::byte*>(out.data()),
  };
  if (core->bytes != 0) {
    std::byte* raw = ws.memory().data();
    const auto address = reinterpret_cast<std::uintptr_t>(raw);
    std::byte* base = raw + (64 - address % 64) % 64;
    at.a = base;
    at.b = base + core->offset_b;
  }
  report("driver", time_ns([&] { d::run_transforms(core->prog, 0, batch, at); }, batch, reps, ms));

  // The body: the records after the batch loops, which for rows are stages only.
  const d::program& p = core->prog;
  std::vector<const d::stage*> body;
  for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
    const d::stage* st = std::get_if<d::stage>(&p.records[i]);
    if (st == nullptr) {
      std::fprintf(stderr, "n=%zu: the body has a loop; skipping kernel levels\n", n);
      return;
    }
    body.push_back(st);
  }
  std::vector<d::pass_args> args;
  args.reserve(body.size());
  for (const d::stage* st : body) {
    args.push_back(st->args);
  }
  const auto dist = static_cast<std::ptrdiff_t>(n * sizeof(C));
  // A stage's kernel on the slots `here`: a leaf with the addresses, a pass through args.
  const auto call = [&](std::size_t k, const d::slot_addresses& here) {
    const d::stage& st = *body[k];
    if (st.leaf != nullptr) {
      st.leaf(&st.args, slot_src(st.src, here), slot_dst(st.dst, here), 0, 0, 1);
      return;
    }
    args[k].src = slot_src(st.src, here);
    args[k].dst = slot_dst(st.dst, here);
    st.run(&args[k], 0, st.units);
  };
  report("kernel", time_ns(
                       [&] {
                         d::slot_addresses here = at;
                         for (std::size_t t = 0; t < batch; ++t) {
                           for (std::size_t k = 0; k < body.size(); ++k) {
                             call(k, here);
                           }
                           here.in += dist;
                           here.out += dist;
                         }
                       },
                       batch, reps, ms));
  report("copy", time_ns(
                     [&] {
                       d::slot_addresses here = at;
                       for (std::size_t t = 0; t < batch; ++t) {
                         for (std::size_t k = 0; k < body.size(); ++k) {
                           const d::stage& st = *body[k];
                           if (st.leaf != nullptr) {
                             call(k, here);
                             continue;
                           }
                           d::pass_args a = st.args;
                           a.src = slot_src(st.src, here);
                           a.dst = slot_dst(st.dst, here);
                           st.run(&a, 0, st.units);
                         }
                         here.in += dist;
                         here.out += dist;
                       }
                     },
                     batch, reps, ms));
  if (body.size() == 1 && body[0]->leaf != nullptr) {
    const d::leaf_fn fn = body[0]->leaf;
    const d::pass_args* a = &body[0]->args;
    report("once", time_ns(
                       [&] {
                         const C* src = in.data();
                         C* dst = out.data();
                         for (std::size_t t = 0; t < batch; ++t) {
                           fn(a, src, dst, 0, 0, 1);
                           src += n;
                           dst += n;
                         }
                       },
                       batch, reps, ms));
  }
}

int run(std::span<char* const> argv) {
  std::vector<std::size_t> ns{2, 4, 8, 16, 32, 64};
  std::vector<std::size_t> batches{1, 64};
  std::vector<std::string> precs{"f64", "f32"};
  int reps = 15;
  double ms = 20.0;
  fftn::isa isa = fftn::isa::best;
  for (std::size_t i = 1; i + 1 < argv.size(); i += 2) {
    const std::string_view k = argv[i];
    const std::string_view v = argv[i + 1];
    if (k == "--n") {
      ns = parse_list(v);
    } else if (k == "--batch") {
      batches = parse_list(v);
    } else if (k == "--prec") {
      precs.clear();
      if (v.find("f64") != std::string_view::npos) {
        precs.emplace_back("f64");
      }
      if (v.find("f32") != std::string_view::npos) {
        precs.emplace_back("f32");
      }
    } else if (k == "--reps") {
      reps = static_cast<int>(std::strtol(argv[i + 1], nullptr, 10));
    } else if (k == "--ms") {
      ms = std::strtod(argv[i + 1], nullptr);
    } else if (k == "--isa") {
      for (auto t = static_cast<unsigned>(fftn::isa::scalar);
           t <= static_cast<unsigned>(fftn::isa::best); ++t) {
        if (v == fftn::isa_name(static_cast<fftn::isa>(t))) {
          isa = static_cast<fftn::isa>(t);
        }
      }
    }
  }
  for (const std::string& prec : precs) {
    for (const std::size_t b : batches) {
      for (const std::size_t n : ns) {
        if (prec == "f64") {
          run_case<double>(n, b, reps, ms, isa);
        } else {
          run_case<float>(n, b, reps, ms, isa);
        }
      }
    }
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(std::span<char* const>(argv, static_cast<std::size_t>(argc)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "x20_overhead: %s\n", e.what());
    return 2;
  }
}

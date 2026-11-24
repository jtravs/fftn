// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The harness of experiment T3 (dev/experiments/T3-kokkos-dispatch.md): what one
// Kokkos::parallel_for per transform costs, and how team_barrier behaves. One JSON line per
// probe, case and arm: the median and spread of the arm's time over repetitions in this
// process, and the median of its paired ratio to the probe's base arm. The arms of a case run
// interleaved. bench/t3/t3.py combines the process rounds.
//
//   t3_kokkos --threads N [--probes dispatch,transform,scale,barrier] [--seconds S]
//
// Built against the test configuration's Kokkos (FFTN_KOKKOS_BACKEND): with the OpenMP back
// end the arms include fftn::omp::executor on the same runtime.

#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#ifdef KOKKOS_ENABLE_OPENMP
#include <fftn/omp.hpp>
#include <omp.h>
#endif

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <ratio>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using exec_space = Kokkos::DefaultHostExecutionSpace;
using team_policy = Kokkos::TeamPolicy<exec_space>;
using team_member = team_policy::member_type;
using cplx = Kokkos::complex<double>;
using rows_view = Kokkos::View<cplx**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using planes_view = Kokkos::View<cplx***, Kokkos::LayoutRight, Kokkos::HostSpace>;
using volumes_view = Kokkos::View<cplx****, Kokkos::LayoutRight, Kokkos::HostSpace>;
using steady = std::chrono::steady_clock;

double seconds_per_case = 1.0;
int threads = 0;

double now_ns() {
  return std::chrono::duration<double, std::nano>(steady::now().time_since_epoch()).count();
}

void spin_for_ns(double ns) {
  const double end = now_ns() + ns;
  while (now_ns() < end) {
  }
}

double median(std::vector<double> v) {
  std::ranges::sort(v);
  return v[v.size() / 2];
}

double quantile(std::vector<double> v, double q) {
  std::ranges::sort(v);
  return v[static_cast<std::size_t>(q * static_cast<double>(v.size() - 1))];
}

struct arm {
  std::string name;
  std::function<void()> run;
  double divisor = 1; // operations in one run
};

// Runs the arms in turn, rotating the first, until the case's time is used; the first arm is
// the base of the paired ratios.
void measure(const std::string& probe, const std::string& name, std::vector<arm> arms) {
  for (const arm& a : arms) {
    a.run(); // first touch, thread start-up
  }
  const double first = now_ns();
  arms[0].run();
  const double one = now_ns() - first;
  const double total = seconds_per_case * 1e9;
  const auto rounds = static_cast<std::size_t>(
      std::clamp(total / (std::max(one, 1.0) * static_cast<double>(arms.size())), 15.0, 20000.0));
  const std::size_t n = arms.size();
  std::vector<std::vector<double>> times(n, std::vector<double>(rounds));
  for (std::size_t r = 0; r < rounds; ++r) {
    for (std::size_t k = 0; k < n; ++k) {
      const std::size_t a = (k + r) % n;
      const double start = now_ns();
      arms[a].run();
      times[a][r] = (now_ns() - start) / arms[a].divisor;
    }
  }
  for (std::size_t a = 0; a < n; ++a) {
    std::vector<double> ratio(rounds);
    for (std::size_t r = 0; r < rounds; ++r) {
      ratio[r] = times[a][r] / times[0][r];
    }
    std::printf("{\"probe\":\"%s\",\"backend\":\"%s\",\"threads\":%d,\"case\":\"%s\","
                "\"arm\":\"%s\",\"reps\":%zu,\"median_ns\":%.1f,\"p10_ns\":%.1f,\"p90_ns\":%.1f,"
                "\"ratio\":%.5f}\n",
                probe.c_str(), exec_space::name(), threads, name.c_str(), arms[a].name.c_str(),
                rounds, median(times[a]), quantile(times[a], 0.1), quantile(times[a], 0.9),
                median(ratio));
  }
  std::fflush(stdout);
}

void empty_task(void* /*arg*/, int /*i*/) {}

// An empty dispatch to every lane, by each route.
void probe_dispatch(const exec_space& exec) {
  const fftn::kokkos::executor<exec_space> ex(exec);
  std::vector<arm> arms;
  arms.push_back({.name = "kokkos", .run = [&] { ex.bulk(threads, empty_task, nullptr); }});
  arms.push_back({.name = "kokkos2", .run = [&] { ex.bulk(threads, empty_task, nullptr); }});
  using policy =
      Kokkos::RangePolicy<exec_space, Kokkos::Schedule<Kokkos::Static>, Kokkos::IndexType<int>>;
  arms.push_back({
      .name = "parallel_for",
      .run =
          [&] {
            Kokkos::parallel_for(
                "fftn::bulk", policy(exec, 0, threads).set_chunk_size(1),
                fftn::kokkos::detail::bulk_call{.task = empty_task, .arg = nullptr});
          },
  });
  arms.push_back({
      .name = "team_region",
      .run =
          [&] {
            Kokkos::parallel_for("fftn::bulk", team_policy(exec, 1, threads),
                                 [](const team_member&) {});
          },
  });
#ifdef KOKKOS_ENABLE_OPENMP
  const fftn::omp::executor omp_ex(threads);
  arms.push_back({"omp", [&] { omp_ex.bulk(threads, empty_task, nullptr); }});
#endif
  measure("dispatch", "empty", std::move(arms));
}

// One transform by each route; the workspaces are per arm and sized for the width.
template <class View, class Plan>
void transform_arms(const std::string& name, const exec_space& exec, const View& x, const View& y,
                    const Plan& p) {
  const fftn::kokkos::executor<exec_space> ex(exec);
  const auto bytes = p.workspace_bytes(threads);
  fftn::workspace wk(bytes);
  fftn::workspace wk2(bytes);
  fftn::workspace ws(p);
  fftn::workspace wall(bytes);
  fftn::workspace wown(bytes);
  fftn::workspace wfour(bytes);
  const auto* in = fftn::kokkos::data_of(x);
  auto* out = fftn::kokkos::data_of(y);
  std::vector<arm> arms;
  arms.push_back({"kokkos", [&] { p.execute(in, out, ex, wk); }});
  arms.push_back({"kokkos2", [&] { p.execute(in, out, ex, wk2); }});
  arms.push_back({
      "team_all",
      [&] {
        Kokkos::parallel_for("fftn team", team_policy(exec, 1, threads), [&](const team_member& m) {
          p.execute(in, out, fftn::kokkos::team_of(m), wall);
        });
      },
  });
  arms.push_back({
      "team_own_barrier",
      [&] {
        Kokkos::parallel_for("fftn team", team_policy(exec, 1, threads), [&](const team_member& m) {
          p.execute(in, out, fftn::kokkos::team_of(m), wown, fftn::completion::own);
          m.team_barrier();
        });
      },
  });
  arms.push_back({
      "team_four_in_a_region",
      [&] {
        Kokkos::parallel_for("fftn team", team_policy(exec, 1, threads), [&](const team_member& m) {
          const fftn::team t = fftn::kokkos::team_of(m);
          for (int i = 0; i < 4; ++i) {
            p.execute(in, out, t, wfour);
          }
        });
      },
      4,
  });
#ifdef KOKKOS_ENABLE_OPENMP
  const fftn::omp::executor omp_ex(threads);
  fftn::workspace wo(bytes);
  arms.push_back({"omp", [&] { p.execute(in, out, omp_ex, wo); }});
  // The members of a plain OpenMP region as the team: team_all without Kokkos' team set-up.
  fftn::workspace wot(bytes);
  arms.push_back({"omp_team", [&] {
#pragma omp parallel num_threads(threads)
                    p.execute(in, out, fftn::omp::this_team(), wot);
                  }});
#endif
  measure("transform", name, std::move(arms));
  // The speed-up, apart from the arms above: a serial run leaves the data in one core's cache,
  // which slows whichever threaded arm runs after it (the first runs of T3 showed that).
  std::vector<arm> pair;
  pair.push_back({"kokkos", [&] { p.execute(in, out, ex, wk); }});
  pair.push_back({"serial", [&] { p.execute(in, out, ws); }});
  measure("speedup", name, std::move(pair));
}

void probe_transform(const exec_space& exec) {
  const auto rows = [&](std::size_t n, std::size_t count) {
    const rows_view x("x", count, n);
    const rows_view y("y", count, n);
    Kokkos::deep_copy(x, cplx(0.25, -0.5));
    const auto p =
        fftn::kokkos::make_plan(x, y, fftn::direction::forward, fftn::kokkos::axis_type<1>{-1});
    transform_arms("rows " + std::to_string(count) + "x" + std::to_string(n), exec, x, y, p);
  };
  rows(1024, 64);
  rows(4096, 64);
  rows(4096, 512);
  rows(65536, 8);
  const auto planes = [&](std::size_t nyx, std::size_t count) {
    const planes_view x("x", count, nyx, nyx);
    const planes_view y("y", count, nyx, nyx);
    Kokkos::deep_copy(x, cplx(0.25, -0.5));
    const auto p =
        fftn::kokkos::make_plan(x, y, fftn::direction::forward, fftn::kokkos::axis_type<2>{-2, -1});
    transform_arms("planes " + std::to_string(count) + "x" + std::to_string(nyx) + "^2", exec, x, y,
                   p);
  };
  planes(192, 16);
  planes(512, 4);
  planes(1024, 1);
  const volumes_view vx("x", 2, 64, 64, 64);
  const volumes_view vy("y", 2, 64, 64, 64);
  Kokkos::deep_copy(vx, cplx(0.25, -0.5));
  const auto vp = fftn::kokkos::make_plan(vx, vy, fftn::direction::forward,
                                          fftn::kokkos::axis_type<3>{-3, -2, -1});
  transform_arms("volumes 2x64^3", exec, vx, vy, vp);
}

// The scaling pass of view_plan::execute: the backward arm's time beyond the unscaled one, at
// sizes either side of the point where it starts to be shared.
void probe_scale(const exec_space& exec) {
  for (const std::size_t n : {2048U, 4096U, 8192U, 16384U, 32768U}) {
    const std::size_t count = 8;
    const rows_view x("x", count, n);
    const rows_view y("y", count, n);
    Kokkos::deep_copy(x, cplx(0.25, -0.5));
    const fftn::kokkos::view_plan p(exec, x, y, fftn::direction::backward, -1);
    std::vector<arm> arms;
    arms.push_back(
        {.name = "none", .run = [&] { p.execute(x, y, fftn::kokkos::normalization::none); }});
    arms.push_back(
        {.name = "scaled", .run = [&] { p.execute(x, y, fftn::kokkos::normalization::backward); }});
    measure("scale", std::to_string(count * n) + " elements", std::move(arms));
  }
}

// team_barrier: the time per barrier as the last member sees it, with one member arriving
// `delay_ns` late, and beside it an OpenMP barrier on the same threads.
void probe_barrier(const exec_space& exec) {
  for (const double delay_ns : {0.0, 10000.0, 100000.0}) {
    const int count = delay_ns > 0 ? 200 : 2000;
    std::vector<arm> arms;
    arms.push_back({
        .name = "team_barrier",
        .run =
            [&, delay_ns, count] {
              Kokkos::parallel_for("fftn barrier", team_policy(exec, 1, threads),
                                   [&, delay_ns, count](const team_member& m) {
                                     for (int i = 0; i < count; ++i) {
                                       if (m.team_rank() == 0 && delay_ns > 0) {
                                         spin_for_ns(delay_ns);
                                       }
                                       m.team_barrier();
                                     }
                                   });
            },
        .divisor = static_cast<double>(count),
    });
#ifdef KOKKOS_ENABLE_OPENMP
    arms.push_back({"omp_barrier",
                    [&, delay_ns, count] {
#pragma omp parallel num_threads(threads)
                      {
                        for (int i = 0; i < count; ++i) {
                          if (omp_get_thread_num() == 0 && delay_ns > 0) {
                            spin_for_ns(delay_ns);
                          }
#pragma omp barrier
                        }
                      }
                    },
                    static_cast<double>(count)});
#endif
    measure("barrier", "delay " + std::to_string(static_cast<int>(delay_ns / 1000)) + " us",
            std::move(arms));
  }
}

} // namespace

int main(int argc, char* const* argv) {
  try {
    std::set<std::string> probes{"dispatch", "transform", "scale", "barrier"};
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string key = argv[i];
      const std::string value = argv[i + 1];
      if (key == "--threads") {
        threads = std::stoi(value);
      } else if (key == "--seconds") {
        seconds_per_case = std::stod(value);
      } else if (key == "--probes") {
        probes.clear();
        std::stringstream list(value);
        for (std::string p; std::getline(list, p, ',');) {
          probes.insert(p);
        }
      } else {
        throw std::invalid_argument("unknown option " + key);
      }
    }
    if (threads < 2) {
      throw std::invalid_argument("--threads N (at least 2) is required");
    }
    const Kokkos::ScopeGuard kokkos(
        Kokkos::InitializationSettings().set_num_threads(threads).set_disable_warnings(true));
    const exec_space exec;
    if (exec.concurrency() != threads) {
      throw std::runtime_error("the back end gave another width than --threads");
    }
    if (probes.contains("dispatch")) {
      probe_dispatch(exec);
    }
    if (probes.contains("transform")) {
      probe_transform(exec);
    }
    if (probes.contains("scale")) {
      probe_scale(exec);
    }
    if (probes.contains("barrier")) {
      probe_barrier(exec);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "t3_kokkos: %s\n", e.what());
    return 1;
  }
}

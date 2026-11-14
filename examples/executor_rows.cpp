// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Executing a plan on an executor: FFTN's own thread pool, and an adapter over a pool of the
// program's own (here a stand-in that starts a std::thread per task), the examples quoted
// in docs/threading.md. Exits with 0 when both results are bit for bit those of serial
// execution.

#include <fftn/fftn.hpp>

#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <functional>
#include <numbers>
#include <thread>
#include <vector>

namespace {

// snippet-begin: adapter
// The program's own pool: anything that can run n jobs and wait for them.
struct my_pool {
  int threads = 4;

  [[nodiscard]] int size() const { return threads; }
  void parallel_for(int n, const std::function<void(int)>& job) const {
    std::vector<std::thread> team;
    team.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
      team.emplace_back([&, t] {
        for (int i = t; i < n; i += threads) {
          job(i);
        }
      });
    }
    for (std::thread& member : team) {
      member.join();
    }
  }
};

// The adapter FFTN needs: concurrency() and bulk(n, task, arg).
struct my_pool_executor {
  const my_pool& pool;
  [[nodiscard]] int concurrency() const { return pool.size(); }
  void bulk(int n, fftn::task_function task, void* arg) const {
    pool.parallel_for(n, [=](int i) { task(arg, i); });
  }
};
// snippet-end: adapter

bool same_bits(const std::vector<std::complex<double>>& a,
               const std::vector<std::complex<double>>& b) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool executors_match_serial() {
  const std::size_t n = 8192;
  const std::size_t count = 5;
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::polar(1.0, 2 * std::numbers::pi * static_cast<double>(i % 89) / 89.0);
  }
  const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  std::vector<std::complex<double>> serial(n * count);
  fftn::workspace serial_ws(plan);
  plan.execute(x.data(), serial.data(), serial_ws);

  // snippet-begin: pool
  // The program starts the pool; FFTN never starts threads by itself.
  const fftn::thread_pool pool(4);

  // A workspace for as many lanes as the executor has.
  fftn::workspace ws(plan, pool.concurrency());

  std::vector<std::complex<double>> y(n * count);
  plan.execute(x.data(), y.data(), pool, ws); // returns when y is written
  // snippet-end: pool

  // snippet-begin: custom
  const my_pool mine;
  std::vector<std::complex<double>> z(n * count);
  plan.execute(x.data(), z.data(), my_pool_executor{mine}, ws);
  // snippet-end: custom

  return same_bits(y, serial) && same_bits(z, serial);
}

} // namespace

int main() {
  try {
    const bool same = executors_match_serial();
    std::printf("executor results %s serial\n", same ? "equal" : "differ from");
    return same ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}

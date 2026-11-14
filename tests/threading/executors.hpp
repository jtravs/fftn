// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Custom executors for the executor tests: the contract is "run n tasks and return when they
// are done", in any order, on any threads, so these run them in order, in reverse, in a
// seeded random order on a few threads, or count what FFTN asked for. Test code: they may
// allocate and lock.

#pragma once

#include "threading/thread_team.hpp"

#include <fftn/fftn.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace fftn::test {

// Runs the tasks one after another on the caller, forwards or backwards; reports any width
// and grouping, and counts its calls.
struct sequential_executor {
  int width = 1;
  int group = 0;
  bool reverse = false;
  int calls = 0;
  int last_n = 0;

  [[nodiscard]] int concurrency() const { return width; }
  [[nodiscard]] int group_size() const { return group; }
  void bulk(int n, fftn::task_function task, void* arg) {
    ++calls;
    last_n = n;
    for (int k = 0; k < n; ++k) {
      task(arg, reverse ? n - 1 - k : k);
    }
  }
};

// Runs the tasks in a seeded random order on `threads` threads (fewer than the tasks, often):
// each thread takes the next task of the shuffled list until none is left. Reports any width
// and grouping.
class random_order_executor {
public:
  random_order_executor(int threads, std::uint64_t seed) : team_(threads), rng_(seed) {}

  int width = 1;
  int group = 0;

  [[nodiscard]] int concurrency() const { return width; }
  [[nodiscard]] int group_size() const { return group; }
  void bulk(int n, fftn::task_function task, void* arg) {
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng_);
    std::atomic<std::size_t> next{0};
    team_.run(team_.threads(), [&](int) {
      for (std::size_t k = next.fetch_add(1); k < order.size(); k = next.fetch_add(1)) {
        task(arg, order[k]);
      }
    });
  }

private:
  thread_team team_;
  std::mt19937_64 rng_;
};

// A thread pool seen with another width and grouping: more tasks than lanes, any grouping.
struct pool_view {
  const fftn::thread_pool* pool = nullptr;
  int width = 1;
  int group = 0;

  [[nodiscard]] int concurrency() const { return width; }
  [[nodiscard]] int group_size() const { return group; }
  void bulk(int n, fftn::task_function task, void* arg) const { pool->bulk(n, task, arg); }
};

// Runs tasks [0, ran) of each call, then throws: an executor that fails part way.
struct throwing_executor {
  int width = 4;
  int ran = 1;

  [[nodiscard]] int concurrency() const { return width; }
  void bulk(int n, fftn::task_function task, void* arg) const {
    for (int k = 0; k < std::min(ran, n); ++k) {
      task(arg, k);
    }
    throw std::runtime_error("executor failed");
  }
};

} // namespace fftn::test

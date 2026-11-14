// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Real threads for the team tests: a fixed set of std::threads, each running member `rank`
// of whatever team a test hands it, so that executions do not pay for creating threads.
// Test code, so it may lock; the members' own calls into FFTN are what is tested.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace fftn::test {

class thread_team {
public:
  explicit thread_team(int threads) {
    for (int r = 0; r < threads; ++r) {
      workers_.emplace_back([this, r] { work(r); });
    }
  }
  thread_team(const thread_team&) = delete;
  thread_team& operator=(const thread_team&) = delete;
  thread_team(thread_team&&) = delete;
  thread_team& operator=(thread_team&&) = delete;
  ~thread_team() {
    {
      const std::scoped_lock lock(mutex_);
      stop_ = true;
    }
    start_.notify_all();
    for (std::thread& t : workers_) {
      t.join();
    }
  }

  [[nodiscard]] int threads() const { return static_cast<int>(workers_.size()); }

  // Runs body(rank) on threads [0, members) at once and returns when all have returned.
  void run(int members, const std::function<void(int)>& body) {
    std::unique_lock lock(mutex_);
    body_ = &body;
    members_ = members;
    running_ = members;
    ++round_;
    start_.notify_all();
    finished_.wait(lock, [this] { return running_ == 0; });
    body_ = nullptr;
  }

private:
  void work(int rank) {
    std::size_t seen = 0;
    for (;;) {
      const std::function<void(int)>* body = nullptr;
      {
        std::unique_lock lock(mutex_);
        start_.wait(lock, [&] { return stop_ || (round_ != seen && rank < members_); });
        if (stop_) {
          return;
        }
        seen = round_;
        body = body_;
      }
      (*body)(rank);
      {
        const std::scoped_lock lock(mutex_);
        if (--running_ == 0) {
          finished_.notify_all();
        }
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable finished_;
  const std::function<void(int)>* body_ = nullptr;
  int members_ = 0;
  int running_ = 0;
  std::size_t round_ = 0;
  bool stop_ = false;
};

} // namespace fftn::test

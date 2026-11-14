// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The claiming engine (DESIGN.md 13.3): what one member of a team does in one execution
// with completion::all. Members claim work rather than wait for each other:
//
// 1. Whole tasks of its own cache group, in runs, from the group's counter.
// 2. The split tasks of its group: it joins one (at most `cap` members each) and claims
//    its chunks one at a time, in order. A chunk of phase p runs once every chunk of phase
//    p - 1 is done, which the task's done counter shows; the chunks of earlier phases were
//    all claimed before it, by members that are running, so the wait never depends on a
//    member that has not arrived. The member that claims chunk 0 owns the task: the task's
//    shared scratch is its scratch, and it leaves only when the task is done.
// 3. Whole tasks of other groups, and the split tasks of groups none of whose members have
//    arrived, run whole. A split task is never shared across groups.
// 4. It waits until every task is done.
//
// Whoever arrives, in whatever order, every task is claimed by a member that is running;
// one member alone does everything, and a member that arrives late finds nothing left.
//
// The engine is a template over its environment, which runs the work and supplies the
// waiting: team_env (team_env.hpp) in the library, and in the tests a simulator that
// interleaves members deterministically and keeps virtual time. Every access to shared
// state is preceded by env.point(), where the simulator may switch members.
#pragma once

#include "threading/schedule.hpp"
#include "threading/team_state.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace fftn::inline v0::detail {

// What one member needs to take part in one execution; the same for every member but rank.
struct team_job {
  const work_graph* graph = nullptr;
  schedule plan;
  lane_map lanes;
  std::byte* state = nullptr; // 128-byte aligned, team_layout{lanes}.bytes() long
  sync_word flag{nullptr};    // the workspace's word (team_state.hpp)
  int rank = 0;
};

template <class Env> class team_engine {
public:
  team_engine(const team_job& job, Env& env) noexcept
      : job_(job), env_(env), layout_{static_cast<std::size_t>(job.lanes.lanes)},
        group_(job.lanes.group_of(job.rank)),
        target_(job.plan.inline_all ? 1 : static_cast<std::uint32_t>(job.plan.tasks)) {}

  // Returns when every task of the execution is done.
  void run() noexcept {
    prepare();
    begin();
    // Each step returns false when it finds a later execution: this one is over.
    if (arrive() && whole_tasks(group_) && split_tasks() && steal()) {
      const sync_word blocked = word(team_layout::done_blocked());
      env_.wait(word(team_layout::done()), &blocked, [this](std::uint64_t v) {
        return is_later(v, epoch_) || count_in(v, epoch_) == target_;
      });
    }
  }

private:
  struct claimed {
    std::uint32_t at = 0;
    std::uint32_t count = 0;
    bool stale = false;
  };

  [[nodiscard]] sync_word word(std::size_t offset) const noexcept {
    // prepare() constructed an atomic at every multiple of 8 of the 128-byte aligned state.
    return sync_word(std::launder(reinterpret_cast<team_word_t*>(job_.state + offset)));
  }

  // Makes the state valid for this lane count: the first member to find it otherwise
  // constructs its words, all zero, while the others wait.
  void prepare() noexcept {
    const std::uint64_t ready = layout_.lanes * 4 + team_ready;
    const std::uint64_t preparing = layout_.lanes * 4 + team_preparing;
    for (;;) {
      env_.point();
      std::uint64_t seen = job_.flag.load(std::memory_order_acquire);
      if (seen == ready) {
        return;
      }
      if ((seen & 3U) == team_preparing) {
        env_.wait(job_.flag, nullptr, [seen](std::uint64_t v) { return v != seen; });
        continue;
      }
      if (job_.flag.compare_exchange(seen, preparing)) {
        for (std::size_t at = 0; at < layout_.bytes(); at += sizeof(team_word_t)) {
          ::new (static_cast<void*>(job_.state + at)) team_word_t(0);
        }
        job_.flag.store(ready);
        env_.wake(job_.flag, nullptr);
        return;
      }
    }
  }

  // This member's epoch: its count of executions on the workspace. Epoch 0 is skipped
  // when the count wraps: it is the zeroed state's.
  void begin() noexcept {
    const sync_word calls = word(team_layout::calls(static_cast<std::size_t>(job_.rank)));
    epoch_ = static_cast<std::uint32_t>(calls.load(std::memory_order_relaxed) + 1);
    if (epoch_ == 0) {
      epoch_ = 1;
    }
    calls.store(epoch_, std::memory_order_relaxed);
  }

  // Claims want(left) more of tagged counter w, which counts up to limit.
  template <class Want> claimed claim(sync_word w, std::uint32_t limit, Want want) noexcept {
    for (;;) {
      env_.point();
      std::uint64_t seen = w.load();
      if (is_later(seen, epoch_)) {
        return {.stale = true};
      }
      const std::uint32_t at = count_in(seen, epoch_);
      if (at >= limit) {
        return {.at = at};
      }
      const std::uint32_t take = want(limit - at);
      if (w.compare_exchange(seen, tagged(epoch_, at + take))) {
        return {.at = at, .count = take};
      }
    }
  }

  claimed claim_one(sync_word w, std::uint32_t limit) noexcept {
    return claim(w, limit, [](std::uint32_t) { return std::uint32_t{1}; });
  }

  // Adds to a tagged counter that cannot be from a later epoch: one that this execution
  // cannot finish without. Returns the new count.
  std::uint32_t add(sync_word w, std::uint32_t amount) noexcept {
    for (;;) {
      env_.point();
      std::uint64_t seen = w.load();
      const std::uint32_t next = count_in(seen, epoch_) + amount;
      if (w.compare_exchange(seen, tagged(epoch_, next))) {
        return next;
      }
    }
  }

  void finish(std::uint32_t tasks) noexcept {
    if (add(word(team_layout::done()), tasks) == target_) {
      const sync_word blocked = word(team_layout::done_blocked());
      env_.wake(word(team_layout::done()), &blocked);
    }
  }

  bool arrive() noexcept {
    return !claim_one(word(layout_.arrived(static_cast<std::size_t>(group_))),
                      std::numeric_limits<std::uint32_t>::max())
                .stale;
  }

  bool whole_tasks(int group) noexcept {
    const group_share share = share_of(job_.plan, job_.lanes, group);
    if (share.whole == 0) {
      return true;
    }
    const sync_word w = word(layout_.whole(static_cast<std::size_t>(group)));
    if (job_.plan.inline_all) {
      // One claim for the whole batch, which may be too large for the counters.
      const claimed c = claim_one(w, 1);
      if (c.count != 0) {
        env_.run_whole(0, job_.plan.tasks, job_.rank);
        finish(1);
      }
      return !c.stale;
    }
    const auto limit = static_cast<std::uint32_t>(share.whole);
    const auto runs = static_cast<std::uint32_t>(std::max(job_.plan.profile.claim_divisor, 1) *
                                                 job_.lanes.size_of(group));
    for (;;) {
      const claimed c =
          claim(w, limit, [runs](std::uint32_t left) { return std::max(left / runs, 1U); });
      if (c.count == 0) {
        return !c.stale;
      }
      env_.run_whole(share.first + c.at, c.count, job_.rank);
      finish(c.count);
    }
  }

  bool split_tasks() noexcept {
    const group_share share = share_of(job_.plan, job_.lanes, group_);
    // Members start on different tasks, so that each task gets going at once.
    const auto in_group = static_cast<std::size_t>(job_.rank - job_.lanes.base(group_));
    for (std::size_t k = 0; k < share.split; ++k) {
      if (!join(share, (in_group + k) % share.split)) {
        return false;
      }
    }
    return true;
  }

  // Takes part in split task j of this member's group until none of its chunks is left.
  bool join(const group_share& share, std::size_t j) noexcept {
    const std::size_t slot = static_cast<std::size_t>(job_.lanes.base(group_)) + j;
    const std::size_t task = share.first + share.whole + j;
    const claimed seat =
        claim_one(word(layout_.members(slot)), static_cast<std::uint32_t>(share.cap));
    if (seat.count == 0) {
      return !seat.stale;
    }
    const auto& phases = job_.graph->phases;
    const auto total = static_cast<std::uint32_t>(job_.graph->chunks);
    const sync_word done = word(layout_.chunks_done(slot));
    const sync_word blocked = word(layout_.task_blocked(slot));
    const sync_word owner_word = word(layout_.owner(slot));
    const std::uint32_t epoch = epoch_;
    int owner = -1;
    std::size_t p = 0;
    for (;;) {
      const claimed c = claim_one(word(layout_.claim(slot)), total);
      if (c.count == 0) {
        if (c.stale) {
          return false;
        }
        break;
      }
      if (c.at == 0) {
        owner = job_.rank;
        env_.point();
        owner_word.store(tagged(epoch, static_cast<std::uint32_t>(owner)));
        env_.wake(owner_word, &blocked);
      } else if (owner < 0) {
        // The owner claimed chunk 0 before this claim and is about to say so.
        env_.wait(owner_word, &blocked, [epoch](std::uint64_t v) { return epoch_of(v) == epoch; });
        owner = static_cast<int>(count_in(owner_word.load(), epoch));
      }
      while (c.at >= phases[p].first + phases[p].chunks) {
        ++p;
      }
      const std::uint64_t from = phases[p].first;
      if (from != 0) {
        env_.wait(done, &blocked,
                  [epoch, from](std::uint64_t v) { return count_in(v, epoch) >= from; });
      }
      env_.run_chunk(task, p, static_cast<std::uint32_t>(c.at - from), owner, job_.rank);
      const std::uint32_t now = add(done, 1);
      if (now == phases[p].first + phases[p].chunks) {
        env_.wake(done, &blocked);
        if (now == total) {
          finish(1);
        }
      }
    }
    if (owner == job_.rank) {
      // The task's shared scratch is this member's: keep it until the last chunk is done.
      env_.wait(done, &blocked, [epoch, total](std::uint64_t v) {
        return is_later(v, epoch) || count_in(v, epoch) == total;
      });
    }
    return true;
  }

  bool steal() noexcept {
    const int groups = job_.plan.groups;
    for (int k = 1; k <= groups; ++k) {
      const int h = (group_ + k) % groups;
      if (h == group_) {
        continue;
      }
      if (!whole_tasks(h)) {
        return false;
      }
      const group_share share = share_of(job_.plan, job_.lanes, h);
      if (share.split == 0) {
        continue;
      }
      // Split tasks are left to their group while any of its members is here.
      env_.point();
      const std::uint64_t arrived = word(layout_.arrived(static_cast<std::size_t>(h))).load();
      if (is_later(arrived, epoch_)) {
        return false;
      }
      if (count_in(arrived, epoch_) != 0) {
        continue;
      }
      for (std::size_t j = 0; j < share.split; ++j) {
        if (!take_whole(share, h, j)) {
          return false;
        }
      }
    }
    return true;
  }

  // Runs split task j of group h whole, unless someone has started it.
  bool take_whole(const group_share& share, int h, std::size_t j) noexcept {
    const std::size_t slot = static_cast<std::size_t>(job_.lanes.base(h)) + j;
    const sync_word w = word(layout_.claim(slot));
    const auto total = static_cast<std::uint32_t>(job_.graph->chunks);
    for (;;) {
      env_.point();
      std::uint64_t seen = w.load();
      if (is_later(seen, epoch_)) {
        return false;
      }
      if (count_in(seen, epoch_) != 0) {
        return true;
      }
      if (w.compare_exchange(seen, tagged(epoch_, total))) {
        break;
      }
    }
    env_.run_whole(share.first + share.whole + j, 1, job_.rank);
    finish(1);
    return true;
  }

  const team_job& job_;
  Env& env_;
  team_layout layout_;
  int group_ = 0;
  std::uint32_t target_ = 0;
  std::uint32_t epoch_ = 0;
};

} // namespace fftn::inline v0::detail

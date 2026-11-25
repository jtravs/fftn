// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The claiming engine (DESIGN.md 13.3): what one member of a team does in one execution
// with completion::all. Members claim work rather than wait for each other:
//
// 1. Whole tasks of its own cache group, in runs, from the group's counter; or, where the
//    profile anchors them (machine_profile::anchored_runs, X73), from its own range of them
//    first, then from the other members' ranges.
// 2. The split tasks of its group: it joins one (at most `cap` members each) and claims
//    its chunks in order, in runs within one phase (a share of the phase's chunks left,
//    shrinking to single chunks; X52). A chunk of phase p runs once every chunk of phase
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
        group_(job.lanes.group_of(job.rank)), target_(done_target(job.plan, job.lanes)) {}

  // Returns when every task of the execution is done.
  void run() noexcept {
    prepare();
    begin();
    // Each step returns false when it finds this execution over.
    if (arrive() && whole_tasks(group_) && split_tasks() && steal()) {
      const sync_word blocked = word(team_layout::done_blocked());
      env_.wait(word(team_layout::done()), &blocked,
                [this](std::uint64_t v) { return count_in(v, tag_) == target_ || over(); });
      publish();
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

  // This member's epoch: its count of executions on the workspace (64 bits: it never wraps).
  // Epochs whose tag is 0 are skipped: that is the zeroed state's tag.
  void begin() noexcept {
    const sync_word calls = word(team_layout::calls(static_cast<std::size_t>(job_.rank)));
    epoch_ = calls.load(std::memory_order_relaxed) + 1;
    if (tag_of_epoch(epoch_) == 0) {
      ++epoch_;
    }
    tag_ = tag_of_epoch(epoch_);
    calls.store(epoch_, std::memory_order_relaxed);
  }

  // Whether this execution is over: some member has returned from it or a later one. Read
  // after a word that carries another epoch's tag: a later execution writes that word only
  // after one of its members published, before starting it, the end of this one.
  [[nodiscard]] bool over() const noexcept {
    return word(team_layout::finished()).load() >= epoch_;
  }

  // Before returning: raises `finished` to this epoch.
  void publish() noexcept {
    const sync_word w = word(team_layout::finished());
    env_.point();
    std::uint64_t seen = w.load();
    while (seen < epoch_ && !w.compare_exchange(seen, epoch_)) {
    }
  }

  // Claims want(left) more of tagged counter w, which counts up to limit.
  template <class Want> claimed claim(sync_word w, std::uint32_t limit, Want want) noexcept {
    for (;;) {
      env_.point();
      std::uint64_t seen = w.load();
      if (tag_of(seen) != tag_ && over()) {
        return {.stale = true};
      }
      const std::uint32_t at = count_in(seen, tag_);
      if (at >= limit) {
        return {.at = at};
      }
      const std::uint32_t take = want(limit - at);
      if (w.compare_exchange(seen, tagged(tag_, at + take))) {
        return {.at = at, .count = take};
      }
    }
  }

  claimed claim_one(sync_word w, std::uint32_t limit) noexcept {
    return claim(w, limit, [](std::uint32_t) { return std::uint32_t{1}; });
  }

  // Adds to a tagged counter of this execution that it cannot finish without. Returns the
  // new count.
  std::uint32_t add(sync_word w, std::uint32_t amount) noexcept {
    for (;;) {
      env_.point();
      std::uint64_t seen = w.load();
      const std::uint32_t next = count_in(seen, tag_) + amount;
      if (w.compare_exchange(seen, tagged(tag_, next))) {
        return next;
      }
    }
  }

  void finish(std::uint32_t units) noexcept {
    if (add(word(team_layout::done()), units) == target_) {
      const sync_word blocked = word(team_layout::done_blocked());
      env_.wake(word(team_layout::done()), &blocked);
    }
  }

  bool arrive() noexcept {
    return !claim_one(word(layout_.arrived(static_cast<std::size_t>(group_))), count_mask).stale;
  }

  bool whole_tasks(int group) noexcept {
    const group_share share = share_of(job_.plan, job_.lanes, group);
    if (share.whole == 0) {
      return true;
    }
    const sync_word w = word(layout_.whole(static_cast<std::size_t>(group)));
    if (job_.plan.inline_all) {
      const claimed c = claim_one(w, 1);
      if (c.count != 0) {
        env_.run_whole(0, job_.plan.tasks, job_.rank);
        finish(1);
      }
      return !c.stale;
    }
    // Units of `grain` consecutive tasks, claimed in runs.
    const std::size_t grain = job_.plan.grain;
    const auto limit = static_cast<std::uint32_t>(whole_units(job_.plan, share));
    if (job_.plan.profile.anchored_runs) {
      return anchored_tasks(group, share, limit);
    }
    const auto runs = static_cast<std::uint32_t>(std::max(job_.plan.profile.claim_divisor, 1) *
                                                 job_.lanes.size_of(group));
    for (;;) {
      const claimed c =
          claim(w, limit, [runs](std::uint32_t left) { return std::max(left / runs, 1U); });
      if (c.count == 0) {
        return !c.stale;
      }
      const std::size_t from = std::size_t{c.at} * grain;
      const std::size_t to = std::min(std::size_t{c.at + c.count} * grain, share.whole);
      env_.run_whole(share.first + from, to - from, job_.rank);
      finish(c.count);
    }
  }

  // Whole tasks anchored to members (machine_profile::anchored_runs; X73): the group's
  // `units` units cut into one contiguous range per member dealt, in order of rank. A member
  // of the group starts with its own range, anyone else at the range of its rank, and each
  // then goes round the ranges in turn, claiming runs of half what is left of the range (at
  // least one unit) until none is left. With every member present each runs the same tasks
  // in every execution (X73: at 16 unpinned lanes on the EPYC 7763 the arrival-order runs put
  // each row of a batch beyond L3 on a different core complex in each execution: out of place
  // 2.3 ms against 1.2 ms for the same rows dealt by rank); a member that is late or absent
  // has its range taken by the others, so no task waits for it.
  bool anchored_tasks(int group, const group_share& share, std::uint32_t units) noexcept {
    const std::size_t grain = job_.plan.grain;
    const int base = job_.lanes.base(group);
    const auto dealt = static_cast<std::size_t>(
        std::max(std::min(base + job_.lanes.size_of(group), job_.plan.lanes_dealt) - base, 1));
    const auto start =
        static_cast<std::size_t>(group == group_ ? job_.rank - base : job_.rank) % dealt;
    for (std::size_t k = 0; k < dealt; ++k) {
      const std::size_t j = (start + k) % dealt;
      const std::size_t from = chunk_start(units, dealt, j);
      const auto length = static_cast<std::uint32_t>(chunk_start(units, dealt, j + 1) - from);
      const sync_word w = word(layout_.anchored(static_cast<std::size_t>(base) + j));
      for (;;) {
        const claimed c =
            claim(w, length, [](std::uint32_t left) { return std::max(left / 2, 1U); });
        if (c.count == 0) {
          if (c.stale) {
            return false;
          }
          break;
        }
        const std::size_t first = (from + c.at) * grain;
        const std::size_t last = std::min((from + c.at + c.count) * grain, share.whole);
        env_.run_whole(share.first + first, last - first, job_.rank);
        finish(c.count);
      }
    }
    return true;
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
    const std::uint64_t tag = tag_;
    int owner = -1;
    std::size_t p = 0;
    // Chunks are claimed in runs within one phase, each a share of the phase's chunks left
    // (left / (cap x max(claim_divisor, 2)), at least one): one claim and one count per run
    // rather than per chunk, and a member's chunks of a phase adjacent (X52 arms G and G2: at
    // 8 lanes Xeon 0.91, EPYC 0.90, Mac at 6 lanes 0.86-0.94 over the planes and volumes that
    // split; on the CCX, whose whole tasks go in runs of left / members, runs that large
    // left a real 256^2 plane 1.035 slower, so chunk runs take at most half that share). A run
    // never crosses a phase: the counter only rises within an epoch, so the value `claim`
    // hands the lambda is at least this member's last claim, in phase p or later, and below
    // `total`.
    // Where a phase's neighbouring lines write the same cache lines and a member's fair share
    // of it is small, a run is that share instead (chunk_run, work_graph.hpp; X59).
    const auto cap = static_cast<std::uint32_t>(std::max(share.cap, 1));
    const auto runs =
        static_cast<std::uint32_t>(std::max(job_.plan.profile.claim_divisor, 2)) * cap;
    for (;;) {
      // While this member holds a claim the execution cannot end, so no word of this task
      // can carry a later tag before the loop leaves.
      const claimed c = claim(
          word(layout_.claim(slot)), total, [&phases, total, runs, cap, p](std::uint32_t left) {
            const std::uint32_t at = total - left;
            std::size_t q = p;
            while (at >= phases[q].first + phases[q].chunks) {
              ++q;
            }
            const graph_phase& ph = phases[q];
            return chunk_run(ph, static_cast<std::uint32_t>(ph.first + ph.chunks) - at, cap, runs);
          });
      if (c.count == 0) {
        if (c.stale) {
          return false;
        }
        break;
      }
      if (c.at == 0) {
        owner = job_.rank;
        env_.point();
        owner_word.store(tagged(tag, static_cast<std::uint32_t>(owner)));
        env_.wake(owner_word, &blocked);
      } else if (owner < 0) {
        // The owner claimed chunk 0 before this claim and is about to say so.
        env_.wait(owner_word, &blocked, [tag](std::uint64_t v) { return tag_of(v) == tag; });
        owner = static_cast<int>(count_in(owner_word.load(), tag));
      }
      while (c.at >= phases[p].first + phases[p].chunks) {
        ++p;
      }
      const std::uint64_t from = phases[p].first;
      if (from != 0) {
        env_.wait(done, &blocked,
                  [tag, from](std::uint64_t v) { return count_in(v, tag) >= from; });
      }
      for (std::uint32_t k = 0; k < c.count; ++k) {
        env_.run_chunk(task, p, static_cast<std::uint32_t>(c.at + k - from), owner, job_.rank);
      }
      const std::uint32_t now = add(done, c.count);
      if (now == phases[p].first + phases[p].chunks) {
        env_.wake(done, &blocked);
        if (now == total) {
          finish(1);
        }
      }
    }
    if (owner == job_.rank) {
      // The task's shared scratch is this member's: keep it until the last chunk is done.
      env_.wait(done, &blocked, [this, tag, total](std::uint64_t v) {
        return count_in(v, tag) == total || over();
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
      if (tag_of(arrived) != tag_ && over()) {
        return false;
      }
      if (count_in(arrived, tag_) != 0) {
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
      if (tag_of(seen) != tag_ && over()) {
        return false;
      }
      if (count_in(seen, tag_) != 0) {
        return true;
      }
      if (w.compare_exchange(seen, tagged(tag_, total))) {
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
  std::uint64_t epoch_ = 0;
  std::uint64_t tag_ = 0;
};

} // namespace fftn::inline v0::detail

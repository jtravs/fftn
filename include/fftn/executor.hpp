// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Executors: anything that can run n tasks and return when they are done.
#pragma once

#include "fftn/config.hpp"
#include "fftn/executor.h"

#include <concepts>
#include <memory>
#include <type_traits>

namespace fftn::inline v0 {

/// The tasks an executor runs: `task(arg, index)`, the same type as the C contract's
/// `fftn_task`.
using task_function = fftn_task;

/// What FFTN needs from a threading system to execute a plan on it (the executor overload
/// of `plan::execute`): an object `e` with
///
/// - `e.concurrency()`, convertible to `int`: the most tasks it can usefully run at once
///   when called from the calling thread now, at least 1. An executor called from inside
///   one of its own tasks (or its runtime's parallel region) returns 1, so that a nested
///   execution runs serially instead of waiting for busy threads.
/// - `e.bulk(n, task, arg)`: runs `task(arg, i)` once for every `i` in `[0, n)` and returns
///   when all have returned. Tasks may run in any order, on any threads, any number at a
///   time, the calling thread among them, even one after another on the caller: FFTN's
///   tasks never wait for a task that has not started. The call of `bulk` happens before
///   each task starts, and each task's return happens before `bulk` returns (as a mutex,
///   a thread join or a release and acquire pair order them). If `bulk` throws, it does so
///   only once every task it started has returned.
///
/// and, optionally, `e.group_size()`, convertible to `int`: how many consecutive task
/// indices share a cache (as `fftn::team::group_size`; 0, the default, puts all in one
/// group). Grouping affects speed only, never results.
///
/// `fftn::thread_pool` and `fftn::omp::executor` (`fftn/omp.hpp`) are executors; so is
/// any adapter of a few lines over another pool (docs/threading.md).
template <class E>
concept executor = requires(E& e, int n, task_function task, void* arg) {
  { e.concurrency() } -> std::convertible_to<int>;
  e.bulk(n, task, arg);
};

/// A non-owning reference to an executor, or to a C pool's `fftn_executor`: what the
/// executor overload of `plan::execute` takes, so that any executor type reaches the
/// compiled library through three function pointers. Cheap to copy.
///
/// Preconditions: the executor outlives every use of the reference (a temporary executor
/// lives until the end of the full expression that names it, which covers a call of
/// `execute`).
class executor_ref {
public:
  /// Refers to executor `e`, which is not copied. Implicit, so that an executor is passed
  /// where an `executor_ref` is expected.
  template <class E>
    requires executor<std::remove_reference_t<E>> &&
             (!std::same_as<std::remove_cvref_t<E>, executor_ref>)
  // NOLINTNEXTLINE(misc-explicit-constructor): implicit, so that executors convert at calls
  executor_ref(E&& e) noexcept
      : c_{
            .self = const_cast<void*>(static_cast<const void*>(std::addressof(e))),
            .concurrency = &concurrency_of<std::remove_reference_t<E>>,
            .bulk = &bulk_of<std::remove_reference_t<E>>,
            .group_size = group_size_function<std::remove_reference_t<E>>(),
        } {}

  /// Refers to the pool that `c` describes, whose `concurrency` and `bulk` are not null.
  /// The struct itself is copied; the pool is not.
  explicit executor_ref(const fftn_executor& c) noexcept : c_(c) {}

  /// The executor's `concurrency()`.
  [[nodiscard]] int concurrency() const { return c_.concurrency(c_.self); }

  /// The executor's `group_size()`, or 0 when it has none.
  [[nodiscard]] int group_size() const {
    return c_.group_size != nullptr ? c_.group_size(c_.self) : 0;
  }

  /// The executor's `bulk(n, task, arg)`.
  void bulk(int n, task_function task, void* arg) const { c_.bulk(c_.self, n, task, arg); }

private:
  using group_size_type = int (*)(void*);

  // E may be const: the pointer was stored without its qualifiers and gets them back here.
  template <class E> static E& object(void* self) noexcept { return *static_cast<E*>(self); }
  template <class E> static int concurrency_of(void* self) {
    return static_cast<int>(object<E>(self).concurrency());
  }
  template <class E> static void bulk_of(void* self, int n, task_function task, void* arg) {
    object<E>(self).bulk(n, task, arg);
  }
  template <class E> static int group_size_of(void* self) {
    return static_cast<int>(object<E>(self).group_size());
  }
  template <class E> static constexpr group_size_type group_size_function() noexcept {
    if constexpr (requires(E& e) {
                    { e.group_size() } -> std::convertible_to<int>;
                  }) {
      return &group_size_of<E>;
    } else {
      return nullptr;
    }
  }

  fftn_executor c_;
};

} // namespace fftn::inline v0

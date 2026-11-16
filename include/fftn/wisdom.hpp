// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Wisdom: plans' programs recorded as text, so that a measured choice can be kept and
/// rebuilt exactly without measuring again.
#pragma once

#include "fftn/config.hpp"
#include "fftn/plan.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fftn::inline v0 {

namespace detail {
struct wisdom_access;
} // namespace detail

/// A set of recorded programs, one entry per plan: the shape and the options that change the
/// program, the library version, the processor model and instruction set it was recorded
/// on, and the program. It is plain data owned by the caller: FFTN keeps no wisdom of its
/// own and no global state. Plans read it through `options::use`.
///
/// The text form is one line per entry, tab-separated, readable but not meant to be edited;
/// lines starting with `#` and blank lines are ignored. An entry names the build that
/// recorded it (the library version and an identifier of its sources), the processor model
/// and the instruction set; entries of other builds or processors are kept as they are,
/// exported again and never used here.
///
/// ```cpp
/// fftn::wisdom w;
/// const auto shape = fftn::rows_c2c<double>(100000, 1, fftn::direction::forward);
/// const fftn::c2c_plan<double> measured(shape, {.effort = fftn::planning::measure});
/// w.add(measured);
/// w.save("plans.wisdom");
/// // Later, in another run on the same machine:
/// const fftn::wisdom saved = fftn::wisdom::load("plans.wisdom");
/// const fftn::c2c_plan<double> p(shape, {.use = &saved}); // the measured program, untimed
/// ```
///
/// A wisdom object is a value: copy it, move it, use one from several threads while no
/// thread modifies it.
class wisdom {
public:
  /// No entries.
  wisdom() = default;

  /// The entries in `text`, as `text()` writes them.
  ///
  /// Throws `std::invalid_argument` naming the first line that is not blank, not a comment
  /// and not an entry: a line must start with a wisdom format tag (`fftn-wisdom-`), and an
  /// entry of this format recorded by this build must be complete and well formed. Lines of
  /// other formats or builds are kept unchanged.
  explicit wisdom(std::string_view text);

  /// The entries of the file at `path` (its whole content read as by the text constructor).
  ///
  /// Throws `std::runtime_error` when the file cannot be read, `std::invalid_argument` when
  /// a line is not an entry (naming the file and the line).
  [[nodiscard]] static wisdom load(const std::filesystem::path& path);

  /// Writes `text()` to the file at `path`, replacing it.
  ///
  /// Throws `std::runtime_error` when the file cannot be written.
  void save(const std::filesystem::path& path) const;

  /// The entries as text: a comment line naming the format, then one line per entry in the
  /// order they were added. `wisdom(w.text())` holds the same entries as `w`.
  [[nodiscard]] std::string text() const;

  /// Records the program of `p`, however it was chosen (rules, measurement or wisdom),
  /// replacing an entry for the same shape, options, library version, processor model and
  /// instruction set. A plan built with `options::use = &w` from the result rebuilds the
  /// same program on this machine, with bitwise identical results.
  template <class In, class Out> void add(const plan<In, Out>& p) { add_entry(entry_of(*p.core_)); }

  /// The entries of `other` added one by one, as `add` does.
  void add(const wisdom& other);

  /// The number of entries.
  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
  static std::string entry_of(const detail::plan_core& core);
  void add_entry(std::string entry);

  std::vector<std::string> entries_;                   // in the order added
  std::unordered_map<std::string, std::size_t> index_; // where an entry applies -> position
  friend struct detail::wisdom_access;
};

} // namespace fftn::inline v0

// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Pieces of measured planning (core/measure.cpp) that the tests check on their own.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace fftn::inline v0::detail {

// The batch counts (outermost first) of a reduced timing shape: when `scaled` bytes (the
// data, in proportion to the number of transforms) plus `fixed` bytes (workspace and tables)
// exceed `limit`, the counts shrink, the outer dimensions first, to the most transforms
// that fit, at least one transform. Counts are returned unchanged when everything fits.
[[nodiscard]] std::vector<std::size_t> reduced_counts(std::span<const std::size_t> counts,
                                                      std::size_t scaled, std::size_t fixed,
                                                      std::size_t limit);

// The byte offsets, multiples of 64 within a 4 KiB page, at which round `round` places the
// input and the output of its timings: they change from round to round, so that no program
// is timed at one lucky or unlucky placement (fbench offsets its buffers in the same way;
// P2's first pass found page-aligned scratch favouring programs fbench timed slower).
struct placement {
  std::size_t in = 0;
  std::size_t out = 0;
};
[[nodiscard]] placement placement_of(std::size_t round) noexcept;

} // namespace fftn::inline v0::detail

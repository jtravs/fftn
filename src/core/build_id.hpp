// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The build identifier: 16 hexadecimal digits hashing the library's sources and the
// configuration that changes its code (cmake/FftnBuildId.cmake writes the definition at build
// time). Wisdom records carry it with the version, so that an entry is used only by the
// build that recorded it: the version stays the same across development commits, and a
// program description need not rebuild the same kernels after one.
#pragma once

namespace fftn::inline v0::detail {

[[nodiscard]] const char* build_id() noexcept;

} // namespace fftn::inline v0::detail

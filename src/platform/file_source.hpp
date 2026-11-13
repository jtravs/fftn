// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Read-only access to a tree of small text files by path relative to a root
// ("sys/devices/system/cpu/online"): what the Linux topology reader reads through. The live
// system is a directory (the root "/"); a recorded machine is a sysfs snapshot, one text
// file holding every file the reader needs.
//
// Sysfs snapshot format (written by tools/capture_platform.py):
//   - The first line is exactly "# fftn-sysfs-snapshot 1".
//   - Every further non-empty line is "<path>\t<content>". The path is relative, uses "/",
//     and has no empty, "." or ".." component, no tab and no backslash. The content is the
//     file's text without its final newline, with three escapes: "\\" for a backslash,
//     "\t" for a tab and "\n" for a newline. Any other backslash sequence is an error.
//   - A path appears once. Directories are implied by the paths. A carriage return ending
//     a line is ignored (a checkout with CRLF line ends); other control characters in
//     content are refused by the writer.
#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::inline v0::detail {

class file_source {
public:
  file_source() = default;
  file_source(const file_source&) = default;
  file_source(file_source&&) = default;
  file_source& operator=(const file_source&) = default;
  file_source& operator=(file_source&&) = default;
  virtual ~file_source() = default;

  // The whole content of a regular file; nullopt if it is absent or unreadable.
  [[nodiscard]] virtual std::optional<std::string> read(std::string_view path) const = 0;
  // The names of the entries of a directory ("" is the root), ascending; empty if absent.
  [[nodiscard]] virtual std::vector<std::string> list(std::string_view directory) const = 0;
};

// Files under a directory of the running system.
class directory_files final : public file_source {
public:
  explicit directory_files(std::filesystem::path root);
  [[nodiscard]] std::optional<std::string> read(std::string_view path) const override;
  [[nodiscard]] std::vector<std::string> list(std::string_view directory) const override;

private:
  std::filesystem::path root_;
};

inline constexpr std::string_view sysfs_snapshot_header = "# fftn-sysfs-snapshot 1";

// Whether `text` starts with the sysfs snapshot header line (any version).
[[nodiscard]] bool is_sysfs_snapshot(std::string_view text) noexcept;

// The files of a sysfs snapshot, held in memory.
class snapshot_files final : public file_source {
public:
  // Throws std::runtime_error naming the line for anything the format above does not allow.
  explicit snapshot_files(std::string_view text);
  [[nodiscard]] std::optional<std::string> read(std::string_view path) const override;
  [[nodiscard]] std::vector<std::string> list(std::string_view directory) const override;

private:
  std::map<std::string, std::string, std::less<>> files_;
};

} // namespace fftn::inline v0::detail

// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/file_source.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

directory_files::directory_files(std::filesystem::path root) : root_(std::move(root)) {}

std::optional<std::string> directory_files::read(std::string_view path) const {
  const std::filesystem::path full = root_ / path;
  std::error_code ec;
  if (!std::filesystem::is_regular_file(full, ec)) {
    return std::nullopt;
  }
  const std::ifstream file(full);
  std::ostringstream text;
  text << file.rdbuf();
  if (!file) {
    return std::nullopt;
  }
  return std::move(text).str();
}

std::vector<std::string> directory_files::list(std::string_view directory) const {
  std::vector<std::string> names;
  std::error_code ec;
  for (std::filesystem::directory_iterator it(root_ / directory, ec), end; !ec && it != end;
       it.increment(ec)) {
    names.push_back(it->path().filename().string());
  }
  std::ranges::sort(names);
  return names;
}

bool is_sysfs_snapshot(std::string_view text) noexcept {
  constexpr std::string_view stem = "# fftn-sysfs-snapshot ";
  return text.starts_with(stem);
}

namespace {

[[noreturn]] void malformed(std::size_t line, std::string_view why) {
  throw std::runtime_error("fftn: sysfs snapshot line " + std::to_string(line) + ": " +
                           std::string(why));
}

bool valid_path(std::string_view path) {
  if (path.empty() || path.find_first_of("\t\\") != std::string_view::npos) {
    return false;
  }
  while (true) {
    const auto slash = path.find('/');
    const std::string_view part = path.substr(0, slash);
    if (part.empty() || part == "." || part == "..") {
      return false;
    }
    if (slash == std::string_view::npos) {
      return true;
    }
    path.remove_prefix(slash + 1);
  }
}

std::string unescape(std::string_view text, std::size_t line) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '\\') {
      out += text[i];
      continue;
    }
    if (++i == text.size()) {
      malformed(line, "a backslash ends the line");
    }
    switch (text[i]) {
    case '\\':
      out += '\\';
      break;
    case 't':
      out += '\t';
      break;
    case 'n':
      out += '\n';
      break;
    default:
      malformed(line, "unknown escape");
    }
  }
  return out;
}

} // namespace

snapshot_files::snapshot_files(std::string_view text) {
  std::size_t number = 0;
  while (!text.empty()) {
    const auto newline = text.find('\n');
    std::string_view line = text.substr(0, newline);
    text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
    ++number;
    if (line.ends_with('\r')) {
      line.remove_suffix(1);
    }
    if (number == 1) {
      if (line != sysfs_snapshot_header) {
        malformed(number, "expected the header \"# fftn-sysfs-snapshot 1\"");
      }
      continue;
    }
    if (line.empty()) {
      continue;
    }
    const auto tab = line.find('\t');
    if (tab == std::string_view::npos) {
      malformed(number, "no tab between path and content");
    }
    const std::string_view path = line.substr(0, tab);
    if (!valid_path(path)) {
      malformed(number, "invalid path");
    }
    if (!files_.emplace(std::string(path), unescape(line.substr(tab + 1), number)).second) {
      malformed(number, "the path appeared before");
    }
  }
  if (number == 0) {
    malformed(1, "empty snapshot");
  }
}

std::optional<std::string> snapshot_files::read(std::string_view path) const {
  const auto it = files_.find(path);
  if (it == files_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<std::string> snapshot_files::list(std::string_view directory) const {
  const std::string prefix = directory.empty() ? std::string() : std::string(directory) + "/";
  std::vector<std::string> names;
  for (auto it = files_.lower_bound(prefix); it != files_.end() && it->first.starts_with(prefix);
       ++it) {
    const std::string_view rest = std::string_view(it->first).substr(prefix.size());
    std::string name(rest.substr(0, rest.find('/')));
    if (names.empty() || names.back() != name) {
      names.push_back(std::move(name));
    }
  }
  // Keys sort "a/b" before "a.x/c" but "a" is listed when first met, so sort the names.
  std::ranges::sort(names);
  const auto repeated = std::ranges::unique(names);
  names.erase(repeated.begin(), repeated.end());
  return names;
}

} // namespace fftn::inline v0::detail

// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/file_source.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fftn::detail::directory_files;
using fftn::detail::is_sysfs_snapshot;
using fftn::detail::snapshot_files;
using names = std::vector<std::string>;

TEST(SnapshotFiles, ReadsPathsAndContent) {
  const snapshot_files files("# fftn-sysfs-snapshot 1\r\n"
                             "sys/devices/system/cpu/online\t0-71\r\n"
                             "\n"
                             "sys/devices/system/cpu/cpu0/uevent\t\n"
                             "proc/self/status\tCpus_allowed_list:\\t0-1\\nA\\\\B\n");
  EXPECT_EQ(files.read("sys/devices/system/cpu/online"), "0-71");
  EXPECT_EQ(files.read("sys/devices/system/cpu/cpu0/uevent"), "");
  EXPECT_EQ(files.read("proc/self/status"), "Cpus_allowed_list:\t0-1\nA\\B");
  EXPECT_EQ(files.read("sys/devices/system/cpu"), std::nullopt); // a directory, not a file
  EXPECT_EQ(files.read("missing"), std::nullopt);
}

TEST(SnapshotFiles, ListsImpliedDirectories) {
  const snapshot_files files("# fftn-sysfs-snapshot 1\n"
                             "a/cpu1/x\t1\n"
                             "a/cpu0/x\t0\n"
                             "a/cpu0/y\t0\n"
                             "a/cpu.max\tmax\n"
                             "a/online\t0-1\n"
                             "b\t2\n");
  EXPECT_EQ(files.list(""), (names{"a", "b"}));
  EXPECT_EQ(files.list("a"), (names{"cpu.max", "cpu0", "cpu1", "online"}));
  EXPECT_EQ(files.list("a/cpu0"), (names{"x", "y"}));
  EXPECT_TRUE(files.list("c").empty());
  EXPECT_TRUE(files.list("a/cpu").empty()); // a prefix of a name is not a directory
}

TEST(SnapshotFiles, RefusesMalformedText) {
  const std::string header = "# fftn-sysfs-snapshot 1\n";
  for (const std::string& bad : {
           std::string(),
           std::string("a\t1\n"),                    // no header
           std::string("# fftn-sysfs-snapshot 2\n"), // unknown version
           header + "no-tab\n",
           header + "/abs\t1\n",
           header + "a//b\t1\n",
           header + "a/../b\t1\n",
           header + "./a\t1\n",
           header + "a/\t1\n",
           header + "a\\b\t1\n",
           header + "a\tbad \\x escape\n",
           header + "a\tends in \\\n",
           header + "a\t1\na\t2\n", // a path twice
       }) {
    EXPECT_THROW(snapshot_files{bad}, std::runtime_error) << bad;
  }
}

TEST(SnapshotFiles, HeaderIdentifiesTheFormat) {
  EXPECT_TRUE(is_sysfs_snapshot("# fftn-sysfs-snapshot 1\n"));
  EXPECT_TRUE(is_sysfs_snapshot("# fftn-sysfs-snapshot 7\n")); // refused later, by version
  EXPECT_FALSE(is_sysfs_snapshot("hw.ncpu: 18\n"));
  EXPECT_FALSE(is_sysfs_snapshot(""));
}

TEST(DirectoryFiles, ReadsAndListsARealTree) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "fftn-platform-test-directory-files";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "sys/cpu1");
  std::filesystem::create_directories(root / "sys/cpu0");
  std::ofstream(root / "sys/online") << "0-1\n";
  {
    const std::ofstream empty(root / "sys/cpu0/empty");
  }
  const directory_files files(root);
  EXPECT_EQ(files.read("sys/online"), "0-1\n");
  EXPECT_EQ(files.read("sys/cpu0/empty"), "");
  EXPECT_EQ(files.read("sys/cpu0"), std::nullopt); // a directory
  EXPECT_EQ(files.read("sys/none"), std::nullopt);
  EXPECT_EQ(files.list("sys"), (names{"cpu0", "cpu1", "online"}));
  EXPECT_TRUE(files.list("none").empty());
  std::filesystem::remove_all(root);
}

} // namespace

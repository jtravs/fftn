#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Installs one LLVM release series on an Ubuntu runner from apt.llvm.org: the compiler,
# its sanitizer and profile runtimes, and the coverage tools beside it in
# /usr/lib/llvm-<version>/bin. The repository's signing key is pinned by its SHA-256; the
# packages are verified by apt against it.
#
#   .github/scripts/install-llvm.sh <major version>
set -euo pipefail
version=$1
key_sha256=8b2a587ffd672c4687e7581dad4b2f6c1bb2ad6b480cd9771ba2ff48e0b8c75d
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tmp=$(mktemp -d)
"$here/fetch.sh" https://apt.llvm.org/llvm-snapshot.gpg.key "$key_sha256" "$tmp/llvm.asc"
sudo install -m 0644 "$tmp/llvm.asc" /usr/share/keyrings/apt-llvm.asc
codename=$(sed -n 's/^VERSION_CODENAME=//p' /etc/os-release)
echo "deb [signed-by=/usr/share/keyrings/apt-llvm.asc] https://apt.llvm.org/$codename/" \
  "llvm-toolchain-$codename-$version main" | sudo tee /etc/apt/sources.list.d/apt-llvm.list
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  "clang-$version" "llvm-$version" "libclang-rt-$version-dev"
"/usr/lib/llvm-$version/bin/clang++" --version

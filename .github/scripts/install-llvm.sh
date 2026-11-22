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
bash "$here/fetch.sh" https://apt.llvm.org/llvm-snapshot.gpg.key "$key_sha256" "$tmp/llvm.asc"
sudo install -m 0644 "$tmp/llvm.asc" /usr/share/keyrings/apt-llvm.asc
codename=$(sed -n 's/^VERSION_CODENAME=//p' /etc/os-release)
echo "deb [signed-by=/usr/share/keyrings/apt-llvm.asc] https://apt.llvm.org/$codename/" \
  "llvm-toolchain-$codename-$version main" | sudo tee /etc/apt/sources.list.d/apt-llvm.list
# A mirror that stalls holds apt-get without end: on 2026-11-19 the required sanitizer job sat
# 90 minutes in `apt-get update` until the job's timeout cancelled it (run 44478097749). Each
# attempt is cut off after 5 minutes (10 for the install) and tried up to three times.
apt_retry() {
  local limit=$1 attempt
  shift
  for attempt in 1 2 3; do
    if sudo timeout "$limit" apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=60 \
      -o Acquire::https::Timeout=60 "$@"; then
      return 0
    fi
    echo "apt-get $1: attempt $attempt failed or timed out after $limit" >&2
  done
  return 1
}
apt_retry 5m update
apt_retry 10m install -y --no-install-recommends \
  "clang-$version" "llvm-$version" "libclang-rt-$version-dev"
"/usr/lib/llvm-$version/bin/clang++" --version

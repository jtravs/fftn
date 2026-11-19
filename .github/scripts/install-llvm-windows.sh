#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Installs a pinned LLVM release on a Windows runner, from the project's release archive
# (URL and SHA-256 below; the hashes are the ones GitHub lists for the release assets), in
# C:/fftn-llvm, where the Windows matrices of ci.yml and extended.yml name the compiler. Its
# bin directory goes on PATH for the later steps: the tests load libomp.dll from there.
#
# The LLVM on the images is not used: each image moves it and Visual Studio separately, and
# the Microsoft standard library refuses to compile with a Clang older than it expects.
#
#   .github/scripts/install-llvm-windows.sh x64|arm64
set -euo pipefail
arch=$1
version=23.1.5
case $arch in
  x64)
    triple=x86_64-pc-windows-msvc
    sha256=b2be33385ddc86e83511417a060de927ae8b09d1f3017cc3cdce5c73cd5029d7
    ;;
  arm64)
    triple=aarch64-pc-windows-msvc
    sha256=cb523582d9fe00862c5584ce5f7715f963037c11bba95181fa98425f48e06863
    ;;
  *)
    echo "install-llvm-windows: unknown architecture $arch" >&2
    exit 2
    ;;
esac
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tmp=$(mktemp -d)
bash "$here/fetch.sh" \
  "https://github.com/llvm/llvm-project/releases/download/llvmorg-$version/clang+llvm-$version-$triple.tar.zst" \
  "$sha256" "$tmp/llvm.tar.zst"
mkdir -p /c/fftn-llvm
tar --use-compress-program=zstd -xf "$tmp/llvm.tar.zst" -C /c/fftn-llvm --strip-components=1
rm -rf "$tmp"
cygpath -w /c/fftn-llvm/bin >> "$GITHUB_PATH"
/c/fftn-llvm/bin/clang --version

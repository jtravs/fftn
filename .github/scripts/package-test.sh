#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The packaging check (.github/workflows/package.yml; DESIGN.md 14.3, docs/building.md): installs
# FFTN from a build that has the tests and their dependencies (Kokkos, GoogleTest), checks that
# the install tree holds only FFTN's own files, then builds and runs the separate project in
# examples/package three ways: find_package against the install tree, pkg-config, FetchContent.
# The install has the C API's shared library too (FFTN_BUILD_C_SHARED), and each way builds a C
# program on the C API.
#
#   .github/scripts/package-test.sh [--omp] [--no-pkg-config]
#
#   --omp            also build the OpenMP adapter's example through fftn::omp
#   --no-pkg-config  leave out the pkg-config build (Windows)
# The compiler comes from CXX (and CC); FFTN_CHECK_JOBS limits parallel compiles.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
# CMake on Windows takes D:/... paths, not Git Bash's /d/...
if pwd -W > /dev/null 2>&1; then root=$(pwd -W); fi
omp=OFF
pc=1
for arg in "$@"; do
  case $arg in
    --omp) omp=ON ;;
    --no-pkg-config) pc=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
jobs=${FFTN_CHECK_JOBS:-4}
prefix=$root/build/package-install
say() { printf '\n== %s\n' "$*"; }
run_tests() { ctest --test-dir "$1" --output-on-failure --no-tests=error; }

say "install from a build with the tests (their dependencies must stay out of the tree)"
rm -rf "$prefix"
cmake -S . -B build/package -G Ninja -DCMAKE_BUILD_TYPE=Release -DFFTN_BUILD_TESTS=ON \
  -DFFTN_BUILD_BENCH=OFF -DFFTN_TEST_WITH_FFTW=OFF -DFFTN_BUILD_C_SHARED=ON
cmake --build build/package --target fftn fftn_c -j "$jobs"
cmake --install build/package --prefix "$prefix"

say "installed files"
(cd "$prefix" && find . -type f | sed 's|^\./||' | sort) > build/package-files.txt
cat build/package-files.txt
allowed='^(include/fftn/([a-z0-9_]+/)*[a-z0-9_]+\.(h|hpp)|lib(64)?/(libfftn\.a|fftn\.lib)'
# The shared library's file carries its full version (libfftn_c.so.0.1.0, libfftn_c.0.1.0.dylib);
# the soname and linker names beside it are links, which find -type f does not list.
allowed+='|lib(64)?/(libfftn_c\.so(\.[0-9]+)+|libfftn_c(\.[0-9]+)+\.dylib|fftn_c\.lib)'
allowed+='|bin/fftn_c\.dll'
allowed+='|lib(64)?/cmake/fftn/fftn(Config|ConfigVersion|Targets|Targets-[a-z]+)\.cmake'
allowed+='|lib(64)?/pkgconfig/fftn\.pc|share/doc/fftn/LICENSE)$'
unexpected=$(grep -vE "$allowed" build/package-files.txt || true)
if [[ -n $unexpected ]]; then
  printf 'files that are not FFTN'"'"'s own were installed:\n%s\n' "$unexpected"
  exit 1
fi

say "find_package(fftn) against the install tree"
cmake -S examples/package -B build/package-cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$prefix" -DFFTN_PACKAGE_OMP=$omp
cmake --build build/package-cmake -j "$jobs"
run_tests build/package-cmake

if ((pc)); then
  say "pkg-config"
  pc_file=$(find "$prefix" -name fftn.pc)
  flags=$(PKG_CONFIG_PATH=$(dirname "$pc_file") pkg-config --cflags --libs fftn)
  echo "pkg-config --cflags --libs fftn: $flags"
  # shellcheck disable=SC2086 # the flags are words
  "$CXX" -std=c++20 -O2 examples/rows_c2c.cpp $flags -o build/package-rows_c2c
  build/package-rows_c2c
  # The C program: compiled by the C compiler, linked by the C++ one (the C++ standard library).
  # shellcheck disable=SC2086
  "$CC" -std=c99 -O2 -c examples/c_api.c $flags -o build/package-c_api.o
  # shellcheck disable=SC2086
  "$CXX" build/package-c_api.o $flags -o build/package-c_api
  build/package-c_api
fi

say "FetchContent"
cmake -S examples/package -B build/package-fetch -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFFTN_PACKAGE_FROM=fetch -DFETCHCONTENT_SOURCE_DIR_FFTN="$root" -DFFTN_PACKAGE_OMP=$omp
cmake --build build/package-fetch -j "$jobs"
run_tests build/package-fetch

say "packaging check passed"

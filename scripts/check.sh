#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# FFTN's quality gate. A change is finished when this is green (AGENTS.md, "The gate").
#
#   scripts/check.sh                              # every stage that applies
#   FFTN_CHECK_STAGES="format build" scripts/check.sh
#
# Stages, in order:
#   spdx      every source file carries the SPDX header, and shell scripts are
#             executable in git                                    (tools/repo_lint.py)
#   docs      Markdown links resolve, PLAN.md keeps its sections   (tools/repo_lint.py)
#   knobs     experimental switches are registered and not expired (tools/repo_lint.py)
#   format    clang-format (pinned), C++ sources and headers
#   python    ruff (pinned): lint and format of tools/, gen/, bench/
#   shell     bash -n on scripts (shellcheck too when it is installed)
#   gen       generated codelets are reproducible: regenerate and compare with the tree;
#             the generator's own tests (gen/tests)
#   apidoc    every public declaration is documented (Doxygen as a linter; no output kept)
#             and the code snippets in docs/ match the compiled files in examples/
#   build     Release, warnings as errors, tests under every kernel variant the CPU has
#   tidy      clang-tidy (pinned), warnings as errors
#   sanitize  ASan + UBSan (leak detection on), tests under every kernel variant
#   tsan      ThreadSanitizer over the threading tests
#   coverage  line and branch coverage of hand-written code against the threshold
#   bench     fbench smoke: FFTN agrees with the reference; no timing claims
#
# A stage whose inputs do not exist yet reports "nothing to check" and passes; the code
# stages switch on when CMakeLists.txt and the named CMake presets appear (dev/PLAN.md,
# phase 0). Never pipe this script into tail or head inside an && chain: that hides its
# exit status.
#
# The test suites themselves force every kernel variant and instruction set the CPU has
# (AGENTS.md, section 5), so no stage needs a variant list.
#
#   FFTN_CHECK_JOBS            parallel build jobs (default: all CPUs)
#   FFTN_COVERAGE_MIN_LINES    override the coverage threshold below (to raise it only)
# The compiler and any offline dependency directory are taken by the CMake presets from
# the usual CMake and environment variables (CXX, FETCHCONTENT_SOURCE_DIR_<NAME>).
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
all="spdx docs knobs format python shell gen apidoc build tidy sanitize tsan coverage bench"
stages=" ${FFTN_CHECK_STAGES:-$all} "
stage() { [[ $stages == *" $1 "* ]]; }
say() { printf '\n== %s\n' "$*"; }
skip() { printf '   nothing to check yet (%s)\n' "$*"; }

CLANG_FORMAT_VERSION=23.1.0
CLANG_TIDY_VERSION=23.1.4  # the PyPI wheels have no 23.1.0
RUFF_VERSION=0.17.4
# Line coverage of hand-written code, in percent. Raise it as coverage rises; never lower it.
COVERAGE_MIN_LINES=${FFTN_COVERAGE_MIN_LINES:-98}
jobs=${FFTN_CHECK_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

# Python tools run in the project's uv environment (pyproject.toml, uv.lock).
py() { uv run --frozen --quiet python "$@"; }
files() { git ls-files --cached --others --exclude-standard -- "$@"; }
have_code() { [[ -f CMakeLists.txt ]]; }
have_preset() { have_code && cmake --list-presets 2>/dev/null | grep -q "\"$1\""; }

# Configure one CMake preset and report the compiler; logs go to build/<preset>.log.
configure() {
  local name=$1
  mkdir -p build
  if ! cmake --preset "$name" > "build/$name.log" 2>&1; then
    tail -40 "build/$name.log"
    echo "configure of preset $name failed: build/$name.log"
    exit 1
  fi
  printf '   compiler: %s (%s)\n' "$(cache_value "$name" FFTN_CXX_COMPILER)" \
    "$(cache_value "$name" CMAKE_CXX_COMPILER)"
}
cache_value() {
  sed -n "s/^$2:[A-Z]*=//p" "build/$1/CMakeCache.txt" | head -1
}
# Build and test a configured preset; the error lines are shown on failure.
build_and_test() {
  local name=$1
  if ! cmake --build --preset "$name" -j "$jobs" >> "build/$name.log" 2>&1; then
    grep -E "error|warning:" -A4 "build/$name.log" | head -80 || true
    echo "build of preset $name failed: build/$name.log"
    exit 1
  fi
  ${test_prefix[@]+"${test_prefix[@]}"} ctest --preset "$name" --output-on-failure
}
test_prefix=()
preset() { configure "$1" && build_and_test "$1"; }

if stage spdx; then say "SPDX headers"; py tools/repo_lint.py spdx; fi
if stage docs; then say "documents"; py tools/repo_lint.py docs; fi
if stage knobs; then say "experimental switches"; py tools/repo_lint.py knobs; fi

if stage format; then
  say "clang-format $CLANG_FORMAT_VERSION"
  cpp=$(files '*.cpp' '*.hpp' '*.h' '*.inl' | grep -v '^src/gen/' || true)
  if [[ -n $cpp ]]; then
    # shellcheck disable=SC2086
    uvx "clang-format==$CLANG_FORMAT_VERSION" --dry-run --Werror $cpp
  else skip "no C++ sources"; fi
fi

if stage python; then
  say "ruff $RUFF_VERSION"
  py=$(files '*.py')
  if [[ -n $py ]]; then
    # shellcheck disable=SC2086
    uvx "ruff==$RUFF_VERSION" check $py
    # shellcheck disable=SC2086
    uvx "ruff==$RUFF_VERSION" format --check $py
  else skip "no Python sources"; fi
fi

if stage shell; then
  say "shell scripts"
  for f in $(files '*.sh' '*.sbatch'); do
    bash -n "$f"
    if command -v shellcheck > /dev/null; then shellcheck "$f"; fi
  done
fi

if stage gen; then
  say "generated codelets are reproducible; generator tests"
  if [[ -f gen/fftn_gen/__main__.py ]]; then
    # Builds and verifies every codelet in memory and compares the text with src/gen/ byte
    # for byte; it writes nothing.
    py -m gen.fftn_gen --check src/gen
    py -m unittest discover -s gen/tests -t gen
  else skip "no generator"; fi
fi

if stage apidoc; then
  say "public API documentation and example snippets"
  if [[ -f docs/Doxyfile ]]; then
    mkdir -p build
    doxygen docs/Doxyfile > build/apidoc.log 2>&1 || { cat build/apidoc.log; exit 1; }
    py tools/repo_lint.py snippets
  else skip "no docs/Doxyfile"; fi
fi

if stage build; then
  say "build (Release, warnings as errors) and tests"
  if have_preset check-release; then preset check-release; else skip "no check-release preset"; fi
fi

if stage tidy; then
  say "clang-tidy $CLANG_TIDY_VERSION"
  if have_preset check-tidy; then
    FFTN_CLANG_TIDY="uvx;clang-tidy==$CLANG_TIDY_VERSION" preset check-tidy
  else skip "no check-tidy preset"; fi
fi

if stage sanitize; then
  say "ASan + UBSan"
  if have_preset check-sanitize; then
    configure check-sanitize
    # Leak detection is on wherever the runtime has it; Apple Clang's does not.
    leaks=1
    if [[ $(cache_value check-sanitize FFTN_CXX_COMPILER) == AppleClang* ]]; then
      leaks=0
      echo "   leak detection: not available with Apple Clang (use Homebrew LLVM for it)"
    fi
    # GCC's runtime detects use after return by default, with a fake stack aligned to 32
    # bytes; AVX-512 code spills zmm registers with aligned 64-byte stores and faults there
    # (Xeon Gold 6240, GCC 14). Clang's runtime leaves it off by default; so does the gate.
    asan="detect_leaks=$leaks:detect_stack_use_after_return=0"
    ASAN_OPTIONS=${ASAN_OPTIONS:-$asan} \
      UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1} \
      build_and_test check-sanitize
  else skip "no check-sanitize preset"; fi
fi

if stage tsan; then
  say "ThreadSanitizer"
  if have_preset check-tsan; then
    # Older TSan runtimes (GCC 14's among them) abort on the high-entropy address
    # randomisation of recent Linux kernels ("unexpected memory mapping"); tests run with
    # randomisation off there, which changes nothing else about them.
    if [[ $(uname -s) == Linux ]] && command -v setarch > /dev/null; then
      test_prefix=(setarch "$(uname -m)" -R)
    fi
    TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1} preset check-tsan
    test_prefix=()
  else skip "no check-tsan preset"; fi
fi

if stage coverage; then
  say "coverage (hand-written code, at least $COVERAGE_MIN_LINES % of lines)"
  if have_preset check-coverage; then
    # Profiles from earlier runs would count lines this run did not execute.
    rm -rf build/check-coverage/profiles
    if [[ -d build/check-coverage ]]; then find build/check-coverage -name '*.gcda' -delete; fi
    preset check-coverage
    py tools/coverage_gate.py build/check-coverage --min-lines "$COVERAGE_MIN_LINES" \
      --dump build/check-coverage/tests/platform/fftn_platform_dump
  else skip "no check-coverage preset"; fi
fi

if stage bench; then
  say "fbench smoke (correctness only)"
  if [[ -f bench/fbench.py ]] && have_preset check-release; then
    py bench/fbench.py smoke --build build/check-release --strict
  else skip "no fbench"; fi
fi

say "all requested stages passed:${stages}"

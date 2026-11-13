#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs a build's test programs on an emulated processor (DESIGN.md 15.4): Intel SDE for
# x86-64 models, QEMU user mode for AArch64 ones.
#
#   .github/scripts/emulate.sh <build dir> <expected lines> <emulator command...>
#
# First fftn_platform_dump runs on the emulated processor, and every line of <expected
# lines> (separated by ';', for example "avx2 0;best_target scalar") must appear in its
# output exactly: that checks feature detection and the target a plan would choose. Then
# the instruction-set test programs run (EMULATED_TESTS below), forcing in turn each target
# that processor can execute. The long oracle suites run natively in ci.yml; under emulation
# they cost an hour per model and check arithmetic, not instruction sets. A test program
# with kernels for a target (phase 1 on) belongs in EMULATED_TESTS. The
# emulators reject any instruction the emulated processor lacks (SDE's chip check, QEMU's
# undefined-instruction trap), so a newer instruction reaching code that runs there fails
# the run, wherever it sits.
set -euo pipefail
build=$1 expected=$2
shift 2
emulator=("$@")
failed=()

run() {
  echo "::group::$*"
  local status=0
  "${emulator[@]}" "$@" || status=$?
  echo "::endgroup::"
  if ((status != 0)); then
    echo "emulate: exit status $status from $*"
    failed+=("$(basename "$1")")
  fi
}

echo "emulator: ${emulator[*]}"
if ! dump=$("${emulator[@]}" "$build/tests/platform/fftn_platform_dump"); then
  echo "emulate: fftn_platform_dump failed under ${emulator[*]}"
  exit 1
fi
sed -n '1,/^best_target /p' <<< "$dump"
IFS=';' read -ra want <<< "$expected"
for line in "${want[@]}"; do
  if ! grep -qxF "$line" <<< "$dump"; then
    echo "emulate: fftn_platform_dump does not print '$line'"
    failed+=("detection")
  fi
done

shopt -s nullglob
EMULATED_TESTS=(isa/fftn_isa_tests platform/fftn_platform_tests core/fftn_core_tests
  support/fftn_support_tests kokkos/fftn_kokkos_tests)
tests=()
for t in "${EMULATED_TESTS[@]}"; do
  if [[ -x $build/tests/$t ]]; then tests+=("$build/tests/$t"); fi
done
if ((${#tests[@]} == 0)); then
  echo "emulate: no test programs under $build/tests"
  exit 1
fi
for t in "${tests[@]}"; do run "$t" --gtest_brief=1; done
run "$build/tests/api/fftn_headers_alone"
if [[ -x $build/bench/fbench-fftn ]]; then run "$build/bench/fbench-fftn" selftest; fi

if ((${#failed[@]} > 0)); then
  echo "emulate: FAILED under ${emulator[*]}: ${failed[*]}"
  exit 1
fi
echo "emulate: all passed under ${emulator[*]}"

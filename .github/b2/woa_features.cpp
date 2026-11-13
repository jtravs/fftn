// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Experiment B2 probe (dev/experiments/B2-isa-regions-windows-clang.md): which interfaces
// tell a Windows on Arm program that the processor has FCMA and pointer authentication.
// Built and run by the CI's windows-11-arm job; not part of the library or its tests.
//
// Prints, for comparison with the ground truth at the end:
//   - every IsProcessorFeaturePresent index from 0 to 127 that reports 1;
//   - the "CP xxxx" values of HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0, which
//     hold copies of the AArch64 ID registers (the name encodes op0, op1, CRn, CRm, op2),
//     with the FCMA and pointer-authentication fields of ID_AA64ISAR1_EL1 (CP 4031) and
//     ID_AA64ISAR2_EL1 (CP 4032) decoded;
//   - whether reading ID_AA64ISAR1_EL1 directly (MRS) works in user mode;
//   - ground truth: whether an FCMLA and an XPACI instruction execute (both are undefined
//     without FCMA and pointer authentication respectively).

#if !defined(_WIN32) || !defined(_M_ARM64)
#error "woa_features.cpp is for Windows on Arm only"
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <intrin.h>

#include <cstdint>
#include <cstdio>

namespace {

unsigned field(std::uint64_t reg, unsigned lsb) {
  return static_cast<unsigned>((reg >> lsb) & 0xFU);
}

bool read_cp(HKEY key, const wchar_t* name, std::uint64_t& value) {
  DWORD size = sizeof(value);
  DWORD type = 0;
  const LSTATUS status =
      RegQueryValueExW(key, name, nullptr, &type, reinterpret_cast<BYTE*>(&value), &size);
  return status == ERROR_SUCCESS && type == REG_QWORD;
}

void registry() {
  HKEY key = nullptr;
  if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                    KEY_READ, &key) != ERROR_SUCCESS) {
    std::puts("registry: CentralProcessor\\0 cannot be opened");
    return;
  }
  for (DWORD i = 0;; ++i) {
    wchar_t name[256];
    DWORD name_size = 256;
    DWORD type = 0;
    std::uint64_t value = 0;
    DWORD size = sizeof(value);
    const LSTATUS status = RegEnumValueW(key, i, name, &name_size, nullptr, &type,
                                         reinterpret_cast<BYTE*>(&value), &size);
    if (status == ERROR_NO_MORE_ITEMS) {
      break;
    }
    if (status == ERROR_SUCCESS && type == REG_QWORD && name[0] == L'C' && name[1] == L'P') {
      std::printf("registry: %ls = 0x%016llx\n", name, static_cast<unsigned long long>(value));
    }
  }
  std::uint64_t isar1 = 0;
  if (read_cp(key, L"CP 4031", isar1)) {
    std::printf("registry ID_AA64ISAR1_EL1: FCMA=%u APA=%u API=%u GPA=%u GPI=%u JSCVT=%u\n",
                field(isar1, 16), field(isar1, 4), field(isar1, 8), field(isar1, 24),
                field(isar1, 28), field(isar1, 12));
  } else {
    std::puts("registry: no CP 4031 (ID_AA64ISAR1_EL1)");
  }
  std::uint64_t isar2 = 0;
  if (read_cp(key, L"CP 4032", isar2)) {
    std::printf("registry ID_AA64ISAR2_EL1: APA3=%u GPA3=%u\n", field(isar2, 12), field(isar2, 8));
  } else {
    std::puts("registry: no CP 4032 (ID_AA64ISAR2_EL1)");
  }
  RegCloseKey(key);
}

// Each probe runs one instruction under structured exception handling; an undefined
// instruction raises EXCEPTION_ILLEGAL_INSTRUCTION.
int filter(DWORD code) {
  return code == EXCEPTION_ILLEGAL_INSTRUCTION || code == EXCEPTION_PRIV_INSTRUCTION
             ? EXCEPTION_EXECUTE_HANDLER
             : EXCEPTION_CONTINUE_SEARCH;
}

void mrs_isar1() {
  __try {
    const std::uint64_t isar1 = _ReadStatusReg(ARM64_SYSREG(3, 0, 0, 6, 1));
    std::printf("mrs ID_AA64ISAR1_EL1: 0x%016llx FCMA=%u APA=%u API=%u\n",
                static_cast<unsigned long long>(isar1), field(isar1, 16), field(isar1, 4),
                field(isar1, 8));
  } __except (filter(GetExceptionCode())) {
    std::puts("mrs ID_AA64ISAR1_EL1: trapped");
  }
}

void execute_fcmla() {
  __try {
    // FCMLA V0.2D, V0.2D, V1.2D, #0
    __asm__ volatile(".inst 0x6ec1c400" ::: "v0");
    std::puts("truth: FCMLA executes");
  } __except (filter(GetExceptionCode())) {
    std::puts("truth: FCMLA is undefined");
  }
}

void execute_xpaci() {
  __try {
    // XPACI X0: outside the hint space, so undefined without pointer authentication.
    __asm__ volatile(".inst 0xdac143e0" ::: "x0");
    std::puts("truth: XPACI executes");
  } __except (filter(GetExceptionCode())) {
    std::puts("truth: XPACI is undefined");
  }
}

} // namespace

int main() {
  for (DWORD pf = 0; pf < 128; ++pf) {
    if (IsProcessorFeaturePresent(pf) != 0) {
      std::printf("IsProcessorFeaturePresent(%lu) = 1\n", static_cast<unsigned long>(pf));
    }
  }
  registry();
  mrs_isar1();
  execute_fcmla();
  execute_xpaci();
  return 0;
}

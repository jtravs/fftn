// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Experiment B2 probe (dev/experiments/B2-isa-regions-windows-clang.md): which interfaces
// tell a Windows on Arm program that the processor has FCMA and pointer authentication.
// Built and run by the CI's windows-11-arm job; not part of the library or its tests.
//
// Prints, one method per section, for comparison with the ground truth at the end:
//   1. every IsProcessorFeaturePresent index from 0 to 127 that reports 1 (the job prints
//      the SDK's PF_ARM_* names beside them);
//   2. the processor's identification strings and every "CP xxxx" value of
//      HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0: copies of the AArch64 ID
//      registers, the name being the MRS encoding (op0 & 1, op1, CRn, CRm, op2) in hex, so
//      CP 4030, 4031 and 4032 are ID_AA64ISAR0/1/2_EL1. FCMA (ISAR1 bits 19:16) and the
//      pointer-authentication fields (ISAR1 APA 7:4, API 11:8, GPA 27:24, GPI 31:28; ISAR2
//      APA3 15:12, GPA3 11:8) are decoded;
//   3. ground truth: whether MRS ID_AA64ISAR1_EL1, an FCMLA and an XPACI execute in user
//      mode. FCMLA is undefined without FCMA and XPACI (outside the hint space) without
//      pointer authentication. A vectored exception handler catches the fault and steps
//      over the instruction, so a missing feature cannot kill the probe;
//   4. whether structured exception handling (__try around a call to the instruction)
//      also catches the fault under clang-cl, the method a library could use without
//      installing a process-wide handler. If it misses, the unhandled-exception filter
//      steps over the instruction and the probe says so.
//
// The first run put the instructions inline in __try bodies and died of the first fault
// (CI run 43487560137), printing nothing because stdout was buffered. Clang's __try
// catches faults raised in the calls it makes, not in instructions of its own body (that
// would need asynchronous exceptions), so each instruction now lives in a function of its
// own, and output is unbuffered.

#if !defined(_WIN32) || !defined(_M_ARM64)
#error "woa_features.cpp is for Windows on Arm only"
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <cstdio>

namespace {

unsigned field(std::uint64_t reg, unsigned lsb) {
  return static_cast<unsigned>((reg >> lsb) & 0xFU);
}

// --- 1. IsProcessorFeaturePresent ------------------------------------------------------

void processor_features() {
  for (DWORD pf = 0; pf < 128; ++pf) {
    if (IsProcessorFeaturePresent(pf) != 0) {
      std::printf("IsProcessorFeaturePresent(%lu) = 1\n", static_cast<unsigned long>(pf));
    }
  }
}

// --- 2. The registry --------------------------------------------------------------------

void print_string(HKEY key, const wchar_t* name) {
  wchar_t text[256] = {};
  DWORD size = sizeof(text);
  if (RegGetValueW(key, nullptr, name, RRF_RT_REG_SZ, nullptr, text, &size) == ERROR_SUCCESS) {
    std::printf("registry: %ls = %ls\n", name, text);
  } else {
    std::printf("registry: no %ls\n", name);
  }
}

bool read_cp(HKEY key, const wchar_t* name, std::uint64_t& value) {
  DWORD size = sizeof(value);
  return RegGetValueW(key, nullptr, name, RRF_RT_REG_QWORD, nullptr, &value, &size) ==
         ERROR_SUCCESS;
}

void registry() {
  HKEY key = nullptr;
  if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                    KEY_READ, &key) != ERROR_SUCCESS) {
    std::puts("registry: CentralProcessor\\0 cannot be opened");
    return;
  }
  print_string(key, L"ProcessorNameString");
  print_string(key, L"Identifier");
  print_string(key, L"VendorIdentifier");
  unsigned cp_values = 0;
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
    // ERROR_MORE_DATA: a value larger than eight bytes (a string), not an ID register.
    if (status == ERROR_SUCCESS && type == REG_QWORD && name[0] == L'C' && name[1] == L'P') {
      std::printf("registry: %ls = 0x%016llx\n", name, static_cast<unsigned long long>(value));
      ++cp_values;
    }
  }
  std::printf("registry: %u CP values\n", cp_values);
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

// --- 3. Ground truth, under a vectored exception handler ---------------------------------

// Set by the handlers when they step over a faulting instruction.
volatile LONG faulted = 0;
volatile LONG stepping = 0;

bool is_instruction_fault(DWORD code) {
  return code == EXCEPTION_ILLEGAL_INSTRUCTION || code == EXCEPTION_PRIV_INSTRUCTION;
}

// Every AArch64 instruction is four bytes.
LONG step_over(EXCEPTION_POINTERS* e) {
  if (stepping != 0 && is_instruction_fault(e->ExceptionRecord->ExceptionCode)) {
    faulted = 1;
    e->ContextRecord->Pc += 4;
    return EXCEPTION_CONTINUE_EXECUTION;
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

LONG CALLBACK vectored(EXCEPTION_POINTERS* e) { return step_over(e); }

// The instructions, each alone in a function that is never inlined.
__declspec(noinline) std::uint64_t mrs_isar1() {
  std::uint64_t v = 0;
  __asm__ volatile("mrs %0, S3_0_C0_C6_1" : "=r"(v));
  return v;
}

__declspec(noinline) void fcmla() {
  // FCMLA V0.2D, V0.2D, V1.2D, #0
  __asm__ volatile(".inst 0x6ec1c400" ::: "v0");
}

__declspec(noinline) void xpaci() {
  // XPACI X0
  __asm__ volatile(".inst 0xdac143e0" ::: "x0");
}

void ground_truth() {
  void* handler = AddVectoredExceptionHandler(1, vectored);
  if (handler == nullptr) {
    std::puts("truth: AddVectoredExceptionHandler failed");
    return;
  }
  stepping = 1;

  faulted = 0;
  const std::uint64_t isar1 = mrs_isar1();
  if (faulted != 0) {
    std::puts("truth: MRS ID_AA64ISAR1_EL1 faults in user mode");
  } else {
    std::printf("truth: MRS ID_AA64ISAR1_EL1 = 0x%016llx FCMA=%u APA=%u API=%u\n",
                static_cast<unsigned long long>(isar1), field(isar1, 16), field(isar1, 4),
                field(isar1, 8));
  }

  faulted = 0;
  fcmla();
  std::puts(faulted != 0 ? "truth: FCMLA is undefined" : "truth: FCMLA executes");

  faulted = 0;
  xpaci();
  std::puts(faulted != 0 ? "truth: XPACI is undefined" : "truth: XPACI executes");

  stepping = 0;
  RemoveVectoredExceptionHandler(handler);
}

// --- 4. Structured exception handling ----------------------------------------------------

LONG WINAPI unhandled(EXCEPTION_POINTERS* e) { return step_over(e); }

int seh_filter(DWORD code) {
  return is_instruction_fault(code) ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH;
}

// 1: the instruction ran; 0: __except caught its fault.
int seh_try(void (*instruction)()) {
  __try {
    instruction();
    return 1;
  } __except (seh_filter(GetExceptionCode())) {
    return 0;
  }
}

void seh(const char* name, void (*instruction)()) {
  faulted = 0;
  const int ran = seh_try(instruction);
  if (faulted != 0) {
    std::printf("seh: %s faulted and __except missed it (unhandled filter stepped over)\n", name);
  } else {
    std::printf("seh: %s %s\n", name, ran != 0 ? "ran" : "faulted; __except caught it");
  }
}

void mrs_isar1_discard() { static_cast<void>(mrs_isar1()); }

void structured() {
  SetUnhandledExceptionFilter(unhandled);
  stepping = 1;
  seh("MRS ID_AA64ISAR1_EL1", mrs_isar1_discard);
  seh("FCMLA", fcmla);
  seh("XPACI", xpaci);
  stepping = 0;
}

} // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  processor_features();
  registry();
  ground_truth();
  structured();
  return 0;
}

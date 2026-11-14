# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""asm_check's rule for calls, on functions built by hand: which jumps are identical-code
aliases and which are calls that must have been inlined. (The fault objects of
tests/isa/asm_faults.cpp test the rest of the check on real compiler output.)"""

from __future__ import annotations

import unittest

from asm_check import Call, Function, call_errors

PREFIX = "_ZN4fftn2v06detail{n}{t}12_GLOBAL__N_1L"


def entry(target: str, index: int, name: str = "run_codelet") -> str:
    return f"{PREFIX.format(n=len(target), t=target)}{len(name)}{name}IdLm{index}EEEvPKT_mPS5_mS7_"


def function(name: str, insns: list[tuple[str, str]], calls: list[Call]) -> Function:
    return Function([name], "codelets_scalar.cpp.o", ".text", 0, insns, calls)


def jump_to(sym: str, mnemonic: str = "b") -> Call:
    return Call(mnemonic, sym, None)


class IdenticalCodeAlias(unittest.TestCase):
    def test_a_lone_jump_to_a_sibling_instantiation_is_an_alias(self) -> None:
        call = jump_to(entry("scalar", 0))
        f = function(entry("scalar", 1), [("b", "0x0")], [call])
        self.assertEqual(call_errors("scalar", f), [])

    def test_a_landing_pad_before_the_jump_is_allowed(self) -> None:
        call = jump_to(entry("avx2", 0), "jmpq")
        f = function(entry("avx2", 1), [("endbr64", ""), ("jmpq", "0x0")], [call])
        self.assertEqual(call_errors("avx2", f), [])

    def test_padding_after_the_jump_is_allowed(self) -> None:
        # GCC 14 and 15 on x86-64 under the sanitizers: the folded alias is a jump followed
        # by alignment padding that is never executed.
        call = jump_to(entry("avx2", 0), "jmpq")
        f = function(
            entry("avx2", 1), [("jmpq", "0x0"), ("data16", "nopw %cs:0x0(%rax,%rax,1)")], [call]
        )
        self.assertEqual(call_errors("avx2", f), [])

    def test_padding_does_not_excuse_code_before_the_jump(self) -> None:
        call = jump_to(entry("avx2", 0), "jmpq")
        f = function(
            entry("avx2", 1),
            [("shl", "$0x4,%rsi"), ("jmpq", "0x0"), ("data16", "nopw %cs:0x0(%rax,%rax,1)")],
            [call],
        )
        self.assertEqual(len(call_errors("avx2", f)), 1)


class StillCalls(unittest.TestCase):
    def assert_reported(self, target: str, f: Function) -> None:
        errors = call_errors(target, f)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("must inline", errors[0])

    def test_any_other_instruction_makes_it_a_call(self) -> None:
        call = jump_to(entry("scalar", 0))
        f = function(entry("scalar", 1), [("lsl", "x1, x1, #4"), ("b", "0x0")], [call])
        self.assert_reported("scalar", f)

    def test_a_jump_to_a_different_function_is_a_call(self) -> None:
        helper = entry("scalar", 0, name="small_dft")
        self.assert_reported("scalar", function(entry("scalar", 1), [("b", "")], [jump_to(helper)]))

    def test_a_jump_to_another_targets_copy_is_a_call(self) -> None:
        f = function(entry("scalar", 1), [("b", "")], [jump_to(entry("avx2", 0))])
        self.assert_reported("scalar", f)

    def test_a_call_with_return_is_a_call(self) -> None:
        call = jump_to(entry("scalar", 0), "bl")
        self.assert_reported("scalar", function(entry("scalar", 1), [("bl", "")], [call]))

    def test_a_jump_into_the_standard_library_is_a_call(self) -> None:
        f = function(entry("scalar", 1), [("b", "")], [jump_to("_ZNSt6vectorIdSaIdEE5beginEv")])
        self.assert_reported("scalar", f)

    def test_microsoft_names_get_no_exception(self) -> None:
        a = "?run_codelet@?A0x1@scalar@detail@v0@fftn@@YAXPEBN_KPEAN1PEBN@Z"
        b = "?run_codelet@?A0x1@scalar@detail@v0@fftn@@YAXPEBM_KPEAM1PEBM@Z"
        self.assert_reported("scalar", function(a, [("jmp", "")], [jump_to(b, "jmp")]))


if __name__ == "__main__":
    unittest.main()

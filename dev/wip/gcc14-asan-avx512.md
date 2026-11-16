# WIP: GCC 14 ASan segfault in avx512 code on the Xeon (to be folded into dev/LESSONS.md)

Found so far (Xeon, GCC 14.4, RelWithDebInfo, -fsanitize=address,undefined, main e2f5e92):

- Faulting instruction: `vmovapd %zmm4,-0x180(%r14)` in
  `leaf_lines<double,5,1,...>` (store_first's 64-byte `T tile[2*C]`; other failures are
  store_part's tile in bluestein_in). An aligned 512-bit store to a slot that is
  32 mod 64: a #GP, which ASan reports as SEGV "READ" on an unknown address.
- GCC raises the 64-byte local array to 64-byte alignment (cache-line rule for locals) and
  so turns `_mm512_storeu_pd` into `vmovapd`. It lays the ASan frame out with its base 32
  mod 64 on the real stack (base = rsp + 0xa0 after `and $-64,%rsp`), so the tile is
  aligned there; with `detect_stack_use_after_return=1` (the runtime default on Linux) the
  frame comes from `__asan_stack_malloc_4`, 64-aligned, and the tile lands 32 mod 64.
- `ASAN_OPTIONS=detect_stack_use_after_return=0`: the test passes (job 15980).
- A minimal one-function reproducer faults with GCC 13.3 and passes with 14.4, 15.2, 16.2
  and Clang 23 (job 15981): GCC PR 110027 (fixed for 14) covers the simple case, not this
  frame layout. Not yet minimised for 14.4.
- Verdict so far: a GCC sanitizer code-generation bug (class 2), not an FFTN bug.

Queued on the Xeon: 15983 (GCC 15 then 16, ASan, failing tests, both UAR settings),
15984 (GCC 14 on main 5b7a6ae, then Clang 23). Logs in ~/awork/fftn-asan512/.
CI: no extended run ever ran (GCC ASan on x86 never ran in CI); ci.yml's clang-23 asan
job lands on AVX-512 runners some of the time (best_target avx512 in 2 of 3 runs checked).

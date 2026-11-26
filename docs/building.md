# Building and using FFTN

FFTN is a static library, `libfftn` (`fftn.lib` on Windows), with headers in
`include/fftn/`. It has no dependencies beyond the C++ standard library and the platform's
thread library. Its users compile as C++20 or later.

## Requirements

- CMake 3.25 or newer, and a generator (Ninja is what CI uses).
- A C++20 compiler: GCC 14 or newer, Clang 16 or newer, Apple Clang, or Clang on Windows
  (`clang-cl`, or `clang++` with the MSVC standard library). MSVC's own compiler is not
  supported. Intel's `icpx` works but is not recommended over GCC or Clang.
- Platforms: Linux and macOS on x86-64 and AArch64, Windows on x64 and Arm64.

## Building and installing

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/prefix
```

A top-level build also builds the tests and the benchmark harness; add
`-DFFTN_BUILD_TESTS=OFF -DFFTN_BUILD_BENCH=OFF` to build only the library. The kernels are
compiled optimised whatever the build type, since unoptimised kernels run 30–60 times slower.

The install holds the headers (`include/fftn/`), the library (`lib/`), the CMake package
(`lib/cmake/fftn/`), `lib/pkgconfig/fftn.pc` and the licence (`share/doc/fftn/`), and
nothing else: the test dependencies are never installed. The tree can be moved after
installation.

## Using an installed FFTN

With CMake, put the install prefix on `CMAKE_PREFIX_PATH` and link `fftn::fftn`:

```cmake
find_package(fftn 0.0 REQUIRED)
target_link_libraries(my_app PRIVATE fftn::fftn)
```

`fftn::fftn` brings the include directory, the C++20 requirement and the thread library. The
version check accepts the same major and minor version only: until 1.0 a minor version may
change the API.

The optional adapters are components, each a target that adds its runtime to `fftn::fftn`.
They are header-only and compiled in your own translation units, against your runtime
([threading](threading.md), [Kokkos](kokkos.md)):

| Component | Target | Header | Adds |
|---|---|---|---|
| `omp` | `fftn::omp` | `fftn/omp.hpp` | the compiler's OpenMP (`OpenMP::OpenMP_CXX`) |
| `kokkos` | `fftn::kokkos` | `fftn/kokkos.hpp` | your Kokkos (`Kokkos::kokkos`, found unless it already exists) |

```cmake
find_package(fftn 0.0 REQUIRED COMPONENTS omp)
target_link_libraries(my_app PRIVATE fftn::omp)
```

A component whose runtime is not found is missing: `find_package` fails if it was required
and leaves its target undefined if it was optional (`OPTIONAL_COMPONENTS`).

With pkg-config:

```bash
c++ -std=c++20 -O2 my_app.cpp $(pkg-config --cflags --libs fftn)
```

## FFTN as a subproject

FetchContent or `add_subdirectory` builds FFTN inside your project. As a subproject it builds
only the library (no tests, no benchmarks) and installs nothing unless `FFTN_INSTALL` is on.

```cmake
include(FetchContent)
FetchContent_Declare(fftn GIT_REPOSITORY https://github.com/jtravs/fftn.git GIT_TAG main)
FetchContent_MakeAvailable(fftn)
target_link_libraries(my_app PRIVATE fftn::fftn)
```

`fftn::omp` exists when FFTN's configuration finds the compiler's OpenMP, and `fftn::kokkos`
when `Kokkos::kokkos` exists before FFTN is added. `FETCHCONTENT_SOURCE_DIR_FFTN` points the
declaration at a local copy for offline builds.

[examples/package](../examples/package/CMakeLists.txt) is a complete project that uses FFTN
either way; CI builds it against an install tree, with pkg-config and through FetchContent on
Linux, macOS and Windows (`.github/workflows/package.yml`).

## Instruction sets

One build serves every processor of its architecture: no `-march` or ISA option is needed,
and none should be added for FFTN. The library contains kernels for each instruction-set
target of the architecture (`fftn::isa` in `include/fftn/isa.hpp`: scalar, `x86_v2`, `avx`,
`avx2` and `avx512` on x86-64; scalar, `neon` and `neon_fcma` on AArch64). When a plan is
constructed it takes the fastest target that the processor and the operating system support
for its transform. `plan::instruction_set()` reports the choice and `options::force_isa`
overrides it, for tests and benchmarks.

## Options

The build options are the cache variables at the top of
[CMakeLists.txt](../CMakeLists.txt), each described there. Those a packager may want:

- `FFTN_INSTALL` (on for a top-level build): the install rules.
- `FFTN_OLD_X86_TARGETS` (on): the `x86_v2` and `avx` targets for x86-64 processors from
  before 2013, about half of an x86-64 library's text. Off leaves them out, and such
  processors run the scalar kernels; keep the default for a distributed build.
- `FFTN_MASKED_TAILS` (`AUTO`): masked partial vector runs on x86, on with GCC by default.
  Results are bitwise the same either way.

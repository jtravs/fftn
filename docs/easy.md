# The convenience layer

`fftn/easy.hpp` gives numpy.fft's functions in one call each: `fft`, `ifft`, `rfft` and
`irfft` over spans, their 2-D and N-D forms over strided arrays, the three norms, and the
helpers `fftfreq`, `rfftfreq`, `fftshift` and `ifftshift`; and scipy.fft's real-to-real
functions, `dct`, `idct`, `dst` and `idst` with their N-D forms, plus the DHT. Each call
finds or builds a core plan (`fftn::plan`, [conventions.md](conventions.md)) and executes
it, keeping the plan and its scratch for the next call of the same layout. Nothing else is
added: no kernels, no planning of its own. The declarations, preconditions and guarantees are in
`include/fftn/easy.hpp`; this page says what the functions compute and what the cache keeps.

The layer is **provisional** (steps 6.6 and 6.7, the proposals of dev/DESIGN.md 5.4 and 18.2,
"Phase 6"): its names and semantics await the maintainer's review.

## One dimension

```cpp file=examples/easy.cpp region=one
// A tone at bin 3 of 16 samples.
std::vector<double> x(16);
for (std::size_t j = 0; j < x.size(); ++j) {
  x[j] = std::cos(2 * std::numbers::pi * 3 * static_cast<double>(j) / 16);
}

// numpy.fft.rfft(x): 9 values, 8 at bin 3 (unnormalised forward, numpy's default).
const std::vector<std::complex<double>> spectrum = fftn::easy::rfft(x);

// numpy.fft.irfft(spectrum, n=16): the tone again, scaled by 1/16 on the way back.
std::vector<double> back(16);
fftn::easy::irfft(spectrum, back);

// Unitary transforms instead: norm="ortho".
std::vector<std::complex<double>> unitary(9);
fftn::easy::rfft(x, unitary, fftn::easy::norm::ortho);
```

Each function has two forms. One writes into the caller's span and checks its size: `fft`
and `ifft` write as many values as they read, `rfft` writes n/2 + 1, and `irfft` takes its
length n from the output (numpy's `n`), reading n/2 + 1 values. The other returns a new
`std::vector`, which it allocates on every call; its `irfft` takes `n` as an argument, 0
meaning numpy's default 2(m - 1). Both precisions are overloads, so a `std::vector` or a
`std::array` passes directly. The input may be the output's memory for `fft` and `ifft` (in
place); the real transforms have no in-place form, as numpy has none.

What numpy's `n` does beyond this, cropping or zero-padding the input to another length,
is not offered: the sizes must agree, and a mismatch is refused.

## Several dimensions

The N-D functions take `fftn::easy::view<T, Rank>`: a pointer, the extents slowest first,
and the strides in elements (all 0, the default, for tight C order; negative strides
allowed). A `view` of `T` converts to a `view` of `const T`, and where the standard library
has `std::mdspan` (C++23) a strided `mdspan` converts too. The functions deduce their element
types from views, so an `mdspan` is passed as `fftn::easy::view(m)`: the conversion is not
applied during deduction.

```cpp file=examples/easy.cpp region=views
// A batch of 4 real images of 32 x 48, C-ordered: rfft2 transforms the last two dimensions.
const std::size_t batch = 4;
const std::size_t ny = 32;
const std::size_t nx = 48;
std::vector<float> images(batch * ny * nx, 1.0F);
std::vector<std::complex<float>> spectra(batch * ny * (nx / 2 + 1));
fftn::easy::rfft2(
    fftn::easy::view<const float, 3>(images.data(), {batch, ny, nx}),
    fftn::easy::view<std::complex<float>, 3>(spectra.data(), {batch, ny, nx / 2 + 1}));

// The same along dimensions 0 and 2 of a padded array (rows 64 apart), the last listed
// (2) halved, as numpy's axes=(0, 2) or axes=(0, -1).
std::vector<float> padded(ny * 64, 1.0F);
std::vector<std::complex<float>> half(ny * (nx / 2 + 1));
fftn::easy::rfftn(fftn::easy::view<const float, 2>(padded.data(), {ny, nx}, {64, 1}),
                  fftn::easy::view<std::complex<float>, 2>(half.data(), {ny, nx / 2 + 1}),
                  {0, -1});
```

The rules are numpy's:

- **Axes.** `axes` lists the dimensions to transform, a negative index counting from the
  end; the other dimensions are looped over. By default `fftn`, `ifftn`, `rfftn` and
  `irfftn` transform every dimension, and the `2` forms the last two. A transform takes one
  to three dimensions (the core's limit, dev/DESIGN.md 5.6), so an array of four or more
  dimensions needs `axes`. A dimension listed twice is refused (numpy would transform it
  twice).
- **Real transforms.** The last dimension listed is the halved one: `rfftn`'s output and
  `irfftn`'s input have n/2 + 1 elements along it and the real array's extents elsewhere.
  `irfftn` takes the real extents from its output (numpy's `s`), runs the inverse along the
  other dimensions first and the halved one last, and ignores the imaginary parts the real
  transform ignores, as numpy does; its input is preserved.
- **Empty arrays.** A transformed dimension of extent 0 is refused; a looped dimension of
  extent 0 makes the call do nothing.

## Norms

`norm::backward` (the default) leaves the forward transforms unscaled and scales the inverse
ones by 1/N, N the number of points transformed; `norm::ortho` scales both by 1/sqrt(N);
`norm::forward` scales the forward ones by 1/N and leaves the inverse ones unscaled. The
factor is the plan's `options::scale`, fused into the pass that writes the output, so it
costs nothing extra and the result is bit for bit the core plan's with that scale.

## Real-to-real transforms

```cpp file=examples/easy.cpp region=r2r
// scipy.fft.dct(x, norm="ortho"), the orthonormal DCT-II: 5, -2.23, 0, -0.16. idct undoes it.
const std::vector<double> x{1, 2, 3, 4};
const std::vector<double> c = fftn::easy::dct(x, 2, fftn::easy::norm::ortho);
const std::vector<double> back = fftn::easy::idct(c, 2, fftn::easy::norm::ortho);

// scipy.fft.dstn(a, type=1, axes=[0]) of a 3 x 5 array: the DST-I down its columns.
std::vector<double> a(15, 1.0);
std::vector<double> s(15);
fftn::easy::dstn(fftn::easy::view<const double, 2>(a.data(), {3, 5}),
                 fftn::easy::view<double, 2>(s.data(), {3, 5}), 1, {0});
```

`dct`, `idct`, `dst` and `idst` take scipy's `type`, 1 to 4 (2 by default), and `norm`; their
results are scipy.fft's. Unnormalised (`norm::backward` on the forward functions) they are
FFTW's REDFT and RODFT transforms, the kinds `dct1` to `dst4` of the core
([conventions.md](conventions.md), "Real-to-real transforms"), whose table gives each sum.
The inverse functions compute the inverse type (types 1 and 4 are their own inverses, 2 and
3 each other's), scaled by 1/N by default, where N is scipy's logical size: 2(n - 1) for
type 1 of the DCT, 2(n + 1) for type 1 of the DST and 2n otherwise. `norm::forward` moves
that factor to the forward function, and `norm::ortho` scales both by 1/sqrt(N) and applies
scipy's orthogonalising weights, as scipy does by default (`orthogonalize=True`): the DCT of
type 1 multiplies the first and last input points by sqrt 2 and divides the first and last
outputs by it, type 2 divides the first output, type 3 multiplies the first input, the DST
of type 2 divides the last output and type 3 multiplies the last input; types 1 (DST) and 4
need none. The DCT of type 1 needs at least two points.

`dctn`, `idctn`, `dstn` and `idstn` transform the dimensions `axes` of a view (all by default,
at most three) with the same type along each, as scipy's do: the separable product, with N the
product of the dimensions' logical sizes and each dimension's weights under `norm::ortho`.
`dht` is the discrete Hartley transform, H_k = sum_j x_j cas(2 pi jk / n) with cas = cos + sin
(scipy.fft has none): a forward transform for the norms (unscaled by default), and its own
inverse up to n, so `dht` with `norm::forward` undoes `dht`, and with `norm::ortho` it undoes
itself. Every real-to-real function works in place when the output is the input's memory with
the same strides. scipy's `n` and `s` (cropping and padding), its `orthogonalize=False` with
`norm="ortho"`, and its `overwrite_x` and `workers` are not offered.

The weights cost almost nothing. Along one axis they are real factors on the plan's edges
(dev/DESIGN.md 12.1), an array of n values kept with the plan in the cache, so such a call is
bit for bit the core plan with those edges and the norm's scale. Over several axes a point's
weight is the product of its axes' weights, which differ from 1 only on the first or last
hyperplane of each axis. So the output's weights are a pass over those hyperplanes after the
transform. Input weights (the DCT of types 1 and 3, the DST of type 3) work differently: the
input is first copied into the output, its hyperplanes are weighted there, and the plan then
runs in place on the output, so the input is never written. That costs no memory beyond the
plan, and N/n points per axis of work plus the copy; the passes and the copy run on the
calling thread.

## Frequencies and shifts

```cpp file=examples/easy.cpp region=helpers
// numpy.fft.fftfreq(8, d=0.5) and the spectrum's zero frequency moved to the middle.
std::vector<double> f(8);
fftn::easy::fftfreq(8, f, 0.5); // 0, 0.25, 0.5, 0.75, -1, -0.75, -0.5, -0.25
std::vector<double> centred(8);
fftn::easy::fftshift(fftn::easy::view<const double, 1>(f.data(), {8}),
                     fftn::easy::view<double, 1>(centred.data(), {8})); // -1, ..., 0.75
```

`fftfreq` and `rfftfreq` compute numpy's values the way numpy does (the integer frequency
times 1/(n d), in double). `fftshift` and `ifftshift` work on views of any element type,
along chosen axes (all by default), in place when the output is the input's view.

## The cache

The plans and scratch live in a `fftn::easy::context`. The free functions use
`fftn::easy::thread_context()`, the calling thread's own: a `thread_local`, created on the
thread's first call and destroyed when the thread exits. Threads therefore share nothing
and take no lock, and a context must not be used by two threads at once. This per-thread
context is the only state FFTN keeps behind the caller's back (AGENTS.md section 4 allows
it to the convenience layer alone).

A context keys its plans by kind, precision, direction, norm, the arrays' extents and
strides on both sides, the axes, and whether the call is in place. A call whose key is
cached executes that plan and allocates nothing. A miss builds the plan with
`planning::rules`, keeps it, and grows the context's one workspace if the plan needs more.
At most `max_plans()` plans are kept, 64 unless `set_max_plans` says otherwise; beyond that
the least recently used plan is dropped. The memory a context holds is its plans' tables
and one workspace as large as the largest call has needed; `clear()` releases both.

```cpp file=examples/easy.cpp region=context
// A context of one's own: at most 8 plans, released together with the context.
fftn::easy::context ctx;
ctx.set_max_plans(8);
std::vector<std::complex<double>> a(1000, 1.0);
std::vector<std::complex<double>> b(1000);
ctx.fft(a, b);  // builds and keeps the plan of 1000 points
ctx.ifft(b, a); // another plan: the inverse
ctx.fft(a, b);  // finds the first plan: allocates nothing
```

A call must not re-enter the context it runs on (a task on a context's executor calling the
free functions on the same thread, say), and the free functions must not be called from
another `thread_local`'s destructor once the thread is exiting.

A context built from an executor (`fftn::easy::context ctx(pool)`) executes every call on
it, with the core's guarantee that the results are those of serial execution bit for bit
([threading.md](threading.md)); the executor must outlive the context.

## What the layer does not do

- Team execution, plan edges, pipelines and callbacks: they need the core API, whose plans
  and workspaces the caller holds.
- Measured planning and wisdom: plans are built with the rules only
  ([planning.md](planning.md) says when measuring pays; it pays for plans kept long, which
  the core API serves).
- Real-to-real transforms with a different kind on each axis: the core's `axes_r2r` takes them.

## Errors

Every refusal is `std::invalid_argument` whose message begins with the function's name and
names the argument: `out.size()`, `in.extents[1]`, `axes`, `out.strides[0]`. Layouts that
the core refuses (output elements that overlap, say) come back with the core's message
quoted. Nothing is written when a call is refused.

# Distributed Multidimensional FFT - Implementation Record

Progress record for the parallel FFT subsystem in CLUSTR. Each section
covers one phase of the roadmap: what was built, the exact APIs, design
decisions made and why, validation results, and known limitations. Use
this document to audit the architecture or trace a bug back to its
design origin.

Last updated: 2026-04-11 (Phase 7 complete, Phase 8 pending).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 0 - DistArray](#2-phase-0---distarray)
3. [Phase 3 - Subarray Descriptors](#3-phase-3---subarray-descriptors)
4. [Phase 4 - alltoallw Collective](#4-phase-4---alltoallw-collective)
5. [Phase 5 - Cartesian Topology and Sub-communicators](#5-phase-5---cartesian-topology-and-sub-communicators)
6. [Phase 6 - RedistributePlan](#6-phase-6---redistributeplan)
7. [Phase 7 - 2D Slab FFT](#7-phase-7---2d-slab-fft)
8. [Phase 8 - 3D Pencil FFT (Pending)](#8-phase-8---3d-pencil-fft-pending)
9. [Compile-Time Configuration](#9-compile-time-configuration)
10. [Header Dependency Chain](#10-header-dependency-chain)
11. [Tag Reservation Table](#11-tag-reservation-table)
12. [Balanced Block Decomposition](#12-balanced-block-decomposition)
13. [Test Matrix](#13-test-matrix)
14. [Known Limitations](#14-known-limitations)

---

## 1. Architecture Overview

The FFT subsystem is a vertical stack of composable, header-only C++20
templates. Each layer depends only on the one below it. Communication
uses CLUSTR's custom MPI implementation (ASIO TCP, C++20 coroutines)
rather than a standard MPI library.

```
ParallelFFT2D<T>           (Phase 7 - orchestration)
  |
  +-- PocketFFT c2c<T>     (serial 1D transforms)
  +-- RedistributePlan<T>  (Phase 6 - axis swap)
        |
        +-- alltoallw<T>   (Phase 4 - generalized all-to-all)
              |
              +-- Subarray<T>  (Phase 3 - zero-copy geometry)
              +-- Comm         (Phase 1/2 - TCP mesh, send_raw/recv_raw)
                    |
                    +-- DistArray<T>  (Phase 0 - row-major storage)
```

All transforms operate on `std::complex<T>` elements where T is the
scalar type (double or float). Strides are stored in bytes throughout
the stack so they pass directly to PocketFFT without conversion.

Every function that touches the network returns `asio::awaitable<void>`
and is called with `co_await`. Local-only operations (PocketFFT, shape
math, Subarray construction) are synchronous.

---

## 2. Phase 0 - DistArray

**File:** `include/dist_array.h`

Row-major multidimensional array with an optional distributed-axis
annotation. This is the fundamental data container every FFT primitive
operates on.

### API

```cpp
namespace clustr {

inline constexpr int kNoDistributedAxis = -1;

template <typename T>
class DistArray {
public:
    using shape_t  = std::vector<std::size_t>;
    using stride_t = std::vector<std::ptrdiff_t>;   // BYTES

    // Serial constructor: distributed_axis = kNoDistributedAxis,
    // local_shape == global_shape.
    static DistArray serial(shape_t shape);

    // Distributed constructor.
    DistArray(shape_t global_shape,
              shape_t local_shape,
              int     distributed_axis);

    const shape_t&  global_shape()      const noexcept;
    const shape_t&  local_shape()       const noexcept;
    int             distributed_axis()  const noexcept;
    std::size_t     ndim()              const noexcept;
    std::size_t     size()              const noexcept;
    const stride_t& strides_bytes()     const noexcept;

    T*       data()       noexcept;
    const T* data() const noexcept;

    template <typename... I> T&       at(I... idx);
    template <typename... I> const T& at(I... idx) const;
};

}
```

### Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Strides in bytes, not elements | PocketFFT's c2c signature takes byte strides. Storing in bytes avoids a per-call conversion on the hot path. |
| 2 | `distributed_axis` annotation | Purely metadata. The array itself is always a flat `std::vector<T>` sized to `product(local_shape)`. The tag exists so downstream code (RedistributePlan, ParallelFFT2D) can validate that the caller wired the right buffer to the right slot. |
| 3 | Row-major (C-order) | PocketFFT assumes row-major. The innermost axis changes fastest. Stride computation: `strides[i] = product(shape[i+1..n-1]) * sizeof(T)`. |
| 4 | No allocator parameter | All storage is `std::vector<T>` with the default allocator. Pinned/device memory is a Phase 9+ concern. |

### Validation

- Empty shape throws `std::invalid_argument`.
- `local_shape.size() != global_shape.size()` throws `std::invalid_argument`.
- `at()` checks index count and bounds; throws `std::out_of_range`.

---

## 3. Phase 3 - Subarray Descriptors

**File:** `include/clustr/subarray.h`

Non-owning box descriptor over a DistArray. Given a start + extent box
inside a row-major DistArray, it produces an ASIO buffer sequence whose
entries are coalesced runs of contiguous bytes. This is the C++ analog
of `MPI_Type_create_subarray` from FAST-FFT.md Section 3.

### API

```cpp
namespace clustr {

template <typename T>
class Subarray {
public:
    Subarray(DistArray<T>& arr, shape_t start, shape_t extent);

    // Non-copyable, non-movable (cache holds raw pointers).
    Subarray(const Subarray&)            = delete;
    Subarray(Subarray&&)                 = delete;

    std::size_t     ndim()          const noexcept;
    const shape_t&  start()         const noexcept;
    const shape_t&  extent()        const noexcept;
    std::size_t     total_bytes()   const noexcept;
    std::size_t     coalesced_axis() const noexcept;
    std::size_t     fragment_count() const noexcept;

    const std::vector<asio::const_buffer>&   as_const_buffers();
    const std::vector<asio::mutable_buffer>& as_mutable_buffers();
};

}
```

### Coalesce Rule

Walk axes from innermost outward. Fold axis k into the fragment if and
only if every axis strictly inside k has full extent
(`extent[j] == local_shape[j]` for all `j > k`). The inner axis is
always part of the fragment, even if partial.

```
Shape 64x64x64, extent 64x64x64 -> coalesced_axis=0, 1 fragment
Shape 64x64x64, extent 32x64x64 -> coalesced_axis=0, 1 fragment (*)
Shape 64x64x64, extent 64x32x64 -> coalesced_axis=1, 64 fragments
Shape 64x64x64, extent 64x64x32 -> coalesced_axis=2, 4096 fragments

(*) outer partial with both inner axes full still coalesces to 1.
```

### Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Non-copyable, non-movable | The buffer cache stores raw pointers into the parent DistArray. Move/copy would silently invalidate them. |
| 2 | Lazy materialization | `as_const_buffers()` / `as_mutable_buffers()` build the fragment table on first call and cache it. Amortizes cost when the same Subarray is used for multiple sends (not the case today, but future-safe). |
| 3 | Zero-extent produces 0 fragments | `total_bytes() == 0` short-circuits alltoallw without a branch. No null-pointer checks needed in the inner loop. |

### Validation

- `start.size()` or `extent.size()` != parent ndim: throws `std::invalid_argument`.
- `start[i] + extent[i] > local_shape[i]` on any axis: throws `std::out_of_range` with axis number.

### Test Coverage

**File:** `jobs/subarray_coalesce_test.cpp` (Ranks=1, no MPI)

| Test | Array | Box | Expected Fragments |
|------|-------|-----|--------------------|
| Full 3D | 64^3 | full | 1 |
| Partial outer | 64^3 | 32x64x64 | 1 |
| Partial middle | 64^3 | 64x32x64 | 64 |
| Partial inner | 64^3 | 64x64x32 | 4096 |
| Offset origin | 4x4x4 | (1,2,0) 2x2x4 | verified byte_offset |
| 1D full | 100 | full | 1 |
| 1D partial | 100 | 10..40 | 1 |
| Empty extent | 4x4x4 | 2x0x4 | 0 |
| Out of range | 4x4x4 | exceeds | throws |

**File:** `jobs/subarray_send_test.cpp` (Ranks>=2, MPI wire round-trip)

| Scenario | Box | Fragments | Validates |
|----------|-----|-----------|-----------|
| Full 8^3 | 8x8x8 | 1 | Single-fragment zero-copy send |
| Partial outer | 4x8x8 @ (2,0,0) | 1 | Offset + coalesce |
| Partial inner | 8x8x4 | 64 | Multi-fragment scatter-gather |

---

## 4. Phase 4 - alltoallw Collective

**File:** `include/clustr/alltoallw.h`

Generalized all-to-all over per-peer Subarray descriptors. Each rank
exchanges one Subarray with every peer concurrently. This is the
communication engine that RedistributePlan calls.

### API

```cpp
namespace clustr {

inline constexpr std::int32_t kAlltoallwTag = -6;

template <typename T>
asio::awaitable<void> alltoallw(
    Comm& comm,
    std::span<Subarray<T>* const> send_types,   // size == comm.size()
    std::span<Subarray<T>* const> recv_types);   // size == comm.size()

}
```

### Concurrency Model

1. **Self-loopback** (p == rank): inline memcpy via two-cursor
   fragment-stream copy. No self-socket exists in the connect mesh, so
   this path is mandatory.

2. **Remote peers**: each (send, recv) pair is a coroutine spawned with
   `co_spawn(..., asio::deferred)`. All deferred ops are collected and
   awaited via `asio::experimental::make_parallel_group(...).async_wait(
   wait_for_all(), use_awaitable)`.

3. **Zero-byte slots** (`total_bytes() == 0`): skipped without posting
   an op. The slot must still be a valid Subarray pointer (not nullptr).

### Self-Loopback Detail

`copy_subarray_to_subarray(src, dst)` walks both buffer vectors with
two independent cursors, copying `min(remaining_src, remaining_dst)` at
each step. This handles send and recv subarrays that have different
fragment layouts (e.g. fully coalesced send vs. partial-inner recv).

### Wire Format

Each remote send posts a `PeerHeader` with:
- `type = PeerMsgType::SEND`
- `tag = kAlltoallwTag (-6)`
- `comm_id = comm.comm_id()`
- `payload_len = sub->total_bytes()`

followed by the Subarray's `as_const_buffers()` scatter-gather list.

The receiver calls `comm.recv_raw(src, kAlltoallwTag, SEND)` to get a
contiguous `vector<uint8_t>`, then `scatter_bytes_into()` walks the
recv Subarray's `as_mutable_buffers()` and copies the flat byte stream
back into the scattered geometry.

### Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | No nullptr slots | Keeps the inner loop branch-free. "No exchange" is extent=0, not nullptr. |
| 2 | ranged_parallel_group | Single co_await for all peers. Simpler than manual join logic and rethrows the first exception. |
| 3 | Tag -6 | Next free negative tag after barrier(-1), bcast(-2), reduce(-3), scatter(-4), gather(-5). |

### Validation

- `send_types.size() != comm.size()` or `recv_types.size() != comm.size()`: throws `std::invalid_argument`.
- `scatter_bytes_into`: received byte count != `dst.total_bytes()`: throws `std::runtime_error`.
- `copy_subarray_to_subarray`: src/dst `total_bytes()` mismatch: throws `std::runtime_error`.
- First non-null exception from parallel_group is rethrown.

### Test Coverage

**File:** `jobs/alltoallw_test.cpp` (Ranks>=2)

Each rank owns an 8x8x8 input with deterministic fill:
`f(producer, i, j, k) = producer*1e6 + i*1e4 + j*1e2 + k`.

Box kind assigned by delta = (peer - rank) mod N:

| Kind | Delta | Box | Fragments | Path |
|------|-------|-----|-----------|------|
| 0 | 0 (self) | 8x8x8 | 1 | loopback memcpy |
| 1 | 1 mod 3 | 8x8x8 | 1 | full coalesce, wire |
| 2 | 2 mod 3 | 8x8x4 | 64 | partial inner, wire |
| 3 | 0 mod 3, d!=0 | 0x0x0 | 0 | no exchange |

Rank count determines which kinds are exercised: 2 ranks = {0,1},
3 ranks = {0,1,2}, 4+ ranks = all four cycled.

---

## 5. Phase 5 - Cartesian Topology and Sub-communicators

**Files:**
- `include/clustr_mpi.h` (cart_create, cart_sub declarations, sub-comm private constructor)
- `include/clustr/cart.hpp` (cart_sub implementation, FNV hash)

Phase 5 is prerequisite infrastructure for Phase 8 (3D pencil FFT).
It is not used by Phase 7 (2D slab FFT operates on the world comm).

### 5A - cart_create

```cpp
void Comm::cart_create(const std::vector<int>& dims);
```

Assigns Cartesian coordinates to each rank using row-major layout.
Product of dims must equal `comm.size()`. Sets `dims_` and `coords_`
vectors on the Comm.

Example for `dims = {2, 3}` on 6 ranks:

| World Rank | coords[0] | coords[1] |
|------------|-----------|-----------|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 1 | 0 |
| 4 | 1 | 1 |
| 5 | 1 | 2 |

### 5B - Collectives on Sub-comms

All collectives (bcast, reduce, scatter, gather, barrier) use
`my_logical_rank_` / `my_size_` instead of `roster_.my_rank` /
`roster_.size` so they work correctly on sub-communicators without
code changes.

### 5C - cart_sub

```cpp
Comm Comm::cart_sub(int axis);
```

Projects the Cartesian grid onto a 1D sub-communicator by varying the
specified axis and fixing all other coordinates. Ranks sharing the same
non-projected coordinates form a cohort.

Requires `CLUSTR_RECV == CLUSTR_RECV_CENTRAL` (static_assert under
INLINE mode). Under the CENTRAL dispatch model, sub-comms share the
parent's TCP sockets via `shared_ptr` cloning. The central dispatch
loop demultiplexes messages by `(comm_id, src_rank, tag, type)` tuples.

**Comm ID allocation:** FNV-1a 64-bit hash of `(parent_comm_id, axis,
non_projected_coords)`, folded to 32 bits. Debug builds scan for
collisions. This is deterministic: the same grid geometry produces the
same comm_id on every rank without negotiation.

**Socket sharing:** The sub-comm constructor clones the parent's socket
shared_ptrs. No new TCP connections are opened. The parent's
`child_count_` is incremented (throw-safe: incremented last after all
allocations succeed).

**recv_raw parent chain:** When a sub-comm calls `recv_raw`, it walks
the `parent_` pointer chain to the root Comm before looking up the
central channel map. This ensures all sub-comms and the world comm
share one channel table per physical connection.

### 5D - Validation Test

**File:** `jobs/cart_sub_test.cpp` (Ranks=6)
**Runner:** `tests/run_cart_sub_local.sh` (6 localhost processes)

2x3 grid, 5 phases:

| Phase | Operation | Validates |
|-------|-----------|-----------|
| 1 | cart_create | Correct coordinates for all 6 ranks |
| 2 | cart_sub(1) row bcast | Size-3 row cohorts, sub-rank-0 broadcasts row-specific payload |
| 3 | cart_sub(0) col bcast | Size-2 col cohorts, sub-rank-0 broadcasts col-specific payload |
| 4 | Isolation | Row + col bcast at same default tag (-2). Only comm_id distinguishes them. Proves central dispatch demux is correct. |
| 5 | Sub-comm barriers | Row, col, and world barriers independently |

Builds under CLUSTR_TRANSPORT x {1,2}, CLUSTR_RECV=2 (CENTRAL only).

---

## 6. Phase 6 - RedistributePlan

**File:** `include/clustr/redistribute.h`

Core FFT primitive. Swaps the distributed axis of a DistArray from `v`
to `w` within a 1D process group. This is the Section 3 routine from
FAST-FFT.md.

### API

```cpp
namespace clustr {

namespace redistribute_detail {
    inline std::pair<std::size_t, std::size_t>
    balanced_block(std::size_t n, std::size_t p, std::size_t r);
}

template <typename T>
class RedistributePlan {
public:
    using shape_t = typename DistArray<T>::shape_t;

    RedistributePlan(Comm& comm, shape_t global_shape, int v, int w);

    const shape_t& global_shape()       const noexcept;
    const shape_t& input_local_shape()  const noexcept;
    const shape_t& output_local_shape() const noexcept;
    int            v()                  const noexcept;
    int            w()                  const noexcept;
    std::size_t    my_v_start()         const noexcept;
    std::size_t    my_v_size()          const noexcept;
    std::size_t    my_w_start()         const noexcept;
    std::size_t    my_w_size()          const noexcept;

    asio::awaitable<void> execute(DistArray<T>& input,
                                  DistArray<T>& output);
};

}
```

### Plan-and-Execute Pattern

The constructor does all expensive shape math once:

1. Compute my own v-slab and w-slab via `balanced_block`.
2. Derive `input_local_shape` (global with axis v shrunk) and
   `output_local_shape` (global with axis w shrunk).
3. Precompute N per-peer `(start, extent)` vector pairs for both
   send and recv subarrays.

`execute()` is the hot path:

1. Validate input/output shapes and distributed_axis annotations.
2. Build N `unique_ptr<Subarray<T>>` for send and N for recv using
   the cached start/extent vectors.
3. Pass raw pointer spans to `alltoallw<T>`.

The Subarray objects cannot be cached in the plan (they hold raw data
pointers that change between calls). But the *geometry* is cached.
execute() is O(N) pointer plumbing around one alltoallw.

### Per-Peer Geometry

**Send to peer p** (in MY INPUT local coordinates):
- axis v: full local slab (start=0, extent=my_v_size)
- axis w: peer p's w-slab in global coords
- all other axes: full global extent

**Recv from peer p** (in MY OUTPUT local coordinates):
- axis v: peer p's v-slab in global coords
- axis w: full local slab (start=0, extent=my_w_size)
- all other axes: full global extent

### Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Template on T, not restricted to complex | Redistribute is a data-layout operation. It works on any element type (int, double, complex). The FFT layer provides the complex constraint. |
| 2 | DistArray<T>& API, not void* | Type safety. Catches shape/stride mismatches at compile time (template) and run time (validation). |
| 3 | Cache geometry vectors, not Subarrays | Subarray is non-copyable and holds a raw data pointer that changes per execute() call. Caching the (start, extent) vectors and rebuilding Subarrays at execute time is the only viable pattern. |
| 4 | v == w throws | Not a redistribute. Caller bug. |
| 5 | Empty-slab ranks handled implicitly | When `global_shape[axis] < comm.size()`, some ranks get extent 0. Subarray reports `total_bytes() == 0`, alltoallw short-circuits. No special-case code needed here. |
| 6 | 1D process group only | Multi-dim grids are Phase 7/8's job. The caller projects the world Comm onto a 1D axis via cart_sub and hands the sub-comm here. |

### Validation

- Empty global_shape: `std::invalid_argument`
- v or w out of [0, ndim): `std::out_of_range`
- v == w: `std::invalid_argument`
- Empty comm: `std::invalid_argument`
- execute() shape mismatch: `std::invalid_argument`
- execute() distributed_axis mismatch: `std::invalid_argument`

### Test Coverage

**File:** `jobs/redistribute_test.cpp` (Ranks>=2)
**Runner:** `tests/run_redistribute_local.sh` (3 ranks, all 4 presets)

Global shape {9, 8}. With 3 ranks:
- axis 0: 9/3 = 3 per rank (uniform, no remainder)
- axis 1: 8/3 = 2 rem 2, so ranks 0,1 get 3 cols, rank 2 gets 2

| Phase | Operation | Validates |
|-------|-----------|-----------|
| Forward | v=0 -> w=1 | Each output element at local (i, jl) == f(i, my_w_start + jl) where f(i,j) = i*1000 + j |
| Round-trip | v=1 -> w=0 | Output is bit-exact equal to original input |

Cluster validation: 3 machines, rank 2 confirmed 2 local cols
(remainder slab).

---

## 7. Phase 7 - 2D Slab FFT

**File:** `include/clustr/parallel_fft_2d.h`

First working parallel FFT. 2D complex-to-complex transform using slab
decomposition over a 1D process group (the world communicator).

### API

```cpp
namespace clustr {

template <typename T>
class ParallelFFT2D {
    static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (double, float)");

public:
    using complex_t = std::complex<T>;
    using shape_t   = typename DistArray<complex_t>::shape_t;

    ParallelFFT2D(Comm& comm, shape_t global_shape);

    const shape_t& global_shape()       const noexcept;
    const shape_t& input_local_shape()  const noexcept;
    const shape_t& output_local_shape() const noexcept;

    const RedistributePlan<complex_t>& fwd_plan() const noexcept;
    const RedistributePlan<complex_t>& inv_plan() const noexcept;

    // Destroys input. Output distributed on axis 1.
    asio::awaitable<void> forward(DistArray<complex_t>& input,
                                  DistArray<complex_t>& output);

    // Destroys input. Output distributed on axis 0.
    asio::awaitable<void> inverse(DistArray<complex_t>& input,
                                  DistArray<complex_t>& output);
};

}
```

### Algorithm

**Forward** (input distributed on axis 0, output distributed on axis 1):

```
1. pocketfft::c2c(input.local_shape, strides, strides,
                  axes={1}, forward=true, input, input, fct=1.0)
   -- In-place 1D FFT along axis 1 (contiguous, no communication).
   -- Destroys input: axis 1 is now frequency-domain.

2. co_await fwd_plan_.execute(input, output)
   -- Redistribute axis 0 -> axis 1 via alltoallw.
   -- After this, axis 0 is fully local in the output buffer.

3. pocketfft::c2c(output.local_shape, strides, strides,
                  axes={0}, forward=true, output, output, fct=1.0)
   -- In-place 1D FFT along axis 0.
   -- Output is now fully transformed, distributed on axis 1.
```

**Inverse** (input distributed on axis 1, output distributed on axis 0):

```
1. pocketfft::c2c(input.local_shape, strides, strides,
                  axes={0}, forward=false, input, input,
                  fct=1.0/global_shape[0])
   -- In-place 1D IFFT along axis 0. Normalize by 1/N0.

2. co_await inv_plan_.execute(input, output)
   -- Redistribute axis 1 -> axis 0.

3. pocketfft::c2c(output.local_shape, strides, strides,
                  axes={1}, forward=false, output, output,
                  fct=1.0/global_shape[1])
   -- In-place 1D IFFT along axis 1. Normalize by 1/N1.
   -- Combined normalization: 1/(N0*N1). Round-trip is identity.
```

One alltoallw per direction. Forward + inverse = 2 alltoallw total.

### Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Two-argument API (input, output) | Matches RedistributePlan::execute(). No internal scratch buffers needed -- the output IS the second buffer. |
| 2 | Natural output distribution | Forward produces axis-1-distributed output, inverse produces axis-0. Round-trip restores original. Avoids wasting a second alltoallw to redistribute back. |
| 3 | Destructive to input | First local FFT operates in-place before redistribute. Most memory-efficient. Matches FFTW convention. |
| 4 | Split normalization per-axis | Inverse step 1 normalizes by 1/N0, step 3 by 1/N1. Combined: 1/(N0*N1). Each local IFFT is self-contained and correct in isolation. |
| 5 | Template on scalar T | `ParallelFFT2D<double>` operates on `DistArray<complex<double>>`. Matches PocketFFT's `c2c<T>()` API. `static_assert(is_floating_point_v<T>)` for clear errors. |
| 6 | Two RedistributePlans (fwd + inv) | Forward plan: v=0, w=1. Inverse plan: v=1, w=0. Both constructed once in the ParallelFFT2D constructor and reused on every forward/inverse call. |

### Internal State

```cpp
Comm&       comm_;
shape_t     global_shape_;
RedistributePlan<complex_t> fwd_plan_;   // v=0 -> w=1
RedistributePlan<complex_t> inv_plan_;   // v=1 -> w=0
T           inv_norm_axis0_;             // 1.0 / global_shape_[0]
T           inv_norm_axis1_;             // 1.0 / global_shape_[1]
```

Constructor order: `validate_2d(global_shape)` runs as part of
`global_shape_` initialization (before plan construction) so a non-2D
shape throws before any plans are allocated.

### Validation

- Non-2D global_shape: throws in constructor via `validate_2d`.
- forward(): input.local_shape != plan expectation: throws.
- forward(): input.distributed_axis != 0: throws.
- forward(): output.distributed_axis != 1: throws.
- inverse(): mirror checks for axis 1 input, axis 0 output.

### Test Coverage

**File:** `jobs/fft_2d_test.cpp` (Ranks>=2)
**Runner:** `tests/run_fft_2d_local.sh` (ranks 2,3,4,8 x 4 presets)

Test signal: single complex sinusoid at bin (K0, K1) = (3, 7).

```
x[i][j] = exp(2*pi*i * (3*i/256 + 7*j/256))
```

After forward FFT: spectrum should have a single peak at (3, 7) with
magnitude N0*N1 = 65536 and all other bins at machine-epsilon noise.

| Check | Threshold | Observed |
|-------|-----------|----------|
| Peak magnitude at (3,7) | within 1e-6 of 65536 | exact 65536 |
| All other bins | < 1e-6 | max ~3.6e-11 |
| Round-trip max error | < 1e-10 | ~9.2e-16 |

Tested at 4 rank counts x 4 presets = 16 runs, all passed.

Cluster validation: Ranks=2 on real network, all checks passed.
Round-trip error ~1.0e-15.

---

## 8. Phase 8 - 3D Pencil FFT (Pending)

3D complex-to-complex transform using pencil decomposition over a 2D
Cartesian process grid.

### Planned Algorithm

From FAST-FFT.md Section 4.2. Assume 3D array on process grid with
row-subgroups P0 and column-subgroups P1.

| Step | Operation | Communicator |
|------|-----------|--------------|
| 1 | Local FFT along axis 2 | none |
| 2 | Redistribute axis 2 -> axis 1 | P1 (column sub-comm) |
| 3 | Local FFT along axis 1 | none |
| 4 | Redistribute axis 1 -> axis 0 | P0 (row sub-comm) |
| 5 | Local FFT along axis 0 | none |

Two alltoallw calls per direction (one on each sub-comm). Inverse
mirrors the sequence.

### Prerequisites

- Phase 5C cart_sub (complete, validated)
- Phase 6 RedistributePlan (complete)
- Phase 7 ParallelFFT2D (complete, validates the 2D building block)

### Open Questions

- Test shape: 64x64x64 or 128x128x128?
- Process grid: {2, 3} on 6 ranks (requires cart_sub_local.sh) or
  smaller grids testable on 3 machines?
- Exit criteria error tolerance: same 1e-10 as Phase 7?

---

## 9. Compile-Time Configuration

The MPI layer has two compile-time selectors that affect all collective
operations including the FFT redistribute path.

### CLUSTR_TRANSPORT

| Value | Macro | Behavior |
|-------|-------|----------|
| 1 | CLUSTR_TRANSPORT_ZERO_COPY | Sends Subarray's scatter-gather buffer list directly via ASIO's scatter/gather I/O. Zero local copies. |
| 2 | CLUSTR_TRANSPORT_PACK_UNPACK | Packs Subarray fragments into a contiguous buffer before send, unpacks after recv. One extra memcpy per direction. |

Both produce identical results. Zero-copy is faster on large messages;
pack-unpack may be faster on many small fragments due to fewer syscalls.

### CLUSTR_RECV

| Value | Macro | Behavior |
|-------|-------|----------|
| 1 | CLUSTR_RECV_INLINE | Each `recv_raw` call reads directly from the peer socket. Simple but does not support sub-communicators (messages from different comms share sockets). |
| 2 | CLUSTR_RECV_CENTRAL | One detached coroutine per remote socket reads all incoming frames and demultiplexes by `(comm_id, src_rank, tag, type)` into per-key `asio::experimental::channel`. Required for cart_sub and Phase 8. |

Phase 6 and 7 work under all 4 combinations (the world comm has no
sub-comm demux issues). Phase 5C/8 require CENTRAL.

### Build Flags

```
-DCLUSTR_TRANSPORT=1  -DCLUSTR_RECV=1   # zc + inline
-DCLUSTR_TRANSPORT=1  -DCLUSTR_RECV=2   # zc + central
-DCLUSTR_TRANSPORT=2  -DCLUSTR_RECV=1   # pu + inline
-DCLUSTR_TRANSPORT=2  -DCLUSTR_RECV=2   # pu + central
```

All test runners build and run every combination.

---

## 10. Header Dependency Chain

```
parallel_fft_2d.h
  +-- pocketfft_hdronly.h     (serial FFT, MIT, header-only)
  +-- redistribute.h
        +-- alltoallw.h
        |     +-- subarray.h
        |     |     +-- dist_array.h
        |     |     +-- asio.hpp
        |     +-- clustr_mpi.h
        |     +-- asio/experimental/parallel_group.hpp
        +-- dist_array.h
        +-- clustr_mpi.h
        +-- asio.hpp
```

All headers use `#pragma once`. Transitive includes are idempotent.

---

## 11. Tag Reservation Table

Negative tags are reserved for collective operations. User point-to-point
tags must be >= 0.

| Tag | Collective | File |
|-----|------------|------|
| -1 | barrier | clustr_mpi.h |
| -2 | bcast | clustr_mpi.h |
| -3 | reduce | clustr_mpi.h |
| -4 | scatter | clustr_mpi.h |
| -5 | gather | clustr_mpi.h |
| -6 | alltoallw | alltoallw.h |

---

## 12. Balanced Block Decomposition

Used throughout the FFT stack whenever an axis of length N is
partitioned across P ranks. Defined in `redistribute_detail::balanced_block`.

```
base = floor(N / P)
rem  = N mod P

if rank < rem:
    size  = base + 1
    start = rank * (base + 1)
else:
    size  = base
    start = rem * (base + 1) + (rank - rem) * base
```

**Invariants:**
- `sum(size[r] for r in 0..P-1) == N`
- `start[r+1] == start[r] + size[r]`
- `size[r] >= 0`; `size[r] == 0` iff `N < P` and `rank >= N`
- First `rem` ranks get the fatter blocks (base+1); rest get base.

**Example:** N=256, P=3

| Rank | Start | Size |
|------|-------|------|
| 0 | 0 | 86 |
| 1 | 86 | 85 |
| 2 | 171 | 85 |

**Example:** N=8, P=3

| Rank | Start | Size |
|------|-------|------|
| 0 | 0 | 3 |
| 1 | 3 | 3 |
| 2 | 6 | 2 |

---

## 13. Test Matrix

### Local Runners (tests/*.sh)

| Runner | Ranks | Presets | Total Runs |
|--------|-------|---------|------------|
| run_cart_sub_local.sh | 6 | 2 (CENTRAL only) | 2 |
| run_redistribute_local.sh | 3 | 4 | 4 |
| run_fft_2d_local.sh | 2,3,4,8 | 4 | 16 |

### Cluster Validated

| Test | Ranks | Preset | Result |
|------|-------|--------|--------|
| cart_sub_test | 6 | zc+central (localhost) | PASS |
| redistribute_test | 3 | zc+central | PASS |
| fft_2d_test | 2 | zc+central | PASS |

### Serial Tests (Ranks=1, no MPI)

| Test | Validates |
|------|-----------|
| fft_serial_test | PocketFFT c2c round-trip on 8x16x32, error < 1e-10 |
| subarray_coalesce_test | Fragment geometry, coalesce rule, edge cases |

---

## 14. Known Limitations

1. **No float specialization tested.** All tests use double.
   `ParallelFFT2D<float>` should work (PocketFFT supports it, all
   templates propagate T) but has zero test coverage.

2. **No in-place transform.** Both forward and inverse require separate
   input and output DistArrays. A `forward_copy()` variant that
   preserves the input is not implemented.

3. **PocketFFT is single-threaded.** The `nthreads` parameter is always
   1 (default). On a node with many cores and few ranks, intra-node
   parallelism is left on the table. Could be exposed as a plan
   parameter.

4. **No FFTW backend.** PocketFFT is correct and portable but slower
   than FFTW on x86 for large transforms. The header-only design makes
   swapping backends easy (same c2c signature).

5. **World comm only for Phase 7.** The 2D slab FFT uses the world
   communicator as its 1D process group. It cannot currently operate on
   a cart_sub sub-communicator (though nothing in the code prevents it;
   it just hasn't been tested).

6. **No real-to-complex (r2c) / complex-to-real (c2r) transforms.**
   Only c2c is implemented. r2c would halve the memory and
   communication for real-valued inputs but requires asymmetric shapes
   (N/2+1 on the frequency axis).

7. **No checkpoint/restart across redistribute.** If a rank dies
   mid-alltoallw, the entire job fails. The fail-stop model is inherited
   from the MPI layer.

8. **3-machine cluster limits testing.** Ranks=4 and Ranks=8 are only
   validated via localhost oversubscription, not on real network with
   distinct machines. Latency/bandwidth characteristics differ.

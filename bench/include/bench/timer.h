#pragma once

// bench/timer.h - honest-timing primitives.
//
// Design notes (what makes a benchmark honest):
//
//  * steady_clock is monotonic; on x86_64 Linux/macOS it resolves to
//    CLOCK_MONOTONIC, typically backed by TSC with ~1-10 ns resolution.
//    system_clock is adjustable by NTP and MUST NOT be used for intervals.
//
//  * DoNotOptimize is a sink: feeding a value through an asm("" : : "r"(&x) : "memory")
//    tells the compiler that an outside party may observe x, which blocks
//    dead-code elimination and constant propagation. Without it, the
//    optimizer will happily delete the work being measured.
//
//  * ClobberMemory inserts a full compiler memory fence. It prevents the
//    compiler from reordering memory operations across the fence; combined
//    with DoNotOptimize on the result, the "work" between the two fences
//    cannot be moved or removed.
//
//  * These are compiler-level fences, NOT cpu-level fences. For CPU-side
//    ordering (TSO, MFENCE, etc.) use std::atomic_thread_fence. Here we're
//    measuring wall-time of network/memcpy-dominated operations where the
//    compiler fence is what actually matters.
//
// Usage:
//
//   bench::Timer t;
//   for (auto& x : result) bench::DoNotOptimize(x);
//   t.start();
//   do_work();
//   bench::ClobberMemory();
//   auto ns = t.stop_ns();
//
// The DoNotOptimize/ClobberMemory pattern is lifted from Google Benchmark,
// which got it from Chandler Carruth's talk "Tuning C++: Benchmarks, and
// CPUs, and Compilers! Oh My!" (CppCon 2015).

#include <chrono>
#include <cstdint>

namespace bench {

using clock_t = std::chrono::steady_clock;

// Minimal stopwatch. Intentionally NOT RAII - the stop point must be
// explicit so the benchmark author can place memory fences deliberately
// rather than have the destructor fire wherever the scope happens to end.
class Timer {
public:
    inline void start() noexcept { t0_ = clock_t::now(); }

    inline std::int64_t stop_ns() noexcept {
        auto t1 = clock_t::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count();
    }

    // For cases where the caller wants to pre-compute the end time before
    // additional work (e.g. logging) adds latency to stop_ns().
    inline clock_t::time_point now() const noexcept { return clock_t::now(); }

private:
    clock_t::time_point t0_{};
};

// ============================================================================
// DoNotOptimize / ClobberMemory
// ============================================================================
//
// Templated on T& so the compiler's view of the object is "it might be
// modified by an external party" - that forces reloads and prevents the
// whole chain of dependent computation from being elided.

#if defined(__GNUC__) || defined(__clang__)
template <typename T>
inline void DoNotOptimize(T const& value) noexcept {
    asm volatile("" : : "r,m"(value) : "memory");
}

template <typename T>
inline void DoNotOptimize(T& value) noexcept {
    asm volatile("" : "+r,m"(value) : : "memory");
}

inline void ClobberMemory() noexcept {
    asm volatile("" : : : "memory");
}
#else
// MSVC or other: fall back to a volatile read through a function the
// compiler cannot see. Worse fence than the inline-asm version but still
// prevents most DCE.
template <typename T>
inline void DoNotOptimize(T const& value) noexcept {
    volatile auto sink = &value; (void)sink;
}
template <typename T>
inline void DoNotOptimize(T& value) noexcept {
    volatile auto sink = &value; (void)sink;
}
inline void ClobberMemory() noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
#endif

}  // namespace bench

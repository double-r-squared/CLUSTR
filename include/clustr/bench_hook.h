#pragma once

// clustr/bench_hook.h - compile-time-polymorphic benchmark hooks.
//
// Core idea:
//
//   Template an instrumented class on a Hook policy type. Every timing
//   site calls `hook.begin("name")` and `hook.end("name")`. In the default
//   build, Hook == NullBenchHook, whose methods are empty inline functions
//   that the optimizer deletes. When instrumentation is wanted, pass
//   TextBenchHook (the historical #ifdef CLUSTR_BENCHMARK behavior) or
//   JsonSinkBenchHook (records for bench/runner.h to emit as JSON).
//
// Why compile-time polymorphism and not #ifdef?
//
//   * A Hook parameter makes the instrumentation a first-class type you
//     can test without recompiling the world.
//   * It composes: a benchmark that times the FFT externally can still
//     feed a hook in to capture per-step breakdowns in the same run.
//   * No preprocessor dependency bleed across TUs: the default hook
//     really is zero cost (empty struct, empty methods), not "cost if
//     you forget a #define".
//
// Why keep #ifdef CLUSTR_BENCHMARK as the *default selector*?
//
//   Existing job files (fft_3d_test_auto.cpp) compiled with the
//   -DCLUSTR_BENCHMARK flag expect the old printf behavior. The selector
//   alias `DefaultBenchHook` below picks TextBenchHook when that flag is
//   set and NullBenchHook otherwise, so every existing command line keeps
//   working byte-for-byte.

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace clustr {

// ── NullBenchHook ──────────────────────────────────────────────────────────
//
// Every method is `constexpr inline` so the optimizer can delete it even
// at -O0 for calls through a concrete Hook type. In release builds the
// whole hook parameter collapses to nothing: no stack slots, no per-step
// chrono calls, no format strings in the binary.

struct NullBenchHook {
    struct Token {};

    constexpr NullBenchHook() noexcept = default;

    [[nodiscard]] constexpr Token begin(std::string_view /*label*/) const noexcept { return {}; }
    constexpr void                end(Token /*t*/, std::string_view /*label*/) const noexcept {}

    // Optional grouping - called at the start/end of a logical block
    // (forward(), inverse(), etc.). Default hook does nothing.
    constexpr void group_begin(std::string_view /*label*/) const noexcept {}
    constexpr void group_end  (std::string_view /*label*/) const noexcept {}
};

// ── TextBenchHook ──────────────────────────────────────────────────────────
//
// Prints microseconds to stdout after each step. Reproduces the behavior
// of the original #ifdef CLUSTR_BENCHMARK path so `make_file_data_msg` style
// text reports still appear in logs.
//
// The `rank` parameter exists so multi-rank runs only print from a single
// rank (typically 0). TextBenchHook is stateless; the only instance state
// is `rank_` and a prefix, both copy-cheap.

struct TextBenchHook {
    using clock_t = std::chrono::steady_clock;

    struct Token { clock_t::time_point t0; };

    explicit TextBenchHook(int rank = 0, std::string prefix = "bench")
        : rank_(rank), prefix_(std::move(prefix)) {}

    [[nodiscard]] Token begin(std::string_view /*label*/) const noexcept {
        return Token{clock_t::now()};
    }
    void end(Token t, std::string_view label) const {
        if (rank_ != 0) return;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            clock_t::now() - t.t0).count();
        std::cout << "[" << prefix_ << "] " << label << " = " << us << "us\n";
    }
    void group_begin(std::string_view label) const {
        if (rank_ == 0) std::cout << "[" << prefix_ << "] >> " << label << "\n";
    }
    void group_end(std::string_view label) const {
        if (rank_ == 0) std::cout << "[" << prefix_ << "] << " << label << "\n";
    }

private:
    int         rank_;
    std::string prefix_;
};

// ── JsonSinkBenchHook ──────────────────────────────────────────────────────
//
// Records every sample into an in-memory sink keyed by label. The caller
// extracts results after the run via `hook.sink().samples("label")` and
// feeds them to bench::StatAccumulator / bench::write_json_file.
//
// Separating the sink from the hook lets the same sink collect data from
// multiple hook instances (forward pass + inverse pass + collectives
// all targeting one file).

struct BenchSink {
    struct Series {
        std::string          label;
        std::vector<std::int64_t> samples_ns;
    };

    Series& series(std::string_view label) {
        for (auto& s : series_) if (s.label == label) return s;
        series_.push_back(Series{std::string(label), {}});
        return series_.back();
    }

    const std::vector<Series>& all() const noexcept { return series_; }

    void clear() { series_.clear(); }

private:
    std::vector<Series> series_;
};

struct JsonSinkBenchHook {
    using clock_t = std::chrono::steady_clock;
    struct Token { clock_t::time_point t0; };

    explicit JsonSinkBenchHook(BenchSink* sink, int rank = 0)
        : sink_(sink), rank_(rank) {}

    [[nodiscard]] Token begin(std::string_view /*label*/) const noexcept {
        return Token{clock_t::now()};
    }
    void end(Token t, std::string_view label) const {
        if (!sink_ || rank_ != 0) return;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock_t::now() - t.t0).count();
        sink_->series(label).samples_ns.push_back(ns);
    }
    constexpr void group_begin(std::string_view) const noexcept {}
    constexpr void group_end  (std::string_view) const noexcept {}

    BenchSink*       sink()       noexcept { return sink_; }
    const BenchSink* sink() const noexcept { return sink_; }

private:
    BenchSink* sink_;
    int        rank_;
};

// ── DefaultBenchHook ───────────────────────────────────────────────────────
//
// Picked by the default template argument of instrumented classes. Honors
// the historical -DCLUSTR_BENCHMARK command-line flag so legacy users get
// unchanged behavior without editing their code.

#ifdef CLUSTR_BENCHMARK
using DefaultBenchHook = TextBenchHook;
#else
using DefaultBenchHook = NullBenchHook;
#endif

}  // namespace clustr

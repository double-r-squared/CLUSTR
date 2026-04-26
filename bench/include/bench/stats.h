#pragma once

// bench/stats.h - percentile statistics accumulator.
//
// We record every sample and compute order statistics at the end. This
// costs N doubles of memory but makes percentile reporting exact.
//
// Running-mean/welford-variance would be cheaper but hides tail latency,
// which is the interesting part of a distributed-system benchmark.
// p50 can look fine while p99 is 10x worse, and knowing that is the whole
// reason to benchmark at all.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace bench {

struct Stats {
    std::size_t n       = 0;
    double      min_ns  = 0.0;
    double      max_ns  = 0.0;
    double      mean_ns = 0.0;
    double      stdev_ns = 0.0;
    double      p50_ns  = 0.0;
    double      p90_ns  = 0.0;
    double      p99_ns  = 0.0;
};

class StatAccumulator {
public:
    void reserve(std::size_t n) { samples_.reserve(n); }

    void record(std::int64_t ns) { samples_.push_back(static_cast<double>(ns)); }

    [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }

    // Sort a copy (non-destructive) and return order statistics.
    [[nodiscard]] Stats summarize() const {
        if (samples_.empty())
            throw std::runtime_error("StatAccumulator::summarize: no samples");

        std::vector<double> s = samples_;
        std::sort(s.begin(), s.end());

        Stats out;
        out.n      = s.size();
        out.min_ns = s.front();
        out.max_ns = s.back();

        const double sum = std::accumulate(s.begin(), s.end(), 0.0);
        out.mean_ns = sum / static_cast<double>(s.size());

        double sq = 0.0;
        for (double x : s) { double d = x - out.mean_ns; sq += d * d; }
        out.stdev_ns = std::sqrt(sq / static_cast<double>(s.size()));

        out.p50_ns = percentile(s, 0.50);
        out.p90_ns = percentile(s, 0.90);
        out.p99_ns = percentile(s, 0.99);
        return out;
    }

    [[nodiscard]] const std::vector<double>& raw() const noexcept { return samples_; }

private:
    // Linear-interpolation percentile (NIST type 7, matches numpy default).
    static double percentile(const std::vector<double>& sorted, double q) {
        if (sorted.empty()) return 0.0;
        if (sorted.size() == 1) return sorted.front();
        const double h = q * static_cast<double>(sorted.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(h));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(h));
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
    }

    std::vector<double> samples_;
};

}  // namespace bench

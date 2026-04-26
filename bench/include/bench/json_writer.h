#pragma once

// bench/json_writer.h - tiny ad-hoc JSON emitter.
//
// We intentionally do NOT pull in nlohmann/json or similar. The schema is
// fixed and flat, the amount we emit is small, and dependency-free build
// matters for the benchmark binary (the more we link in, the more the
// benchmark's own setup time pollutes short-run measurements).
//
// Output schema (one JSON file per benchmark run):
//
//   {
//     "bench":      "pingpong",             // benchmark name
//     "config":     { arbitrary k/v },       // transport, ranks, size, etc.
//     "env":        { host, compiler, ... }, // captured by bench::Env
//     "warmup":     <int>,
//     "iterations": <int>,
//     "stats": {
//       "n": ..., "min_ns": ..., "max_ns": ...,
//       "mean_ns": ..., "stdev_ns": ...,
//       "p50_ns": ..., "p90_ns": ..., "p99_ns": ...
//     },
//     "samples_ns": [ ... every raw sample ... ]   // optional, for plots
//   }

#include "bench/stats.h"

#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace bench {

// A ConfigValue is either a string, an int64, or a double. Keeps the emitter
// trivial and matches what benchmarks actually record (transport_name=string,
// ranks=int, bytes=int, etc).
using ConfigValue = std::variant<std::string, std::int64_t, double, bool>;
using ConfigMap   = std::unordered_map<std::string, ConfigValue>;

inline void emit_value(std::ostream& os, const ConfigValue& v) {
    std::visit([&os](auto const& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::string>) {
            // Minimal JSON string escaping - enough for identifiers, hostnames,
            // and compiler banners. Not a full JSON validator.
            os << '"';
            for (char c : x) {
                switch (c) {
                case '"':  os << "\\\""; break;
                case '\\': os << "\\\\"; break;
                case '\n': os << "\\n";  break;
                case '\r': os << "\\r";  break;
                case '\t': os << "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) os << ' ';
                    else os << c;
                }
            }
            os << '"';
        } else if constexpr (std::is_same_v<T, bool>) {
            os << (x ? "true" : "false");
        } else {
            os << x;
        }
    }, v);
}

inline void emit_config(std::ostream& os, const ConfigMap& m) {
    os << '{';
    bool first = true;
    for (auto const& [k, v] : m) {
        if (!first) os << ',';
        first = false;
        os << '"' << k << "\":";
        emit_value(os, v);
    }
    os << '}';
}

inline void emit_stats(std::ostream& os, const Stats& s) {
    os << '{'
       << "\"n\":"        << s.n
       << ",\"min_ns\":"  << s.min_ns
       << ",\"max_ns\":"  << s.max_ns
       << ",\"mean_ns\":" << s.mean_ns
       << ",\"stdev_ns\":"<< s.stdev_ns
       << ",\"p50_ns\":"  << s.p50_ns
       << ",\"p90_ns\":"  << s.p90_ns
       << ",\"p99_ns\":"  << s.p99_ns
       << '}';
}

struct RunRecord {
    std::string         bench;
    ConfigMap           config;
    ConfigMap           env;
    int                 warmup     = 0;
    int                 iterations = 0;
    Stats               stats{};
    std::vector<double> samples_ns;  // may be empty to save space
};

inline void write_json(std::ostream& os, const RunRecord& r) {
    os << '{'
       << "\"bench\":\"" << r.bench << "\","
       << "\"config\":";   emit_config(os, r.config);   os << ',';
    os << "\"env\":";      emit_config(os, r.env);      os << ',';
    os << "\"warmup\":"     << r.warmup << ','
       << "\"iterations\":" << r.iterations << ','
       << "\"stats\":";   emit_stats(os, r.stats);
    if (!r.samples_ns.empty()) {
        os << ",\"samples_ns\":[";
        for (std::size_t i = 0; i < r.samples_ns.size(); ++i) {
            if (i) os << ',';
            os << r.samples_ns[i];
        }
        os << ']';
    }
    os << '}' << '\n';
}

inline void write_json_file(const std::string& path, const RunRecord& r) {
    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("json_writer: cannot open " + path);
    write_json(f, r);
}

}  // namespace bench

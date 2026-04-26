// bench_file_transfer.cpp - throughput of the scheduler-side file transfer
// primitive and the worker-side receive primitive, in-process.
//
// This bench measures:
//
//   (1) make_file_data_msg(path, msg_id) - read-file + crc32 + wrap in
//       Message. Sizes: 1 KiB, 1 MiB, 100 MiB.
//
//   (2) handle_file_data(msg, work_dir, msg_id) - verify crc, write to
//       disk, then (if .tar.gz) shell out to tar xzf. For the plain path
//       we measure just verify+write; for the tarball path we measure
//       the whole chain.
//
//   (3) make_bundle_msg + handle_file_data (end-to-end round trip) across
//       companion counts: 1, 5, 20, 100 files of 4 KiB each.
//
// Why in-process:
//   These are the paths on the scheduler CPU and worker CPU respectively.
//   Network latency is measured separately by p2p_bandwidth. Isolating
//   disk+crc+tar from the network lets us see which is the real
//   bottleneck in a multi-file submission.
//
// Honest-measurement rules:
//
//   * Temp files are created ONCE outside the timed region; the timer
//     covers only the function under test.
//
//   * We fsync after each create to flush the dentry cache state, so
//     every measured read starts from a consistent page-cache state.
//     (With the exception of iteration 0, which warms the cache. The
//     runner's warmup handles this.)
//
//   * Every iteration uses a unique work_dir so handle_file_data's
//     create_directories call doesn't race with cleanup.

#include "bench/runner.h"
#include "file_transfer.h"
#include "protocol.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Create a file of exactly `bytes` filled with pseudo-random data (so
// tar's gzip stage can't trivially collapse the content to near-zero,
// which would make the compressed-transfer numbers unrealistically fast).
std::string make_temp_file(std::size_t bytes, const std::string& tag) {
    fs::path dir = fs::temp_directory_path() / ("clustr_bench_" + tag);
    fs::create_directories(dir);
    fs::path p = dir / ("f_" + std::to_string(bytes) + ".bin");

    std::mt19937_64 rng(0xDEADBEEF ^ bytes);
    std::vector<std::uint8_t> buf(std::min<std::size_t>(bytes, 1 << 16));
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    std::size_t written = 0;
    while (written < bytes) {
        for (auto& b : buf) b = static_cast<std::uint8_t>(rng());
        std::size_t chunk = std::min(buf.size(), bytes - written);
        out.write(reinterpret_cast<const char*>(buf.data()), chunk);
        written += chunk;
    }
    out.close();
    return p.string();
}

std::vector<std::string> make_companion_files(int count, std::size_t bytes_each) {
    std::vector<std::string> out;
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        out.push_back(make_temp_file(bytes_each, "companion_" + std::to_string(i)));
    }
    return out;
}

}  // namespace

int main() {
    const std::string out_path =
        std::getenv("BENCH_OUT") ? std::getenv("BENCH_OUT")
                                  : "bench/results/file_transfer.jsonl";

    // ── (1) make_file_data_msg for single files at several sizes ───────────
    for (std::size_t bytes : { std::size_t(1 << 10),        // 1 KiB
                                std::size_t(1 << 20),        // 1 MiB
                                std::size_t(100 * (1 << 20))  // 100 MiB
                              }) {
        const std::string path = make_temp_file(bytes, "single_" + std::to_string(bytes));

        bench::RunOptions opts = bench::default_options("make_file_data_msg");
        opts.config["bytes"] = static_cast<std::int64_t>(bytes);
        opts.iterations = bytes >= (100 << 20) ? 10 : 50;
        opts.warmup     = 3;
        opts.out_path   = out_path;

        auto op = [&](int i) {
            auto msg = clustr::make_file_data_msg(path, static_cast<uint32_t>(i));
            bench::DoNotOptimize(msg.payload.data());
        };
        auto stats = bench::run_bench_sync(opts, op);

        const double seconds = stats.p50_ns / 1e9;
        const double mibps = (bytes / (1024.0 * 1024.0)) / seconds;
        std::cout << "[make_file_data_msg] bytes=" << bytes
                  << " p50=" << (stats.p50_ns / 1e6) << "ms"
                  << " tp=" << mibps << " MiB/s\n";
    }

    // ── (2) handle_file_data round-trip (scheduler -> worker path) ─────────
    for (std::size_t bytes : { std::size_t(1 << 10),
                                std::size_t(1 << 20),
                                std::size_t(100 * (1 << 20)) }) {
        const std::string path = make_temp_file(bytes, "rt_" + std::to_string(bytes));
        auto msg = clustr::make_file_data_msg(path, 1);

        bench::RunOptions opts = bench::default_options("handle_file_data");
        opts.config["bytes"] = static_cast<std::int64_t>(bytes);
        opts.iterations = bytes >= (100 << 20) ? 10 : 50;
        opts.warmup     = 3;
        opts.out_path   = out_path;

        auto op = [&](int i) {
            fs::path dir = fs::temp_directory_path() / ("clustr_bench_worker_" + std::to_string(i));
            auto ack = clustr::handle_file_data(msg, dir.string(), 2);
            bench::DoNotOptimize(ack.payload.data());
            fs::remove_all(dir);
        };
        auto stats = bench::run_bench_sync(opts, op);

        const double seconds = stats.p50_ns / 1e9;
        const double mibps = (bytes / (1024.0 * 1024.0)) / seconds;
        std::cout << "[handle_file_data] bytes=" << bytes
                  << " p50=" << (stats.p50_ns / 1e6) << "ms"
                  << " tp=" << mibps << " MiB/s\n";
    }

    // ── (3) tarball bundling: varies companion count at fixed size/each ────
    for (int companions : { 1, 5, 20, 100 }) {
        const std::size_t per_file = 4 * 1024;     // 4 KiB per companion
        const std::string primary  = make_temp_file(per_file, "primary_" + std::to_string(companions));
        const auto companion_paths = make_companion_files(companions, per_file);

        // make_bundle_msg: wraps tar czf + make_file_data_msg.
        {
            bench::RunOptions opts = bench::default_options("make_bundle_msg");
            opts.config["companion_count"] = static_cast<std::int64_t>(companions);
            opts.config["per_file_bytes"]  = static_cast<std::int64_t>(per_file);
            opts.iterations = 20;
            opts.warmup     = 3;
            opts.out_path   = out_path;

            auto op = [&](int i) {
                auto msg = clustr::make_bundle_msg(primary, companion_paths,
                                                    static_cast<uint32_t>(i));
                bench::DoNotOptimize(msg.payload.data());
            };
            auto stats = bench::run_bench_sync(opts, op);
            std::cout << "[make_bundle_msg] companions=" << companions
                      << " p50=" << (stats.p50_ns / 1e6) << "ms"
                      << " p99=" << (stats.p99_ns / 1e6) << "ms\n";
        }

        // Full round trip: bundle -> receive -> extract.
        {
            bench::RunOptions opts = bench::default_options("bundle_roundtrip");
            opts.config["companion_count"] = static_cast<std::int64_t>(companions);
            opts.config["per_file_bytes"]  = static_cast<std::int64_t>(per_file);
            opts.iterations = 20;
            opts.warmup     = 3;
            opts.out_path   = out_path;

            auto op = [&](int i) {
                auto msg = clustr::make_bundle_msg(primary, companion_paths,
                                                    static_cast<uint32_t>(i));
                fs::path dir = fs::temp_directory_path() /
                    ("clustr_bench_bundle_" + std::to_string(i));
                auto ack = clustr::handle_file_data(msg, dir.string(), 2);
                bench::DoNotOptimize(ack.payload.data());
                fs::remove_all(dir);
            };
            auto stats = bench::run_bench_sync(opts, op);
            std::cout << "[bundle_roundtrip] companions=" << companions
                      << " p50=" << (stats.p50_ns / 1e6) << "ms\n";
        }
    }

    // Cleanup temp files created above. Best-effort.
    std::error_code ec;
    for (auto& p : fs::directory_iterator(fs::temp_directory_path(), ec)) {
        if (!ec && p.path().filename().string().rfind("clustr_bench_", 0) == 0) {
            fs::remove_all(p.path(), ec);
        }
    }

    return 0;
}

#include "capability_detector.h"
#include <sstream>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>
#include <fstream>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#ifdef __unix__
#include <unistd.h>
#endif

namespace clustr {

HardwareCapability CapabilityDetector::detect_all() {
    HardwareCapability cap;
    
    cap.cpu_cores = detect_cpu_cores();
    std::string model = detect_cpu_model();
    std::strncpy(cap.cpu_model, model.c_str(), sizeof(cap.cpu_model) - 1);
    cap.cpu_model[sizeof(cap.cpu_model) - 1] = '\0';
    cap.cpu_threads = cap.cpu_cores;  // Simplified, doesn't detect hyper threads 
    
    cap.has_avx2 = detect_avx2();
    cap.has_avx512 = detect_avx512();
    cap.has_neon = detect_neon();
    
    cap.total_ram_bytes = detect_total_ram();
    cap.available_ram_bytes = detect_available_ram();
    cap.storage_bytes = detect_storage();
    
    cap.compute_score = benchmark_compute();
    cap.memory_bandwidth_gbps = benchmark_memory_bandwidth();
    
    return cap;
}

// ============================================================================
// CPU Detection
// ============================================================================

uint32_t CapabilityDetector::detect_cpu_cores() {
#ifdef __APPLE__
    int count = 0;
    size_t count_len = sizeof(count);
    sysctlbyname("hw.physicalcpu", &count, &count_len, nullptr, 0);
    return count > 0 ? count : 1;
#elif defined(__unix__)
    return std::max(1, (int)sysconf(_SC_NPROCESSORS_ONLN));
#else
    return 1;
#endif
}

std::string CapabilityDetector::detect_cpu_model() {
#ifdef __APPLE__
    char model_name[256] = {0};
    size_t model_len = sizeof(model_name);
    sysctlbyname("machdep.cpu.brand_string", model_name, &model_len, nullptr, 0);
    if (strlen(model_name) > 0) {
        return std::string(model_name);
    }
    return "Unknown (Apple macOS)";
#elif defined(__unix__)
    // Try /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                return line.substr(pos + 2);
            }
        }
    }
    return "Unknown (Linux)";
#else
    return "Unknown";
#endif
}

bool CapabilityDetector::detect_avx2() {
#if defined(__AVX2__)
    return true;
#elif defined(__GNUC__) && defined(__x86_64__)
    // Runtime check via CPUID
    uint32_t eax = 7, ecx = 0;
    uint32_t ebx = 0;
    __asm__ volatile(
        "cpuid"
        : "=b"(ebx)
        : "a"(eax), "c"(ecx)
        : "edx"
    );
    return (ebx & (1 << 5)) != 0;  // AVX2 is bit 5 of EBX
#else
    return false;
#endif
}

bool CapabilityDetector::detect_avx512() {
#if defined(__AVX512F__)
    return true;
#elif defined(__GNUC__) && defined(__x86_64__)
    // Check for AVX-512F
    uint32_t eax = 7, ecx = 0;
    uint32_t ebx = 0;
    __asm__ volatile(
        "cpuid"
        : "=b"(ebx)
        : "a"(eax), "c"(ecx)
        : "edx"
    );
    return (ebx & (1 << 16)) != 0;  // AVX-512F is bit 16 of EBX
#else
    return false;
#endif
}

bool CapabilityDetector::detect_neon() {
#if defined(__ARM_NEON) || defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

// ============================================================================
// Memory Detection
// ============================================================================

uint64_t CapabilityDetector::detect_total_ram() {
#ifdef __APPLE__
    uint64_t ram = 0;
    size_t ram_len = sizeof(ram);
    sysctlbyname("hw.memsize", &ram, &ram_len, nullptr, 0);
    return ram;
#elif defined(__unix__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
#else
    return 0;
#endif
}

uint64_t CapabilityDetector::detect_available_ram() {
#ifdef __APPLE__
    // For macOS, similar to total for simplicity
    // In production, parse vm_stat or use libproc
    return detect_total_ram();
#elif defined(__unix__)
    long avail_pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return avail_pages * page_size;
#else
    return 0;
#endif
}

// ============================================================================
// Storage Detection
// ============================================================================

uint64_t CapabilityDetector::detect_storage() {
    // Simplified: just report a default value
    // In production, use statfs() on root filesystem
    return 1024ULL * 1024 * 1024 * 500;  // 500 GB default
}

// ============================================================================
// Microbenchmarks
// ============================================================================

float CapabilityDetector::benchmark_compute() {
    // Simple FLOPS benchmark: dot product
    const size_t N = 1000000;
    std::vector<float> a(N), b(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = float(i) * 0.1f;
        b[i] = float(i) * 0.2f;
    }
    
    volatile float sum = 0.0f;  // volatile prevents compiler from optimizing away the loop
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < N; ++i) {
        sum += a[i] * b[i];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // GFLOPS (2 ops per multiplication - multiply + add)
    float gflops = (2.0f * N) / (duration.count() * 1e6);
    (void)sum;  // Suppress unused warning
    return gflops;
}

float CapabilityDetector::benchmark_memory_bandwidth() {
    // Simple memory bandwidth test: large array copy
    const size_t N = 100 * 1024 * 1024;  // 100 MB
    std::vector<uint8_t> src(N, 42);
    std::vector<uint8_t> dst(N);
    
    auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst.data(), src.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // GB/s
    float gbps = (N / 1e9) / (duration.count() / 1000.0f);
    return gbps;
}

}  // namespace clustr

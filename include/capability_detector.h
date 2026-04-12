#pragma once

#include "protocol.h"
#include <string>

namespace clustr {

class CapabilityDetector {
public:
    // Detect all hardware capabilities
    static HardwareCapability detect_all();
    
private:
    // CPU detection
    static uint32_t detect_cpu_cores();
    static std::string detect_cpu_model();
    static bool detect_avx2();
    static bool detect_avx512();
    static bool detect_neon();
    
    // Memory detection
    static uint64_t detect_total_ram();
    static uint64_t detect_available_ram();
    
    // Storage detection
    static uint64_t detect_storage();
    
    // Simple microbenchmark
    static float benchmark_compute();
    static float benchmark_memory_bandwidth();
};

}  // namespace clustr

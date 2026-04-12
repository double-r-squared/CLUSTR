#include "protocol.h"
#include "capability_detector.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <iomanip>

using namespace clustr;

// Global verbose flag
bool g_verbose = false;

// Helper function to print hex dump
void print_hex_dump(const std::string& label, const std::vector<uint8_t>& data) {
    if (!g_verbose) return;
    
    std::cout << "  [HEX] " << label << " (" << data.size() << " bytes):" << std::endl;
    std::cout << "    ";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << std::hex << std::setfill('0') << std::setw(2) 
                  << static_cast<int>(data[i]) << " ";
        if ((i + 1) % 16 == 0 && i + 1 < data.size()) {
            std::cout << std::endl << "    ";
        }
    }
    std::cout << std::dec << std::endl;
}

// Helper function to print timing
void print_timing(const std::string& label, double ms) {
    if (!g_verbose) return;
    std::cout << "  [TIMING] " << label << ": " << std::fixed << std::setprecision(3) 
              << ms << " ms" << std::endl;
}

void test_hello_serialization() {
    std::cout << "Testing HELLO message serialization (binary)..." << std::endl;
    
    Message msg;
    msg.type = MessageType::HELLO;
    msg.message_id = 1;
    
    HelloPayload payload;
    std::strncpy(payload.worker_id, "test_worker_123", sizeof(payload.worker_id) - 1);
    payload.protocol_version = PROTOCOL_VERSION;
    std::strncpy(payload.worker_version, "0.1.0", sizeof(payload.worker_version) - 1);
    
    msg.payload.resize(sizeof(HelloPayload));
    std::memcpy(msg.payload.data(), &payload, sizeof(HelloPayload));
    
    if (g_verbose) {
        std::cout << "  [PAYLOAD] Worker ID: " << payload.worker_id << std::endl;
        std::cout << "  [PAYLOAD] Version: " << payload.worker_version << std::endl;
        std::cout << "  [PAYLOAD] Protocol Version: " << payload.protocol_version << std::endl;
    }
    
    auto t_start = std::chrono::high_resolution_clock::now();
    auto serialized = msg.serialize();
    auto t_end = std::chrono::high_resolution_clock::now();
    double serialize_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    std::cout << "  Serialized message size: " << serialized.size() << " bytes" << std::endl;
    print_hex_dump("Serialized Data", serialized);
    print_timing("Serialization", serialize_ms);
    
    // Deserialize
    t_start = std::chrono::high_resolution_clock::now();
    auto msg2 = Message::deserialize(serialized);
    t_end = std::chrono::high_resolution_clock::now();
    double deserialize_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    assert(msg2.type == MessageType::HELLO);
    assert(msg2.message_id == 1);
    
    HelloPayload payload2;
    std::memcpy(&payload2, msg2.payload.data(), sizeof(HelloPayload));
    assert(std::string(payload2.worker_id) == "test_worker_123");
    assert(payload2.protocol_version == PROTOCOL_VERSION);
    
    print_timing("Deserialization", deserialize_ms);
    
    if (g_verbose) {
        std::cout << "  [VERIFY] Checksum: " << std::hex << msg2.checksum << std::dec << std::endl;
    }
    
    std::cout << "  ✓ HELLO serialization test passed" << std::endl;
}

void test_capability_serialization() {
    std::cout << "Testing CAPABILITY_REPORT message serialization (with real hardware detection)..." << std::endl;
    
    // Detect actual hardware capabilities
    HardwareCapability detected_cap = CapabilityDetector::detect_all();
    
    Message msg;
    msg.type = MessageType::CAPABILITY_REPORT;
    msg.message_id = 2;
    
    CapabilityPayload payload;
    payload.capability = detected_cap;  // Use real detected capabilities
    
    msg.payload.resize(sizeof(CapabilityPayload));
    std::memcpy(msg.payload.data(), &payload, sizeof(CapabilityPayload));
    
    if (g_verbose) {
        std::cout << "  [PAYLOAD] CPU Cores: " << payload.capability.cpu_cores << std::endl;
        std::cout << "  [PAYLOAD] CPU Threads: " << payload.capability.cpu_threads << std::endl;
        std::cout << "  [PAYLOAD] CPU Model: " << payload.capability.cpu_model << std::endl;
        std::cout << "  [PAYLOAD] AVX2: " << (payload.capability.has_avx2 ? "Yes" : "No") << std::endl;
        std::cout << "  [PAYLOAD] AVX512: " << (payload.capability.has_avx512 ? "Yes" : "No") << std::endl;
        std::cout << "  [PAYLOAD] NEON: " << (payload.capability.has_neon ? "Yes" : "No") << std::endl;
        std::cout << "  [PAYLOAD] Total RAM: " << (payload.capability.total_ram_bytes / (1024*1024*1024)) << " GB" << std::endl;
        std::cout << "  [PAYLOAD] Available RAM: " << (payload.capability.available_ram_bytes / (1024*1024*1024)) << " GB" << std::endl;
        std::cout << "  [PAYLOAD] Storage: " << (payload.capability.storage_bytes / (1024*1024*1024)) << " GB" << std::endl;
        std::cout << "  [PAYLOAD] Compute Score: " << std::fixed << std::setprecision(2) << payload.capability.compute_score << " GFLOPS" << std::endl;
        std::cout << "  [PAYLOAD] Memory Bandwidth: " << std::fixed << std::setprecision(2) << payload.capability.memory_bandwidth_gbps << " GB/s" << std::endl;
    }
    
    auto t_start = std::chrono::high_resolution_clock::now();
    auto serialized = msg.serialize();
    auto t_end = std::chrono::high_resolution_clock::now();
    double serialize_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    std::cout << "  Serialized message size: " << serialized.size() << " bytes" << std::endl;
    print_hex_dump("Serialized Data", serialized);
    print_timing("Serialization", serialize_ms);
    
    // Deserialize
    t_start = std::chrono::high_resolution_clock::now();
    auto msg2 = Message::deserialize(serialized);
    t_end = std::chrono::high_resolution_clock::now();
    double deserialize_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    assert(msg2.type == MessageType::CAPABILITY_REPORT);
    assert(msg2.message_id == 2);
    
    CapabilityPayload payload2;
    std::memcpy(&payload2, msg2.payload.data(), sizeof(CapabilityPayload));
    
    // Verify that the deserialized payload matches the detected capabilities
    assert(payload2.capability.cpu_cores == detected_cap.cpu_cores);
    assert(payload2.capability.cpu_threads == detected_cap.cpu_threads);
    assert(payload2.capability.has_avx2 == detected_cap.has_avx2);
    assert(payload2.capability.has_avx512 == detected_cap.has_avx512);
    assert(payload2.capability.has_neon == detected_cap.has_neon);
    assert(payload2.capability.total_ram_bytes == detected_cap.total_ram_bytes);
    assert(payload2.capability.available_ram_bytes == detected_cap.available_ram_bytes);
    
    print_timing("Deserialization", deserialize_ms);
    
    if (g_verbose) {
        std::cout << "  [VERIFY] Checksum: " << std::hex << msg2.checksum << std::dec << std::endl;
    }
    
    std::cout << "  ✓ CAPABILITY serialization test passed" << std::endl;
}

void test_capability_detection() {
    std::cout << "Testing Hardware Capability Detection..." << std::endl;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    HardwareCapability cap = CapabilityDetector::detect_all();
    auto t_end = std::chrono::high_resolution_clock::now();
    double detect_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    // Basic sanity checks
    assert(cap.cpu_cores > 0);
    assert(cap.cpu_threads > 0);
    assert(cap.total_ram_bytes > 0);
    assert(cap.compute_score > 0);
    assert(cap.memory_bandwidth_gbps > 0);
    
    std::cout << "  CPU Cores: " << cap.cpu_cores << std::endl;
    std::cout << "  CPU Threads: " << cap.cpu_threads << std::endl;
    std::cout << "  CPU Model: " << cap.cpu_model << std::endl;
    std::cout << "  Total RAM: " << (cap.total_ram_bytes / (1024*1024*1024)) << " GB" << std::endl;
    std::cout << "  Available RAM: " << (cap.available_ram_bytes / (1024*1024*1024)) << " GB" << std::endl;
    std::cout << "  Storage: " << (cap.storage_bytes / (1024*1024*1024)) << " GB" << std::endl;
    std::cout << "  SIMD Support:";
    if (cap.has_avx2) std::cout << " AVX2";
    if (cap.has_avx512) std::cout << " AVX512";
    if (cap.has_neon) std::cout << " NEON";
    if (!cap.has_avx2 && !cap.has_avx512 && !cap.has_neon) std::cout << " None";
    std::cout << std::endl;
    std::cout << "  Compute Score: " << std::fixed << std::setprecision(2) << cap.compute_score << " GFLOPS" << std::endl;
    std::cout << "  Memory Bandwidth: " << std::fixed << std::setprecision(2) << cap.memory_bandwidth_gbps << " GB/s" << std::endl;
    
    print_timing("Detection & Benchmarking", detect_ms);
    
    // Test serialization of detected capabilities
    if (g_verbose) {
        std::cout << "  [PAYLOAD] Verifying serialization of detected capabilities..." << std::endl;
        
        Message msg;
        msg.type = MessageType::CAPABILITY_REPORT;
        msg.message_id = 99;
        
        CapabilityPayload payload;
        payload.capability = cap;
        msg.payload.resize(sizeof(CapabilityPayload));
        std::memcpy(msg.payload.data(), &payload, sizeof(CapabilityPayload));
        
        auto serialized = msg.serialize();
        auto msg_verify = Message::deserialize(serialized);
        
        CapabilityPayload payload_verify;
        std::memcpy(&payload_verify, msg_verify.payload.data(), sizeof(CapabilityPayload));
        
        assert(payload_verify.capability.cpu_cores == cap.cpu_cores);
        assert(payload_verify.capability.total_ram_bytes == cap.total_ram_bytes);
        assert(payload_verify.capability.has_avx2 == cap.has_avx2);
        
        std::cout << "    ✓ Serialization verified" << std::endl;
    }
    
    std::cout << "  ✓ Capability detection test passed" << std::endl;
}

void test_crc32() {
    std::cout << "Testing CRC32..." << std::endl;
    
    std::string test_data = "Hello, World!";
    std::vector<uint8_t> data(test_data.begin(), test_data.end());
    
    if (g_verbose) {
        std::cout << "  [INPUT] Test data: " << test_data << std::endl;
        print_hex_dump("Input Data", data);
    }
    
    auto t_start = std::chrono::high_resolution_clock::now();
    uint32_t crc = crc32(data);
    auto t_end = std::chrono::high_resolution_clock::now();
    double crc_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    uint32_t crc2 = crc32(data);
    
    assert(crc == crc2);
    assert(crc != 0);
    
    std::cout << "  CRC32: " << std::hex << crc << std::dec << std::endl;
    print_timing("CRC32 Computation", crc_ms);
    
    std::cout << "  ✓ CRC32 test passed" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            g_verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: test_protocol [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --verbose, -v   Enable verbose output (hex dumps, timing, payload details)" << std::endl;
            std::cout << "  --help, -h      Show this help message" << std::endl;
            return 0;
        }
    }
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                  CLUSTR Protocol Tests                        ║" << std::endl;
    if (g_verbose) {
        std::cout << "║                    (VERBOSE MODE ENABLED)                    ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;
    
    try {
        test_capability_detection();
        std::cout << std::endl;
        
        test_crc32();
        std::cout << std::endl;
        
        test_hello_serialization();
        std::cout << std::endl;
        
        test_capability_serialization();
        std::cout << std::endl;
        
        std::cout << "╔════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                   ✓ All tests passed!                         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}

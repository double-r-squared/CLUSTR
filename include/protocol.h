#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cstring>

namespace clustr {

// ============================================================================
// Protocol Constants
// ============================================================================

constexpr uint16_t PROTOCOL_VERSION = 1;
constexpr uint16_t DEFAULT_PORT = 9999;
constexpr size_t MAX_MESSAGE_SIZE = 10 * 1024 * 1024;  // 10 MB

// ============================================================================
// Message Types
// ============================================================================

enum class MessageType : uint8_t {
    HELLO = 0x01,              // Worker → Scheduler: "I'm here"
    HELLO_ACK = 0x02,          // Scheduler → Worker: "Welcome"
    CAPABILITY_REPORT = 0x03,  // Worker → Scheduler: Hardware info
    TASK_ASSIGN = 0x04,        // Scheduler → Worker: Take this task
    TASK_UPDATE = 0x05,        // Worker → Scheduler: Progress checkpoint
    TASK_RESULT = 0x06,        // Worker → Scheduler: Task complete with result
    HEARTBEAT = 0x07,          // Bidirectional: "Still alive"
    GOODBYE = 0x08,            // Worker → Scheduler: "Shutting down"

    // File transfer
    FILE_DATA = 0x09,          // Scheduler → Worker: Source file payload
    FILE_ACK  = 0x0A,          // Worker → Scheduler: File received + checksum ok

    // Remote execution
    EXEC_CMD    = 0x0B,        // Scheduler → Worker: Run a shell command
    EXEC_RESULT = 0x0C,        // Worker → Scheduler: stdout/stderr/exit code

    // Process management
    PROCESS_SPAWNED    = 0x0D, // Worker → Scheduler: PID after fork
    PROCESS_KILL       = 0x0E, // Scheduler → Worker: Send signal to PID
    PROCESS_STATUS_REQ = 0x0F, // Scheduler → Worker: Query resource usage
    PROCESS_STATUS_RESP= 0x10, // Worker → Scheduler: CPU%, mem, state

    // MPI peer coordination (scheduler ↔ worker, not peer ↔ peer)
    PEER_ROSTER        = 0x20, // Scheduler → Worker: rank assignment + peer addresses
    PEER_READY         = 0x21, // Worker → Scheduler: peer connections established, OS port
    RANK_DONE          = 0x22, // Worker → Scheduler: non-root rank finished (root sends TASK_RESULT)

    ERROR_MSG = 0xFF,          // Bidirectional: Something went wrong
};

// ============================================================================
// Hardware Capability Structure (Binary, packed)
// ============================================================================

#pragma pack(push, 1)

struct HardwareCapability {
    uint32_t cpu_cores;
    uint32_t cpu_threads;
    char cpu_model[256];        // Null-terminated string
    uint8_t has_avx2;
    uint8_t has_avx512;
    uint8_t has_neon;
    uint64_t total_ram_bytes;
    uint64_t available_ram_bytes;
    uint64_t storage_bytes;
    float compute_score;
    float memory_bandwidth_gbps;
};

#pragma pack(pop)

// ============================================================================
// Binary Payload Structures
// ============================================================================

#pragma pack(push, 1)

struct HelloPayload {
    char worker_id[64];        // Unique worker ID (null-terminated)
    uint16_t protocol_version;
    char worker_version[16];   // e.g. "0.1.0" (null-terminated)
};

struct HelloAckPayload {
    char scheduler_id[64];
    uint32_t task_timeout_seconds;
};

struct CapabilityPayload {
    HardwareCapability capability;
};

struct TaskAssignPayload {
    char task_id[64];
    uint32_t timeout_seconds;
    uint32_t task_data_size;
    // task_data follows immediately after (variable length)
};

struct TaskUpdatePayload {
    char task_id[64];
    uint32_t progress_percent;
    char message[256];  // Progress message (null-terminated)
};

struct TaskResultPayload {
    char task_id[64];
    uint8_t success;
    uint32_t result_data_size;
    char error_message[512];    // If success == false (null-terminated)
    // result_data follows immediately after (variable length)
};

// ---- File Transfer ----

struct FileDataPayload {
    char filename[256];         // Relative path, e.g. "main.cpp"
    uint32_t file_size;         // Total bytes of file content
    uint32_t checksum;          // CRC32 of file content
    // file content follows immediately (file_size bytes)
};

struct FileAckPayload {
    char filename[256];
    uint8_t success;            // 1 = checksum matched, 0 = mismatch
};

// ---- Remote Execution ----

struct ExecCmdPayload {
    char task_id[64];           // Which task this execution belongs to
    char command[1024];         // Shell command, e.g. "g++ -O2 main.cpp -o main"
    char working_dir[256];      // Working directory on the worker
    uint32_t timeout_seconds;   // Kill if exceeds this
    uint8_t detach;             // 0 = run sync, return EXEC_RESULT
                                // 1 = run async, return PROCESS_SPAWNED
};

struct ExecResultPayload {
    int32_t  exit_code;
    uint32_t stdout_size;       // Bytes of stdout appended after struct
    uint32_t stderr_size;       // Bytes of stderr appended after stdout
    // stdout bytes follow, then stderr bytes
};

// ---- Process Management ----

struct ProcessSpawnedPayload {
    char     task_id[64];
    uint32_t pid;
};

struct ProcessKillPayload {
    uint32_t pid;
    uint8_t  signal;            // POSIX signal: 15=SIGTERM, 9=SIGKILL
};

struct ProcessStatusReqPayload {
    uint32_t pid;
};

struct ProcessStatusRespPayload {
    uint32_t pid;
    float    cpu_percent;
    uint64_t mem_bytes;
    uint8_t  state;             // Cast to ProcessState
};

// ---- MPI Peer Coordination ----
//
// Layout: Scheduler → Worker (PEER_ROSTER)
//   Tells a worker its rank within the job group, total group size,
//   and the IP + OS-assigned peer port of every other rank.
//   Worker uses this to open N-1 peer TCP connections before execution.
//
// Layout: Worker → Scheduler (PEER_READY)
//   Sent once all peer connections are established.
//   Carries the OS-assigned port this worker is listening on so the
//   scheduler can include it in rosters sent to other ranks.
//
// Layout: Worker → Scheduler (RANK_DONE)
//   Non-root ranks send this instead of TASK_RESULT when the job exits.
//   Lets the scheduler detect partial failures in a group.

constexpr uint32_t MAX_MPI_RANKS = 64;

struct PeerEntry {
    uint32_t rank;
    char     ip[64];            // Null-terminated IPv4/IPv6 string
    uint16_t peer_port;         // OS-assigned peer listener port
};

struct PeerRosterPayload {
    char     job_id[64];        // Which job this roster belongs to
    uint32_t my_rank;           // This worker's rank within the group
    uint32_t num_ranks;         // Total group size
    PeerEntry peers[MAX_MPI_RANKS]; // All peers (entry my_rank is self — skip)
};

struct PeerReadyPayload {
    char     job_id[64];
    uint32_t my_rank;
    uint16_t peer_port;         // The port this worker is listening on for peers
};

struct RankDonePayload {
    char     job_id[64];
    uint32_t rank;
    int32_t  exit_code;
    uint32_t output_size;   // bytes of stdout/stderr appended after this struct
};

#pragma pack(pop)

// ---- Process State (outside pack block — enum cannot be packed) ----

enum class ProcessState : uint8_t {
    RUNNING  = 0,
    SLEEPING = 1,
    STOPPED  = 2,
    ZOMBIE   = 3,
    DEAD     = 4,
    UNKNOWN  = 5,
};

// ============================================================================
// Message Frame Structure
// ============================================================================
//
// Binary layout:
// [0-3]   : Total message size (excluding this 4-byte field)
// [4-5]   : Protocol version (u16, big-endian)
// [6]     : Message type (u8)
// [7-10]  : Message ID (u32, big-endian)
// [11-14] : CRC32 checksum (computed over payload only)
// [15-N]  : Payload (binary, depends on message type)
//

struct Message {
    uint16_t protocol_version;
    MessageType type;
    uint32_t message_id;
    std::vector<uint8_t> payload;
    uint32_t checksum;  // Computed during serialization
    
    // Serialize to binary
    std::vector<uint8_t> serialize() const;
    
    // Deserialize from binary
    static Message deserialize(const std::vector<uint8_t>& data);
};

// ============================================================================
// Utility Functions
// ============================================================================

uint32_t crc32(const std::vector<uint8_t>& data);
std::string generate_worker_id();

}  // namespace clustr


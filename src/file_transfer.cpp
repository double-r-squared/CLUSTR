#include "file_transfer.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace clustr {

// ============================================================================
// Scheduler Side
// ============================================================================

Message make_file_data_msg(const std::string& filepath, uint32_t msg_id) {
    // Read entire file into memory
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open source file: " + filepath);

    std::vector<uint8_t> file_bytes(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    // Build fixed header
    FileDataPayload header{};
    std::string filename = std::filesystem::path(filepath).filename().string();
    std::strncpy(header.filename, filename.c_str(), sizeof(header.filename) - 1);
    header.file_size = static_cast<uint32_t>(file_bytes.size());
    header.checksum  = crc32(file_bytes);

    // Payload = header + raw file bytes
    Message msg;
    msg.type             = MessageType::FILE_DATA;
    msg.message_id       = msg_id;
    msg.protocol_version = PROTOCOL_VERSION;
    msg.payload.resize(sizeof(FileDataPayload) + file_bytes.size());
    std::memcpy(msg.payload.data(), &header, sizeof(FileDataPayload));
    std::memcpy(msg.payload.data() + sizeof(FileDataPayload),
                file_bytes.data(), file_bytes.size());

    return msg;
}

// ============================================================================
// Worker Side
// ============================================================================

Message handle_file_data(const Message& msg, const std::string& work_dir, uint32_t msg_id) {
    if (msg.payload.size() < sizeof(FileDataPayload))
        throw std::runtime_error("FILE_DATA payload too small");

    FileDataPayload header{};
    std::memcpy(&header, msg.payload.data(), sizeof(FileDataPayload));

    if (msg.payload.size() < sizeof(FileDataPayload) + header.file_size)
        throw std::runtime_error("FILE_DATA payload truncated");

    // Extract file bytes
    const uint8_t* data_start = msg.payload.data() + sizeof(FileDataPayload);
    std::vector<uint8_t> file_bytes(data_start, data_start + header.file_size);

    // Verify CRC32 before writing anything to disk
    uint32_t computed = crc32(file_bytes);
    bool ok = (computed == header.checksum);

    if (ok) {
        std::filesystem::create_directories(work_dir);
        std::string dest = work_dir + "/" + std::string(header.filename);
        std::ofstream out(dest, std::ios::binary | std::ios::trunc);
        if (!out.is_open())
            throw std::runtime_error("Cannot write file: " + dest);
        out.write(reinterpret_cast<const char*>(file_bytes.data()), file_bytes.size());
    }

    // Build FILE_ACK
    FileAckPayload ack{};
    std::strncpy(ack.filename, header.filename, sizeof(ack.filename) - 1);
    ack.success = ok ? 1 : 0;

    Message ack_msg;
    ack_msg.type             = MessageType::FILE_ACK;
    ack_msg.message_id       = msg_id;
    ack_msg.protocol_version = PROTOCOL_VERSION;
    ack_msg.payload.resize(sizeof(FileAckPayload));
    std::memcpy(ack_msg.payload.data(), &ack, sizeof(FileAckPayload));

    return ack_msg;
}

}  // namespace clustr

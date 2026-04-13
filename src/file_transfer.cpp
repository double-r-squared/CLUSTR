#include "file_transfer.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace clustr {

// ============================================================================
// Helpers
// ============================================================================

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

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

Message make_bundle_msg(const std::string& source_file,
                        const std::vector<std::string>& companion_files,
                        uint32_t msg_id) {
    namespace fs = std::filesystem;

    // Archive name: <source_stem>_bundle.tar.gz
    std::string stem = fs::path(source_file).stem().string();
    std::string archive_name = stem + "_bundle.tar.gz";
    std::string tmp_tar = "/tmp/" + archive_name;

    // Build tar command — each file added with -C <abs_dir> <basename> so the
    // archive contains flat basenames, no directory prefixes.
    // Paths must be absolute because each -C is relative to the previous one.
    auto abs_dir = [](const std::string& path) -> std::string {
        return fs::absolute(fs::path(path).parent_path()).string();
    };

    std::string cmd = "tar czf " + tmp_tar;
    cmd += " -C " + abs_dir(source_file)
         + " " + fs::path(source_file).filename().string();
    for (const auto& cf : companion_files) {
        cmd += " -C " + abs_dir(cf)
             + " " + fs::path(cf).filename().string();
    }

    int rc = std::system(cmd.c_str());
    if (rc != 0)
        throw std::runtime_error("make_bundle_msg: tar failed (exit " +
                                 std::to_string(rc) + "): " + cmd);

    // Reuse make_file_data_msg to build the actual MESSAGE
    Message msg_out = make_file_data_msg(tmp_tar, msg_id);

    // Clean up temp tarball
    std::remove(tmp_tar.c_str());
    return msg_out;
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
        out.close();

        // If this is a tarball, extract it into work_dir and remove the archive
        if (ends_with(dest, ".tar.gz")) {
            std::string cmd = "tar xzf " + dest + " -C " + work_dir;
            int rc = std::system(cmd.c_str());
            if (rc != 0)
                throw std::runtime_error("handle_file_data: tar extract failed: " + cmd);
            std::remove(dest.c_str());
        }
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

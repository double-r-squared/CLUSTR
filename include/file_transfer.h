#pragma once

#include "protocol.h"
#include <string>

namespace clustr {

// ============================================================================
// File Transfer — Scheduler Side
// ============================================================================

// Read a file from disk and build a FILE_DATA message.
// The filename stored in the payload is the basename only (e.g. "job.cpp"),
// not the full path. The file bytes and CRC32 are appended to the payload.
// Throws std::runtime_error if the file cannot be opened or read.
Message make_file_data_msg(const std::string& filepath, uint32_t msg_id);

// ============================================================================
// File Transfer — Worker Side
// ============================================================================

// Receive a FILE_DATA message, write the file into work_dir, and verify CRC32.
// Returns a FILE_ACK message with success=1 on checksum match, success=0 on mismatch.
// Throws std::runtime_error if the payload is malformed or the file cannot be written.
Message handle_file_data(const Message& msg, const std::string& work_dir, uint32_t msg_id);

}  // namespace clustr

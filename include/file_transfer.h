#pragma once

#include "protocol.h"
#include <string>
#include <vector>

namespace clustr {

// ============================================================================
// File Transfer — Scheduler Side
// ============================================================================

// Read a file from disk and build a FILE_DATA message.
// The filename stored in the payload is the basename only (e.g. "job.cpp"),
// not the full path. The file bytes and CRC32 are appended to the payload.
// Throws std::runtime_error if the file cannot be opened or read.
Message make_file_data_msg(const std::string& filepath, uint32_t msg_id);

// Bundle multiple files into a single .tar.gz and build a FILE_DATA message.
// The source_file and all companion_files are packed into one archive whose
// basename is "<source_stem>_bundle.tar.gz".  The worker detects the .tar.gz
// extension on receipt and extracts it, so the protocol stays one-file/one-ACK.
// Throws std::runtime_error if tar fails or any file cannot be read.
Message make_bundle_msg(const std::string& source_file,
                        const std::vector<std::string>& companion_files,
                        uint32_t msg_id);

// ============================================================================
// File Transfer — Worker Side
// ============================================================================

// Receive a FILE_DATA message, write the file into work_dir, and verify CRC32.
// If the file is a .tar.gz, it is automatically extracted into work_dir after
// writing, then the archive is removed.
// Returns a FILE_ACK message with success=1 on checksum match, success=0 on mismatch.
// Throws std::runtime_error if the payload is malformed or the file cannot be written.
Message handle_file_data(const Message& msg, const std::string& work_dir, uint32_t msg_id);

}  // namespace clustr

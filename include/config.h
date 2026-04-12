#pragma once

#include "protocol.h"
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>

namespace clustr {

// ============================================================================
// Config — loads key=value pairs from a config file
//
// Example clustr.conf:
//   scheduler_ip=192.168.1.5
//   scheduler_port=9999
//   worker_id=worker_a1b2c3d4    # optional — auto-generated if absent
//   work_dir=/tmp/clustr          # directory for received source files
// ============================================================================

class Config {
public:
    // Load from file. Throws std::runtime_error if file cannot be opened.
    static Config from_file(const std::string& path) {
        Config cfg;
        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error("Cannot open config file: " + path);

        std::string line;
        while (std::getline(file, line)) {
            // Strip comments and whitespace
            auto comment = line.find('#');
            if (comment != std::string::npos)
                line = line.substr(0, comment);

            auto eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key   = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));

            if (!key.empty() && !value.empty())
                cfg.entries_[key] = value;
        }
        return cfg;
    }

    // Required string value — throws if missing.
    std::string require(const std::string& key) const {
        auto it = entries_.find(key);
        if (it == entries_.end())
            throw std::runtime_error("Config missing required key: " + key);
        return it->second;
    }

    // Optional string value with default.
    std::string get(const std::string& key, const std::string& default_val = "") const {
        auto it = entries_.find(key);
        return (it != entries_.end()) ? it->second : default_val;
    }

    // Optional uint16 value with default.
    uint16_t get_uint16(const std::string& key, uint16_t default_val = 0) const {
        auto it = entries_.find(key);
        if (it == entries_.end()) return default_val;
        return static_cast<uint16_t>(std::stoul(it->second));
    }

    // ---- Typed accessors for known keys ----

    std::string scheduler_ip() const {
        return require("scheduler_ip");
    }

    uint16_t scheduler_port() const {
        return get_uint16("scheduler_port", DEFAULT_PORT);
    }

    // Returns stored worker_id or generates one if absent.
    std::string worker_id() const {
        return get("worker_id", generate_worker_id());
    }

    // Directory where received source files and compiled binaries are stored.
    std::string work_dir() const {
        return get("work_dir", "/tmp/clustr");
    }

    // ---- Deploy credentials (initial bootstrap only) ----

    // Username on the remote worker machine (e.g. "ubuntu", "pi")
    std::string deploy_user() const { return require("deploy_user"); }

    // IP or hostname of the remote worker machine
    std::string deploy_host() const { return require("deploy_host"); }

    // Path to SSH private key. Defaults to ~/.ssh/id_rsa if not set.
    std::string ssh_key_path() const {
        return get("ssh_key_path", "~/.ssh/id_rsa");
    }

    // Remote install path for the worker binary.
    std::string remote_install_path() const {
        return get("remote_install_path", "/usr/local/bin/worker");
    }

private:
    std::unordered_map<std::string, std::string> entries_;

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end   = s.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        return s.substr(start, end - start + 1);
    }
};

}  // namespace clustr

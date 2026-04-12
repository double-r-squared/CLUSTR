#pragma once

#include "protocol.h"
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace clustr {

// ============================================================================
// KnownWorker — one worker entry from system.conf
// ============================================================================

struct KnownWorker {
    std::string name;                                      // Section header, e.g. "maturin"
    std::string deploy_host;                               // Remote IP (not a hostname)
    std::string deploy_user;
    std::string ssh_key_path        = "~/.ssh/id_rsa";
    std::string remote_install_path = "/var/tmp/clustr_worker";
    std::string work_dir            = "/tmp/clustr";
    std::string scheduler_ip;                              // Written to worker's remote conf
    uint16_t    scheduler_port      = DEFAULT_PORT;
};

// ============================================================================
// SystemConf — single source of truth for the entire cluster.
//
// system.conf format (INI-style):
//
//   # Global — scheduler identity and shared defaults for all workers
//   scheduler_ip=192.168.1.5        # this machine's IP, written to remote workers
//   scheduler_port=9999
//   ssh_key_path=~/.ssh/id_rsa
//   remote_install_path=/usr/local/bin/worker
//   work_dir=/tmp/clustr
//
//   [maturin]
//   deploy_host=192.168.1.10        # must be an IP address
//   deploy_user=nate
//
//   [neuromancer]
//   deploy_host=192.168.1.11
//   deploy_user=nate
//   ssh_key_path=~/.ssh/id_rsa_neu  # override global for this worker only
//
// deploy_host and deploy_user are required in each worker section.
// All other fields inherit from the global section if not overridden.
// ============================================================================

class SystemConf {
public:
    // Global scheduler settings
    std::string  scheduler_ip;
    uint16_t     scheduler_port        = DEFAULT_PORT;
    std::string  ssh_key_path          = "~/.ssh/id_rsa";
    std::string  remote_install_path   = "/var/tmp/clustr_worker";
    std::string  work_dir              = "/tmp/clustr";

    // Known worker nodes
    std::vector<KnownWorker> workers;

    // Load from file.  Returns an empty SystemConf if the file does not exist.
    static SystemConf load(const std::string& path) {
        SystemConf conf;
        std::ifstream file(path);
        if (!file.is_open()) return conf;

        KnownWorker* current = nullptr;
        std::string  line;

        while (std::getline(file, line)) {
            auto comment = line.find('#');
            if (comment != std::string::npos) line = line.substr(0, comment);
            line = trim(line);
            if (line.empty()) continue;

            // Section header
            if (line.front() == '[' && line.back() == ']') {
                std::string section = trim(line.substr(1, line.size() - 2));
                if (section == "global") {
                    current = nullptr;
                } else {
                    KnownWorker kw;
                    kw.name                = section;
                    // Inherit globals
                    kw.scheduler_ip        = conf.scheduler_ip;
                    kw.scheduler_port      = conf.scheduler_port;
                    kw.ssh_key_path        = conf.ssh_key_path;
                    kw.remote_install_path = conf.remote_install_path;
                    kw.work_dir            = conf.work_dir;
                    conf.workers.push_back(std::move(kw));
                    current = &conf.workers.back();
                }
                continue;
            }

            // key=value
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            if (key.empty() || val.empty()) continue;

            if (current == nullptr) {
                // Global section
                if      (key == "scheduler_ip")        conf.scheduler_ip = val;
                else if (key == "scheduler_port")      conf.scheduler_port = parse_port(val);
                else if (key == "ssh_key_path")        conf.ssh_key_path = val;
                else if (key == "remote_install_path") conf.remote_install_path = val;
                else if (key == "work_dir")            conf.work_dir = val;
            } else {
                // Per-worker section
                if      (key == "deploy_host")         current->deploy_host = val;
                else if (key == "deploy_user")         current->deploy_user = val;
                else if (key == "ssh_key_path")        current->ssh_key_path = val;
                else if (key == "remote_install_path") current->remote_install_path = val;
                else if (key == "work_dir")            current->work_dir = val;
                else if (key == "scheduler_ip")        current->scheduler_ip = val;
                else if (key == "scheduler_port")      current->scheduler_port = parse_port(val);
            }
        }
        return conf;
    }

    // Find by TCP source IP (used when a worker connects).
    const KnownWorker* find_by_ip(const std::string& ip) const {
        for (const auto& w : workers)
            if (w.deploy_host == ip) return &w;
        return nullptr;
    }

    // Find by display name (section header).
    const KnownWorker* find_by_name(const std::string& name) const {
        for (const auto& w : workers)
            if (w.name == name) return &w;
        return nullptr;
    }

private:
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end   = s.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        return s.substr(start, end - start + 1);
    }

    static uint16_t parse_port(const std::string& s) {
        try { return static_cast<uint16_t>(std::stoul(s)); }
        catch (...) { return DEFAULT_PORT; }
    }
};

}  // namespace clustr

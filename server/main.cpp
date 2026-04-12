#include "scheduler.h"
#include "system_conf.h"
#include "tui.h"
#include <thread>

using namespace clustr;

int main(int argc, char* argv[]) {
    // Args: [system.conf]
    std::string conf_path = "system.conf";
    if (argc > 1) conf_path = argv[1];

    SystemConf conf = SystemConf::load(conf_path);
    uint16_t   port = conf.scheduler_port;

    TuiState ui;
    {
        std::lock_guard<std::mutex> lk(ui.mutex);
        ui.port          = port;
        ui.known_workers = conf.workers;
        ui.add_log("CLUSTR Scheduler  port=" + std::to_string(port));
        if (!conf.workers.empty())
            ui.add_log("Loaded " + std::to_string(conf.workers.size()) +
                       " known worker(s) from " + conf_path);
        else
            ui.add_log("No workers found in " + conf_path);
    }

    asio::io_context io;

    Scheduler scheduler(io, port, ui, std::move(conf));
    scheduler.start();

    std::thread net([&io]() { io.run(); });

    TuiCommands cmds;
    cmds.submit_job = [&](const std::string& pin) {
        asio::post(io, [&scheduler, pin]() { scheduler.submit_job(pin); });
    };
    cmds.force_dispatch = [&](const std::string& jid, const std::string& wid) {
        asio::post(io, [&scheduler, jid, wid]() { scheduler.force_dispatch(jid, wid); });
    };
    cmds.kill_job = [&](const std::string& wid) {
        asio::post(io, [&scheduler, wid]() { scheduler.kill_job(wid); });
    };
    cmds.request_status = [&](const std::string& wid) {
        asio::post(io, [&scheduler, wid]() { scheduler.request_status(wid); });
    };
    cmds.cancel_job = [&](const std::string& jid) {
        asio::post(io, [&scheduler, jid]() { scheduler.cancel_job(jid); });
    };
    cmds.set_strategy = [&](const std::string& name) {
        asio::post(io, [&scheduler, name]() { scheduler.set_strategy(name); });
    };
    cmds.deploy_worker = [&](const std::string& name) {
        scheduler.deploy_worker(name);
    };
    cmds.reload_config = [&]() {
        asio::post(io, [&scheduler, conf_path]() { scheduler.reload_config(conf_path); });
    };
    cmds.quit = [&io]() { io.stop(); };

    Tui tui(ui, std::move(cmds), conf_path);
    tui.run();

    io.stop();
    net.join();
    return 0;
}

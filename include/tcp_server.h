#pragma once

#include "protocol.h"
#include <asio.hpp>
#include <memory>
#include <functional>
#include <map>

namespace clustr {

using asio::ip::tcp;

// ============================================================================
// Base Connection Handler
// ============================================================================

class Connection : public std::enable_shared_from_this<Connection> {
public:
    using Ptr = std::shared_ptr<Connection>;
    
    explicit Connection(asio::io_context& io_context)
        : socket_(io_context), read_buffer_(65536) {}
    
    static Ptr create(asio::io_context& io_context) {
        return std::make_shared<Connection>(io_context);
    }
    
    tcp::socket& socket() {
        return socket_;
    }
    
    void start();
    void send_message(const Message& msg);
    
    // Callback when message received
    std::function<void(Ptr, const Message&)> on_message;
    std::function<void(Ptr)> on_disconnect;
    
private:
    void async_read_header();
    void async_read_payload(uint32_t payload_size);
    
    tcp::socket socket_;
    std::vector<uint8_t> read_buffer_;
    std::vector<uint8_t> write_buffer_;
};

// ============================================================================
// TCP Server
// ============================================================================

class TcpServer {
public:
    TcpServer(asio::io_context& io_context, uint16_t port = DEFAULT_PORT)
        : io_context_(io_context),
          acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {}
    
    void start();
    void stop();
    
    // Callback when new connection arrives
    std::function<void(Connection::Ptr)> on_connection;
    
private:
    void async_accept();
    
    asio::io_context& io_context_;
    tcp::acceptor acceptor_;
};

// ============================================================================
// TCP Client
// ============================================================================

class TcpClient {
public:
    TcpClient(asio::io_context& io_context)
        : io_context_(io_context),
          connection_(Connection::create(io_context)) {}
    
    void connect(const std::string& host, uint16_t port, 
                 std::function<void(bool)> on_connected);
    
    void send_message(const Message& msg);
    void disconnect();
    
    // Callback when message received
    std::function<void(const Message&)> on_message;
    
private:
    asio::io_context& io_context_;
    Connection::Ptr connection_;
};

}  // namespace clustr

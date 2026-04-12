#include "tcp_server.h"
#include <iostream>

namespace clustr {

// ============================================================================
// Connection Implementation
// ============================================================================

void Connection::start() {
    async_read_header();
}

void Connection::send_message(const Message& msg) {
    auto data = msg.serialize();
    asio::async_write(
        socket_,
        asio::buffer(data),
        [this, data](const asio::error_code& ec, std::size_t /*bytes_transferred*/) {
            if (ec) {
                std::cerr << "Write error: " << ec.message() << std::endl;
                if (on_disconnect) {
                    on_disconnect(shared_from_this());
                }
            }
        }
    );
}

void Connection::async_read_header() {
    // Read 4-byte message size
    asio::async_read(
        socket_,
        asio::buffer(read_buffer_, 4),
        [this](const asio::error_code& ec, std::size_t /*bytes_transferred*/) {
            if (ec) {
                std::cerr << "Read header error: " << ec.message() << std::endl;
                if (on_disconnect) {
                    on_disconnect(shared_from_this());
                }
                return;
            }
            
            // Parse size (big-endian)
            uint32_t payload_size = 
                ((read_buffer_[0] << 24) |
                 (read_buffer_[1] << 16) |
                 (read_buffer_[2] << 8) |
                 read_buffer_[3]);
            
            if (payload_size > MAX_MESSAGE_SIZE) {
                std::cerr << "Message too large: " << payload_size << std::endl;
                if (on_disconnect) {
                    on_disconnect(shared_from_this());
                }
                return;
            }
            
            async_read_payload(payload_size);
        }
    );
}

void Connection::async_read_payload(uint32_t payload_size) {
    asio::async_read(
        socket_,
        asio::buffer(read_buffer_, payload_size),
        [this, payload_size](const asio::error_code& ec, std::size_t /*bytes_transferred*/) {
            if (ec) {
                std::cerr << "Read payload error: " << ec.message() << std::endl;
                if (on_disconnect) {
                    on_disconnect(shared_from_this());
                }
                return;
            }
            
            try {
                // Reconstruct full message: prepend size + payload
                std::vector<uint8_t> full_msg;
                full_msg.push_back((payload_size >> 24) & 0xff);
                full_msg.push_back((payload_size >> 16) & 0xff);
                full_msg.push_back((payload_size >> 8) & 0xff);
                full_msg.push_back(payload_size & 0xff);
                full_msg.insert(full_msg.end(), 
                               read_buffer_.begin(), 
                               read_buffer_.begin() + payload_size);
                
                Message msg = Message::deserialize(full_msg);
                
                if (on_message) {
                    on_message(shared_from_this(), msg);
                }
                
                // Continue reading next message
                async_read_header();
                
            } catch (const std::exception& e) {
                std::cerr << "Message deserialization error: " << e.what() << std::endl;
                if (on_disconnect) {
                    on_disconnect(shared_from_this());
                }
            }
        }
    );
}

// ============================================================================
// TCP Server Implementation
// ============================================================================

void TcpServer::start() {
    async_accept();
}

void TcpServer::stop() {
    acceptor_.close();
}

void TcpServer::async_accept() {
    auto new_conn = Connection::create(io_context_);
    
    acceptor_.async_accept(
        new_conn->socket(),
        [this, new_conn](const asio::error_code& ec) {
            if (!ec) {
                std::cout << "New connection from " 
                         << new_conn->socket().remote_endpoint() << std::endl;
                
                new_conn->start();
                
                if (on_connection) {
                    on_connection(new_conn);
                }
            } else {
                std::cerr << "Accept error: " << ec.message() << std::endl;
            }
            
            // Continue accepting
            async_accept();
        }
    );
}

// ============================================================================
// TCP Client Implementation
// ============================================================================

void TcpClient::connect(const std::string& host, uint16_t port,
                       std::function<void(bool)> on_connected) {
    tcp::resolver resolver(io_context_);
    auto endpoints = resolver.resolve(host, std::to_string(port));
    
    asio::async_connect(
        connection_->socket(),
        endpoints,
        [this, on_connected](const asio::error_code& ec, 
                           const tcp::endpoint& ep) {
            if (!ec) {
                std::cout << "Connected to " << ep << std::endl;
                connection_->on_message = [this](Connection::Ptr /*conn*/, const Message& msg) {
                    if (on_message) {
                        on_message(msg);
                    }
                };
                
                connection_->start();
                on_connected(true);
            } else {
                std::cerr << "Connect error: " << ec.message() << std::endl;
                on_connected(false);
            }
        }
    );
}

void TcpClient::send_message(const Message& msg) {
    connection_->send_message(msg);
}

void TcpClient::disconnect() {
    connection_->socket().close();
}

}  // namespace clustr

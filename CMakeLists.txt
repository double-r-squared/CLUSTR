cmake_minimum_required(VERSION 3.20)
project(clustr CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# ============================================================================
# Dependencies (Auto-downloaded via FetchContent if not found)
# ============================================================================

include(FetchContent)

# Set modern FetchContent policy
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# ASIO (header-only networking library)
find_package(asio CONFIG QUIET)
if(NOT asio_FOUND)
    message(STATUS "ASIO not found locally, downloading from GitHub...")
    FetchContent_Declare(asio
        URL https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-28-0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    FetchContent_MakeAvailable(asio)
    set(asio_INCLUDE_DIR "${asio_SOURCE_DIR}/asio/include")
else()
    message(STATUS "Using system-installed ASIO")
endif()

# ncurses (required for scheduler TUI)
find_package(Curses REQUIRED)

# Optional: spdlog for future logging
find_package(spdlog CONFIG QUIET)

# ============================================================================
# Library: libclustr
# ============================================================================

add_library(clustr STATIC
    src/protocol.cpp
    src/tcp_server.cpp
    src/capability_detector.cpp
    src/file_transfer.cpp
    src/remote_exec.cpp
    src/process_monitor.cpp
)

target_include_directories(clustr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add ASIO includes
if(asio_FOUND)
    target_link_libraries(clustr PUBLIC asio::asio)
else()
    target_include_directories(clustr PUBLIC ${asio_INCLUDE_DIR})
    # ASIO is header-only, no linking needed
endif()

# Platform-specific linking
if(APPLE)
    target_link_libraries(clustr PUBLIC "-framework CoreFoundation")
    # proc_pidinfo() lives in libproc (ships with macOS, no install needed)
    target_link_libraries(clustr PUBLIC "-lproc")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(clustr PUBLIC pthread)
endif()

# ============================================================================
# Executable: scheduler (server)
# ============================================================================

add_executable(scheduler
    server/main.cpp
    src/scheduler.cpp
    src/tui/tui_state.cpp
    src/tui/tui_draw.cpp
    src/tui/tui_input.cpp
    src/tui/tui_dialogs.cpp
)
target_include_directories(scheduler PRIVATE
    ${CURSES_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tui   # gives tui_impl.h visibility
)
target_link_libraries(scheduler PRIVATE clustr ${CURSES_LIBRARIES})

# ============================================================================
# Executable: worker (client)
# ============================================================================

add_executable(worker client/client.cpp)
target_link_libraries(worker PRIVATE clustr)

# ============================================================================
# Optional: Tests
# ============================================================================

enable_testing()

# A simple test binary for protocol serialization
add_executable(test_protocol 
    tests/test_protocol.cpp
)
target_link_libraries(test_protocol PRIVATE clustr)
add_test(NAME ProtocolTest COMMAND test_protocol)

# ============================================================================
# Installation
# ============================================================================
#
# Default install prefix is /usr/local, so binaries land in /usr/local/bin.
# To install:
#   cmake --build build/
#   sudo cmake --install build/
#
# To override the prefix:
#   sudo cmake --install build/ --prefix /usr
#
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Install prefix" FORCE)
endif()

install(TARGETS scheduler worker
    DESTINATION bin           # → /usr/local/bin/scheduler and /usr/local/bin/worker
)

install(DIRECTORY include/
    DESTINATION include/clustr
)

install(FILES system.conf
    DESTINATION etc/clustr    # → /usr/local/etc/clustr/system.conf (example)
)

# ============================================================================
# Print Summary
# ============================================================================

message(STATUS "")
message(STATUS "  CLUSTR Build Configuration")
message(STATUS "  ==========================")
message(STATUS "  CXX Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform:         ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")
message(STATUS "  Dependencies:")
if(asio_FOUND)
    message(STATUS "    • ASIO:           Found (system)")
else()
    message(STATUS "    • ASIO:           Will download from GitHub")
endif()
message(STATUS "")
message(STATUS "  Targets:")
message(STATUS "    - scheduler (Server/Scheduler)")
message(STATUS "    - worker (Client/Worker)")
message(STATUS "    - test_protocol (Protocol tests)")
message(STATUS "")


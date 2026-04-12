#pragma once

// transport_common.hpp — shared transport infrastructure.
//
// This header is included by BOTH transport implementations
// (transport_zero_copy.hpp and transport_pack_unpack.hpp) and contains code
// that must behave identically across them:
//
//   - Runtime IOV_MAX query via sysconf(_SC_IOV_MAX). POSIX explicitly warns
//     against hardcoding a constant here; the kernel value can vary, and
//     exceeding it returns -1/EINVAL from writev/readv.
//
//   - The (ptr, len) convenience form of send_raw, which delegates to the
//     buffer-sequence form that each transport provides.
//
// Do not define transport-divergent logic here. If a change affects only one
// implementation, it belongs in the corresponding split header.

#include <cstddef>
#include <unistd.h>  // sysconf, _SC_IOV_MAX

namespace clustr {

// POSIX guarantees _POSIX_IOV_MAX >= 16. sysconf returns -1 on failure or if
// the limit is indeterminate - we conservatively fall back to that floor,
// never to a guess like 1024.
inline std::size_t query_iov_max() noexcept {
    long cap = ::sysconf(_SC_IOV_MAX);
    return (cap > 0) ? static_cast<std::size_t>(cap) : 16u;
}

}  // namespace clustr

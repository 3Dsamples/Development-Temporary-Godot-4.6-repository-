//File 0500 : xtensor-io/xio_config.hpp
//Configuration header: version, exception macros, C++17 support, and platform detection for xtensor-io.
#ifndef XTENSOR_IO_CONFIG_HPP
#define XTENSOR_IO_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#define XTENSOR_IO_VERSION_MAJOR 0
#define XTENSOR_IO_VERSION_MINOR 3
#define XTENSOR_IO_VERSION_PATCH 0

#if (!defined(__cpp_exceptions) && !defined(__EXCEPTIONS) && !defined(_CPPUNWIND))
    #define XTENSOR_IO_DISABLE_EXCEPTIONS
#endif

#if defined(XTENSOR_IO_DISABLE_EXCEPTIONS)
    #include <iostream>
    #include <cstdlib>
    #define XTENSOR_IO_THROW(exception, msg) \
        do { \
            std::cerr << msg << std::endl; \
            std::abort(); \
        } while(0)
#else
    #define XTENSOR_IO_THROW(exception, msg) throw exception(msg)
#endif

namespace xt {
namespace io {

    // Default buffer size for binary I/O
    constexpr std::size_t default_io_buffer_size = 65536;

    // Alignment for SIMD I/O operations
    constexpr std::size_t simd_io_alignment = 64;

    // Magic bytes for format detection
    namespace magic {
        constexpr unsigned char npy_magic[6] = {0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59}; // "\x93NUMPY"
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_CONFIG_HPP
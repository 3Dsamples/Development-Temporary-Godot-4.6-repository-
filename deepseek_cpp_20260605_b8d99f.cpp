//File 0523 : xtensor-io/xtensor_io_config.hpp.in
//CMake‑generated configuration header for xtensor‑io: version numbers, optional dependency toggles, and platform detection.
#ifndef XTENSOR_IO_CONFIG_HPP_IN
#define XTENSOR_IO_CONFIG_HPP_IN

#define XTENSOR_IO_VERSION_MAJOR @xtensor_io_VERSION_MAJOR@
#define XTENSOR_IO_VERSION_MINOR @xtensor_io_VERSION_MINOR@
#define XTENSOR_IO_VERSION_PATCH @xtensor_io_VERSION_PATCH@

#cmakedefine XTENSOR_IO_USE_FREEIMAGE
#cmakedefine XTENSOR_IO_USE_GDAL
#cmakedefine XTENSOR_IO_USE_HIGHFIVE
#cmakedefine XTENSOR_IO_USE_BLOSC
#cmakedefine XTENSOR_IO_USE_ZLIB
#cmakedefine XTENSOR_IO_USE_AWS_SDK
#cmakedefine XTENSOR_IO_USE_GOOGLE_CLOUD_CPP
#cmakedefine XTENSOR_IO_USE_SNDFILE

#cmakedefine XTENSOR_IO_HAS_EXCEPTIONS
#cmakedefine XTENSOR_IO_HAS_CXX17

#include <cstddef>
#include <cstdint>

namespace xt {
namespace io {

    constexpr std::size_t default_io_buffer_size = 65536;
    constexpr std::size_t simd_io_alignment = 64;

    namespace magic {
        constexpr unsigned char npy_magic[6] = {0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59};
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_CONFIG_HPP_IN
//File 0600 : xtensor-signal/xtensor_signal_config.hpp
//Configuration header for xtensor-signal: version, C++17 detection, exception handling, and feature macros.
#ifndef XTENSOR_SIGNAL_CONFIG_HPP
#define XTENSOR_SIGNAL_CONFIG_HPP

#define XTENSOR_SIGNAL_VERSION_MAJOR 0
#define XTENSOR_SIGNAL_VERSION_MINOR 1
#define XTENSOR_SIGNAL_VERSION_PATCH 0

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if (!defined(__cpp_exceptions) && !defined(__EXCEPTIONS) && !defined(_CPPUNWIND))
    #define XTENSOR_SIGNAL_DISABLE_EXCEPTIONS
#endif

#if defined(XTENSOR_SIGNAL_DISABLE_EXCEPTIONS)
    #include <iostream>
    #include <cstdlib>
    #define XTENSOR_SIGNAL_THROW(exception, msg) \
        do { std::cerr << msg << std::endl; std::abort(); } while(0)
#else
    #define XTENSOR_SIGNAL_THROW(exception, msg) throw exception(msg)
#endif

namespace xt {
namespace signal {

    // Default padding modes for convolution
    enum class padding_mode : int {
        full,
        same,
        valid
    };

    // Window types
    enum class window_type : int {
        hamming,
        hanning,
        blackman,
        bartlett,
        kaiser,
        rectangular,
        flattop
    };

    // Resampling quality
    enum class resample_quality : int {
        low,
        medium,
        high,
        ultra
    };

    // Default SIMD block size for signal processing
    constexpr std::size_t signal_simd_block_size = 64;

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_CONFIG_HPP
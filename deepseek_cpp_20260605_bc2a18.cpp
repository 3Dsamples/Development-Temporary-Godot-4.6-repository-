//File 0400 : xtensor-fftw/xtensor_fftw_config.hpp
//Configuration header: version, exception handling, C++17 feature detection, FFTW3 library integration for xtensor-fftw.
#ifndef XTENSOR_FFTW_CONFIG_HPP
#define XTENSOR_FFTW_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#define XTENSOR_FFTW_VERSION_MAJOR 0
#define XTENSOR_FFTW_VERSION_MINOR 3
#define XTENSOR_FFTW_VERSION_PATCH 0

#if (!defined(__cpp_exceptions) && !defined(__EXCEPTIONS) && !defined(_CPPUNWIND))
    #define XTENSOR_FFTW_DISABLE_EXCEPTIONS
#endif

#if defined(XTENSOR_FFTW_DISABLE_EXCEPTIONS)
    #include <iostream>
    #include <cstdlib>
    #define XTENSOR_FFTW_THROW(exception, msg) \
        do { \
            std::cerr << msg << std::endl; \
            std::abort(); \
        } while(0)
#else
    #define XTENSOR_FFTW_THROW(exception, msg) throw exception(msg)
#endif

#if defined(__has_include) && __has_include(<fftw3.h>)
    #define XTENSOR_FFTW_HAS_FFTW3 1
    #include <fftw3.h>
#else
    #error "FFTW3 headers not found. Please install FFTW3 and ensure fftw3.h is in your include path."
#endif

namespace xt {
namespace fftw {

    constexpr std::size_t default_fft_threads = 1;

    enum class fft_direction : int {
        forward = FFTW_FORWARD,
        backward = FFTW_BACKWARD
    };

    enum class fft_flag : unsigned {
        estimate = FFTW_ESTIMATE,
        measure = FFTW_MEASURE,
        patient = FFTW_PATIENT,
        exhaustive = FFTW_EXHAUSTIVE,
        wisdom_only = FFTW_WISDOM_ONLY,
        destroy_input = FFTW_DESTROY_INPUT,
        preserve_input = FFTW_PRESERVE_INPUT,
        unaligned = FFTW_UNALIGNED
    };

    constexpr unsigned int default_flags = FFTW_ESTIMATE;

    inline constexpr unsigned int combine_flags(fft_flag f1, fft_flag f2) noexcept {
        return static_cast<unsigned>(f1) | static_cast<unsigned>(f2);
    }

} // namespace fftw
} // namespace xt

#endif // XTENSOR_FFTW_CONFIG_HPP
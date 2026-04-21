// core/xtensor_config.hpp
#ifndef XTENSOR_CONFIG_HPP
#define XTENSOR_CONFIG_HPP

// ----------------------------------------------------------------------------
// xtensor_config.hpp – Global configuration for xtensor + BigNumber + FFT
// ----------------------------------------------------------------------------
// Defines platform, compiler, SIMD, and feature macros. Enables BigNumber as
// default value type, configures FFT acceleration thresholds, and sets
// performance‑critical parameters for 120 fps CPU/GPU scientific simulation.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <limits>

// ----------------------------------------------------------------------------
// BigNumber and FFT integration
// ----------------------------------------------------------------------------
#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

#ifndef XTENSOR_DEFAULT_VALUE_TYPE
    #define XTENSOR_DEFAULT_VALUE_TYPE bignumber::BigNumber
#endif

#ifndef XTENSOR_FFT_MULTIPLY_THRESHOLD_LIMBS
    #define XTENSOR_FFT_MULTIPLY_THRESHOLD_LIMBS 32
#endif

#ifndef XTENSOR_DISABLE_FFT_MULTIPLY
    #define XTENSOR_USE_FFT_MULTIPLY 1
#else
    #define XTENSOR_USE_FFT_MULTIPLY 0
#endif

#ifndef XTENSOR_FFT_CACHE_TWIDDLE_FACTORS
    #define XTENSOR_FFT_CACHE_TWIDDLE_FACTORS 1
#endif

// ----------------------------------------------------------------------------
// Version
// ----------------------------------------------------------------------------
#define XTENSOR_VERSION_MAJOR 1
#define XTENSOR_VERSION_MINOR 0
#define XTENSOR_VERSION_PATCH 0
#define XTENSOR_VERSION "1.0.0"

// ----------------------------------------------------------------------------
// Compiler detection
// ----------------------------------------------------------------------------
#if defined(_MSC_VER)
    #define XTENSOR_COMPILER_MSVC
#elif defined(__clang__)
    #define XTENSOR_COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
    #define XTENSOR_COMPILER_GCC
#endif

// ----------------------------------------------------------------------------
// Platform
// ----------------------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
    #define XTENSOR_PLATFORM_WINDOWS
#elif defined(__linux__)
    #define XTENSOR_PLATFORM_LINUX
#elif defined(__APPLE__)
    #define XTENSOR_PLATFORM_MACOS
#elif defined(__ANDROID__)
    #define XTENSOR_PLATFORM_ANDROID
#endif

// ----------------------------------------------------------------------------
// Architecture and SIMD
// ----------------------------------------------------------------------------
#if defined(__AVX512F__)
    #define XTENSOR_SIMD_AVX512
    #define XTENSOR_SIMD_WIDTH 64
#elif defined(__AVX2__)
    #define XTENSOR_SIMD_AVX2
    #define XTENSOR_SIMD_WIDTH 32
#elif defined(__SSE2__) || defined(_M_X64)
    #define XTENSOR_SIMD_SSE2
    #define XTENSOR_SIMD_WIDTH 16
#elif defined(__ARM_NEON)
    #define XTENSOR_SIMD_NEON
    #define XTENSOR_SIMD_WIDTH 16
#endif

// ----------------------------------------------------------------------------
// Alignment for SIMD
// ----------------------------------------------------------------------------
#ifndef XTENSOR_ALIGNMENT
    #ifdef XTENSOR_SIMD_AVX512
        #define XTENSOR_ALIGNMENT 64
    #elif defined(XTENSOR_SIMD_AVX2)
        #define XTENSOR_ALIGNMENT 32
    #else
        #define XTENSOR_ALIGNMENT 16
    #endif
#endif

// ----------------------------------------------------------------------------
// Threading (TBB or OpenMP)
// ----------------------------------------------------------------------------
#ifndef XTENSOR_USE_TBB
    #if __has_include(<tbb/tbb.h>)
        #define XTENSOR_USE_TBB 1
    #else
        #define XTENSOR_USE_TBB 0
    #endif
#endif

#ifndef XTENSOR_USE_OPENMP
    #ifdef _OPENMP
        #define XTENSOR_USE_OPENMP 1
    #else
        #define XTENSOR_USE_OPENMP 0
    #endif
#endif

// ----------------------------------------------------------------------------
// Default index / size types
// ----------------------------------------------------------------------------
#ifndef XTENSOR_DEFAULT_INDEX_TYPE
    #define XTENSOR_DEFAULT_INDEX_TYPE std::ptrdiff_t
#endif
#ifndef XTENSOR_DEFAULT_SIZE_TYPE
    #define XTENSOR_DEFAULT_SIZE_TYPE std::size_t
#endif

// ----------------------------------------------------------------------------
// Maximum dimensions and small vector optimization
// ----------------------------------------------------------------------------
#ifndef XTENSOR_MAX_DIMENSIONS
    #define XTENSOR_MAX_DIMENSIONS 16
#endif
#ifndef XTENSOR_SMALL_VECTOR_SIZE
    #define XTENSOR_SMALL_VECTOR_SIZE 8
#endif

// ----------------------------------------------------------------------------
// Print options
// ----------------------------------------------------------------------------
#ifndef XTENSOR_PRINT_THRESHOLD
    #define XTENSOR_PRINT_THRESHOLD 1000
#endif
#ifndef XTENSOR_PRINT_EDGE_ITEMS
    #define XTENSOR_PRINT_EDGE_ITEMS 3
#endif

// ----------------------------------------------------------------------------
// Exception handling
// ----------------------------------------------------------------------------
#ifdef XTENSOR_DISABLE_EXCEPTIONS
    #define XTENSOR_NO_EXCEPTIONS 1
    #define XTENSOR_THROW(ex, msg) std::abort()
#else
    #define XTENSOR_NO_EXCEPTIONS 0
    #define XTENSOR_THROW(ex, msg) throw ex(msg)
#endif

// ----------------------------------------------------------------------------
// Assertions
// ----------------------------------------------------------------------------
#ifdef XTENSOR_ENABLE_ASSERT
    #define XTENSOR_ASSERT(cond) do { if(!(cond)) XTENSOR_THROW(std::runtime_error, "Assertion failed: " #cond); } while(0)
#else
    #define XTENSOR_ASSERT(cond) ((void)0)
#endif

// ----------------------------------------------------------------------------
// Force inline and branch prediction
// ----------------------------------------------------------------------------
#ifdef _MSC_VER
    #define XTENSOR_FORCE_INLINE __forceinline
    #define XTENSOR_LIKELY(x) (x)
    #define XTENSOR_UNLIKELY(x) (x)
#else
    #define XTENSOR_FORCE_INLINE __attribute__((always_inline)) inline
    #define XTENSOR_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define XTENSOR_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// ----------------------------------------------------------------------------
// Namespace
// ----------------------------------------------------------------------------
#define XTENSOR_NAMESPACE xt

// ----------------------------------------------------------------------------
// Type aliases for configuration
// ----------------------------------------------------------------------------
namespace xt {
namespace config {
    using value_type = XTENSOR_DEFAULT_VALUE_TYPE;
    using index_type = XTENSOR_DEFAULT_INDEX_TYPE;
    using size_type  = XTENSOR_DEFAULT_SIZE_TYPE;

    enum class layout_type { row_major, column_major, dynamic };
    static constexpr layout_type default_layout = layout_type::row_major;
    static constexpr std::size_t max_dimensions = XTENSOR_MAX_DIMENSIONS;
    static constexpr std::size_t alignment = XTENSOR_ALIGNMENT;
    static constexpr bool use_fft_multiply = (XTENSOR_USE_FFT_MULTIPLY == 1);
    static constexpr std::size_t fft_multiply_threshold_limbs = XTENSOR_FFT_MULTIPLY_THRESHOLD_LIMBS;
    static constexpr bool fft_cache_twiddle = (XTENSOR_FFT_CACHE_TWIDDLE_FACTORS == 1);
} // namespace config
} // namespace xt

#endif // XTENSOR_CONFIG_HPP
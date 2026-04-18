// core/xtensor_config.hpp

#ifndef XTENSOR_CONFIG_HPP
#define XTENSOR_CONFIG_HPP

// Standard library includes
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <version>

// ----------------------------------------------------------------------------
// XTENSOR_VERSION - Semantic versioning
// ----------------------------------------------------------------------------
#define XTENSOR_VERSION_MAJOR 0
#define XTENSOR_VERSION_MINOR 26
#define XTENSOR_VERSION_PATCH 0
#define XTENSOR_VERSION "0.26.0"

// ----------------------------------------------------------------------------
// Compiler detection
// ----------------------------------------------------------------------------
#if defined(_MSC_VER)
    #define XTENSOR_COMPILER_MSVC
    #define XTENSOR_COMPILER_VERSION _MSC_VER
#elif defined(__clang__)
    #define XTENSOR_COMPILER_CLANG
    #define XTENSOR_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__) || defined(__GNUG__)
    #define XTENSOR_COMPILER_GCC
    #define XTENSOR_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#else
    #define XTENSOR_COMPILER_UNKNOWN
    #define XTENSOR_COMPILER_VERSION 0
#endif

// ----------------------------------------------------------------------------
// Platform detection
// ----------------------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
    #define XTENSOR_PLATFORM_WINDOWS
#elif defined(__linux__)
    #define XTENSOR_PLATFORM_LINUX
#elif defined(__APPLE__) && defined(__MACH__)
    #define XTENSOR_PLATFORM_MACOS
#elif defined(__ANDROID__)
    #define XTENSOR_PLATFORM_ANDROID
#elif defined(__EMSCRIPTEN__)
    #define XTENSOR_PLATFORM_EMSCRIPTEN
#else
    #define XTENSOR_PLATFORM_UNKNOWN
#endif

// ----------------------------------------------------------------------------
// Architecture detection
// ----------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
    #define XTENSOR_ARCH_X86_64
    #define XTENSOR_ARCH_NAME "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
    #define XTENSOR_ARCH_X86
    #define XTENSOR_ARCH_NAME "x86"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define XTENSOR_ARCH_ARM64
    #define XTENSOR_ARCH_NAME "ARM64"
#elif defined(__arm__) || defined(_M_ARM)
    #define XTENSOR_ARCH_ARM
    #define XTENSOR_ARCH_NAME "ARM"
#elif defined(__powerpc64__) || defined(__ppc64__)
    #define XTENSOR_ARCH_PPC64
    #define XTENSOR_ARCH_NAME "PowerPC64"
#elif defined(__powerpc__) || defined(__ppc__)
    #define XTENSOR_ARCH_PPC
    #define XTENSOR_ARCH_NAME "PowerPC"
#elif defined(__riscv)
    #define XTENSOR_ARCH_RISCV
    #if __riscv_xlen == 64
        #define XTENSOR_ARCH_NAME "RISC-V64"
    #else
        #define XTENSOR_ARCH_NAME "RISC-V32"
    #endif
#else
    #define XTENSOR_ARCH_UNKNOWN
    #define XTENSOR_ARCH_NAME "Unknown"
#endif

// ----------------------------------------------------------------------------
// C++ Standard version detection
// ----------------------------------------------------------------------------
#if __cplusplus >= 202302L
    #define XTENSOR_CPP_STANDARD 23
    #define XTENSOR_CPP_STANDARD_NAME "C++23"
#elif __cplusplus >= 202002L
    #define XTENSOR_CPP_STANDARD 20
    #define XTENSOR_CPP_STANDARD_NAME "C++20"
#elif __cplusplus >= 201703L
    #define XTENSOR_CPP_STANDARD 17
    #define XTENSOR_CPP_STANDARD_NAME "C++17"
#elif __cplusplus >= 201402L
    #define XTENSOR_CPP_STANDARD 14
    #define XTENSOR_CPP_STANDARD_NAME "C++14"
#elif __cplusplus >= 201103L
    #define XTENSOR_CPP_STANDARD 11
    #define XTENSOR_CPP_STANDARD_NAME "C++11"
#else
    #error "xtensor requires C++11 or higher"
#endif

// ----------------------------------------------------------------------------
// Feature detection macros
// ----------------------------------------------------------------------------
#if __has_include(<optional>) && (__cplusplus >= 201703L)
    #define XTENSOR_HAS_OPTIONAL 1
#else
    #define XTENSOR_HAS_OPTIONAL 0
#endif

#if __has_include(<variant>) && (__cplusplus >= 201703L)
    #define XTENSOR_HAS_VARIANT 1
#else
    #define XTENSOR_HAS_VARIANT 0
#endif

#if __has_include(<string_view>) && (__cplusplus >= 201703L)
    #define XTENSOR_HAS_STRING_VIEW 1
#else
    #define XTENSOR_HAS_STRING_VIEW 0
#endif

#if __has_include(<filesystem>) && (__cplusplus >= 201703L)
    #define XTENSOR_HAS_FILESYSTEM 1
#else
    #define XTENSOR_HAS_FILESYSTEM 0
#endif

#if __has_include(<span>) && (__cplusplus >= 202002L)
    #define XTENSOR_HAS_SPAN 1
#else
    #define XTENSOR_HAS_SPAN 0
#endif

#if __has_include(<bit>) && (__cplusplus >= 202002L)
    #define XTENSOR_HAS_BIT 1
#else
    #define XTENSOR_HAS_BIT 0
#endif

#if __has_include(<concepts>) && (__cplusplus >= 202002L)
    #define XTENSOR_HAS_CONCEPTS 1
#else
    #define XTENSOR_HAS_CONCEPTS 0
#endif

#if __has_include(<ranges>) && (__cplusplus >= 202002L)
    #define XTENSOR_HAS_RANGES 1
#else
    #define XTENSOR_HAS_RANGES 0
#endif

#if __has_include(<coroutine>) && (__cplusplus >= 202002L)
    #define XTENSOR_HAS_COROUTINE 1
#else
    #define XTENSOR_HAS_COROUTINE 0
#endif

// ----------------------------------------------------------------------------
// OpenMP support detection
// ----------------------------------------------------------------------------
#if defined(_OPENMP)
    #define XTENSOR_HAS_OPENMP 1
    #if _OPENMP >= 201511
        #define XTENSOR_OPENMP_VERSION "4.5+"
    #elif _OPENMP >= 201307
        #define XTENSOR_OPENMP_VERSION "4.0"
    #elif _OPENMP >= 201107
        #define XTENSOR_OPENMP_VERSION "3.1"
    #elif _OPENMP >= 200805
        #define XTENSOR_OPENMP_VERSION "3.0"
    #elif _OPENMP >= 200505
        #define XTENSOR_OPENMP_VERSION "2.5"
    #else
        #define XTENSOR_OPENMP_VERSION "2.0"
    #endif
#else
    #define XTENSOR_HAS_OPENMP 0
    #define XTENSOR_OPENMP_VERSION "Not available"
#endif

// ----------------------------------------------------------------------------
// SIMD support detection (xsimd)
// ----------------------------------------------------------------------------
#if __has_include(<xsimd/xsimd.hpp>)
    #define XTENSOR_HAS_XSIMD 1
#else
    #define XTENSOR_HAS_XSIMD 0
#endif

// ----------------------------------------------------------------------------
// BLAS/LAPACK support detection
// ----------------------------------------------------------------------------
#if __has_include(<cxxblas.hpp>) && __has_include(<cxxlapack.hpp>)
    #define XTENSOR_HAS_BLAS 1
#else
    #define XTENSOR_HAS_BLAS 0
#endif

// ----------------------------------------------------------------------------
// Exception handling configuration
// ----------------------------------------------------------------------------
#if defined(XTENSOR_DISABLE_EXCEPTIONS)
    #define XTENSOR_NO_EXCEPTIONS 1
    #define XTENSOR_THROW(exception, message) std::abort()
    #define XTENSOR_TRY if(true)
    #define XTENSOR_CATCH(exception) if(false)
    #define XTENSOR_CATCH_ALL if(false)
    #define XTENSOR_RETHROW std::abort()
#else
    #define XTENSOR_NO_EXCEPTIONS 0
    #define XTENSOR_THROW(exception, message) throw exception(message)
    #define XTENSOR_TRY try
    #define XTENSOR_CATCH(exception) catch(const exception&)
    #define XTENSOR_CATCH_ALL catch(...)
    #define XTENSOR_RETHROW throw
#endif

// ----------------------------------------------------------------------------
// Assertion configuration
// ----------------------------------------------------------------------------
#if defined(XTENSOR_ENABLE_ASSERT)
    #define XTENSOR_ASSERT(condition) \
        do { \
            if (!(condition)) { \
                XTENSOR_THROW(std::runtime_error, "Assertion failed: " #condition); \
            } \
        } while(0)
    #define XTENSOR_ASSERT_MSG(condition, message) \
        do { \
            if (!(condition)) { \
                XTENSOR_THROW(std::runtime_error, message); \
            } \
        } while(0)
#else
    #define XTENSOR_ASSERT(condition) ((void)0)
    #define XTENSOR_ASSERT_MSG(condition, message) ((void)0)
#endif

// ----------------------------------------------------------------------------
// Index type configuration
// ----------------------------------------------------------------------------
#ifndef XTENSOR_DEFAULT_INDEX_TYPE
    #define XTENSOR_DEFAULT_INDEX_TYPE std::ptrdiff_t
#endif

#ifndef XTENSOR_DEFAULT_SIZE_TYPE
    #define XTENSOR_DEFAULT_SIZE_TYPE std::size_t
#endif

#ifndef XTENSOR_DEFAULT_SHAPE_TYPE
    #define XTENSOR_DEFAULT_SHAPE_TYPE std::vector<XTENSOR_DEFAULT_SIZE_TYPE>
#endif

#ifndef XTENSOR_DEFAULT_STRIDES_TYPE
    #define XTENSOR_DEFAULT_STRIDES_TYPE std::vector<XTENSOR_DEFAULT_INDEX_TYPE>
#endif

#ifndef XTENSOR_DEFAULT_LAYOUT
    #define XTENSOR_DEFAULT_LAYOUT layout_type::row_major
#endif

#ifndef XTENSOR_DEFAULT_ALLOCATOR
    #define XTENSOR_DEFAULT_ALLOCATOR(T) std::allocator<T>
#endif

// ----------------------------------------------------------------------------
// Maximum dimensions configuration
// ----------------------------------------------------------------------------
#ifndef XTENSOR_MAX_DIMENSIONS
    #define XTENSOR_MAX_DIMENSIONS 16
#endif

// ----------------------------------------------------------------------------
// Small vector optimization configuration
// ----------------------------------------------------------------------------
#ifndef XTENSOR_SMALL_VECTOR_SIZE
    #define XTENSOR_SMALL_VECTOR_SIZE 8
#endif

// ----------------------------------------------------------------------------
// Print options configuration
// ----------------------------------------------------------------------------
#ifndef XTENSOR_PRINT_THRESHOLD
    #define XTENSOR_PRINT_THRESHOLD 1000
#endif

#ifndef XTENSOR_PRINT_EDGE_ITEMS
    #define XTENSOR_PRINT_EDGE_ITEMS 3
#endif

#ifndef XTENSOR_PRINT_LINE_WIDTH
    #define XTENSOR_PRINT_LINE_WIDTH 75
#endif

#ifndef XTENSOR_PRINT_PRECISION
    #define XTENSOR_PRINT_PRECISION 6
#endif

// ----------------------------------------------------------------------------
// Thread pool configuration
// ----------------------------------------------------------------------------
#ifndef XTENSOR_THREAD_POOL_SIZE
    #define XTENSOR_THREAD_POOL_SIZE 0  // 0 = automatic (std::thread::hardware_concurrency())
#endif

#ifndef XTENSOR_MIN_PARALLEL_SIZE
    #define XTENSOR_MIN_PARALLEL_SIZE 32768  // Minimum elements for parallel execution
#endif

// ----------------------------------------------------------------------------
// Memory alignment configuration
// ----------------------------------------------------------------------------
#ifndef XTENSOR_ALIGNMENT
    #if defined(XTENSOR_ARCH_X86_64) || defined(XTENSOR_ARCH_X86)
        #define XTENSOR_ALIGNMENT 64  // AVX-512 alignment
    #elif defined(XTENSOR_ARCH_ARM64)
        #define XTENSOR_ALIGNMENT 32  // NEON alignment
    #else
        #define XTENSOR_ALIGNMENT 16
    #endif
#endif

// ----------------------------------------------------------------------------
// Namespace configuration
// ----------------------------------------------------------------------------
#define XTENSOR_NAMESPACE xt

// ----------------------------------------------------------------------------
// Inline namespace for ABI versioning
// ----------------------------------------------------------------------------
#define XTENSOR_INLINE_NAMESPACE inline namespace v0_26

// ----------------------------------------------------------------------------
// Function attributes
// ----------------------------------------------------------------------------
#if defined(XTENSOR_COMPILER_MSVC)
    #define XTENSOR_FORCE_INLINE __forceinline
    #define XTENSOR_NO_INLINE __declspec(noinline)
    #define XTENSOR_RESTRICT __restrict
    #define XTENSOR_UNUSED
    #define XTENSOR_DEPRECATED(message) __declspec(deprecated(message))
    #define XTENSOR_NORETURN __declspec(noreturn)
#elif defined(XTENSOR_COMPILER_GCC) || defined(XTENSOR_COMPILER_CLANG)
    #define XTENSOR_FORCE_INLINE __attribute__((always_inline)) inline
    #define XTENSOR_NO_INLINE __attribute__((noinline))
    #define XTENSOR_RESTRICT __restrict__
    #define XTENSOR_UNUSED __attribute__((unused))
    #define XTENSOR_DEPRECATED(message) __attribute__((deprecated(message)))
    #define XTENSOR_NORETURN __attribute__((noreturn))
#else
    #define XTENSOR_FORCE_INLINE inline
    #define XTENSOR_NO_INLINE
    #define XTENSOR_RESTRICT
    #define XTENSOR_UNUSED
    #define XTENSOR_DEPRECATED(message)
    #define XTENSOR_NORETURN
#endif

// ----------------------------------------------------------------------------
// Likely/unlikely branch prediction hints
// ----------------------------------------------------------------------------
#if defined(XTENSOR_COMPILER_GCC) || defined(XTENSOR_COMPILER_CLANG)
    #define XTENSOR_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define XTENSOR_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define XTENSOR_LIKELY(x)   (x)
    #define XTENSOR_UNLIKELY(x) (x)
#endif

// ----------------------------------------------------------------------------
// Prefetch hint
// ----------------------------------------------------------------------------
#if defined(XTENSOR_COMPILER_GCC) || defined(XTENSOR_COMPILER_CLANG)
    #define XTENSOR_PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#elif defined(XTENSOR_COMPILER_MSVC)
    #define XTENSOR_PREFETCH(addr, rw, locality) _mm_prefetch(reinterpret_cast<const char*>(addr), (locality))
#else
    #define XTENSOR_PREFETCH(addr, rw, locality) ((void)0)
#endif

// ----------------------------------------------------------------------------
// Assume hint
// ----------------------------------------------------------------------------
#if defined(XTENSOR_COMPILER_GCC) && (__GNUC__ >= 13)
    #define XTENSOR_ASSUME(expr) __attribute__((assume(expr)))
#elif defined(XTENSOR_COMPILER_CLANG)
    #define XTENSOR_ASSUME(expr) __builtin_assume(expr)
#elif defined(XTENSOR_COMPILER_MSVC)
    #define XTENSOR_ASSUME(expr) __assume(expr)
#else
    #define XTENSOR_ASSUME(expr) ((void)0)
#endif

// ----------------------------------------------------------------------------
// Unreachable hint
// ----------------------------------------------------------------------------
#if defined(XTENSOR_COMPILER_GCC) || defined(XTENSOR_COMPILER_CLANG)
    #define XTENSOR_UNREACHABLE() __builtin_unreachable()
#elif defined(XTENSOR_COMPILER_MSVC)
    #define XTENSOR_UNREACHABLE() __assume(0)
#else
    #define XTENSOR_UNREACHABLE() std::abort()
#endif

// ----------------------------------------------------------------------------
// Type traits and utilities
// ----------------------------------------------------------------------------
namespace xt {
namespace config {

// Check if type is a valid index type
template<typename T>
struct is_valid_index_type : std::disjunction<
    std::is_integral<T>,
    std::is_same<T, XTENSOR_DEFAULT_INDEX_TYPE>
> {};

template<typename T>
inline constexpr bool is_valid_index_type_v = is_valid_index_type<T>::value;

// Check if type is a valid size type
template<typename T>
struct is_valid_size_type : std::disjunction<
    std::is_unsigned<T>,
    std::is_same<T, XTENSOR_DEFAULT_SIZE_TYPE>
> {};

template<typename T>
inline constexpr bool is_valid_size_type_v = is_valid_size_type<T>::value;

// Check if container is a valid shape container
template<typename T>
struct is_valid_shape_container : std::conjunction<
    std::is_default_constructible<T>,
    std::is_copy_constructible<T>,
    std::is_destructible<T>
> {};

template<typename T>
inline constexpr bool is_valid_shape_container_v = is_valid_shape_container<T>::value;

// Get the index type for a given size type
template<typename SizeType>
using index_type_from_size = std::make_signed_t<SizeType>;

// Get the size type for a given index type
template<typename IndexType>
using size_type_from_index = std::make_unsigned_t<IndexType>;

// Maximum number of dimensions supported
inline constexpr std::size_t max_dimensions = XTENSOR_MAX_DIMENSIONS;

// Small vector size for optimization
inline constexpr std::size_t small_vector_size = XTENSOR_SMALL_VECTOR_SIZE;

// Alignment value
inline constexpr std::size_t alignment = XTENSOR_ALIGNMENT;

// Minimum size for parallel execution
inline constexpr std::size_t min_parallel_size = XTENSOR_MIN_PARALLEL_SIZE;

// Thread pool size
inline constexpr std::size_t thread_pool_size = XTENSOR_THREAD_POOL_SIZE;

// Print options
struct print_options {
    std::size_t threshold = XTENSOR_PRINT_THRESHOLD;
    std::size_t edge_items = XTENSOR_PRINT_EDGE_ITEMS;
    std::size_t line_width = XTENSOR_PRINT_LINE_WIDTH;
    int precision = XTENSOR_PRINT_PRECISION;
};

// Default layout type
enum class layout_type {
    row_major,
    column_major,
    dynamic
};

inline constexpr layout_type default_layout = XTENSOR_DEFAULT_LAYOUT;

// Configuration information structure
struct build_info {
    static constexpr const char* version = XTENSOR_VERSION;
    static constexpr int version_major = XTENSOR_VERSION_MAJOR;
    static constexpr int version_minor = XTENSOR_VERSION_MINOR;
    static constexpr int version_patch = XTENSOR_VERSION_PATCH;
    
    static constexpr const char* compiler = 
#if defined(XTENSOR_COMPILER_MSVC)
        "MSVC";
#elif defined(XTENSOR_COMPILER_CLANG)
        "Clang";
#elif defined(XTENSOR_COMPILER_GCC)
        "GCC";
#else
        "Unknown";
#endif
    
    static constexpr int compiler_version = XTENSOR_COMPILER_VERSION;
    
    static constexpr const char* platform = 
#if defined(XTENSOR_PLATFORM_WINDOWS)
        "Windows";
#elif defined(XTENSOR_PLATFORM_LINUX)
        "Linux";
#elif defined(XTENSOR_PLATFORM_MACOS)
        "macOS";
#elif defined(XTENSOR_PLATFORM_ANDROID)
        "Android";
#elif defined(XTENSOR_PLATFORM_EMSCRIPTEN)
        "Emscripten";
#else
        "Unknown";
#endif
    
    static constexpr const char* architecture = XTENSOR_ARCH_NAME;
    
    static constexpr const char* cpp_standard = XTENSOR_CPP_STANDARD_NAME;
    
    static constexpr bool has_openmp = XTENSOR_HAS_OPENMP;
    static constexpr const char* openmp_version = XTENSOR_OPENMP_VERSION;
    
    static constexpr bool has_xsimd = XTENSOR_HAS_XSIMD;
    static constexpr bool has_blas = XTENSOR_HAS_BLAS;
    static constexpr bool exceptions_enabled = !XTENSOR_NO_EXCEPTIONS;
    static constexpr bool assertions_enabled = 
#ifdef XTENSOR_ENABLE_ASSERT
        true;
#else
        false;
#endif
};

} // namespace config
} // namespace xt

// ----------------------------------------------------------------------------
// Include additional configuration from xtl
// ----------------------------------------------------------------------------
#if __has_include(<xtl/xtl_config.hpp>)
    #include <xtl/xtl_config.hpp>
#endif

// ----------------------------------------------------------------------------
// Ensure necessary includes
// ----------------------------------------------------------------------------
#include <vector>
#include <functional>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdlib>

// ----------------------------------------------------------------------------
// Define common type aliases for convenience
// ----------------------------------------------------------------------------
namespace xt {
    using index_type = XTENSOR_DEFAULT_INDEX_TYPE;
    using size_type = XTENSOR_DEFAULT_SIZE_TYPE;
    using shape_type = XTENSOR_DEFAULT_SHAPE_TYPE;
    using strides_type = XTENSOR_DEFAULT_STRIDES_TYPE;
    
    template<typename T>
    using default_allocator = XTENSOR_DEFAULT_ALLOCATOR(T);
    
    // Common layout types
    using layout = config::layout_type;
    inline constexpr layout row_major = layout::row_major;
    inline constexpr layout column_major = layout::column_major;
    inline constexpr layout dynamic = layout::dynamic;
}

// ----------------------------------------------------------------------------
// Debug macros
// ----------------------------------------------------------------------------
#ifdef XTENSOR_DEBUG
    #define XTENSOR_LOG_DEBUG(msg) \
        do { \
            std::cerr << "[XTENSOR DEBUG] " << msg << std::endl; \
        } while(0)
    #define XTENSOR_LOG_INFO(msg) \
        do { \
            std::cout << "[XTENSOR INFO] " << msg << std::endl; \
        } while(0)
    #define XTENSOR_LOG_WARNING(msg) \
        do { \
            std::cerr << "[XTENSOR WARNING] " << msg << std::endl; \
        } while(0)
    #define XTENSOR_LOG_ERROR(msg) \
        do { \
            std::cerr << "[XTENSOR ERROR] " << msg << std::endl; \
        } while(0)
#else
    #define XTENSOR_LOG_DEBUG(msg) ((void)0)
    #define XTENSOR_LOG_INFO(msg) ((void)0)
    #define XTENSOR_LOG_WARNING(msg) ((void)0)
    #define XTENSOR_LOG_ERROR(msg) ((void)0)
#endif

#endif // XTENSOR_CONFIG_HPP
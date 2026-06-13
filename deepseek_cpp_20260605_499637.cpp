//File 0300 : xframe/xframe_config.hpp
//xframe configuration: C++17 detection, SIMD alignment, memory allocation, dimension and coordinate types.
#ifndef XFRAME_CONFIG_HPP
#define XFRAME_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <string>
#include <xsimd/xsimd.hpp>

#if defined(__GNUC__) || defined(__clang__)
    #define XFRAME_COMPILER_GCC 1
#elif defined(_MSC_VER)
    #define XFRAME_COMPILER_MSVC 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define XFRAME_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define XFRAME_ARCH_ARM64 1
#endif

namespace xframe
{
    // Layout for underlying array storage
    enum class layout { row_major, column_major };
    constexpr layout default_layout = layout::row_major;

    // SIMD architecture selection (same as xtensor)
    #if defined(__AVX512F__)
        using default_simd_arch = xsimd::avx512f;
    #elif defined(__AVX2__)
        using default_simd_arch = xsimd::avx2;
    #elif defined(__SSE4_2__)
        using default_simd_arch = xsimd::sse4_2;
    #elif defined(__SSE2__)
        using default_simd_arch = xsimd::sse2;
    #elif defined(__ARM_NEON)
        using default_simd_arch = xsimd::neon64;
    #else
        using default_simd_arch = xsimd::generic;
    #endif

    template <class T>
    inline constexpr bool simd_enabled_v = xsimd::has_simd_register<T, default_simd_arch>::value;

    constexpr std::size_t simd_alignment = 64;

    // Default label type for dimensions and coordinates
    using label_type = std::string;

    // Default container for labels
    template <class T>
    using label_container = std::vector<T>;

    // Forward declarations
    struct expression_tag {};
    template <class D> class expression;

    template <class T, class L = label_type>
    class variable;

    template <class T>
    class coordinate;

    template <class L = label_type>
    class dimension;

    template <class... V>
    class xframe;

    template <class... V>
    class xframe_view;
}

#endif
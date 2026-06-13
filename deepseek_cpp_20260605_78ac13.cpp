//File 0101 : numdot/config.h
//Compiler and architecture detection, SIMD selection, C++17 feature enablement, memory alignment, and global configuration for NumDot.
#ifndef NUMDOT_CONFIG_H
#define NUMDOT_CONFIG_H

#if defined(__GNUC__) || defined(__clang__)
    #define NUMDOT_COMPILER_GCC 1
#elif defined(_MSC_VER)
    #define NUMDOT_COMPILER_MSVC 1
#endif

#include <xsimd/xsimd.hpp>

#if defined(__x86_64__) || defined(_M_X64)
    #define NUMDOT_ARCH_X86_64 1
    #if defined(__AVX512F__)
        #define NUMDOT_SIMD_AVX512 1
    #elif defined(__AVX2__)
        #define NUMDOT_SIMD_AVX2 1
    #elif defined(__SSE4_2__)
        #define NUMDOT_SIMD_SSE4_2 1
    #else
        #define NUMDOT_SIMD_SSE2 1
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define NUMDOT_ARCH_ARM64 1
    #if defined(__ARM_NEON)
        #define NUMDOT_SIMD_NEON 1
    #endif
#elif defined(_M_ARM)
    #define NUMDOT_ARCH_ARM32 1
    #if defined(__ARM_NEON__)
        #define NUMDOT_SIMD_NEON 1
    #endif
#endif

#ifndef NUMDOT_SIMD_AVX512
    #ifndef NUMDOT_SIMD_AVX2
        #ifndef NUMDOT_SIMD_SSE4_2
            #ifndef NUMDOT_SIMD_SSE2
                #ifndef NUMDOT_SIMD_NEON
                    #define NUMDOT_NO_SIMD 1
                #endif
            #endif
        #endif
    #endif
#endif

namespace numdot
{
    // Layout options
    enum class layout { row_major, column_major, dynamic };
    constexpr layout default_layout = layout::row_major;

    // Select default SIMD architecture
    #if defined(NUMDOT_SIMD_AVX512)
        using default_simd_arch = xsimd::avx512f;
    #elif defined(NUMDOT_SIMD_AVX2)
        using default_simd_arch = xsimd::avx2;
    #elif defined(NUMDOT_SIMD_SSE4_2)
        using default_simd_arch = xsimd::sse4_2;
    #elif defined(NUMDOT_SIMD_SSE2)
        using default_simd_arch = xsimd::sse2;
    #elif defined(NUMDOT_SIMD_NEON)
        using default_simd_arch = xsimd::neon64;
    #else
        using default_simd_arch = xsimd::generic;
    #endif

    // Detect SIMD availability for a type
    template <class T>
    inline constexpr bool simd_enabled_v = xsimd::has_simd_register<T, default_simd_arch>::value;

    // Memory alignment for SIMD
    constexpr std::size_t cache_line_size = 64;
    constexpr std::size_t simd_alignment = 64;

    // Forward declarations
    template <class T> struct type_traits;

    // Aligned allocator for containers
    template <class T, std::size_t Align = simd_alignment>
    struct aligned_allocator {
        using value_type = T;
        aligned_allocator() = default;
        template <class U> aligned_allocator(const aligned_allocator<U, Align>&) noexcept {}
        T* allocate(std::size_t n) {
            void* ptr = ::operator new(n * sizeof(T), std::align_val_t(Align));
            return static_cast<T*>(ptr);
        }
        void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t(Align)); }
        template <class U> struct rebind { using other = aligned_allocator<U, Align>; };
    };

    // Default vector type
    template <class T>
    using uvector = std::vector<T, aligned_allocator<T>>;

    // Helper to disable expressions for SFINAE
    template <class T>
    using disable_if_expression = std::enable_if_t<!std::is_base_of_v<expression_tag, T>>;
}

#endif // NUMDOT_CONFIG_H
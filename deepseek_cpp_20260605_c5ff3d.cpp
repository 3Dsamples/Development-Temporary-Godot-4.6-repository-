//File 0015 : core/xtensor_config.hpp
//Compiler detection, SIMD architecture selection, and preprocessor utilities for portable 64-bit SIMD.
#ifndef XTENSOR_CONFIG_HPP
#define XTENSOR_CONFIG_HPP

#if defined(__GNUC__) || defined(__clang__)
    #define XTENSOR_GCC_COMPILER 1
#elif defined(_MSC_VER)
    #define XTENSOR_MSVC_COMPILER 1
#endif

#include <xsimd/xsimd.hpp>

#if defined(__x86_64__) || defined(_M_X64)
    #define XTENSOR_ARCH_X86_64 1
    #if defined(__AVX512F__)
        #define XTENSOR_SIMD_AVX512 1
    #elif defined(__AVX2__)
        #define XTENSOR_SIMD_AVX2 1
    #elif defined(__SSE4_2__)
        #define XTENSOR_SIMD_SSE4_2 1
    #else
        #define XTENSOR_SIMD_SSE2 1
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define XTENSOR_ARCH_ARM64 1
    #if defined(__ARM_NEON)
        #define XTENSOR_SIMD_NEON 1
    #endif
#elif defined(_M_ARM)
    #define XTENSOR_ARCH_ARM32 1
    #if defined(__ARM_NEON__)
        #define XTENSOR_SIMD_NEON 1
    #endif
#endif

#ifndef XTENSOR_SIMD_AVX512
    #ifndef XTENSOR_SIMD_AVX2
        #ifndef XTENSOR_SIMD_SSE4_2
            #ifndef XTENSOR_SIMD_SSE2
                #ifndef XTENSOR_SIMD_NEON
                    #define XTENSOR_NO_SIMD 1
                #endif
            #endif
        #endif
    #endif
#endif

namespace xt
{
    // Default layout for arrays
    enum class layout_type
    {
        row_major,
        column_major,
        dynamic
    };
    constexpr layout_type DEFAULT_LAYOUT = layout_type::row_major;

    // SIMD architecture alias
    #if defined(XTENSOR_SIMD_AVX512)
        using default_simd_arch = xsimd::avx512f;
    #elif defined(XTENSOR_SIMD_AVX2)
        using default_simd_arch = xsimd::avx2;
    #elif defined(XTENSOR_SIMD_SSE4_2)
        using default_simd_arch = xsimd::sse4_2;
    #elif defined(XTENSOR_SIMD_SSE2)
        using default_simd_arch = xsimd::sse2;
    #elif defined(XTENSOR_SIMD_NEON)
        using default_simd_arch = xsimd::neon64;
    #else
        using default_simd_arch = xsimd::generic;
    #endif

    // Check if SIMD is enabled for a given type
    template <class T>
    inline constexpr bool is_simd_enabled_v = xsimd::has_simd_register<T, default_simd_arch>::value;

    // Cache line size (safe default 64 bytes)
    constexpr std::size_t CACHE_LINE_SIZE = 64;

    // Alignment for SIMD memory allocations
    constexpr std::size_t SIMD_ALIGNMENT = 64;

    // Forward declarations for key components
    template <class D>
    class xcontainer_semantic;

    template <class D>
    class xview_semantic;

    template <class T>
    struct xcontainer_inner_types;

    struct xtensor_expression_tag;

    // Default shape container type (dynamic)
    template <class T>
    using default_shape_container = std::vector<typename T::size_type>;

    // Configuration macro helpers
    template <class T>
    using disable_xexpression = std::enable_if_t<!std::is_base_of<xtensor_expression_tag, T>::value>;

    // Alignment utility functions
    inline std::size_t align_up(std::size_t size, std::size_t alignment) noexcept
    {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    inline void* aligned_alloc(std::size_t size, std::size_t alignment = SIMD_ALIGNMENT)
    {
        #ifdef _MSC_VER
            return _aligned_malloc(size, alignment);
        #else
            return std::aligned_alloc(alignment, size);
        #endif
    }

    inline void aligned_free(void* ptr) noexcept
    {
        #ifdef _MSC_VER
            _aligned_free(ptr);
        #else
            std::free(ptr);
        #endif
    }

    // Uvector allocator that uses SIMD alignment (placeholder for custom allocator)
    template <class T, std::size_t Align = SIMD_ALIGNMENT>
    struct aligned_allocator
    {
        using value_type = T;
        aligned_allocator() = default;
        template <class U>
        aligned_allocator(const aligned_allocator<U, Align>&) noexcept {}

        T* allocate(std::size_t n)
        {
            void* ptr = aligned_alloc(n * sizeof(T), Align);
            if (!ptr) throw std::bad_alloc();
            return static_cast<T*>(ptr);
        }

        void deallocate(T* p, std::size_t) noexcept
        {
            aligned_free(p);
        }

        template <class U>
        struct rebind
        {
            using other = aligned_allocator<U, Align>;
        };
    };

    // Uvector definition with aligned allocator
    template <class T>
    using uvector = std::vector<T, aligned_allocator<T>>;
}

#endif // XTENSOR_CONFIG_HPP
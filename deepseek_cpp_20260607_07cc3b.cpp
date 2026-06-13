/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file build_config.h
 * @brief Compile‑time configuration, feature detection, and platform abstractions.
 *
 * This file sets up the build environment for OrthoTree: compiler checks,
 * SIMD instruction set detection, endianness, alignment requirements,
 * debug/release modes, and customizable performance tuning knobs.
 *
 * It also defines macros for exporting symbols (shared library), inlining,
 * and attributes for hot/cold functions. All math‑related precision and
 * rounding policies are configured here.
 */

#ifndef ORTHOTREE_CORE_BUILD_CONFIG_H_INCLUDED
#define ORTHOTREE_CORE_BUILD_CONFIG_H_INCLUDED

// ============================================================================
//  Compiler detection
// ============================================================================
#if defined(_MSC_VER)
#   define ORTHOTREE_COMPILER_MSVC 1
#   define ORTHOTREE_COMPILER_NAME "MSVC"
#elif defined(__GNUC__)
#   define ORTHOTREE_COMPILER_GCC 1
#   define ORTHOTREE_COMPILER_NAME "GCC"
#   define ORTHOTREE_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#elif defined(__clang__)
#   define ORTHOTREE_COMPILER_CLANG 1
#   define ORTHOTREE_COMPILER_NAME "Clang"
#   define ORTHOTREE_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#else
#   define ORTHOTREE_COMPILER_UNKNOWN 1
#   define ORTHOTREE_COMPILER_NAME "Unknown"
#endif

// ============================================================================
//  C++17 standard check
// ============================================================================
#if !defined(__cplusplus) || __cplusplus < 201703L
#   error "OrthoTree requires C++17 or later"
#endif

// ============================================================================
//  Platform detection (OS)
// ============================================================================
#if defined(_WIN32) || defined(_WIN64)
#   define ORTHOTREE_OS_WINDOWS 1
#elif defined(__linux__)
#   define ORTHOTREE_OS_LINUX 1
#elif defined(__APPLE__) && defined(__MACH__)
#   define ORTHOTREE_OS_MACOS 1
#elif defined(__ANDROID__)
#   define ORTHOTREE_OS_ANDROID 1
#elif defined(__unix__)
#   define ORTHOTREE_OS_UNIX 1
#else
#   define ORTHOTREE_OS_OTHER 1
#endif

// ============================================================================
//  Endianness detection
// ============================================================================
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#   if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#       define ORTHOTREE_LITTLE_ENDIAN 1
#   else
#       define ORTHOTREE_BIG_ENDIAN 1
#   endif
#elif defined(_WIN32) || defined(__i386__) || defined(__x86_64__) || defined(__ARMEL__)
#   define ORTHOTREE_LITTLE_ENDIAN 1
#else
#   define ORTHOTREE_BIG_ENDIAN 1
#endif

// ============================================================================
//  SIMD architecture detection (for accelerated math)
// ============================================================================
#if defined(__AVX512F__)
#   define ORTHOTREE_SIMD_LEVEL 512
#   define ORTHOTREE_SIMD_AVX512 1
#elif defined(__AVX2__)
#   define ORTHOTREE_SIMD_LEVEL 256
#   define ORTHOTREE_SIMD_AVX2 1
#elif defined(__AVX__)
#   define ORTHOTREE_SIMD_LEVEL 256
#   define ORTHOTREE_SIMD_AVX 1
#elif defined(__SSE4_2__)
#   define ORTHOTREE_SIMD_LEVEL 128
#   define ORTHOTREE_SIMD_SSE4 1
#elif defined(__SSE3__)
#   define ORTHOTREE_SIMD_LEVEL 128
#   define ORTHOTREE_SIMD_SSE3 1
#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#   define ORTHOTREE_SIMD_LEVEL 128
#   define ORTHOTREE_SIMD_SSE2 1
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#   define ORTHOTREE_SIMD_LEVEL 128
#   define ORTHOTREE_SIMD_NEON 1
#elif defined(__VSX__)
#   define ORTHOTREE_SIMD_LEVEL 128
#   define ORTHOTREE_SIMD_VSX 1
#else
#   define ORTHOTREE_SIMD_LEVEL 0
#   define ORTHOTREE_SIMD_NONE 1
#endif

// ============================================================================
//  Alignment requirements (cache line, typical 64 bytes)
// ============================================================================
#define ORTHOTREE_CACHE_LINE_SIZE 64
#define ORTHOTREE_SIMD_ALIGNMENT 32   // AVX2 alignment, safe fallback

#if defined(ORTHOTREE_SIMD_AVX512)
#   undef ORTHOTREE_SIMD_ALIGNMENT
#   define ORTHOTREE_SIMD_ALIGNMENT 64
#endif

// Helper macro for cache‑line aligning variables
#define ORTHOTREE_ALIGN_CACHE alignas(ORTHOTREE_CACHE_LINE_SIZE)
#define ORTHOTREE_ALIGN_SIMD alignas(ORTHOTREE_SIMD_ALIGNMENT)

// ============================================================================
//  Debug vs Release macros
// ============================================================================
#if defined(NDEBUG)
#   define ORTHOTREE_RELEASE 1
#   define ORTHOTREE_DEBUG 0
#else
#   define ORTHOTREE_RELEASE 0
#   define ORTHOTREE_DEBUG 1
#endif

// ============================================================================
//  Inline hints
// ============================================================================
#if defined(ORTHOTREE_COMPILER_MSVC)
#   define ORTHOTREE_FORCE_INLINE __forceinline
#   define ORTHOTREE_NO_INLINE __declspec(noinline)
#elif defined(ORTHOTREE_COMPILER_GCC) || defined(ORTHOTREE_COMPILER_CLANG)
#   define ORTHOTREE_FORCE_INLINE __attribute__((always_inline)) inline
#   define ORTHOTREE_NO_INLINE __attribute__((noinline))
#else
#   define ORTHOTREE_FORCE_INLINE inline
#   define ORTHOTREE_NO_INLINE
#endif

#define ORTHOTREE_HOT __attribute__((hot))
#define ORTHOTREE_COLD __attribute__((cold))

// ============================================================================
//  Symbol visibility (for shared libraries)
// ============================================================================
#if defined(_WIN32) || defined(__CYGWIN__)
#   ifdef ORTHOTREE_BUILD_DLL
#       define ORTHOTREE_API __declspec(dllexport)
#   else
#       define ORTHOTREE_API __declspec(dllimport)
#   endif
#   define ORTHOTREE_LOCAL
#else
#   if __GNUC__ >= 4
#       define ORTHOTREE_API __attribute__((visibility("default")))
#       define ORTHOTREE_LOCAL __attribute__((visibility("hidden")))
#   else
#       define ORTHOTREE_API
#       define ORTHOTREE_LOCAL
#   endif
#endif

// ============================================================================
//  Math precision and rounding control
// ============================================================================
#ifndef ORTHOTREE_DEFAULT_FLOAT_TYPE
#   define ORTHOTREE_DEFAULT_FLOAT_TYPE float
#endif

#ifndef ORTHOTREE_DEFAULT_DOUBLE_TYPE
#   define ORTHOTREE_DEFAULT_DOUBLE_TYPE double
#endif

// Rounding mode (for interval arithmetic, etc.)
#ifndef ORTHOTREE_MATH_ROUNDING
#   ifdef __STDC_IEC_559__
#       define ORTHOTREE_MATH_ROUNDING 1   // Use std::fesetround if available
#   else
#       define ORTHOTREE_MATH_ROUNDING 0
#   endif
#endif

// ============================================================================
//  Performance tuning knobs
// ============================================================================

/**
 * Maximum depth of octree (default 16). Larger values allow finer subdivision
 * but increase node count.
 */
#ifndef ORTHOTREE_DEFAULT_MAX_DEPTH
#   define ORTHOTREE_DEFAULT_MAX_DEPTH 16
#endif

/**
 * Bucket size: max entities per leaf before splitting.
 * Smaller values produce deeper trees, larger values produce shallower trees.
 * Recommended: 4-16.
 */
#ifndef ORTHOTREE_DEFAULT_BUCKET_SIZE
#   define ORTHOTREE_DEFAULT_BUCKET_SIZE 8
#endif

/**
 * Auto‑expand world bounds when inserting outside initial volume.
 * Enabled by default for dynamic scenes.
 */
#ifndef ORTHOTREE_AUTO_EXPAND_BOUNDS
#   define ORTHOTREE_AUTO_EXPAND_BOUNDS 1
#endif

/**
 * Use double buffering for thread‑safe read queries while updates occur.
 * Adds slight overhead but increases safety.
 */
#ifndef ORTHOTREE_USE_DOUBLE_BUFFERING
#   define ORTHOTREE_USE_DOUBLE_BUFFERING 1
#endif

/**
 * Enable PMR (polymorphic memory resources) for custom allocators.
 * Disable only if PMR is not available on your platform.
 */
#ifndef ORTHOTREE_ENABLE_PMR
#   define ORTHOTREE_ENABLE_PMR 1
#endif

/**
 * Use 64‑bit Morton codes (vs 32‑bit). Always recommended for large worlds.
 */
#ifndef ORTHOTREE_USE_64BIT_MORTON
#   define ORTHOTREE_USE_64BIT_MORTON 1
#endif

/**
 * Branchless traversal optimization (avoids conditional jumps inside loops).
 * Enabled by default on supported compilers.
 */
#ifndef ORTHOTREE_BRANCHLESS_TRAVERSAL
#   if defined(ORTHOTREE_COMPILER_CLANG) || defined(ORTHOTREE_COMPILER_GCC)
#       define ORTHOTREE_BRANCHLESS_TRAVERSAL 1
#   else
#       define ORTHOTREE_BRANCHLESS_TRAVERSAL 0
#   endif
#endif

// ============================================================================
//  Assertion macro (customizable)
// ============================================================================
#ifndef ORTHOTREE_ASSERT
#   include <cassert>
#   define ORTHOTREE_ASSERT(expr) assert(expr)
#endif

// ============================================================================
//  Compile‑time feature tests
// ============================================================================
#define ORTHOTREE_HAS_CPP17 (__cplusplus >= 201703L)
#define ORTHOTREE_HAS_CPP20 (__cplusplus >= 202002L)

#if defined(__has_include)
#   define ORTHOTREE_HAS_INCLUDE(header) __has_include(header)
#else
#   define ORTHOTREE_HAS_INCLUDE(header) 0
#endif

// Detect std::pmr availability
#if ORTHOTREE_ENABLE_PMR && ORTHOTREE_HAS_INCLUDE(<memory_resource>)
#   define ORTHOTREE_HAS_PMR 1
#else
#   define ORTHOTREE_HAS_PMR 0
#endif

// Detect if we can use aligned new/delete (C++17)
#if ORTHOTREE_HAS_CPP17
#   define ORTHOTREE_HAS_ALIGNED_NEW 1
#else
#   define ORTHOTREE_HAS_ALIGNED_NEW 0
#endif

// ============================================================================
//  Macro for deprecation warnings
// ============================================================================
#if defined(ORTHOTREE_COMPILER_MSVC)
#   define ORTHOTREE_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(ORTHOTREE_COMPILER_GCC) || defined(ORTHOTREE_COMPILER_CLANG)
#   define ORTHOTREE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#   define ORTHOTREE_DEPRECATED(msg)
#endif

// ============================================================================
//  Final configuration summary (printed in debug build if requested)
// ============================================================================
#if ORTHOTREE_DEBUG && defined(ORTHOTREE_VERBOSE_CONFIG)
#pragma message("OrthoTree Build Configuration:")
#pragma message("  Compiler: " ORTHOTREE_COMPILER_NAME)
#pragma message("  SIMD Level: " ORTHOTREE_SIMD_LEVEL)
#pragma message("  Endianness: " (ORTHOTREE_LITTLE_ENDIAN ? "Little" : "Big"))
#pragma message("  Max Depth: " ORTHOTREE_DEFAULT_MAX_DEPTH)
#pragma message("  Bucket Size: " ORTHOTREE_DEFAULT_BUCKET_SIZE)
#pragma message("  PMR Enabled: " ORTHOTREE_HAS_PMR)
#endif

#endif // ORTHOTREE_CORE_BUILD_CONFIG_H_INCLUDED
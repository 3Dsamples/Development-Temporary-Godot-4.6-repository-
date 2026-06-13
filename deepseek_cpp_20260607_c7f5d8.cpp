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
 * @file undefs.h
 * @brief Cleanup of temporary macros used in the OrthoTree implementation.
 *
 * This file undefines all helper macros that were defined in other detail headers
 * to prevent namespace pollution. It is intended to be included at the end of
 * any header that uses internal macros, after all definitions have been processed.
 *
 * Usage:
 *   #include "detail/undefs.h"
 *
 * The macros cleaned up include:
 * - Temporary macros for compiler-specific workarounds
 * - Debug/assert helpers that are not part of the public API
 * - Inline hints and portability macros
 * - Anything that was defined with a limited scope
 */

#ifndef ORTHOTREE_DETAIL_UNDEFS_H_INCLUDED
#define ORTHOTREE_DETAIL_UNDEFS_H_INCLUDED

// ----------------------------------------------------------------------------
//  Macros that may have been defined in build_config.h or common.h
//  These are internal implementation details; we undefine them to avoid
//  leaking into user code.
// ----------------------------------------------------------------------------

// Compiler detection macros (only needed during build, not at runtime)
#ifdef ORTHOTREE_COMPILER_MSVC
#undef ORTHOTREE_COMPILER_MSVC
#endif
#ifdef ORTHOTREE_COMPILER_GCC
#undef ORTHOTREE_COMPILER_GCC
#endif
#ifdef ORTHOTREE_COMPILER_CLANG
#undef ORTHOTREE_COMPILER_CLANG
#endif
#ifdef ORTHOTREE_COMPILER_UNKNOWN
#undef ORTHOTREE_COMPILER_UNKNOWN
#endif
#ifdef ORTHOTREE_COMPILER_NAME
#undef ORTHOTREE_COMPILER_NAME
#endif
#ifdef ORTHOTREE_COMPILER_VERSION
#undef ORTHOTREE_COMPILER_VERSION
#endif

// OS detection macros
#ifdef ORTHOTREE_OS_WINDOWS
#undef ORTHOTREE_OS_WINDOWS
#endif
#ifdef ORTHOTREE_OS_LINUX
#undef ORTHOTREE_OS_LINUX
#endif
#ifdef ORTHOTREE_OS_MACOS
#undef ORTHOTREE_OS_MACOS
#endif
#ifdef ORTHOTREE_OS_ANDROID
#undef ORTHOTREE_OS_ANDROID
#endif
#ifdef ORTHOTREE_OS_UNIX
#undef ORTHOTREE_OS_UNIX
#endif
#ifdef ORTHOTREE_OS_OTHER
#undef ORTHOTREE_OS_OTHER
#endif

// Endianness macros
#ifdef ORTHOTREE_LITTLE_ENDIAN
#undef ORTHOTREE_LITTLE_ENDIAN
#endif
#ifdef ORTHOTREE_BIG_ENDIAN
#undef ORTHOTREE_BIG_ENDIAN
#endif

// SIMD level macros
#ifdef ORTHOTREE_SIMD_LEVEL
#undef ORTHOTREE_SIMD_LEVEL
#endif
#ifdef ORTHOTREE_SIMD_AVX512
#undef ORTHOTREE_SIMD_AVX512
#endif
#ifdef ORTHOTREE_SIMD_AVX2
#undef ORTHOTREE_SIMD_AVX2
#endif
#ifdef ORTHOTREE_SIMD_AVX
#undef ORTHOTREE_SIMD_AVX
#endif
#ifdef ORTHOTREE_SIMD_SSE4
#undef ORTHOTREE_SIMD_SSE4
#endif
#ifdef ORTHOTREE_SIMD_SSE3
#undef ORTHOTREE_SIMD_SSE3
#endif
#ifdef ORTHOTREE_SIMD_SSE2
#undef ORTHOTREE_SIMD_SSE2
#endif
#ifdef ORTHOTREE_SIMD_NEON
#undef ORTHOTREE_SIMD_NEON
#endif
#ifdef ORTHOTREE_SIMD_VSX
#undef ORTHOTREE_SIMD_VSX
#endif
#ifdef ORTHOTREE_SIMD_NONE
#undef ORTHOTREE_SIMD_NONE
#endif

// Alignment macros
#ifdef ORTHOTREE_CACHE_LINE_SIZE
#undef ORTHOTREE_CACHE_LINE_SIZE
#endif
#ifdef ORTHOTREE_SIMD_ALIGNMENT
#undef ORTHOTREE_SIMD_ALIGNMENT
#endif
#ifdef ORTHOTREE_ALIGN_CACHE
#undef ORTHOTREE_ALIGN_CACHE
#endif
#ifdef ORTHOTREE_ALIGN_SIMD
#undef ORTHOTREE_ALIGN_SIMD
#endif

// Debug/release macros
#ifdef ORTHOTREE_RELEASE
#undef ORTHOTREE_RELEASE
#endif
#ifdef ORTHOTREE_DEBUG
#undef ORTHOTREE_DEBUG
#endif

// Inline hints
#ifdef ORTHOTREE_FORCE_INLINE
#undef ORTHOTREE_FORCE_INLINE
#endif
#ifdef ORTHOTREE_NO_INLINE
#undef ORTHOTREE_NO_INLINE
#endif
#ifdef ORTHOTREE_HOT
#undef ORTHOTREE_HOT
#endif
#ifdef ORTHOTREE_COLD
#undef ORTHOTREE_COLD
#endif

// Symbol visibility
#ifdef ORTHOTREE_API
#undef ORTHOTREE_API
#endif
#ifdef ORTHOTREE_LOCAL
#undef ORTHOTREE_LOCAL
#endif

// Performance tuning macros
#ifdef ORTHOTREE_DEFAULT_MAX_DEPTH
#undef ORTHOTREE_DEFAULT_MAX_DEPTH
#endif
#ifdef ORTHOTREE_DEFAULT_BUCKET_SIZE
#undef ORTHOTREE_DEFAULT_BUCKET_SIZE
#endif
#ifdef ORTHOTREE_AUTO_EXPAND_BOUNDS
#undef ORTHOTREE_AUTO_EXPAND_BOUNDS
#endif
#ifdef ORTHOTREE_USE_DOUBLE_BUFFERING
#undef ORTHOTREE_USE_DOUBLE_BUFFERING
#endif
#ifdef ORTHOTREE_ENABLE_PMR
#undef ORTHOTREE_ENABLE_PMR
#endif
#ifdef ORTHOTREE_USE_64BIT_MORTON
#undef ORTHOTREE_USE_64BIT_MORTON
#endif
#ifdef ORTHOTREE_BRANCHLESS_TRAVERSAL
#undef ORTHOTREE_BRANCHLESS_TRAVERSAL
#endif

// Assertion macro
#ifdef ORTHOTREE_ASSERT
#undef ORTHOTREE_ASSERT
#endif
#ifdef ORTHOTREE_ASSERT_MSG
#undef ORTHOTREE_ASSERT_MSG
#endif

// Feature test macros
#ifdef ORTHOTREE_HAS_CPP17
#undef ORTHOTREE_HAS_CPP17
#endif
#ifdef ORTHOTREE_HAS_CPP20
#undef ORTHOTREE_HAS_CPP20
#endif
#ifdef ORTHOTREE_HAS_INCLUDE
#undef ORTHOTREE_HAS_INCLUDE
#endif
#ifdef ORTHOTREE_HAS_PMR
#undef ORTHOTREE_HAS_PMR
#endif
#ifdef ORTHOTREE_HAS_ALIGNED_NEW
#undef ORTHOTREE_HAS_ALIGNED_NEW
#endif

// Deprecation macro
#ifdef ORTHOTREE_DEPRECATED
#undef ORTHOTREE_DEPRECATED
#endif

// Branch prediction hints
#ifdef ORTHOTREE_LIKELY
#undef ORTHOTREE_LIKELY
#endif
#ifdef ORTHOTREE_UNLIKELY
#undef ORTHOTREE_UNLIKELY
#endif

// Unreachable annotation
#ifdef ORTHOTREE_UNREACHABLE
#undef ORTHOTREE_UNREACHABLE
#endif

// Static assertion macro
#ifdef ORTHOTREE_STATIC_ASSERT
#undef ORTHOTREE_STATIC_ASSERT
#endif

// Tag dispatch helpers (Priority, etc. – not macro but we can undef if needed)
// No macros there, so fine.

// Helper macros from bitset_arithmetic and others (none left)

// Finally, we do not undef ORTHOTREE_VERBOSE_CONFIG because it is user-defined.

#endif // ORTHOTREE_DETAIL_UNDEFS_H_INCLUDED
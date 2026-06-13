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
 * @file types.h
 * @brief Fundamental type definitions and aliases for the OrthoTree library.
 *
 * This file defines the basic scalar types, dimensional constants, index types,
 * and compile‑time tags used throughout the spatial indexing system. It also
 * provides type traits for detecting dimensions, scalar types, and memory
 * alignment requirements. All components rely on these consistent type aliases.
 *
 * Key features:
 * - `Dimension` enum (Dim2, Dim3) for 2D/3D polymorphism
 * - `Scalar` type alias (default `float`) for generic code
 * - `Index` type (`uint32_t`) for compact node/entity references
 * - `MortonCode` type (`uint64_t`) for space‑filling curves
 * - Alignment helpers for SIMD and cache‑line optimised structures
 * - Type trait `is_dimension_v`, `is_scalar_v`, `has_aligned_storage_v`
 */

#pragma once
#ifndef ORTHOTREE_CORE_TYPES_H_INCLUDED
#define ORTHOTREE_CORE_TYPES_H_INCLUDED

#include "build_config.h"

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <limits>

namespace OrthoTree {

// ============================================================================
//  Dimension tag (compile‑time 2D/3D selection)
// ============================================================================

/**
 * @brief Dimension specifier for 2D (quadtree) or 3D (octree).
 */
enum Dimension : uint8_t {
    Dim2 = 2,
    Dim3 = 3
};

/**
 * @brief Convert dimension enum to compile‑time integral constant.
 */
template <Dimension D>
struct DimensionConstant {
    static constexpr Dimension value = D;
    static constexpr int dim = static_cast<int>(D);
};

// ============================================================================
//  Basic scalar types (with user‑configurable defaults)
// ============================================================================

/**
 * @brief Default floating‑point type for coordinates and distances.
 *
 * Override by defining ORTHOTREE_SCALAR_TYPE before including orthotree headers.
 */
#ifndef ORTHOTREE_SCALAR_TYPE
    #define ORTHOTREE_SCALAR_TYPE float
#endif

using Scalar = ORTHOTREE_SCALAR_TYPE;

/**
 * @brief High‑precision scalar (double) for simulations requiring accuracy.
 */
using HighPrecisionScalar = double;

/**
 * @brief Signed integer type for entity IDs and array indices.
 */
using Index = int32_t;

/**
 * @brief Unsigned index for node indices (compact storage).
 */
using NodeIndex = uint32_t;

/**
 * @brief Unsigned index for entity references inside leaves.
 */
using LocalEntityIndex = uint16_t;

/**
 * @brief Morton code type (space‑filling curve).
 *
 * Uses 64‑bit for sufficient precision even in very deep octrees.
 */
#if ORTHOTREE_USE_64BIT_MORTON
    using MortonCode = uint64_t;
#else
    using MortonCode = uint32_t;
#endif

/**
 * @brief Type for depth values (0..maxDepth ≤ 255).
 */
using DepthType = uint8_t;

/**
 * @brief Type for child counts and small counters.
 */
using SmallCount = uint8_t;

/**
 * @brief Type for storing flags (e.g., isLeaf, hasEntities).
 */
using FlagType = uint8_t;

// ============================================================================
//  Special index constants
// ============================================================================

constexpr NodeIndex INVALID_NODE_INDEX = static_cast<NodeIndex>(-1);
constexpr Index INVALID_INDEX = -1;
constexpr MortonCode INVALID_MORTON = static_cast<MortonCode>(-1);
constexpr DepthType MAX_DEPTH = 255;   // fits in uint8_t

// ============================================================================
//  Alignment requirements (cache‑line and SIMD)
// ============================================================================

/**
 * @brief Guaranteed cache line size (bytes) – usually 64 on modern CPUs.
 */
constexpr std::size_t CACHE_LINE_SIZE = ORTHOTREE_CACHE_LINE_SIZE;

/**
 * @brief Alignment for SIMD vectors (e.g., SSE, AVX).
 */
constexpr std::size_t SIMD_ALIGNMENT = ORTHOTREE_SIMD_ALIGNMENT;

/**
 * @brief Helper to compute padded size to meet alignment.
 */
template <typename T>
static constexpr std::size_t aligned_size(std::size_t alignment = alignof(T)) noexcept {
    return (sizeof(T) + alignment - 1) & ~(alignment - 1);
}

// ============================================================================
//  Type traits
// ============================================================================

/**
 * @brief Trait to detect if a type is a supported scalar (float or double).
 */
template <typename T>
struct is_scalar : std::integral_constant<bool,
                                          std::is_floating_point_v<T> &&
                                              (std::is_same_v<T, float> ||
                                               std::is_same_v<T, double>)> {};

template <typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

/**
 * @brief Trait to detect dimension enum.
 */
template <Dimension D>
struct is_dimension : std::true_type {};

template <int d>
struct is_dimension_int : std::integral_constant<bool, d == 2 || d == 3> {};

template <typename T>
struct is_dimension_type : std::false_type {};

template <>
struct is_dimension_type<Dimension> : std::true_type {};

template <typename T>
inline constexpr bool is_dimension_type_v = is_dimension_type<T>::value;

/**
 * @brief Get dimension as compile‑time integer.
 */
template <Dimension D>
constexpr int dimension_to_int() noexcept {
    return static_cast<int>(D);
}

/**
 * @brief Number of children per node: 4 for 2D, 8 for 3D.
 */
template <Dimension D>
constexpr SmallType num_children() noexcept {
    return (D == Dim2) ? 4 : 8;
}

/**
 * @brief Number of axes: 2 for 2D, 3 for 3D.
 */
template <Dimension D>
constexpr SmallType num_axes() noexcept {
    return static_cast<SmallType>(D);
}

// ============================================================================
//  Tags for policy‑based design (compile‑time dispatch)
// ============================================================================

struct dynamic_tag {};      ///< Dynamic octree (hashed, mutable)
struct static_tag {};       ///< Static BVH (linear, immutable)
struct hybrid_tag {};       ///< Hybrid (managed, auto‑refine)

struct fast_construction_tag {};   ///< Optimise for build speed
struct fast_query_tag {};          ///< Optimise for query speed
struct balanced_tag {};            ///< Balanced trade‑off

struct memory_safe_tag {};    ///< Use bounds checks, assertions
struct performance_tag {};    ///< Assume correct input, skip checks

// ============================================================================
//  Helper for compile‑time dispatching based on dimension
// ============================================================================

template <Dimension D, typename T = void>
struct dimension_dispatch {
    // Provide static methods that take dimension as template parameter.
    static constexpr Dimension value = D;
};

// Specialisations for 2D and 3D (can be extended with specific optimisations)
template <typename T>
struct dimension_dispatch<Dim2, T> {
    static constexpr int dim = 2;
    static constexpr SmallType children = 4;
    using bounds_type = Math::AxisAlignedBox<T, 2>;
};

template <typename T>
struct dimension_dispatch<Dim3, T> {
    static constexpr int dim = 3;
    static constexpr SmallType children = 8;
    using bounds_type = Math::AxisAlignedBox<T, 3>;
};

// ============================================================================
//  Memory hints (for allocators)
// ============================================================================

/**
 * @brief Hints for expected lifetime and usage patterns.
 */
enum class MemoryHint : uint8_t {
    Unknown,
    ShortLived,       ///< Temporary, frequent allocations
    LongLived,        ///< Persistent, rarely changed
    Streaming,        ///< Append‑only, then read‑only
    Interactive       ///< Frequent small updates
};

/**
 * @brief Size classes for small‑buffer optimisation (inplace vectors).
 */
using SmallBufferSize = std::integral_constant<std::size_t, 8>;

// ============================================================================
//  Version type for double‑buffering (concurrency)
// ============================================================================

/**
 * @brief Monotonic version counter for detecting changes.
 */
using VersionType = uint64_t;

constexpr VersionType INVALID_VERSION = 0;

// ============================================================================
//  Utility: safe narrowing casts (debug builds check overflow)
// ============================================================================

template <typename To, typename From>
constexpr To narrow_cast(From value) noexcept {
    static_assert(std::is_integral_v<To> && std::is_integral_v<From>,
                  "narrow_cast only for integral types");
    To result = static_cast<To>(value);
#if ORTHOTREE_DEBUG
    // In debug, ensure no loss of information
    ORTHOTREE_ASSERT(static_cast<From>(result) == value &&
                     ((value < 0) == (result < 0)));
#endif
    return result;
}

// ============================================================================
//  Helper for calculating ideal node size based on dimension and scalar
// ============================================================================

template <Dimension D, typename T>
struct OptimalNodeSize {
    static constexpr std::size_t value =
        (D == Dim2) ? (64 / sizeof(T)) : (128 / sizeof(T));
};

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_TYPES_H_INCLUDED
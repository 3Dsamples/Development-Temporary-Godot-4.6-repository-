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
 * @file orthotree.h
 * @brief Main entry point for OrthoTree library – unified spatial indexing for 2D/3D real-time and simulation.
 *
 * This file brings together all components of the high-performance Octree/BVH implementation.
 * It provides a complete, easy-to-use interface with advanced math, geometry queries,
 * serialization, and custom memory management. Designed for C++17, low memory footprint,
 * and maximum runtime efficiency.
 *
 * Supported features:
 * - 2D/3D dynamic and static octrees (quadtrees/octrees)
 * - Linear BVH (LBVH) for static scenes
 * - Spatial hashing for uniform grids
 * - PMR allocators for deterministic performance
 * - SIMD-aware bounding box and intersection tests
 * - Multi-threading support via parallel algorithms
 * - Serialization (binary, msgpack) with STL and math adapter support
 * - Real-time friendly: incremental updates, fast queries
 */

#pragma once
#ifndef ORTHOTREE__OCTREE_H_INCLUDED
#define ORTHOTREE__OCTREE_H_INCLUDED

// ============================================================================
//  Build configuration & feature test macros
// ============================================================================
#include "core/build_config.h"

// ----------------------------------------------------------------------------
//  New advanced math modules (C++17, high precision, SIMD ready)
// ----------------------------------------------------------------------------
#include "core/math/vector_math.h"
#include "core/math/interval_arithmetic.h"
#include "core/math/quaternion.h"
#include "core/math/transform.h"
#include "core/math/geometry_queries.h"
#include "core/math/numerical_methods.h"

// ----------------------------------------------------------------------------
//  Low-level utilities and data structures
// ----------------------------------------------------------------------------
#include "detail/bitset_arithmetic.h"       // Bitwise operations for morton codes
#include "detail/common.h"                  // Common macros and helpers
#include "detail/inplace_vector.h"          // Stack-allocated vector for small collections
#include "detail/internal_geometry_module.h"// Internal geometry predicates
#include "detail/memory_resource.h"         // Polymorphic memory resource wrappers
#include "detail/partitioning.h"            // Spatial partitioning helpers
#include "detail/sequence_view.h"           // Non-owning view over contiguous memory
#include "detail/si_mortongrid.h"           // Morton code based grid (spatial hashing)
#include "detail/utils.h"                   // General utilities (hash, align, etc.)
#include "detail/zip_view.h"                // Range adaptor for zipping multiple ranges

// ----------------------------------------------------------------------------
//  Core spatial index components
// ----------------------------------------------------------------------------
#include "core/configuration.h"             // Global compile-time tuning knobs
#include "core/types.h"                     // Fundamental type aliases (Scalar, Index, etc.)
#include "core/entity_adapter.h"            // Adapter for user-defined geometry types
#include "core/ot_dynamic_hash_core.h"      // Dynamic octree with hashed nodes
#include "core/ot_static_linear_core.h"     // Static linear BVH (LBVH)
#include "core/ot_managed.h"                // Managed octree with automatic expansion/shrink
#include "core/ot_query.h"                  // Query engine (raycast, range, nearest neighbor)
#include "adapters/general.h"               // Default adapters for built-in types
#include "core/ot_aliases.h"                // Convenience type aliases

// ============================================================================
//  High-level user interface – recommended types
// ============================================================================

namespace OrthoTree {

/**
 * @brief Main octree class for dynamic scenes (2D/3D).
 *
 * Uses hashed node storage (sparse octree) – optimal for large, sparse, and
 * frequently updated scenes. Supports insertion, removal, point/box queries,
 * raycasts, and nearest neighbor search.
 *
 * @tparam Dim Dimension: 2 for quadtree, 3 for octree.
 * @tparam T Scalar type (float/double).
 * @tparam Allocator Allocator type (default: polymorphic memory resource).
 */
template <Dimension Dim, typename T = float,
          typename Allocator = PMRAllocator<std::byte>>
using DynamicOctree = ot_dynamic_hash_core<Dim, T, Allocator>;

/**
 * @brief Static linear BVH (LBVH) for completely static geometry.
 *
 * Builds a very fast, memory‑compact BVH using Morton codes and a linear
 * representation. Ideal for static environments where query performance is
 * critical and updates are not required.
 *
 * @tparam Dim Dimension (2/3).
 * @tparam T Scalar type.
 * @tparam Allocator Allocator.
 */
template <Dimension Dim, typename T = float,
          typename Allocator = PMRAllocator<std::byte>>
using LinearBVH = ot_static_linear_core<Dim, T, Allocator>;

/**
 * @brief Managed octree that automatically refines/coarsens based on load.
 *
 * Combines dynamic hashed octree with a configurable depth limit and bucket
 * size. Useful for adaptive simulation (e.g., particle systems, fluid dynamics).
 *
 * @tparam Dim Dimension.
 * @tparam T Scalar.
 */
template <Dimension Dim, typename T = float>
using ManagedOctree = ot_managed<Dim, T>;

/**
 * @brief Default 3D octree (float) for most applications.
 */
using Octree3f = DynamicOctree<Dim3, float>;

/**
 * @brief Default 2D quadtree (float).
 */
using Quadtree2f = DynamicOctree<Dim2, float>;

/**
 * @brief Double‑precision 3D octree for high accuracy simulations.
 */
using Octree3d = DynamicOctree<Dim3, double>;

/**
 * @brief Static 3D BVH (float) for static meshes.
 */
using StaticBVH3f = LinearBVH<Dim3, float>;

} // namespace OrthoTree

// ============================================================================
//  Serialization (optional, header‑only)
// ============================================================================
#include "serialization.h"

#endif // ORTHOTREE__OCTREE_H_INCLUDED
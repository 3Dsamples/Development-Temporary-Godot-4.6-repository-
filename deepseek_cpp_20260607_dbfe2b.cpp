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
 * @file serialization.h
 * @brief Unified serialization framework for orthotree data structures.
 *
 * This file provides binary and msgpack serialization for octrees, BVHs,
 * and all associated math types (vectors, matrices, quaternions, intervals, etc.).
 * It integrates with STL containers and external math libraries via adapters.
 * Supports endianness handling, versioning, and custom allocators for zero‑copy
 * deserialization.
 *
 * Features:
 * - Binary archive (portable, compact)
 * - MessagePack archive (interoperable, self‑describing)
 * - Adapters for GLM, Eigen, CGAL, Unreal, Boost.Geometry
 * - STL container serialization (vector, map, set, optional, variant)
 * - Version tolerance and backwards compatibility
 * - PMR allocator support for efficient reconstruction
 */

#pragma once
#ifndef ORTHOTREE__SERIALIZATION_H_INCLUDED
#define ORTHOTREE__SERIALIZATION_H_INCLUDED

#include "core/build_config.h"
#include "core/types.h"
#include "core/math/vector_math.h"
#include "core/math/interval_arithmetic.h"
#include "core/math/quaternion.h"
#include "core/math/transform.h"
#include "core/math/geometry_queries.h"
#include "core/math/numerical_methods.h"
#include "detail/memory_resource.h"
#include "serialization/binary_archive.h"
#include "serialization/msgpack_archive.h"
#include "serialization/nvp.h"
#include "serialization/traits.h"
#include "serialization/stl.h"

#include <type_traits>
#include <cstdint>
#include <vector>
#include <map>
#include <unordered_map>
#include <optional>
#include <variant>

namespace OrthoTree {
namespace serialization {

// ----------------------------------------------------------------------------
//  Archive versioning
// ----------------------------------------------------------------------------
constexpr uint32_t ORTHOTREE_SERIALIZATION_VERSION = 1;

// ----------------------------------------------------------------------------
//  Serialization of math types (forward declarations for binary/msgpack)
// ----------------------------------------------------------------------------

// Vector<N,T>
template <typename Archive, typename T, std::size_t N>
typename std::enable_if_t<is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>>
serialize(Archive& ar, Math::Vector<T, N>& vec, const uint32_t version) {
    for (std::size_t i = 0; i < N; ++i) {
        ar & make_nvp(("v" + std::to_string(i)).c_str(), vec[i]);
    }
}

// Interval<T>
template <typename Archive, typename T>
typename std::enable_if_t<is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>>
serialize(Archive& ar, Math::Interval<T>& interval, const uint32_t version) {
    ar & make_nvp("low", interval.low());
    ar & make_nvp("high", interval.high());
}

// Hyperrectangle<T,N>
template <typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, Math::Hyperrectangle<T, N>& rect, const uint32_t version) {
    for (std::size_t i = 0; i < N; ++i) {
        ar & make_nvp(("dim" + std::to_string(i)).c_str(), rect[i]);
    }
}

// Quaternion<T>
template <typename Archive, typename T>
void serialize(Archive& ar, Math::Quaternion<T>& q, const uint32_t version) {
    ar & make_nvp("w", q.w());
    ar & make_nvp("x", q.x());
    ar & make_nvp("y", q.y());
    ar & make_nvp("z", q.z());
}

// Matrix<T,N>
template <typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, Math::Matrix<T, N>& mat, const uint32_t version) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            ar & make_nvp(("m" + std::to_string(i) + std::to_string(j)).c_str(), mat(i, j));
        }
    }
}

// AffineTransform<T,N>
template <typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, Math::AffineTransform<T, N>& tf, const uint32_t version) {
    ar & make_nvp("matrix", tf.matrix());
    ar & make_nvp("translation", tf.translation());
}

// AxisAlignedBox<T,N>
template <typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, Math::AxisAlignedBox<T, N>& aabb, const uint32_t version) {
    ar & make_nvp("min", aabb.min());
    ar & make_nvp("max", aabb.max());
}

// Sphere<T,N>
template <typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, Math::Sphere<T, N>& sphere, const uint32_t version) {
    ar & make_nvp("center", sphere.center());
    ar & make_nvp("radius", sphere.radius());
}

// Ray<T,N>
template <typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, Math::Ray<T, N>& ray, const uint32_t version) {
    ar & make_nvp("origin", ray.origin());
    ar & make_nvp("direction", ray.direction());
}

// Plane<T,N>
template <typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, Math::Plane<T, N>& plane, const uint32_t version) {
    ar & make_nvp("normal", plane.normal());
    ar & make_nvp("d", plane.d());
}

// ----------------------------------------------------------------------------
//  Octree node serialization (used by octree)
// ----------------------------------------------------------------------------
template <typename Archive, Dimension Dim, typename T>
void serialize(Archive& ar, OctreeNode<Dim, T>& node, const uint32_t version) {
    ar & make_nvp("bounds", node.bounds);
    ar & make_nvp("children", node.children);
    ar & make_nvp("firstEntity", node.firstEntity);
    ar & make_nvp("entityCount", node.entityCount);
    ar & make_nvp("depth", node.depth);
    ar & make_nvp("splitAxis", node.splitAxis);
    ar & make_nvp("isLeaf", node.isLeaf);
    ar & make_nvp("hasEntities", node.hasEntities);
}

// ----------------------------------------------------------------------------
//  Octree serialization (core container)
// ----------------------------------------------------------------------------
template <typename Archive, Dimension Dim, typename T, typename Allocator>
void serialize(Archive& ar, Octree<Dim, T, Allocator>& tree, const uint32_t version) {
    // Serialize world bounds, maxDepth, bucketSize
    ar & make_nvp("worldBounds", tree.m_worldBounds);
    ar & make_nvp("maxDepth", tree.m_maxDepth);
    ar & make_nvp("bucketSize", tree.m_bucketSize);
    ar & make_nvp("nodeCount", tree.m_nodeCount);
    ar & make_nvp("entityCount", tree.m_entityCount);
    
    // Serialize nodes
    ar & make_nvp("nodes", tree.m_nodes);
    // Serialize entities
    ar & make_nvp("entities", tree.m_entities);
    // Serialize node map (as vector of pairs)
    if constexpr (is_output_archive_v<Archive>) {
        std::vector<std::pair<uint64_t, uint32_t>> mapPairs(tree.m_nodeMap.begin(), tree.m_nodeMap.end());
        ar & make_nvp("nodeMap", mapPairs);
    } else {
        std::vector<std::pair<uint64_t, uint32_t>> mapPairs;
        ar & make_nvp("nodeMap", mapPairs);
        tree.m_nodeMap.clear();
        for (auto& p : mapPairs) {
            tree.m_nodeMap[p.first] = p.second;
        }
    }
    ar & make_nvp("root", tree.m_root);
    ar & make_nvp("version", tree.m_version);
}

// ----------------------------------------------------------------------------
//  Static linear BVH serialization
// ----------------------------------------------------------------------------
template <typename Archive, Dimension Dim, typename T, typename Allocator>
void serialize(Archive& ar, ot_static_linear_core<Dim, T, Allocator>& bvh, const uint32_t version) {
    ar & make_nvp("bounds", bvh.m_globalBounds);
    ar & make_nvp("nodes", bvh.m_nodes);
    ar & make_nvp("entities", bvh.m_entities);
    ar & make_nvp("primitiveIndices", bvh.m_primitiveIndices);
    ar & make_nvp("rootIndex", bvh.m_rootIndex);
    ar & make_nvp("nodeCount", bvh.m_nodeCount);
}

// ----------------------------------------------------------------------------
//  High‑level save/load helpers
// ----------------------------------------------------------------------------

/**
 * @brief Save an octree to a binary file.
 * @param tree Octree to save.
 * @param filename Output file path.
 * @return True if successful.
 */
template <Dimension Dim, typename T, typename Allocator>
bool saveToBinary(const Octree<Dim, T, Allocator>& tree, const std::string& filename) {
    BinaryOutputArchive ar;
    ar & make_nvp("version", ORTHOTREE_SERIALIZATION_VERSION);
    ar & make_nvp("tree", const_cast<Octree<Dim, T, Allocator>&>(tree));
    return ar.saveToFile(filename);
}

/**
 * @brief Load an octree from a binary file.
 * @param tree Output octree (will be cleared and filled).
 * @param filename Input file path.
 * @return True if successful.
 */
template <Dimension Dim, typename T, typename Allocator>
bool loadFromBinary(Octree<Dim, T, Allocator>& tree, const std::string& filename) {
    BinaryInputArchive ar;
    if (!ar.loadFromFile(filename)) return false;
    uint32_t version = 0;
    ar & make_nvp("version", version);
    if (version > ORTHOTREE_SERIALIZATION_VERSION) return false;
    ar & make_nvp("tree", tree);
    return true;
}

/**
 * @brief Save to MessagePack format (compact, cross‑platform).
 * @param tree Octree.
 * @param filename Output.
 * @return True.
 */
template <Dimension Dim, typename T, typename Allocator>
bool saveToMsgPack(const Octree<Dim, T, Allocator>& tree, const std::string& filename) {
    MsgPackOutputArchive ar;
    ar & make_nvp("version", ORTHOTREE_SERIALIZATION_VERSION);
    ar & make_nvp("tree", const_cast<Octree<Dim, T, Allocator>&>(tree));
    return ar.saveToFile(filename);
}

/**
 * @brief Load from MessagePack.
 */
template <Dimension Dim, typename T, typename Allocator>
bool loadFromMsgPack(Octree<Dim, T, Allocator>& tree, const std::string& filename) {
    MsgPackInputArchive ar;
    if (!ar.loadFromFile(filename)) return false;
    uint32_t version = 0;
    ar & make_nvp("version", version);
    if (version > ORTHOTREE_SERIALIZATION_VERSION) return false;
    ar & make_nvp("tree", tree);
    return true;
}

// ----------------------------------------------------------------------------
//  Adapter includes (user can include selectively)
// ----------------------------------------------------------------------------
#include "serialization/adapters/boost.h"
#include "serialization/adapters/cgal.h"
#include "serialization/adapters/eigen.h"
#include "serialization/adapters/glm.h"
#include "serialization/adapters/xyz.h"

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE__SERIALIZATION_H_INCLUDED
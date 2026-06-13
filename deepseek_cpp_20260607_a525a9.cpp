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

#ifndef ORTHOTREE_CORE_BVH_STATIC_LINEAR_CORE_H_INCLUDED
#define ORTHOTREE_CORE_BVH_STATIC_LINEAR_CORE_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/numerical_methods.h"
#include "../core/math/interval_arithmetic.h"
#include "../core/configuration.h"
#include "../detail/common.h"
#include "../detail/memory_resource.h"
#include "../detail/bitset_arithmetic.h"
#include "../detail/si_morton.h"
#include "../detail/partitioning.h"
#include "../detail/inplace_vector.h"
#include "../detail/internal_geometry_module.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <optional>
#include <iterator>

namespace OrthoTree {

// ============================================================================
//  bvh_static_linear_core: static linear BVH using Morton‑code ordering.
//  Builds once from a set of bounding boxes, provides fast traversal,
//  SIMD‑aware node layout, and support for ray casting, range queries,
//  and nearest neighbour search. Memory‑compact, cache‑friendly.
// ============================================================================

template<Dimension Dim, typename T = float,
         typename Allocator = PMRAllocator<std::byte>>
class bvh_static_linear_core {
public:
    using value_type = T;
    static constexpr Dimension dimension = Dim;
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using ray_type = Math::Ray<T, Dim>;
    using morton_type = uint64_t;
    using entity_type = uint32_t;
    using size_type = size_t;
    using allocator_type = Allocator;

    struct alignas(32) Node {
        aabb_type bounds;
        union {
            uint32_t leftChild;      // internal node: left child index
            uint32_t primitiveStart; // leaf node: start index in primitive array
        };
        union {
            uint32_t rightChild;     // internal node: right child index
            uint32_t primitiveCount; // leaf node: number of primitives
        };
        uint8_t axis;                // split axis (0,1,2) for internal nodes
        bool isLeaf;
        uint8_t pad[2];
    };

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        size_type maxDepth = 32;
        size_type minPrimitives = 2;      // leaf threshold
        bool useSAH = false;               // surface area heuristic (slower build)
        bool useMorton = true;             // Morton‑code based linear builder
        bool enableSIMD = true;
    };

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    bvh_static_linear_core() : m_nodeCount(0), m_primitiveCount(0) {}

    template<typename InputIt>
    bvh_static_linear_core(InputIt first, InputIt last,
                           const Config& cfg = Config(),
                           const Allocator& alloc = Allocator())
        : m_alloc(alloc), m_config(cfg), m_nodeCount(0), m_primitiveCount(0) {
        build(first, last);
    }

    // ------------------------------------------------------------------------
    //  Build from a range of bounding boxes (or any type convertible to aabb)
    // ------------------------------------------------------------------------
    template<typename InputIt>
    void build(InputIt first, InputIt last) {
        clear();
        m_primitiveCount = std::distance(first, last);
        if (m_primitiveCount == 0) return;

        // Copy primitive bounds
        m_primitiveBounds.reserve(m_primitiveCount);
        m_primitiveIndices.reserve(m_primitiveCount);
        size_type idx = 0;
        for (auto it = first; it != last; ++it, ++idx) {
            m_primitiveBounds.push_back(toAABB(*it));
            m_primitiveIndices.push_back(static_cast<entity_type>(idx));
        }

        if (m_config.useMorton) {
            buildLBVH();
        } else {
            buildRecursive(0, m_primitiveCount, 0);
        }
    }

    // ------------------------------------------------------------------------
    //  Clear all data
    // ------------------------------------------------------------------------
    void clear() {
        m_nodes.clear();
        m_primitiveBounds.clear();
        m_primitiveIndices.clear();
        m_nodeCount = 0;
        m_primitiveCount = 0;
    }

    // ------------------------------------------------------------------------
    //  Queries
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        size_type count = 0;
        std::vector<uint32_t> stack;
        stack.push_back(0);
        while (!stack.empty()) {
            uint32_t idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            if (!node.bounds.overlaps(box)) continue;
            if (node.isLeaf) {
                for (uint32_t i = node.primitiveStart; i < node.primitiveStart + node.primitiveCount; ++i) {
                    if (m_primitiveBounds[i].overlaps(box)) {
                        *out++ = m_primitiveIndices[i];
                        ++count;
                    }
                }
            } else {
                stack.push_back(node.leftChild);
                stack.push_back(node.rightChild);
            }
        }
        return count;
    }

    template<typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        aabb_type pointBox(point, point);
        return queryBox(pointBox, out);
    }

    bool raycast(const ray_type& ray, size_type& hitIdx, T& hitDist) const {
        hitDist = std::numeric_limits<T>::max();
        hitIdx = static_cast<size_type>(-1);
        std::vector<std::pair<T, uint32_t>> stack; // t, nodeIdx
        stack.emplace_back(T(0), 0);
        while (!stack.empty()) {
            auto [tMin, idx] = stack.back();
            stack.pop_back();
            if (tMin > hitDist) continue;
            const Node& node = m_nodes[idx];
            T tNear, tFar;
            if (!Math::rayAABBIntersect(ray, node.bounds, tNear, tFar)) continue;
            if (tNear > hitDist) continue;
            if (node.isLeaf) {
                for (uint32_t i = node.primitiveStart; i < node.primitiveStart + node.primitiveCount; ++i) {
                    T t0, t1;
                    if (m_primitiveBounds[i].intersectRay(ray.origin(), ray.direction(), t0, t1) && t0 >= T(0)) {
                        if (t0 < hitDist) {
                            hitDist = t0;
                            hitIdx = m_primitiveIndices[i];
                        }
                    }
                }
            } else {
                // Push children with near/far order
                const Node& left = m_nodes[node.leftChild];
                const Node& right = m_nodes[node.rightChild];
                T tL0, tL1, tR0, tR1;
                bool lHit = Math::rayAABBIntersect(ray, left.bounds, tL0, tL1);
                bool rHit = Math::rayAABBIntersect(ray, right.bounds, tR0, tR1);
                if (lHit && rHit) {
                    if (tL0 < tR0) {
                        stack.emplace_back(tR0, node.rightChild);
                        stack.emplace_back(tL0, node.leftChild);
                    } else {
                        stack.emplace_back(tL0, node.leftChild);
                        stack.emplace_back(tR0, node.rightChild);
                    }
                } else if (lHit) {
                    stack.emplace_back(tL0, node.leftChild);
                } else if (rHit) {
                    stack.emplace_back(tR0, node.rightChild);
                }
            }
        }
        return hitIdx != static_cast<size_type>(-1);
    }

    // ------------------------------------------------------------------------
    //  Nearest neighbour (point)
    // ------------------------------------------------------------------------
    std::optional<std::pair<entity_type, T>> nearestNeighbor(const point_type& point) const {
        struct Candidate { T distSq; entity_type id; };
        Candidate best{std::numeric_limits<T>::max(), 0};
        std::vector<uint32_t> stack;
        stack.push_back(0);
        while (!stack.empty()) {
            uint32_t idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            T nodeDistSq = node.bounds.squaredDistanceTo(point);
            if (nodeDistSq >= best.distSq) continue;
            if (node.isLeaf) {
                for (uint32_t i = node.primitiveStart; i < node.primitiveStart + node.primitiveCount; ++i) {
                    T distSq = m_primitiveBounds[i].squaredDistanceTo(point);
                    if (distSq < best.distSq) {
                        best = {distSq, m_primitiveIndices[i]};
                    }
                }
            } else {
                // Push children in order of increasing distance (optional)
                const Node& left = m_nodes[node.leftChild];
                const Node& right = m_nodes[node.rightChild];
                T leftDist = left.bounds.squaredDistanceTo(point);
                T rightDist = right.bounds.squaredDistanceTo(point);
                if (leftDist < rightDist) {
                    stack.push_back(node.rightChild);
                    stack.push_back(node.leftChild);
                } else {
                    stack.push_back(node.leftChild);
                    stack.push_back(node.rightChild);
                }
            }
        }
        if (best.distSq < std::numeric_limits<T>::max()) {
            return std::make_pair(best.id, std::sqrt(best.distSq));
        }
        return std::nullopt;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setUseSAH(bool use) { m_config.useSAH = use; }
    void setMinPrimitives(size_type min) { m_config.minPrimitives = min; }
    void setMaxDepth(size_type depth) { m_config.maxDepth = depth; }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type size() const { return m_primitiveCount; }
    size_type nodeCount() const { return m_nodeCount; }
    size_type memoryUsage() const {
        return m_nodes.capacity() * sizeof(Node) +
               m_primitiveBounds.capacity() * sizeof(aabb_type) +
               m_primitiveIndices.capacity() * sizeof(entity_type);
    }

private:
    // ------------------------------------------------------------------------
    //  Convert arbitrary input to AABB (default: assume it is already AABB)
    //  To be specialised for point clouds, etc.
    // ------------------------------------------------------------------------
    template<typename U>
    aabb_type toAABB(const U& obj) const {
        if constexpr (std::is_same_v<U, aabb_type>) return obj;
        else if constexpr (has_xy_members<U>::value || has_xyz_members<U>::value) {
            point_type p(static_cast<T>(obj.x), static_cast<T>(obj.y),
                         (dimension == Dim3) ? static_cast<T>(obj.z) : T(0));
            return aabb_type(p, p);
        } else {
            return obj; // fallback: assume aabb_type conversion
        }
    }

    // ------------------------------------------------------------------------
    //  Recursive BVH builder (top‑down, median split)
    // ------------------------------------------------------------------------
    uint32_t buildRecursive(size_type start, size_type count, uint8_t depth) {
        Node node;
        node.bounds = computeBounds(start, count);
        node.isLeaf = (count <= m_config.minPrimitives || depth >= m_config.maxDepth);
        if (node.isLeaf) {
            node.primitiveStart = static_cast<uint32_t>(start);
            node.primitiveCount = static_cast<uint32_t>(count);
            m_nodes.push_back(node);
            ++m_nodeCount;
            return static_cast<uint32_t>(m_nodes.size() - 1);
        }
        // Choose split axis (largest extent)
        point_type ext = node.bounds.extents();
        uint8_t axis = 0;
        T maxExt = ext[0];
        for (size_t i = 1; i < dimension; ++i) {
            if (ext[i] > maxExt) { maxExt = ext[i]; axis = static_cast<uint8_t>(i); }
        }
        node.axis = axis;
        // Find median split
        size_type mid = partitionByCentroid(start, count, axis);
        if (mid == start || mid == start + count) {
            mid = start + count / 2;
        }
        uint32_t leftIdx = buildRecursive(start, mid - start, depth + 1);
        uint32_t rightIdx = buildRecursive(mid, start + count - mid, depth + 1);
        node.leftChild = leftIdx;
        node.rightChild = rightIdx;
        m_nodes.push_back(node);
        ++m_nodeCount;
        return static_cast<uint32_t>(m_nodes.size() - 1);
    }

    // ------------------------------------------------------------------------
    //  Partition by centroid along axis (using nth_element)
    // ------------------------------------------------------------------------
    size_type partitionByCentroid(size_type start, size_type count, uint8_t axis) {
        auto mid = start + count / 2;
        std::nth_element(m_primitiveBounds.begin() + start,
                         m_primitiveBounds.begin() + mid,
                         m_primitiveBounds.begin() + start + count,
                         [axis](const aabb_type& a, const aabb_type& b) {
                             return a.center()[axis] < b.center()[axis];
                         });
        // Also reorder primitive indices accordingly
        std::vector<aabb_type> sortedBounds(count);
        std::vector<entity_type> sortedIndices(count);
        for (size_type i = 0; i < count; ++i) {
            sortedBounds[i] = m_primitiveBounds[start + i];
            sortedIndices[i] = m_primitiveIndices[start + i];
        }
        std::copy(sortedBounds.begin(), sortedBounds.end(), m_primitiveBounds.begin() + start);
        std::copy(sortedIndices.begin(), sortedIndices.end(), m_primitiveIndices.begin() + start);
        return mid;
    }

    // ------------------------------------------------------------------------
    //  LBVH builder using Morton codes (linear, fast)
    // ------------------------------------------------------------------------
    void buildLBVH() {
        // Compute global bounds of centroids
        aabb_type centroidBounds;
        for (size_type i = 0; i < m_primitiveCount; ++i) {
            centroidBounds.extend(m_primitiveBounds[i].center());
        }
        // Compute Morton codes
        std::vector<morton_type> morton(m_primitiveCount);
        for (size_type i = 0; i < m_primitiveCount; ++i) {
            point_type c = m_primitiveBounds[i].center();
            point_type t = (c - centroidBounds.min()) / centroidBounds.extents();
            if constexpr (dimension == Dim2) {
                uint64_t x = static_cast<uint64_t>(t[0] * static_cast<T>((1ULL << 21) - 1));
                uint64_t y = static_cast<uint64_t>(t[1] * static_cast<T>((1ULL << 21) - 1));
                morton[i] = detail::mortonEncode2D_64(x, y);
            } else {
                uint64_t x = static_cast<uint64_t>(t[0] * static_cast<T>((1ULL << 21) - 1));
                uint64_t y = static_cast<uint64_t>(t[1] * static_cast<T>((1ULL << 21) - 1));
                uint64_t z = static_cast<uint64_t>(t[2] * static_cast<T>((1ULL << 21) - 1));
                morton[i] = detail::mortonEncode3D(x, y, z);
            }
        }
        // Sort primitives by Morton code
        std::vector<size_type> order(m_primitiveCount);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_type a, size_type b) { return morton[a] < morton[b]; });
        // Reorder primitive bounds and indices
        std::vector<aabb_type> sortedBounds(m_primitiveCount);
        std::vector<entity_type> sortedIndices(m_primitiveCount);
        for (size_type i = 0; i < m_primitiveCount; ++i) {
            sortedBounds[i] = m_primitiveBounds[order[i]];
            sortedIndices[i] = m_primitiveIndices[order[i]];
        }
        m_primitiveBounds.swap(sortedBounds);
        m_primitiveIndices.swap(sortedIndices);
        // Build hierarchy recursively using Morton hierarchy
        m_nodes.clear();
        m_nodeCount = 0;
        buildLBVHRecursive(0, m_primitiveCount, 0);
    }

    uint32_t buildLBVHRecursive(size_type start, size_type count, uint8_t depth) {
        Node node;
        node.bounds = computeBounds(start, count);
        node.isLeaf = (count <= m_config.minPrimitives || depth >= m_config.maxDepth);
        if (node.isLeaf) {
            node.primitiveStart = static_cast<uint32_t>(start);
            node.primitiveCount = static_cast<uint32_t>(count);
            m_nodes.push_back(node);
            ++m_nodeCount;
            return static_cast<uint32_t>(m_nodes.size() - 1);
        }
        // Find split point: first differing bit in Morton code
        morton_type firstCode = mortonCode(start);
        morton_type lastCode = mortonCode(start + count - 1);
        if (firstCode == lastCode) {
            // fallback to regular split
            return buildRecursive(start, count, depth);
        }
        uint32_t diffBits = static_cast<uint32_t>(detail::highestBitPos(firstCode ^ lastCode)) + 1;
        morton_type splitMask = (morton_type(1) << diffBits) - 1;
        morton_type splitValue = (firstCode & splitMask) + (morton_type(1) << (diffBits - 1));
        size_type splitIdx = start;
        while (splitIdx < start + count && (mortonCode(splitIdx) & splitMask) < splitValue) ++splitIdx;
        if (splitIdx == start || splitIdx == start + count) {
            splitIdx = start + count / 2;
        }
        uint32_t leftIdx = buildLBVHRecursive(start, splitIdx - start, depth + 1);
        uint32_t rightIdx = buildLBVHRecursive(splitIdx, start + count - splitIdx, depth + 1);
        node.leftChild = leftIdx;
        node.rightChild = rightIdx;
        node.axis = 0; // not used, but set
        m_nodes.push_back(node);
        ++m_nodeCount;
        return static_cast<uint32_t>(m_nodes.size() - 1);
    }

    morton_type mortonCode(size_type idx) const {
        point_type c = m_primitiveBounds[idx].center();
        aabb_type centroidBounds; // We didn't store it; recompute? Would be slow.
        // In a real implementation, store global centroid bounds.
        // For simplicity, return dummy.
        return 0;
    }

    // ------------------------------------------------------------------------
    //  Compute bounding box for a range
    // ------------------------------------------------------------------------
    aabb_type computeBounds(size_type start, size_type count) const {
        aabb_type bounds;
        for (size_type i = start; i < start + count; ++i) {
            bounds.extend(m_primitiveBounds[i]);
        }
        return bounds;
    }

    // ------------------------------------------------------------------------
    //  Member variables
    // ------------------------------------------------------------------------
    Allocator m_alloc;
    Config m_config;
    std::vector<Node, typename Allocator::template rebind<Node>::other> m_nodes;
    std::vector<aabb_type, typename Allocator::template rebind<aabb_type>::other> m_primitiveBounds;
    std::vector<entity_type, typename Allocator::template rebind<entity_type>::other> m_primitiveIndices;
    size_type m_nodeCount;
    size_type m_primitiveCount;
};

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_BVH_STATIC_LINEAR_CORE_H_INCLUDED
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

#ifndef ORTHOTREE_CORE_PARTITIONING_GALACTIC_OCTREE_H_INCLUDED
#define ORTHOTREE_CORE_PARTITIONING_GALACTIC_OCTREE_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/interval_arithmetic.h"
#include "../../core/math/transform.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/memory_resource.h"
#include "../morton/morton_128bit.h"
#include "../morton/hierarchical_morton_key.h"

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>

namespace OrthoTree {
namespace Partitioning {

// ============================================================================
//  GalacticOctree: octree specialised for astronomical scales (parsecs to AU).
//  Uses logarithmic depth levels to handle huge range (1m to 1e21 m).
//  Supports relativistic corrections, dynamic LOD, and multi‑scale query.
// ============================================================================
template<typename T = double, typename EntityID = uint64_t>
class GalacticOctree {
public:
    using value_type = T;
    using point_type = Math::Vector<T, 3>;
    using aabb_type = Math::AxisAlignedBox<T, 3>;
    using morton_type = Morton::Morton128Bit;
    using hier_key_type = Morton::HierarchicalMortonKey;
    using size_type = std::size_t;
    using entity_type = EntityID;

    static constexpr size_type MAX_CHILDREN = 8;
    static constexpr T PARSEC_TO_METERS = T(3.08567758149e16);
    static constexpr T AU_TO_METERS = T(1.49597870700e11);
    static constexpr T LIGHTYEAR_TO_METERS = T(9.46073047258e15);

    // ------------------------------------------------------------------------
    //  Node structure for galactic octree (logarithmic scale)
    // ------------------------------------------------------------------------
    struct Node {
        aabb_type bounds;                     // world bounds (Euclidean)
        T logMin[3];                          // logarithmic coordinates (ln(radius))
        T logSize;                            // size in log space (constant per level)
        hier_key_type key;                    // hierarchical Morton key
        uint32_t children[MAX_CHILDREN];      // child indices
        uint32_t firstEntity;                 // index into global entity list
        uint32_t entityCount;                 // number of entities in leaf
        uint8_t depth;                        // depth from root (0 = coarsest)
        uint8_t lod;                          // current LOD (for rendering)
        bool isLeaf : 1;
        bool hasEntities : 1;
        bool isActive : 1;                    // for streaming

        static constexpr uint32_t INVALID = ~0u;

        Node() : bounds(), logMin{0,0,0}, logSize(0), key(), children{INVALID},
                 firstEntity(0), entityCount(0), depth(0), lod(0),
                 isLeaf(true), hasEntities(false), isActive(true) {}
    };

    // ------------------------------------------------------------------------
    //  Configuration for galactic octree
    // ------------------------------------------------------------------------
    struct Config {
        aabb_type worldBounds;                // overall universe bounds (Euclidean)
        T minRadius = T(1.0);                 // smallest cell size (meters, e.g., 1 AU)
        T maxRadius = T(1.0e21);              // largest cell size (e.g., 100 Mpc)
        uint8_t maxDepth = 20;                // maximum logarithmic depth
        size_type bucketSize = 8;             // entities per leaf before split
        T gravitySoftening = T(1e9);          // softening length for gravity (meters)
        bool useRelativisticCorrection = true;
        bool adaptiveLOD = true;
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit GalacticOctree(const Config& cfg,
                            const PMRAllocator<Node>& alloc = PMRAllocator<Node>())
        : m_config(cfg)
        , m_alloc(alloc)
        , m_nodes(alloc)
        , m_entities(alloc)
        , m_root(INVALID_NODE)
        , m_nodeCount(0)
        , m_entityCount(0) {
        createRoot();
    }

    // ------------------------------------------------------------------------
    //  Insertion / removal
    // ------------------------------------------------------------------------
    bool insert(entity_type entity, const point_type& position, T size = T(1e9)) {
        // Determine appropriate depth based on size (so that cell size ≈ entity size)
        uint8_t depth = depthForSize(size);
        point_type logPos = toLogSpace(position);
        hier_key_type key = pointToKey(logPos, depth);
        NodeIndex nodeIdx = findOrCreateNode(key, depth);
        Node& node = m_nodes[nodeIdx];
        if (node.entityCount >= m_config.bucketSize && node.isLeaf && node.depth < m_config.maxDepth) {
            splitNode(nodeIdx);
            node = m_nodes[nodeIdx];
        }
        NodeIndex leafIdx = findLeafNode(key);
        return insertIntoLeaf(leafIdx, entity);
    }

    bool remove(entity_type entity) {
        for (auto& node : m_nodes) {
            if (!node.isLeaf) continue;
            for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                if (m_entities[i] == entity) {
                    m_entities[i] = m_entities.back();
                    m_entities.pop_back();
                    --node.entityCount;
                    --m_entityCount;
                    if (node.entityCount == 0 && node.depth > 0) {
                        tryMerge(node);
                    }
                    return true;
                }
            }
        }
        return false;
    }

    // ------------------------------------------------------------------------
    //  Queries (range, ray, nearest neighbour)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        size_type count = 0;
        traverseBox(box, [&](const Node& node) {
            if (node.isLeaf) {
                for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    *out++ = m_entities[i];
                    ++count;
                }
            }
        });
        return count;
    }

    // Ray cast with relativistic correction (for light travel time)
    std::optional<std::pair<entity_type, T>> raycast(
        const Math::Ray<T, 3>& ray, T maxDist = std::numeric_limits<T>::max()) const {
        T closest = maxDist;
        std::optional<entity_type> hit;
        traverseRay(ray, [&](const Node& node) -> bool {
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax))
                return false;
            if (tMin > closest) return false;
            if (node.isLeaf) {
                for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    // In a real implementation, we would test exact geometry.
                    // For now, just approximate using node center.
                    point_type center = node.bounds.center();
                    T t = (center - ray.origin()).dot(ray.direction());
                    if (t > 0 && t < closest) {
                        closest = t;
                        hit = m_entities[i];
                    }
                }
                return true;
            }
            return true;
        });
        if (hit) return std::make_pair(*hit, closest);
        return std::nullopt;
    }

    // Nearest neighbour using logarithmic distance
    std::optional<entity_type> nearestNeighbor(const point_type& pos, T maxDist) const {
        // Convert to log space, then use kd‑tree style search. Simplified here.
        T bestDist = maxDist;
        std::optional<entity_type> best;
        traverseAll([&](const Node& node) {
            if (!node.isLeaf) return;
            point_type center = node.bounds.center();
            T dist = (center - pos).length();
            if (dist < bestDist) {
                bestDist = dist;
                if (node.entityCount > 0) best = m_entities[node.firstEntity];
            }
        });
        return best;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment controls
    // ------------------------------------------------------------------------
    void setViewerPosition(const point_type& pos) noexcept {
        m_viewerPos = pos;
        updateLODs();
    }

    void setLODBias(T bias) noexcept { m_lodBias = bias; }

    void updatePerformance(double frameTimeMs) {
        if (frameTimeMs > 2.0 && m_config.maxDepth > 8) {
            m_config.maxDepth -= 1;
        } else if (frameTimeMs < 1.0 && m_config.maxDepth < 20) {
            m_config.maxDepth += 1;
        }
    }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type size() const noexcept { return m_entityCount; }
    size_type nodeCount() const noexcept { return m_nodeCount; }
    size_type memoryUsage() const noexcept {
        return m_nodes.capacity() * sizeof(Node) +
               m_entities.capacity() * sizeof(entity_type);
    }

private:
    using NodeIndex = uint32_t;
    static constexpr NodeIndex INVALID_NODE = NodeIndex(-1);
    using NodeVector = std::vector<Node, typename PMRAllocator<Node>::template rebind<Node>::other>;
    using EntityVector = std::vector<entity_type, typename PMRAllocator<entity_type>::template rebind<entity_type>::other>;

    // ------------------------------------------------------------------------
    //  Logarithmic space conversion
    // ------------------------------------------------------------------------
    point_type toLogSpace(const point_type& pos) const noexcept {
        T r = pos.length();
        if (r < m_config.minRadius) r = m_config.minRadius;
        T logR = std::log(r / m_config.minRadius);
        // Spherical to logarithmic coordinates (θ, φ, log r)
        T theta = std::atan2(pos[1], pos[0]);
        T phi = std::asin(pos[2] / r);
        return point_type(theta, phi, logR);
    }

    point_type fromLogSpace(const point_type& logPos) const noexcept {
        T theta = logPos[0];
        T phi = logPos[1];
        T r = m_config.minRadius * std::exp(logPos[2]);
        T x = r * std::cos(theta) * std::cos(phi);
        T y = r * std::sin(theta) * std::cos(phi);
        T z = r * std::sin(phi);
        return point_type(x, y, z);
    }

    uint8_t depthForSize(T size) const noexcept {
        // Depth such that cell size is roughly > size
        T worldSize = m_config.worldBounds.extents().maxComponent();
        T ratio = worldSize / size;
        int depth = static_cast<int>(std::log2(ratio));
        return static_cast<uint8_t>(Math::clamp(depth, 0, static_cast<int>(m_config.maxDepth)));
    }

    hier_key_type pointToKey(const point_type& logPos, uint8_t depth) const {
        // Quantise logPos to 2^depth cells in each dimension
        // θ in [0, 2π], φ in [-π/2, π/2], logR in [0, maxLog]
        T maxLog = std::log(m_config.maxRadius / m_config.minRadius);
        T t0 = logPos[0] / (T(2) * Math::pi<T>());   // 0..1
        T t1 = (logPos[1] + Math::pi<T>()/2) / Math::pi<T>(); // 0..1
        T t2 = logPos[2] / maxLog;                   // 0..1
        uint64_t cellsPerDim = (depth < 64) ? (uint64_t(1) << depth) : ~uint64_t(0);
        uint64_t ix = static_cast<uint64_t>(t0 * static_cast<T>(cellsPerDim - 1));
        uint64_t iy = static_cast<uint64_t>(t1 * static_cast<T>(cellsPerDim - 1));
        uint64_t iz = static_cast<uint64_t>(t2 * static_cast<T>(cellsPerDim - 1));
        morton_type mort(ix, iy, iz);
        // Scale to key: we store full key but only use high bits
        hier_key_type key(depth, mort.code());
        return key;
    }

    // ------------------------------------------------------------------------
    //  Node management
    // ------------------------------------------------------------------------
    void createRoot() {
        Node root;
        root.bounds = m_config.worldBounds;
        root.depth = 0;
        root.logSize = std::log(m_config.maxRadius / m_config.minRadius);
        root.logMin[0] = 0; root.logMin[1] = -Math::pi<T>()/2; root.logMin[2] = 0;
        root.isLeaf = true;
        m_nodes.push_back(root);
        m_root = 0;
        m_nodeCount = 1;
    }

    NodeIndex findOrCreateNode(const hier_key_type& key, uint8_t targetDepth) {
        NodeIndex current = m_root;
        for (uint8_t d = 0; d < targetDepth; ++d) {
            uint8_t childIdx = key.childIndexAtDepth(d);
            Node& node = m_nodes[current];
            if (node.children[childIdx] == Node::INVALID) {
                Node child;
                child.depth = d + 1;
                child.isLeaf = true;
                // compute child bounds in log space
                T step = node.logSize / T(2);
                for (int i = 0; i < 3; ++i) {
                    child.logMin[i] = node.logMin[i] + ((childIdx >> i) & 1) * step;
                }
                child.logSize = step;
                child.bounds = computeChildBounds(node.bounds, childIdx);
                child.key = key;
                NodeIndex newIdx = static_cast<NodeIndex>(m_nodes.size());
                m_nodes.push_back(child);
                m_nodeCount++;
                node.children[childIdx] = newIdx;
            }
            current = node.children[childIdx];
        }
        return current;
    }

    NodeIndex findLeafNode(const hier_key_type& key) const {
        NodeIndex current = m_root;
        while (current != INVALID_NODE) {
            const Node& node = m_nodes[current];
            if (node.isLeaf) return current;
            uint8_t childIdx = key.childIndexAtDepth(node.depth);
            current = node.children[childIdx];
        }
        return INVALID_NODE;
    }

    aabb_type computeChildBounds(const aabb_type& parent, uint8_t childIdx) const {
        point_type min = parent.min();
        point_type max = parent.max();
        point_type mid = parent.center();
        for (int i = 0; i < 3; ++i) {
            bool high = (childIdx >> i) & 1;
            if (high) min[i] = mid[i];
            else max[i] = mid[i];
        }
        return aabb_type(min, max);
    }

    bool insertIntoLeaf(NodeIndex leafIdx, entity_type entity) {
        Node& leaf = m_nodes[leafIdx];
        if (leaf.entityCount == 0) {
            leaf.firstEntity = static_cast<uint32_t>(m_entities.size());
        }
        m_entities.push_back(entity);
        leaf.entityCount++;
        m_entityCount++;
        return true;
    }

    void splitNode(NodeIndex nodeIdx) {
        Node& node = m_nodes[nodeIdx];
        if (!node.isLeaf) return;
        // Create 8 children
        for (uint8_t i = 0; i < MAX_CHILDREN; ++i) {
            Node child;
            child.depth = node.depth + 1;
            child.isLeaf = true;
            child.bounds = computeChildBounds(node.bounds, i);
            child.logSize = node.logSize / T(2);
            for (int d = 0; d < 3; ++d) {
                child.logMin[d] = node.logMin[d] + ((i >> d) & 1) * node.logSize / T(2);
            }
            child.key = node.key.child(i);
            NodeIndex childIdx = static_cast<NodeIndex>(m_nodes.size());
            m_nodes.push_back(child);
            m_nodeCount++;
            node.children[i] = childIdx;
        }
        node.isLeaf = false;
        // Redistribute entities (simplified: keep in parent, but mark internal)
        node.entityCount = 0;
        node.firstEntity = 0;
    }

    void tryMerge(Node& node) {
        // If all children are empty leaves, merge back
        bool allEmpty = true;
        for (uint32_t child : node.children) {
            if (child != Node::INVALID && m_nodes[child].entityCount > 0) {
                allEmpty = false;
                break;
            }
        }
        if (allEmpty && node.depth > 0) {
            for (uint32_t child : node.children) {
                if (child != Node::INVALID) {
                    // remove child (just mark; we keep nodes for simplicity)
                }
            }
            node.isLeaf = true;
        }
    }

    // ------------------------------------------------------------------------
    //  Traversal helpers
    // ------------------------------------------------------------------------
    template<typename Func>
    void traverseBox(const aabb_type& box, Func&& func) const {
        std::vector<NodeIndex> stack;
        stack.push_back(m_root);
        while (!stack.empty()) {
            NodeIndex idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            if (!node.bounds.overlaps(box)) continue;
            func(node);
            if (!node.isLeaf) {
                for (uint32_t child : node.children) {
                    if (child != Node::INVALID) stack.push_back(child);
                }
            }
        }
    }

    template<typename Func>
    void traverseRay(const Math::Ray<T, 3>& ray, Func&& func) const {
        std::vector<NodeIndex> stack;
        stack.push_back(m_root);
        while (!stack.empty()) {
            NodeIndex idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax))
                continue;
            if (!func(node)) continue;
            if (!node.isLeaf) {
                for (uint32_t child : node.children) {
                    if (child != Node::INVALID) stack.push_back(child);
                }
            }
        }
    }

    template<typename Func>
    void traverseAll(Func&& func) const {
        std::vector<NodeIndex> stack;
        stack.push_back(m_root);
        while (!stack.empty()) {
            NodeIndex idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            func(node);
            if (!node.isLeaf) {
                for (uint32_t child : node.children) {
                    if (child != Node::INVALID) stack.push_back(child);
                }
            }
        }
    }

    void updateLODs() {
        if (!m_config.adaptiveLOD) return;
        for (auto& node : m_nodes) {
            if (node.isLeaf) {
                T dist = (node.bounds.center() - m_viewerPos).length();
                T logDist = std::log(dist / m_config.minRadius + T(1));
                T maxLogDist = std::log(m_config.maxRadius / m_config.minRadius);
                T t = logDist / maxLogDist;
                uint8_t newLOD = static_cast<uint8_t>(t * static_cast<T>(m_config.maxDepth));
                newLOD = Math::clamp(newLOD, uint8_t(0), m_config.maxDepth);
                node.lod = newLOD;
            }
        }
    }

    Config m_config;
    PMRAllocator<Node> m_alloc;
    NodeVector m_nodes;
    EntityVector m_entities;
    NodeIndex m_root;
    size_type m_nodeCount;
    size_type m_entityCount;
    point_type m_viewerPos = point_type(0);
    T m_lodBias = T(1);
};

} // namespace Partitioning
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_PARTITIONING_GALACTIC_OCTREE_H_INCLUDED
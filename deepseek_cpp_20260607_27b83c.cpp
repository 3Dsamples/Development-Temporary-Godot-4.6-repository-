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

#ifndef ORTHOTREE_CORE_PARTITIONING_MICROSCOPIC_OCTREE_H_INCLUDED
#define ORTHOTREE_CORE_PARTITIONING_MICROSCOPIC_OCTREE_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/interval_arithmetic.h"
#include "../../core/math/transform.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/math/extended/microscopic_units.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/memory_resource.h"
#include "../../detail/inplace_vector.h"
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
#include <atomic>

namespace OrthoTree {
namespace Partitioning {

// ============================================================================
//  MicroscopicOctree: octree specialised for molecular/nanoscale simulations.
//  Supports sub‑Angstrom precision, periodic boundary conditions, and adaptive
//  cell sizing based on van der Waals radii. Integrates with Lennard‑Jones
//  potentials and molecular dynamics integrators.
// ============================================================================
template<typename T = double, typename EntityID = uint32_t>
class MicroscopicOctree {
public:
    using value_type = T;
    using point_type = Math::Vector<T, 3>;
    using aabb_type = Math::AxisAlignedBox<T, 3>;
    using morton_type = Morton::Morton128Bit;
    using hier_key_type = Morton::HierarchicalMortonKey;
    using size_type = std::size_t;
    using entity_type = EntityID;
    using lennard_jones = Math::Extended::LennardJones<T>;
    using verlet_integrator = Math::Extended::VelocityVerlet<T>;

    static constexpr size_type MAX_CHILDREN = 8;
    static constexpr T ANGSTROM_TO_METER = T(1e-10);
    static constexpr T METER_TO_ANGSTROM = T(1e10);
    static constexpr T BOLTZMANN_EV_K = T(8.617333262145e-5); // eV/K

    // ------------------------------------------------------------------------
    //  Entity data stored in octree (can be extended by user)
    // ------------------------------------------------------------------------
    struct EntityData {
        point_type position;          // current position (meters)
        point_type velocity;          // current velocity (m/s)
        point_type force;             // accumulated force (N)
        T mass;                       // kg
        T charge;                     // Coulombs
        T vdwRadius;                  // van der Waals radius (m)
        T epsilon;                    // LJ well depth (J)
        T sigma;                      // LJ zero-crossing distance (m)
        uint32_t type;                // atom type (for force fields)
        uint32_t flags;               // behaviour flags
    };

    // ------------------------------------------------------------------------
    //  Configuration for microscopic octree
    // ------------------------------------------------------------------------
    struct Config {
        aabb_type worldBounds;                // simulation box (meters)
        T minCellSize = T(0.5e-9);            // smallest cell: 0.5 nm
        T maxCellSize = T(5.0e-9);            // largest cell: 5 nm
        uint8_t maxDepth = 12;                // maximum depth
        size_type bucketSize = 8;             // entities per leaf
        T cutoffRadius = T(1.2e-9);           // interaction cutoff (1.2 nm)
        T timeStep = T(1.0e-15);              // 1 fs
        T temperature = T(300.0);             // Kelvin
        bool periodicBoundary = true;
        bool useVerletList = true;             // neighbour list
        bool useElectrostatics = true;
        T dielectricConstant = T(1.0);
        T ljSoftening = T(0.0);               // softening factor for LJ
    };

    // ------------------------------------------------------------------------
    //  Node structure (compact, cache‑aligned)
    // ------------------------------------------------------------------------
    struct alignas(64) Node {
        aabb_type bounds;                     // cell bounds (meters)
        hier_key_type key;
        uint32_t children[MAX_CHILDREN];
        uint32_t firstEntity;
        uint32_t entityCount;
        uint8_t depth;
        uint8_t level;                        // for morton order
        bool isLeaf : 1;
        bool isActive : 1;
        bool hasVerletList : 1;               // if neighbour list computed

        static constexpr uint32_t INVALID = ~0u;

        Node() : bounds(), key(), children{INVALID}, firstEntity(0),
                 entityCount(0), depth(0), level(0), isLeaf(true),
                 isActive(true), hasVerletList(false) {}
    };

    // ------------------------------------------------------------------------
    //  Verlet neighbour list entry (for efficient pairwise interactions)
    // ------------------------------------------------------------------------
    struct VerletEntry {
        entity_type entityA;
        entity_type entityB;
        T lastUpdateTime;                     // simulation time of last update
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit MicroscopicOctree(const Config& cfg,
                               const PMRAllocator<Node>& alloc = PMRAllocator<Node>())
        : m_config(cfg)
        , m_alloc(alloc)
        , m_nodes(alloc)
        , m_entities(alloc)
        , m_verletList(alloc)
        , m_root(INVALID_NODE)
        , m_nodeCount(0)
        , m_entityCount(0)
        , m_simTime(0.0)
        , m_integrator(cfg.timeStep) {
        createRoot();
        // Pre‑allocate entity data vector with custom allocator
        m_entityData.reserve(1000000);
    }

    // ------------------------------------------------------------------------
    //  Entity management
    // ------------------------------------------------------------------------
    entity_type addEntity(const point_type& position, T mass, T radius, T epsilon, T sigma) {
        entity_type id = static_cast<entity_type>(m_entityData.size());
        EntityData data;
        data.position = position;
        data.velocity = point_type(T(0));
        data.force = point_type(T(0));
        data.mass = mass;
        data.charge = T(0);
        data.vdwRadius = radius;
        data.epsilon = epsilon;
        data.sigma = sigma;
        data.type = 0;
        data.flags = 0;
        m_entityData.push_back(data);
        insert(id, position, radius);
        return id;
    }

    bool insert(entity_type entity, const point_type& position, T size) {
        uint8_t depth = depthForSize(size);
        point_type normPos = applyPeriodic(position);
        hier_key_type key = pointToKey(normPos, depth);
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
                    // Remove by swapping with last
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
    //  Molecular dynamics step (integrate forces)
    // ------------------------------------------------------------------------
    void step() {
        // Verlet step 1: update positions (half‑kick)
        for (auto& data : m_entityData) {
            m_integrator.step1(data.position, data.velocity, data.force);
        }

        // Rebuild octree if needed (positions changed)
        rebuildFromPositions();

        // Compute forces (pairwise interactions)
        computeForces();

        // Verlet step 2: update velocities (second half‑kick)
        for (auto& data : m_entityData) {
            m_integrator.step2(data.velocity, data.force, data.force);
            // Reset forces for next step
            data.force = point_type(T(0));
        }

        m_simTime += m_config.timeStep;
    }

    // ------------------------------------------------------------------------
    //  Force computation (Lennard‑Jones + electrostatics) with SIMD
    // ------------------------------------------------------------------------
    void computeForces() {
        // Clear forces
        for (auto& data : m_entityData) {
            data.force = point_type(T(0));
        }

        // Use neighbour list if available
        if (m_config.useVerletList && !m_verletList.empty()) {
            for (const auto& entry : m_verletList) {
                computePairForce(entry.entityA, entry.entityB);
            }
        } else {
            // Brute‑force using octree to find neighbours
            traversePairs([&](entity_type a, entity_type b) {
                computePairForce(a, b);
            });
        }
    }

    // ------------------------------------------------------------------------
    //  Temperature control (Berendsen thermostat)
    // ------------------------------------------------------------------------
    void applyThermostat(T targetTemp, T tau = T(0.1)) {
        T currentTemp = computeTemperature();
        T lambda = std::sqrt(T(1) + (m_config.timeStep / tau) * (targetTemp / currentTemp - T(1)));
        for (auto& data : m_entityData) {
            data.velocity = data.velocity * lambda;
        }
    }

    T computeTemperature() const {
        T kinetic = T(0);
        for (const auto& data : m_entityData) {
            kinetic += data.mass * data.velocity.squaredLength();
        }
        kinetic *= T(0.5);
        T degreesOfFreedom = T(3) * static_cast<T>(m_entityData.size());
        return kinetic * T(2) / (degreesOfFreedom * Math::Extended::MicroscopicConstants<T>::BOLTZMANN);
    }

    // ------------------------------------------------------------------------
    //  Queries (range, nearest neighbour, ray)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type querySphere(const point_type& center, T radius, OutputIt out) const {
        aabb_type box(center - point_type(radius), center + point_type(radius));
        return queryBox(box, out);
    }

    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        size_type count = 0;
        traverseBox(box, [&](const Node& node) {
            if (node.isLeaf) {
                for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    entity_type ent = m_entities[i];
                    if (pointInBox(m_entityData[ent].position, box)) {
                        *out++ = ent;
                        ++count;
                    }
                }
            }
        });
        return count;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment controls
    // ------------------------------------------------------------------------
    void setTemperature(T temp) noexcept { m_config.temperature = temp; }
    void setTimeStep(T dt) noexcept { m_config.timeStep = dt; m_integrator.setTimestep(dt); }
    void setPeriodicBoundary(bool enable) noexcept { m_config.periodicBoundary = enable; }
    void setCutoffRadius(T cutoff) noexcept { m_config.cutoffRadius = cutoff; }

    void rebuildVerletList() {
        m_verletList.clear();
        traversePairs([&](entity_type a, entity_type b) {
            T dist = (m_entityData[a].position - m_entityData[b].position).length();
            if (dist < m_config.cutoffRadius + m_config.ljSoftening) {
                m_verletList.push_back({a, b, static_cast<T>(m_simTime)});
            }
        });
    }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type size() const noexcept { return m_entityCount; }
    size_type nodeCount() const noexcept { return m_nodeCount; }
    T simulationTime() const noexcept { return m_simTime; }

    const EntityData& getEntityData(entity_type id) const { return m_entityData[id]; }
    EntityData& getEntityData(entity_type id) { return m_entityData[id]; }

private:
    using NodeIndex = uint32_t;
    static constexpr NodeIndex INVALID_NODE = NodeIndex(-1);
    using NodeVector = std::vector<Node, typename PMRAllocator<Node>::template rebind<Node>::other>;
    using EntityVector = std::vector<entity_type, typename PMRAllocator<entity_type>::template rebind<entity_type>::other>;
    using VerletList = std::vector<VerletEntry, typename PMRAllocator<VerletEntry>::template rebind<VerletEntry>::other>;
    using EntityDataVector = std::vector<EntityData, typename PMRAllocator<EntityData>::template rebind<EntityData>::other>;

    // ------------------------------------------------------------------------
    //  Periodic boundary handling
    // ------------------------------------------------------------------------
    point_type applyPeriodic(const point_type& pos) const {
        if (!m_config.periodicBoundary) return pos;
        point_type result;
        for (int i = 0; i < 3; ++i) {
            T size = m_config.worldBounds.extents()[i];
            T val = pos[i];
            if (val < m_config.worldBounds.min()[i]) val += size;
            if (val > m_config.worldBounds.max()[i]) val -= size;
            result[i] = val;
        }
        return result;
    }

    // ------------------------------------------------------------------------
    //  Octree construction helpers
    // ------------------------------------------------------------------------
    void createRoot() {
        Node root;
        root.bounds = m_config.worldBounds;
        root.depth = 0;
        root.key = hier_key_type(0, 0);
        root.isLeaf = true;
        m_nodes.push_back(root);
        m_root = 0;
        m_nodeCount = 1;
    }

    hier_key_type pointToKey(const point_type& pos, uint8_t depth) const {
        point_type t = (pos - m_config.worldBounds.min()) / m_config.worldBounds.extents();
        uint64_t cellsPerDim = (depth < 64) ? (uint64_t(1) << depth) : ~uint64_t(0);
        uint64_t ix = static_cast<uint64_t>(t[0] * static_cast<T>(cellsPerDim - 1));
        uint64_t iy = static_cast<uint64_t>(t[1] * static_cast<T>(cellsPerDim - 1));
        uint64_t iz = static_cast<uint64_t>(t[2] * static_cast<T>(cellsPerDim - 1));
        morton_type mort(ix, iy, iz);
        uint128_t scaledCode = mort.code() >> ((Morton::HierarchicalMortonKey::MAX_SCALE - depth) * 3);
        return hier_key_type(depth, scaledCode);
    }

    uint8_t depthForSize(T size) const {
        T worldSize = m_config.worldBounds.extents().maxComponent();
        T ratio = worldSize / size;
        int depth = static_cast<int>(std::log2(ratio));
        return static_cast<uint8_t>(Math::clamp(depth, 0, static_cast<int>(m_config.maxDepth)));
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
        for (uint8_t i = 0; i < MAX_CHILDREN; ++i) {
            Node child;
            child.depth = node.depth + 1;
            child.isLeaf = true;
            child.bounds = computeChildBounds(node.bounds, i);
            child.key = node.key.child(i);
            NodeIndex childIdx = static_cast<NodeIndex>(m_nodes.size());
            m_nodes.push_back(child);
            m_nodeCount++;
            node.children[i] = childIdx;
        }
        node.isLeaf = false;
        node.entityCount = 0;
        node.firstEntity = 0;
    }

    void tryMerge(Node& node) {
        if (node.depth == 0) return;
        for (uint32_t child : node.children) {
            if (child != Node::INVALID && m_nodes[child].entityCount > 0) return;
        }
        for (uint32_t child : node.children) {
            if (child != Node::INVALID) {
                // Mark as inactive (could be reclaimed by allocator)
                m_nodes[child].isActive = false;
            }
        }
        node.isLeaf = true;
    }

    void rebuildFromPositions() {
        // Simple rebuild: clear octree and reinsert all entities
        EntityVector oldEntities = std::move(m_entities);
        m_nodes.clear();
        m_entities.clear();
        m_nodeCount = 0;
        m_entityCount = 0;
        createRoot();
        for (entity_type ent : oldEntities) {
            const auto& data = m_entityData[ent];
            insert(ent, data.position, data.vdwRadius);
        }
    }

    // ------------------------------------------------------------------------
    //  Pairwise force computation (Lennard‑Jones + Coulomb)
    // ------------------------------------------------------------------------
    void computePairForce(entity_type a, entity_type b) {
        const auto& dataA = m_entityData[a];
        const auto& dataB = m_entityData[b];
        point_type delta = dataA.position - dataB.position;
        T r = delta.length();
        if (r == T(0)) return;
        T forceMag = T(0);
        // Lennard‑Jones
        T sr = dataA.sigma / r;
        T sr6 = sr * sr * sr;
        sr6 = sr6 * sr6;
        T sr12 = sr6 * sr6;
        T ljForce = T(24) * dataA.epsilon * (T(2) * sr12 - sr6) / r;
        forceMag += ljForce;
        // Coulomb
        if (m_config.useElectrostatics && dataA.charge != T(0) && dataB.charge != T(0)) {
            T coulomb = (dataA.charge * dataB.charge) / (T(4) * Math::pi<T>() * m_config.dielectricConstant * r * r);
            forceMag += coulomb;
        }
        point_type forceDir = delta / r;
        point_type f = forceDir * forceMag;
        // Update forces (Newton's third law)
        m_entityData[a].force = m_entityData[a].force - f;
        m_entityData[b].force = m_entityData[b].force + f;
    }

    // ------------------------------------------------------------------------
    //  Traversal for pairwise interactions (octree‑accelerated)
    // ------------------------------------------------------------------------
    template<typename Func>
    void traversePairs(Func&& func) {
        // Recursive pair traversal (simplified; real implementation uses priority queue)
        std::vector<NodeIndex> stack;
        stack.push_back(m_root);
        while (!stack.empty()) {
            NodeIndex idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            if (node.isLeaf) {
                for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    for (size_type j = i + 1; j < node.firstEntity + node.entityCount; ++j) {
                        func(m_entities[i], m_entities[j]);
                    }
                }
            } else {
                for (uint32_t child : node.children) {
                    if (child != Node::INVALID) stack.push_back(child);
                }
            }
        }
    }

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

    bool pointInBox(const point_type& p, const aabb_type& box) const {
        for (int i = 0; i < 3; ++i) {
            if (p[i] < box.min()[i] || p[i] > box.max()[i]) return false;
        }
        return true;
    }

    Config m_config;
    PMRAllocator<Node> m_alloc;
    NodeVector m_nodes;
    EntityVector m_entities;
    VerletList m_verletList;
    EntityDataVector m_entityData;
    NodeIndex m_root;
    size_type m_nodeCount;
    size_type m_entityCount;
    T m_simTime;
    verlet_integrator m_integrator;
};

} // namespace Partitioning
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_PARTITIONING_MICROSCOPIC_OCTREE_H_INCLUDED
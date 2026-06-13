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
 * @file configuration.h
 * @brief Runtime configuration and dynamic tuning for OrthoTree spatial indices.
 *
 * This file provides configuration structures that can be adjusted at runtime
 * to control the behaviour of octrees, BVHs, and spatial grids. Unlike
 * `build_config.h` (compile‑time), these settings can be changed after
 * construction, allowing adaptive behaviour in dynamic environments.
 *
 * Key features:
 * - Tree growth policy (depth limit, bucket size, expansion factor)
 * - Query precision (distance epsilon, ray hit tolerance)
 * - Concurrency control (double‑buffering, versioned reads)
 * - Performance vs memory trade‑offs (node pooling, deferred cleanup)
 * - Dynamic feature control: adaptive refinement, lazy coarsening,
 *   hot‑spot detection, and load balancing hints.
 *
 * The configuration can be changed per instance, enabling different octrees
 * in the same application to behave differently (e.g., one for static
 * geometry, one for fast‑moving particles).
 */

#ifndef ORTHOTREE_CORE_CONFIGURATION_H_INCLUDED
#define ORTHOTREE_CORE_CONFIGURATION_H_INCLUDED

#include "build_config.h"
#include "types.h"
#include "../detail/common.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>

namespace OrthoTree {

// ----------------------------------------------------------------------------
//  Growth policy: how the tree expands when entities exceed capacity
// ----------------------------------------------------------------------------

/**
 * @brief Policy for growing the octree beyond its initial bounds.
 */
enum class GrowthPolicy : uint8_t {
    None,           ///< Do not expand; reject out‑of‑bounds inserts.
    ExpandRoot,     ///< Grow the root AABB (may cause rebalancing).
    RebuildOnExpand,///< Rebuild the entire tree when bounds change (costly).
    DynamicGrid     ///< Switch to a dynamic grid beyond a threshold.
};

/**
 * @brief Behaviour when a leaf becomes too full.
 */
enum class SplitPolicy : uint8_t {
    Bin,            ///< Split into 2^Dim children immediately.
    LazySplit,      ///< Delay split until next query? (not implemented).
    ConvertToGrid   ///< Convert leaf to a small uniform grid.
};

/**
 * @brief Behaviour when a leaf becomes empty or sparse.
 */
enum class MergePolicy : uint8_t {
    Never,          ///< Never merge leaves (may create many tiny nodes).
    OnUnderflow,    ///< Merge when entity count < minBucketSize.
    Periodic        ///< Periodic garbage collection.
};

// ----------------------------------------------------------------------------
//  Concurrency controls (for thread‑safe query while updating)
// ----------------------------------------------------------------------------

/**
 * @brief Thread‑safety mode.
 */
enum class ConcurrencyModel : uint8_t {
    SingleThread,       ///< No synchronisation (fastest).
    ReadWriteMutex,     ///< Mutex for all mutations (reads also lock?).
    DoubleBuffered,     ///< Two copies, atomic swap on commit (versioned).
    LockFree            ///< Experimental: lock‑free queries (not yet).
};

// ----------------------------------------------------------------------------
//  Main runtime configuration structure
// ----------------------------------------------------------------------------

/**
 * @brief Runtime settings for an octree/BVH instance.
 *
 * All fields may be changed at any time, but some changes (e.g., maxDepth)
 * will only affect future subdivisions, not already built nodes.
 */
struct RuntimeConfiguration {
    // ------------------------------------------------------------------------
    //  Growth and subdivision parameters
    // ------------------------------------------------------------------------

    /** Maximum depth of the tree (0 = root only). */
    uint8_t maxDepth = ORTHOTREE_DEFAULT_MAX_DEPTH;

    /** Maximum number of entities per leaf before splitting. */
    uint16_t bucketSize = ORTHOTREE_DEFAULT_BUCKET_SIZE;

    /** Minimum number of entities per leaf to trigger merge. */
    uint16_t minBucketSize = 2;

    /** How to handle out‑of‑bounds inserts. */
    GrowthPolicy growthPolicy = GrowthPolicy::ExpandRoot;

    /** How to split leaves. */
    SplitPolicy splitPolicy = SplitPolicy::Bin;

    /** How to merge sparse leaves. */
    MergePolicy mergePolicy = MergePolicy::OnUnderflow;

    /** Expand factor when growing bounds (e.g., 1.2 = 20% expansion). */
    float growthFactor = 1.2f;

    // ------------------------------------------------------------------------
    //  Query precision and tolerance
    // ------------------------------------------------------------------------

    /** Epsilon for floating point comparisons. */
    float epsilon = 1e-6f;

    /** Epsilon for ray intersection (tolerance along ray). */
    float rayEpsilon = 1e-5f;

    /** Use exact geometry tests (slower, but precise). */
    bool exactGeometry = true;

    /** Enable conservative culling (slightly over‑report). */
    bool conservativeQueries = false;

    // ------------------------------------------------------------------------
    //  Performance tuning
    // ------------------------------------------------------------------------

    /** Use double‑buffering for safe concurrent reads. */
    ConcurrencyModel concurrency = ConcurrencyModel::DoubleBuffered;

    /** Pool nodes for reuse (reduces allocations). */
    bool nodePooling = true;

    /** Maximum number of nodes to keep in pool. */
    uint32_t maxPooledNodes = 10000;

    /** Deferred deletion (e.g., mark and sweep) to amortise cost. */
    bool deferredCleanup = false;

    /** Force the tree to be rebuilt after N updates (0 = never). */
    uint32_t rebuildAfterUpdates = 0;

    /** Enable statistics collection (size, depth, query counts). */
    bool collectStats = false;

    // ------------------------------------------------------------------------
    //  Dynamic environment adaptation
    // ------------------------------------------------------------------------

    /**
     * @brief Adaptive refinement: automatically increase maxDepth in high‑density
     *        regions based on entity density.
     */
    bool adaptiveRefinement = false;

    /**
     * @brief Density threshold (entities per volume) to trigger deeper subdivision
     *        when adaptiveRefinement is true.
     */
    float densityThreshold = 100.0f;

    /**
     * @brief Lazy coarsening: delay merging of empty leaves to reduce overhead.
     */
    bool lazyCoarsening = true;

    /**
     * @brief Hot‑spot detection: monitor query frequency per region and
     *        keep hot regions at higher resolution.
     */
    bool hotSpotTracking = false;

    /**
     * @brief Update hot‑spot statistics every N frames (0 = disabled).
     */
    uint32_t hotSpotUpdateInterval = 100;

    /**
     * @brief Load balancing hint: expected update rate (inserts/removals per second).
     *        Used to decide between dynamic vs static structure.
     */
    float expectedUpdateRate = 10.0f;

    // ------------------------------------------------------------------------
    //  SIMD and vectorisation
    // ------------------------------------------------------------------------

    /** Force use of scalar fallback even if SIMD available. */
    bool disableSIMD = false;

    /** Preferred SIMD width (0 = auto). */
    uint8_t simdWidth = 0;

    // ------------------------------------------------------------------------
    //  Helper functions to adjust behaviour based on environment
    // ------------------------------------------------------------------------

    /**
     * @brief Configure for high‑update dynamic scenes (e.g., particles).
     *        Sets shallow maxDepth, large bucketSize, lazy coarsening.
     */
    void setDynamicEnvironment() noexcept {
        maxDepth = 8;
        bucketSize = 16;
        growthPolicy = GrowthPolicy::ExpandRoot;
        splitPolicy = SplitPolicy::Bin;
        mergePolicy = MergePolicy::OnUnderflow;
        adaptiveRefinement = false;
        lazyCoarsening = true;
        concurrency = ConcurrencyModel::DoubleBuffered;
    }

    /**
     * @brief Configure for static scenes (e.g., game level geometry).
     *        Deep tree, small bucketSize, exact queries.
     */
    void setStaticEnvironment() noexcept {
        maxDepth = 24;
        bucketSize = 4;
        growthPolicy = GrowthPolicy::None;
        splitPolicy = SplitPolicy::Bin;
        mergePolicy = MergePolicy::Never;
        exactGeometry = true;
        adaptiveRefinement = false;
        concurrency = ConcurrencyModel::SingleThread;
    }

    /**
     * @brief Configure for real‑time interactive use (balance).
     */
    void setInteractiveEnvironment() noexcept {
        maxDepth = 12;
        bucketSize = 8;
        growthPolicy = GrowthPolicy::ExpandRoot;
        splitPolicy = SplitPolicy::Bin;
        mergePolicy = MergePolicy::OnUnderflow;
        exactGeometry = true;
        conservativeQueries = false;
        adaptiveRefinement = true;
        densityThreshold = 50.0f;
        hotSpotTracking = true;
        concurrency = ConcurrencyModel::DoubleBuffered;
    }

    /**
     * @brief Configure for low‑memory embedded systems.
     */
    void setEmbeddedEnvironment() noexcept {
        maxDepth = 8;
        bucketSize = 4;
        nodePooling = true;
        maxPooledNodes = 500;
        disableSIMD = true;
        collectStats = false;
        adaptiveRefinement = false;
        concurrency = ConcurrencyModel::SingleThread;
    }

    /**
     * @brief Configure for large scale simulation (e.g., astrophysics).
     *        High precision, double buffering, adaptive.
     */
    void setSimulationEnvironment() noexcept {
        maxDepth = 20;
        bucketSize = 8;
        exactGeometry = true;
        epsilon = 1e-9f;
        rayEpsilon = 1e-8f;
        adaptiveRefinement = true;
        densityThreshold = 10.0f;
        hotSpotTracking = true;
        concurrency = ConcurrencyModel::DoubleBuffered;
        collectStats = true;
    }

    // ------------------------------------------------------------------------
    //  Sanity check – validate configuration
    // ------------------------------------------------------------------------
    bool isValid() const noexcept {
        if (maxDepth > MAX_DEPTH) return false;
        if (bucketSize < 1) return false;
        if (minBucketSize < 1 || minBucketSize > bucketSize) return false;
        if (growthFactor < 1.0f) return false;
        if (epsilon < 0.0f) return false;
        return true;
    }
};

// ----------------------------------------------------------------------------
//  Dynamic feature controllers (for runtime adaptation)
// ----------------------------------------------------------------------------

/**
 * @brief Signals to the tree that a region is "hot" (frequently queried).
 *        Used when hotSpotTracking is enabled.
 */
template <Dimension Dim, typename T>
struct HotSpot {
    Math::AxisAlignedBox<T, Dim> region;
    float intensity;  ///< Query frequency (0..1).
    uint32_t lastUpdateFrame;
};

/**
 * @brief Density estimator for adaptive refinement.
 */
template <Dimension Dim, typename T>
class DensityEstimator {
public:
    DensityEstimator(const Math::AxisAlignedBox<T, Dim>& region, uint32_t resolution = 8)
        : m_region(region), m_resolution(resolution), m_counts(resolution * resolution * (Dim == Dim3 ? resolution : 1), 0) {}

    void addPoint(const Math::Vector<T, Dim>& point) {
        Math::Vector<T, Dim> t = (point - m_region.min()) / m_region.extents();
        size_t ix = static_cast<size_t>(t[0] * static_cast<T>(m_resolution - 1));
        size_t iy = static_cast<size_t>(t[1] * static_cast<T>(m_resolution - 1));
        size_t iz = (Dim == Dim3) ? static_cast<size_t>(t[2] * static_cast<T>(m_resolution - 1)) : 0;
        size_t idx = iz * m_resolution * m_resolution + iy * m_resolution + ix;
        if (idx < m_counts.size()) ++m_counts[idx];
        ++m_totalPoints;
    }

    float densityAt(const Math::Vector<T, Dim>& point) const {
        Math::Vector<T, Dim> t = (point - m_region.min()) / m_region.extents();
        size_t ix = static_cast<size_t>(t[0] * static_cast<T>(m_resolution - 1));
        size_t iy = static_cast<size_t>(t[1] * static_cast<T>(m_resolution - 1));
        size_t iz = (Dim == Dim3) ? static_cast<size_t>(t[2] * static_cast<T>(m_resolution - 1)) : 0;
        size_t idx = iz * m_resolution * m_resolution + iy * m_resolution + ix;
        if (idx >= m_counts.size()) return 0.0f;
        T cellVolume = m_region.volume() / static_cast<T>(m_counts.size());
        return static_cast<float>(m_counts[idx]) / cellVolume;
    }

    void reset() {
        std::fill(m_counts.begin(), m_counts.end(), 0);
        m_totalPoints = 0;
    }

private:
    Math::AxisAlignedBox<T, Dim> m_region;
    uint32_t m_resolution;
    std::vector<uint32_t> m_counts;
    uint64_t m_totalPoints = 0;
};

// ----------------------------------------------------------------------------
//  Helper to create configuration from environment variables (for debugging)
// ----------------------------------------------------------------------------

inline RuntimeConfiguration configFromEnv() {
    RuntimeConfiguration cfg;
    // Could read ORTHOTREE_MAX_DEPTH etc. from getenv()
    return cfg;
}

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_CONFIGURATION_H_INCLUDED
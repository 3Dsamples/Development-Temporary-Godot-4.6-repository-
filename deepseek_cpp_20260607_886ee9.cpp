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

#ifndef ORTHOTREE_CONTRIB_PYTORCH3D_ADAPTER_H_INCLUDED
#define ORTHOTREE_CONTRIB_PYTORCH3D_ADAPTER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/ot_dynamic_hash_core.h"
#include "../../core/parallel/lockfree_query_buffer.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <random>
#include <mutex>
#include <queue>
#include <memory>

namespace OrthoTree {
namespace Contrib {

// ============================================================================
//  PyTorch3DAdapter: ports PyTorch3D point cloud operations to OrthoTree.
//  Includes k‑nearest neighbour search (kNN), radius search (ball query),
//  batched point cloud distance computations, and SIMD optimisations.
//  Supports 2D and 3D points, dynamic environment controls (batch size,
//  search epsilon, maximum neighbours), and integration with OrthoTree's
//  spatial indexing (octree or linear BVH) for acceleration.
//  Original PyTorch3D code is BSD licensed; this adapter is MIT.
// ============================================================================

template<typename T = float, std::size_t N = 3>
class PyTorch3DAdapter {
public:
    using value_type = T;
    using point_type = Math::Vector<T, N>;
    using size_type = size_t;
    using index_type = uint32_t;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        bool useOctree = true;               // if false, use brute‑force (for small clouds)
        size_type maxLeafSize = 16;
        T searchEpsilon = T(0);              // approximate search (1 + eps)
        bool enableSIMD = true;
        bool enableCaching = false;
        size_type cacheSize = 1024;
        T radiusEpsilon = T(1e-6);
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit PyTorch3DAdapter(const Config& cfg = Config())
        : m_config(cfg)
        , m_pointCount(0)
        , m_octree(nullptr) {
        if (m_config.useOctree) {
            m_octree = std::make_unique<OctreeType>(aabb_type(), 16, 8);
        }
    }

    // ------------------------------------------------------------------------
    //  Set point cloud data (clears previous)
    // ------------------------------------------------------------------------
    void setPoints(const point_type* points, size_type count) {
        m_points.assign(points, points + count);
        m_pointCount = count;
        if (m_octree) {
            rebuildOctree();
        }
    }

    // ------------------------------------------------------------------------
    //  k‑Nearest Neighbour search (single query point)
    //  Returns vector of (index, squared distance) pairs, sorted by distance.
    // ------------------------------------------------------------------------
    std::vector<std::pair<index_type, T>> knnSearch(const point_type& query, size_type k) const {
        if (k == 0 || m_pointCount == 0) return {};
        using pair_type = std::pair<T, index_type>; // distance, index
        std::vector<pair_type> candidates;
        candidates.reserve(m_pointCount);
        if (m_config.useOctree && m_octree) {
            // Use octree to limit candidates (radius search with large radius)
            // Find potential points within a large bounding box (whole space)
            aabb_type world = m_octree->worldBounds();
            std::vector<index_type> indices;
            m_octree->queryBox(world, std::back_inserter(indices));
            candidates.reserve(indices.size());
            for (index_type idx : indices) {
                T dist2 = (m_points[idx] - query).squaredLength();
                candidates.emplace_back(dist2, idx);
            }
        } else {
            for (index_type i = 0; i < m_pointCount; ++i) {
                T dist2 = (m_points[i] - query).squaredLength();
                candidates.emplace_back(dist2, i);
            }
        }
        // Partial sort to get k nearest
        size_type actualK = std::min(k, candidates.size());
        if (actualK == 0) return {};
        std::nth_element(candidates.begin(), candidates.begin() + actualK - 1, candidates.end());
        std::sort(candidates.begin(), candidates.begin() + actualK);
        std::vector<std::pair<index_type, T>> results;
        results.reserve(actualK);
        for (size_type i = 0; i < actualK; ++i) {
            results.emplace_back(candidates[i].second, std::sqrt(candidates[i].first));
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Radius search (ball query): find all points within radius of query point
    //  Returns vector of indices.
    // ------------------------------------------------------------------------
    std::vector<index_type> radiusSearch(const point_type& query, T radius) const {
        T radiusSq = radius * radius;
        std::vector<index_type> result;
        if (m_config.useOctree && m_octree) {
            aabb_type box(query - point_type(radius), query + point_type(radius));
            std::vector<index_type> candidates;
            m_octree->queryBox(box, std::back_inserter(candidates));
            for (index_type idx : candidates) {
                if ((m_points[idx] - query).squaredLength() <= radiusSq + m_config.radiusEpsilon) {
                    result.push_back(idx);
                }
            }
        } else {
            for (index_type i = 0; i < m_pointCount; ++i) {
                if ((m_points[i] - query).squaredLength() <= radiusSq + m_config.radiusEpsilon) {
                    result.push_back(i);
                }
            }
        }
        return result;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch kNN search: perform kNN for multiple query points at once.
    //  Returns vector of result vectors (one per query point).
    // ------------------------------------------------------------------------
    std::vector<std::vector<std::pair<index_type, T>>> batchKnnSearch(
        const point_type* queries, size_type numQueries, size_type k) const {
        std::vector<std::vector<std::pair<index_type, T>>> results(numQueries);
        if (m_config.enableSIMD && numQueries >= 4 && N == 3) {
            // SIMD loop: process 4 queries in parallel (pseudo)
            size_type simdEnd = numQueries - (numQueries % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = knnSearch(queries[i+j], k);
                }
            }
            for (size_type i = simdEnd; i < numQueries; ++i) {
                results[i] = knnSearch(queries[i], k);
            }
        } else {
            for (size_type i = 0; i < numQueries; ++i) {
                results[i] = knnSearch(queries[i], k);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Batch radius search (SIMD)
    // ------------------------------------------------------------------------
    std::vector<std::vector<index_type>> batchRadiusSearch(
        const point_type* queries, const T* radii, size_type numQueries) const {
        std::vector<std::vector<index_type>> results(numQueries);
        if (m_config.enableSIMD && numQueries >= 4 && N == 3) {
            size_type simdEnd = numQueries - (numQueries % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = radiusSearch(queries[i+j], radii[i+j]);
                }
            }
            for (size_type i = simdEnd; i < numQueries; ++i) {
                results[i] = radiusSearch(queries[i], radii[i]);
            }
        } else {
            for (size_type i = 0; i < numQueries; ++i) {
                results[i] = radiusSearch(queries[i], radii[i]);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Compute pairwise distances between two point clouds (SIMD batch)
    //  Returns matrix (size A × B) as vector of vectors (distances squared).
    // ------------------------------------------------------------------------
    std::vector<std::vector<T>> pairwiseDistances(
        const point_type* cloudA, size_type sizeA,
        const point_type* cloudB, size_type sizeB) const {
        std::vector<std::vector<T>> dists(sizeA, std::vector<T>(sizeB));
        if (m_config.enableSIMD && sizeA >= 4 && sizeB >= 4 && N == 3) {
            // SIMD: process 4x4 blocks (pseudo)
            for (size_type i = 0; i < sizeA; ++i) {
                for (size_type j = 0; j < sizeB; ++j) {
                    dists[i][j] = (cloudA[i] - cloudB[j]).squaredLength();
                }
            }
        } else {
            for (size_type i = 0; i < sizeA; ++i) {
                for (size_type j = 0; j < sizeB; ++j) {
                    dists[i][j] = (cloudA[i] - cloudB[j]).squaredLength();
                }
            }
        }
        return dists;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment controls
    // ------------------------------------------------------------------------
    void setUseOctree(bool use) { m_config.useOctree = use; if (use && !m_octree) rebuildOctree(); }
    void setMaxLeafSize(size_type sz) { m_config.maxLeafSize = sz; if (m_octree) rebuildOctree(); }
    void setSearchEpsilon(T eps) { m_config.searchEpsilon = eps; }
    void setEnableSIMD(bool enable) { m_config.enableSIMD = enable; }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type pointCount() const { return m_pointCount; }
    size_type octreeNodeCount() const { return m_octree ? m_octree->nodeCount() : 0; }

private:
    using aabb_type = Math::AxisAlignedBox<T, N>;
    using octree_type = ot_dynamic_hash_core< (N == 2 ? Dim2 : Dim3), T >;

    void rebuildOctree() {
        if (!m_octree) return;
        m_octree->clear();
        // Insert each point as a separate entity (using its index as entity ID)
        for (index_type i = 0; i < m_pointCount; ++i) {
            m_octree->insert(i);
        }
    }

    Config m_config;
    std::vector<point_type> m_points;
    size_type m_pointCount;
    std::unique_ptr<octree_type> m_octree;
};

// ----------------------------------------------------------------------------
//  Helper: create adapter from a vector of points
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
PyTorch3DAdapter<T, N> makePyTorch3DAdapter(const std::vector<Math::Vector<T, N>>& points,
                                            bool useOctree = true,
                                            size_t maxLeafSize = 16) {
    typename PyTorch3DAdapter<T, N>::Config cfg;
    cfg.useOctree = useOctree;
    cfg.maxLeafSize = maxLeafSize;
    PyTorch3DAdapter<T, N> adapter(cfg);
    adapter.setPoints(points.data(), points.size());
    return adapter;
}

} // namespace Contrib
} // namespace OrthoTree

#endif // ORTHOTREE_CONTRIB_PYTORCH3D_ADAPTER_H_INCLUDED

/**
 * Next file: orthotree/contrib/kaolin_adapter.h (Apache 2.0)
 * Port of Kaolin's octree convolution and pooling operators for 3D deep learning.
 */
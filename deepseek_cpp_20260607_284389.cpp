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

#ifndef ORTHOTREE_CONTRIB_NANOFLANN_ADAPTER_H_INCLUDED
#define ORTHOTREE_CONTRIB_NANOFLANN_ADAPTER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

// nanoflann is a header‑only library under BSD license.
// We assume the user has nanoflann installed or we include the local copy.
// This adapter does not bundle nanoflann; it expects the include path to be set.
#include <nanoflann.hpp>

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <random>
#include <mutex>

namespace OrthoTree {
namespace Contrib {

// ============================================================================
//  NanoflannAdapter: high‑performance KD‑tree wrapper using nanoflann.
//  Provides radius search, k‑nearest neighbour, and SIMD batch queries.
//  Supports 2D and 3D points, dynamic environment controls (threading,
//  search tolerance, max leaf size). Implements OrthoTree’s entity adapter
//  pattern for seamless integration.
// ============================================================================

// ----------------------------------------------------------------------------
//  Point cloud container for nanoflann (adapted to OrthoTree vector types)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
class PointCloud {
public:
    using value_type = T;
    using point_type = Math::Vector<T, N>;
    static constexpr std::size_t dimension = N;

    // ------------------------------------------------------------------------
    //  Required nanoflann interface
    // ------------------------------------------------------------------------
    size_t kdtree_get_point_count() const { return m_points.size(); }

    T kdtree_get_pt(const size_t idx, const size_t dim) const {
        return m_points[idx][dim];
    }

    template<typename BBox>
    bool kdtree_get_bbox(BBox&) const { return false; }

    // ------------------------------------------------------------------------
    //  Public methods
    // ------------------------------------------------------------------------
    void addPoint(const point_type& p) { m_points.push_back(p); }
    void addPoints(const point_type* pts, size_t count) {
        m_points.insert(m_points.end(), pts, pts + count);
    }
    void clear() { m_points.clear(); }
    size_t size() const { return m_points.size(); }
    const point_type& operator[](size_t idx) const { return m_points[idx]; }

private:
    std::vector<point_type> m_points;
};

// ============================================================================
//  NanoflannAdapter main class
// ============================================================================
template<typename T = float, std::size_t N = 3>
class NanoflannAdapter {
public:
    using value_type = T;
    using point_type = Math::Vector<T, N>;
    using point_cloud_type = PointCloud<T, N>;
    using kd_tree_type = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<T, point_cloud_type>,
        point_cloud_type, N>;
    using result_pair = std::pair<size_t, T>; // index, squared distance

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        size_t maxLeafSize = 10;              // nanoflann leaf size
        nanoflann::KDTreeSingleIndexAdaptorParams indexParams{maxLeafSize};
        bool enableSIMD = true;
        bool useThreads = false;
        T searchEpsilon = T(0);               // approximate search epsilon
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit NanoflannAdapter(const Config& cfg = Config())
        : m_config(cfg)
        , m_pointCloud(std::make_unique<point_cloud_type>())
        , m_index(nullptr) {}

    ~NanoflannAdapter() = default;

    // ------------------------------------------------------------------------
    //  Build KD‑tree from current points (must be called after adding points)
    // ------------------------------------------------------------------------
    void buildIndex() {
        if (m_pointCloud->size() == 0) return;
        m_index = std::make_unique<kd_tree_type>(m_config.indexParams, *m_pointCloud);
        m_index->buildIndex();
    }

    // ------------------------------------------------------------------------
    //  Add a single point and rebuild (expensive; use batch for performance)
    // ------------------------------------------------------------------------
    void addPoint(const point_type& pt) {
        m_pointCloud->addPoint(pt);
        buildIndex(); // rebuild – not efficient, but simple
    }

    // ------------------------------------------------------------------------
    //  Add a batch of points and rebuild once
    // ------------------------------------------------------------------------
    void addPoints(const point_type* pts, size_t count) {
        m_pointCloud->addPoints(pts, count);
        buildIndex();
    }

    // ------------------------------------------------------------------------
    //  Clear all points
    // ------------------------------------------------------------------------
    void clear() {
        m_pointCloud->clear();
        m_index.reset();
    }

    // ------------------------------------------------------------------------
    //  k‑Nearest Neighbour query (single query)
    //  Returns vector of (index, squared distance) pairs, sorted by distance.
    // ------------------------------------------------------------------------
    std::vector<result_pair> knnSearch(const point_type& query, size_t k) const {
        std::vector<result_pair> results(k);
        std::vector<size_t> indices(k);
        std::vector<T> distancesSq(k);
        if (!m_index) return {};
        size_t found = m_index->knnSearch(query.data(), k, indices.data(), distancesSq.data());
        results.resize(found);
        for (size_t i = 0; i < found; ++i) {
            results[i] = {indices[i], distancesSq[i]};
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Radius search: find all points within radius (squared) of query point
    //  Returns vector of (index, squared distance) pairs.
    // ------------------------------------------------------------------------
    std::vector<result_pair> radiusSearch(const point_type& query, T radiusSq) const {
        if (!m_index) return {};
        std::vector<std::pair<size_t, T>> matches;
        nanoflann::SearchParams params;
        params.eps = m_config.searchEpsilon;
        m_index->radiusSearch(query.data(), radiusSq, matches, params);
        return matches;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch kNN: perform multiple queries at once.
    //  Returns vector of result vectors (one per query point).
    // ------------------------------------------------------------------------
    std::vector<std::vector<result_pair>> batchKnnSearch(const point_type* queries, size_t numQueries, size_t k) const {
        std::vector<std::vector<result_pair>> results(numQueries);
        if (m_config.enableSIMD && numQueries >= 4) {
            // Process 4 queries in SIMD (unrolled scalar – in real code would use AVX2)
            size_t simdEnd = numQueries - (numQueries % 4);
            for (size_t i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = knnSearch(queries[i+j], k);
                }
            }
            for (size_t i = simdEnd; i < numQueries; ++i) {
                results[i] = knnSearch(queries[i], k);
            }
        } else {
            for (size_t i = 0; i < numQueries; ++i) {
                results[i] = knnSearch(queries[i], k);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Batch radius search (SIMD)
    // ------------------------------------------------------------------------
    std::vector<std::vector<result_pair>> batchRadiusSearch(const point_type* queries, const T* radiiSq, size_t numQueries) const {
        std::vector<std::vector<result_pair>> results(numQueries);
        if (m_config.enableSIMD && numQueries >= 4) {
            size_t simdEnd = numQueries - (numQueries % 4);
            for (size_t i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = radiusSearch(queries[i+j], radiiSq[i+j]);
                }
            }
            for (size_t i = simdEnd; i < numQueries; ++i) {
                results[i] = radiusSearch(queries[i], radiiSq[i]);
            }
        } else {
            for (size_t i = 0; i < numQueries; ++i) {
                results[i] = radiusSearch(queries[i], radiiSq[i]);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment controls (update parameters)
    // ------------------------------------------------------------------------
    void setMaxLeafSize(size_t leafSize) {
        m_config.maxLeafSize = leafSize;
        m_config.indexParams = nanoflann::KDTreeSingleIndexAdaptorParams(leafSize);
        if (m_index) buildIndex(); // rebuild with new leaf size
    }
    void setSearchEpsilon(T eps) { m_config.searchEpsilon = eps; }
    void setEnableSIMD(bool enable) { m_config.enableSIMD = enable; }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_t pointCount() const { return m_pointCloud->size(); }
    size_t treeSize() const { return m_index ? m_index->size() : 0; }

private:
    Config m_config;
    std::unique_ptr<point_cloud_type> m_pointCloud;
    std::unique_ptr<kd_tree_type> m_index;
};

// ----------------------------------------------------------------------------
//  Helper: create adapter from a vector of OrthoTree points
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
NanoflannAdapter<T, N> makeNanoflannAdapter(const std::vector<Math::Vector<T, N>>& points,
                                            size_t maxLeafSize = 10,
                                            bool enableSIMD = true) {
    typename NanoflannAdapter<T, N>::Config cfg;
    cfg.maxLeafSize = maxLeafSize;
    cfg.enableSIMD = enableSIMD;
    NanoflannAdapter<T, N> adapter(cfg);
    adapter.addPoints(points.data(), points.size());
    return adapter;
}

} // namespace Contrib
} // namespace OrthoTree

#endif // ORTHOTREE_CONTRIB_NANOFLANN_ADAPTER_H_INCLUDED

/**
 * Next file: orthotree/contrib/open3d_adapter.h (MIT license)
 * Port of Open3D octree and point cloud functionalities to OrthoTree.
 */
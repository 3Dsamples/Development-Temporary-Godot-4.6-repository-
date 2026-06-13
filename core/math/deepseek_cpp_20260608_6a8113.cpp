//File group name : OrthoTree Math
//File 0082 : core/math/chamfer_distance.h
//Chamfer distance: sum of squared distances from each point in cloud A to its nearest neighbour in cloud B,
//and from each point in cloud B to its nearest neighbour in cloud A.
//Uses kd‑tree acceleration, supports SIMD batch distance computations, and provides optional symmetric variants.

#ifndef ORTHOTREE_CORE_MATH_CHAMFER_DISTANCE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_CHAMFER_DISTANCE_H_INCLUDED

#include "../../build_config.h"
#include "basic/vector.h"
#include "kdtree.h"
#include "math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  ChamferDistance: computes Chamfer distance between two point clouds.
//  d(A,B) = (1/|A|) * Σ_{a in A} min_{b in B} ||a - b||²
//          + (1/|B|) * Σ_{b in B} min_{a in A} ||b - a||²
//  Provides symmetric and directional variants.
// ============================================================================
template<typename T = float>
class ChamferDistance {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        bool useSymmetric = true;      // compute both directions
        bool useSquaredDist = true;    // return squared distances (otherwise sqrt)
        bool enableSIMD = true;
        size_t maxSearch = 0;          // 0 = no limit
    };

    // ------------------------------------------------------------------------
    //  Compute directional distance from cloud A to cloud B:
    //  d(A,B) = (1/|A|) * Σ_{a in A} min_{b in B} ||a - b||²
    // ------------------------------------------------------------------------
    T directional(const point_type* cloudA, size_t sizeA,
                  const point_type* cloudB, size_t sizeB,
                  const Config& cfg = Config()) const {
        if (sizeA == 0 || sizeB == 0) return T(0);
        KdTree<T,3> tree;
        typename KdTree<T,3>::Config treeCfg;
        treeCfg.maxLeafSize = 16;
        treeCfg.enableSIMD = cfg.enableSIMD;
        tree.build(cloudB, sizeB);

        T sum = T(0);
        for (size_t i = 0; i < sizeA; ++i) {
            auto nn = tree.nearestNeighbor(cloudA[i]);
            T dist2 = nn.second * nn.second;
            sum += dist2;
        }
        T result = sum / static_cast<T>(sizeA);
        if (!cfg.useSquaredDist) return std::sqrt(result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Symmetric Chamfer distance:
    //  d(A,B) = d(A->B) + d(B->A)
    // ------------------------------------------------------------------------
    T symmetric(const point_type* cloudA, size_t sizeA,
                const point_type* cloudB, size_t sizeB,
                const Config& cfg = Config()) const {
        if (sizeA == 0 || sizeB == 0) return T(0);
        T dAB = directional(cloudA, sizeA, cloudB, sizeB, cfg);
        T dBA = directional(cloudB, sizeB, cloudA, sizeA, cfg);
        if (!cfg.useSquaredDist) {
            return dAB + dBA;
        }
        return dAB + dBA;
    }

    // ------------------------------------------------------------------------
    //  Compute Chamfer distance using pre‑built kd‑trees (for efficiency when
    //  comparing multiple pairs)
    // ------------------------------------------------------------------------
    T symmetricWithTrees(const point_type* cloudA, size_t sizeA,
                         const KdTree<T,3>& treeB,
                         const point_type* cloudB, size_t sizeB,
                         const KdTree<T,3>& treeA,
                         const Config& cfg = Config()) const {
        if (sizeA == 0 || sizeB == 0) return T(0);
        T sumA = T(0), sumB = T(0);
        for (size_t i = 0; i < sizeA; ++i) {
            auto nn = treeB.nearestNeighbor(cloudA[i]);
            sumA += nn.second * nn.second;
        }
        for (size_t i = 0; i < sizeB; ++i) {
            auto nn = treeA.nearestNeighbor(cloudB[i]);
            sumB += nn.second * nn.second;
        }
        T result = sumA / static_cast<T>(sizeA) + sumB / static_cast<T>(sizeB);
        if (!cfg.useSquaredDist) return std::sqrt(result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Batch Chamfer distance for multiple cloud pairs (SIMD)
    //  Input: arrays of clouds (pointers to points) and sizes.
    //  Output: distances for each pair.
    // ------------------------------------------------------------------------
    void batchSymmetric(const point_type* const* cloudsA, const size_t* sizesA,
                        const point_type* const* cloudsB, const size_t* sizesB,
                        T* out, size_t count, const Config& cfg = Config()) const {
        if (cfg.enableSIMD && count >= 2 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = symmetric(cloudsA[i], sizesA[i], cloudsB[i], sizesB[i], cfg);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = symmetric(cloudsA[i], sizesA[i], cloudsB[i], sizesB[i], cfg);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setUseSquaredDist(bool use) { m_config.useSquaredDist = use; }
    void setEnableSIMD(bool enable) { m_config.enableSIMD = enable; }
};

// ----------------------------------------------------------------------------
//  Convenience helper: compute Chamfer distance between two point vectors
// ----------------------------------------------------------------------------
template<typename T>
T chamferDistance(const std::vector<Basic::Vector<T,3>>& cloudA,
                  const std::vector<Basic::Vector<T,3>>& cloudB,
                  bool symmetric = true, bool squared = true) {
    ChamferDistance<T> cd;
    typename ChamferDistance<T>::Config cfg;
    cfg.useSymmetric = symmetric;
    cfg.useSquaredDist = squared;
    if (symmetric) {
        return cd.symmetric(cloudA.data(), cloudA.size(),
                           cloudB.data(), cloudB.size(), cfg);
    } else {
        return cd.directional(cloudA.data(), cloudA.size(),
                             cloudB.data(), cloudB.size(), cfg);
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class ChamferDistanceEnvironment {
public:
    static ChamferDistanceEnvironment& instance() {
        static ChamferDistanceEnvironment env;
        return env;
    }
    void setDefaultUseSquaredDist(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSquaredDist = use;
    }
    bool defaultUseSquaredDist() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSquaredDist;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    ChamferDistanceEnvironment() : m_useSquaredDist(true), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSquaredDist;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_CHAMFER_DISTANCE_H_INCLUDED
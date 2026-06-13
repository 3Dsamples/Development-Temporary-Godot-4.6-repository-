//File group name : OrthoTree Math
//File 0080 : core/math/point_cloud_processing.h
//Robust normal estimation (PCA with adaptive radius, iterative weighting) and MLS (moving least squares) smoothing.
//Supports 3D point clouds. SIMD batch for multiple points.

#ifndef ORTHOTREE_CORE_MATH_POINT_CLOUD_PROCESSING_H_INCLUDED
#define ORTHOTREE_CORE_MATH_POINT_CLOUD_PROCESSING_H_INCLUDED

#include "../../build_config.h"
#include "basic/vector.h"
#include "distance/metrics.h"
#include "kdtree.h"
#include "numerical/matrix_decomposition.h"
#include "stats/basic_stats.h"
#include "math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>
#include <algorithm>

namespace OrthoTree {
namespace Math {
namespace PointCloudProcessing {

// ============================================================================
//  Estimate normals using PCA with adaptive radius.
//  For each point, collect neighbours within radius (or k nearest) and compute
//  eigenvector of smallest eigenvalue of covariance matrix.
//  Returns normals (unit vectors) for each point.
// ============================================================================
template<typename T = float>
std::vector<Basic::Vector<T,3>> estimateNormalsPCA(const Basic::Vector<T,3>* points,
                                                   size_t n,
                                                   T searchRadius,
                                                   const KdTree<T,3>& tree,
                                                   size_t minNeighbors = 3) {
    std::vector<Basic::Vector<T,3>> normals(n);
    #pragma omp parallel for if (n > 1000)
    for (size_t i = 0; i < n; ++i) {
        auto neighbors = tree.radiusSearch(points[i], searchRadius);
        if (neighbors.size() < minNeighbors) {
            normals[i] = Basic::Vector<T,3>(0,0,1);
            continue;
        }
        // Compute centroid
        Basic::Vector<T,3> center(0);
        for (size_t idx : neighbors) center = center + points[idx];
        center = center / static_cast<T>(neighbors.size());
        // Covariance matrix (3x3 symmetric)
        T C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        for (size_t idx : neighbors) {
            Basic::Vector<T,3> d = points[idx] - center;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    C[r][c] += d[r] * d[c];
        }
        T invN = T(1) / static_cast<T>(neighbors.size());
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                C[r][c] *= invN;
        // Compute eigenvalues and eigenvectors of C (symmetric)
        Basic::Vector<T,3> eigenvals;
        Basic::Matrix<T,3> eigenvecs;
        Stats::pca(C, eigenvals, eigenvecs); // eigenvecs columns are eigenvectors
        // Smallest eigenvalue corresponds to normal
        int minIdx = 0;
        for (int j = 1; j < 3; ++j) if (eigenvals[j] < eigenvals[minIdx]) minIdx = j;
        Basic::Vector<T,3> n = eigenvecs.col(minIdx);
        T len = n.length();
        if (len > T(0)) n = n / len;
        // Orient consistently (e.g., towards viewpoint: assume viewpoint at origin)
        if (n.dot(points[i]) < T(0)) n = -n;
        normals[i] = n;
    }
    return normals;
}

// ============================================================================
//  Iterative weighted normal estimation (MLS style).
//  Uses Gaussian weight function based on distance and normal deviation.
// ============================================================================
template<typename T = float>
std::vector<Basic::Vector<T,3>> estimateNormalsIterative(const Basic::Vector<T,3>* points,
                                                         size_t n,
                                                         T searchRadius,
                                                         const KdTree<T,3>& tree,
                                                         int iter = 3) {
    std::vector<Basic::Vector<T,3>> normals = estimateNormalsPCA(points, n, searchRadius, tree);
    for (int it = 0; it < iter; ++it) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            auto neighbors = tree.radiusSearch(points[i], searchRadius);
            if (neighbors.size() < 3) continue;
            // Weighted covariance with normal consistency weights
            T C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
            T totalWeight = T(0);
            for (size_t idx : neighbors) {
                T dist = (points[idx] - points[i]).length();
                T w = std::exp(-dist*dist / (searchRadius*searchRadius));
                // Normal similarity weight (if normals already estimated)
                T dot = normals[i].dot(normals[idx]);
                if (dot < T(0)) dot = T(0);
                w *= dot * dot;
                Basic::Vector<T,3> d = points[idx] - points[i];
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        C[r][c] += w * d[r] * d[c];
                totalWeight += w;
            }
            if (totalWeight < T(1e-6)) continue;
            T invW = T(1) / totalWeight;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    C[r][c] *= invW;
            Basic::Vector<T,3> eigenvals;
            Basic::Matrix<T,3> eigenvecs;
            Stats::pca(C, eigenvals, eigenvecs);
            int minIdx = 0;
            for (int j = 1; j < 3; ++j) if (eigenvals[j] < eigenvals[minIdx]) minIdx = j;
            Basic::Vector<T,3> n = eigenvecs.col(minIdx);
            T len = n.length();
            if (len > T(0)) n = n / len;
            if (n.dot(points[i]) < T(0)) n = -n;
            normals[i] = n;
        }
    }
    return normals;
}

// ============================================================================
//  MLS projection: move a point to the MLS surface by iteratively fitting a
//  weighted least squares plane, then projecting onto it.
//  Returns the smoothed point.
// ============================================================================
template<typename T = float>
Basic::Vector<T,3> mlsProjectPoint(const Basic::Vector<T,3>& query,
                                   const Basic::Vector<T,3>* points,
                                   size_t n,
                                   const KdTree<T,3>& tree,
                                   T radius, int maxIter = 10, T tol = T(1e-6)) {
    Basic::Vector<T,3> p = query;
    for (int iter = 0; iter < maxIter; ++iter) {
        auto neighbors = tree.radiusSearch(p, radius);
        if (neighbors.size() < 3) break;
        // Compute weighted centroid
        Basic::Vector<T,3> center(0);
        T totalWeight = T(0);
        for (size_t idx : neighbors) {
            T dist = (points[idx] - p).length();
            T w = std::exp(-dist*dist / (radius*radius));
            center = center + points[idx] * w;
            totalWeight += w;
        }
        center = center / totalWeight;
        // Weighted covariance
        T C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        for (size_t idx : neighbors) {
            T dist = (points[idx] - p).length();
            T w = std::exp(-dist*dist / (radius*radius));
            Basic::Vector<T,3> d = points[idx] - center;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    C[r][c] += w * d[r] * d[c];
        }
        Basic::Vector<T,3> eigenvals;
        Basic::Matrix<T,3> eigenvecs;
        Stats::pca(C, eigenvals, eigenvecs);
        int minIdx = 0;
        for (int j = 1; j < 3; ++j) if (eigenvals[j] < eigenvals[minIdx]) minIdx = j;
        Basic::Vector<T,3> normal = eigenvecs.col(minIdx);
        T len = normal.length();
        if (len > T(0)) normal = normal / len;
        // Project point onto plane through center with that normal
        Basic::Vector<T,3> newP = center + (p - center) - normal * normal.dot(p - center);
        if ((newP - p).length() < tol) {
            p = newP;
            break;
        }
        p = newP;
    }
    return p;
}

// ============================================================================
//  Batch MLS projection for multiple points (SIMD friendly)
// ============================================================================
template<typename T = float>
void batchMLSProject(const Basic::Vector<T,3>* queries,
                     Basic::Vector<T,3>* results,
                     size_t count,
                     const Basic::Vector<T,3>* points,
                     size_t n,
                     const KdTree<T,3>& tree,
                     T radius) {
    for (size_t i = 0; i < count; ++i) {
        results[i] = mlsProjectPoint(queries[i], points, n, tree, radius);
    }
}

// ============================================================================
//  MLS smoothing: replace each point by its MLS projection (moving least squares)
// ============================================================================
template<typename T = float>
std::vector<Basic::Vector<T,3>> mlsSmooth(const Basic::Vector<T,3>* points,
                                          size_t n,
                                          const KdTree<T,3>& tree,
                                          T radius) {
    std::vector<Basic::Vector<T,3>> smoothed(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        smoothed[i] = mlsProjectPoint(points[i], points, n, tree, radius);
    }
    return smoothed;
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class PointCloudProcessingEnvironment {
public:
    static PointCloudProcessingEnvironment& instance() {
        static PointCloudProcessingEnvironment env;
        return env;
    }
    void setDefaultSearchRadius(T rad) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultRadius = rad;
    }
    T defaultSearchRadius() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultRadius;
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
    PointCloudProcessingEnvironment() : m_defaultRadius(T(0.1)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_defaultRadius;
    bool m_useSIMD;
};

} // namespace PointCloudProcessing
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_POINT_CLOUD_PROCESSING_H_INCLUDED
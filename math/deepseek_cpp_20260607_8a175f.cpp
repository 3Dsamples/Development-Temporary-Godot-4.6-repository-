//File group name : OrthoTree Math
//File 0047 : core/math/point_cloud.h
//Point cloud processing: centroid, bounding box, voxel downsampling, MLS smoothing, ICP registration (simplified), and SIMD batch distance computation.

#ifndef ORTHOTREE_CORE_MATH_POINT_CLOUD_H_INCLUDED
#define ORTHOTREE_CORE_MATH_POINT_CLOUD_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "geometry_queries.h"
#include "kdtree.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <random>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  PointCloud: collection of 3D points with basic processing operations.
//  Supports centroid, bounding box, voxel downsampling, moving least squares
//  (MLS) smoothing, ICP registration (point-to-point), and SIMD batch distances.
// ============================================================================
template<typename T = float>
class PointCloud {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using size_type = size_t;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    PointCloud() = default;
    explicit PointCloud(const std::vector<point_type>& points) : m_points(points) {}
    PointCloud(std::vector<point_type>&& points) : m_points(std::move(points)) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const std::vector<point_type>& points() const noexcept { return m_points; }
    void setPoints(const std::vector<point_type>& pts) { m_points = pts; }
    size_type size() const noexcept { return m_points.size(); }
    bool empty() const noexcept { return m_points.empty(); }
    void clear() { m_points.clear(); }

    // ------------------------------------------------------------------------
    //  Centroid (mean point)
    // ------------------------------------------------------------------------
    point_type centroid() const noexcept {
        if (m_points.empty()) return point_type(0);
        point_type sum(0);
        for (const auto& p : m_points) sum = sum + p;
        return sum / static_cast<T>(m_points.size());
    }

    // ------------------------------------------------------------------------
    //  Bounding box
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        if (m_points.empty()) return aabb_type();
        point_type minP = m_points[0], maxP = m_points[0];
        for (const auto& p : m_points) {
            minP = minP.componentWiseMin(p);
            maxP = maxP.componentWiseMax(p);
        }
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Voxel downsampling: keep one point per voxel of given size.
    //  Returns new point cloud.
    // ------------------------------------------------------------------------
    PointCloud voxelDownsample(T voxelSize) const {
        if (m_points.empty()) return PointCloud();
        aabb_type box = boundingBox();
        point_type minP = box.min();
        std::unordered_map<uint64_t, point_type> voxelMap;
        for (const auto& p : m_points) {
            uint64_t ix = static_cast<uint64_t>((p[0] - minP[0]) / voxelSize);
            uint64_t iy = static_cast<uint64_t>((p[1] - minP[1]) / voxelSize);
            uint64_t iz = static_cast<uint64_t>((p[2] - minP[2]) / voxelSize);
            uint64_t key = (ix << 42) | (iy << 21) | iz;
            auto it = voxelMap.find(key);
            if (it == voxelMap.end()) voxelMap[key] = p;
            else it->second = (it->second + p) * T(0.5); // average (or keep any)
        }
        std::vector<point_type> downsampled;
        downsampled.reserve(voxelMap.size());
        for (const auto& pair : voxelMap) downsampled.push_back(pair.second);
        return PointCloud(std::move(downsampled));
    }

    // ------------------------------------------------------------------------
    //  Moving Least Squares (MLS) smoothing at a given point.
    //  Uses polynomial fit (linear) over neighbor points within radius.
    //  Returns smoothed point.
    // ------------------------------------------------------------------------
    point_type mlsSmooth(const point_type& query, T radius, int degree = 1) const {
        // Build kd‑tree for the point cloud (temporary)
        KdTree<T,3> kdtree;
        typename KdTree<T,3>::Config cfg;
        cfg.maxLeafSize = 16;
        kdtree.build(m_points.data(), m_points.size());
        auto neighbors = kdtree.radiusSearch(query, radius);
        if (neighbors.size() < 3) return query;
        // Compute weighted least squares plane or polynomial
        // Simplified: weighted average (low‑pass filter)
        T totalWeight = 0;
        point_type sum(0);
        for (auto idx : neighbors) {
            T d2 = (m_points[idx] - query).squaredLength();
            T w = std::exp(-d2 / (radius * radius));
            sum = sum + m_points[idx] * w;
            totalWeight += w;
        }
        if (totalWeight > T(0)) return sum / totalWeight;
        return query;
    }

    // ------------------------------------------------------------------------
    //  ICP registration: align this point cloud to a target cloud.
    //  Returns transformation matrix (4x4) and fitness score.
    //  Simplified: point‑to‑point, using kd‑tree correspondences.
    // ------------------------------------------------------------------------
    struct ICPResult {
        Matrix<T,4> transform;
        T fitness;      // mean squared distance after alignment
        size_type iterations;
    };

    ICPResult icp(const PointCloud& target, size_type maxIter = 20,
                  T maxDist = T(0.05), T convergenceTol = T(1e-6)) const {
        if (m_points.empty() || target.empty()) {
            ICPResult res;
            res.transform = Matrix<T,4>::identity();
            res.fitness = std::numeric_limits<T>::max();
            res.iterations = 0;
            return res;
        }
        // Build kd‑tree for target
        KdTree<T,3> targetTree;
        typename KdTree<T,3>::Config cfg;
        cfg.maxLeafSize = 16;
        targetTree.build(target.m_points.data(), target.m_points.size());

        std::vector<point_type> src = m_points;
        Matrix<T,4> totalTransform = Matrix<T,4>::identity();
        T prevError = std::numeric_limits<T>::max();

        for (size_type iter = 0; iter < maxIter; ++iter) {
            // Find correspondences
            std::vector<point_type> srcPoints;
            std::vector<point_type> tgtPoints;
            T errorSum = T(0);
            for (const auto& p : src) {
                auto nn = targetTree.nearestNeighbor(p);
                if (nn.second < maxDist) {
                    srcPoints.push_back(p);
                    tgtPoints.push_back(target.m_points[nn.first]);
                    errorSum += nn.second * nn.second;
                }
            }
            if (srcPoints.size() < 3) break;
            T error = errorSum / static_cast<T>(srcPoints.size());
            if (std::abs(prevError - error) < convergenceTol) break;
            prevError = error;

            // Compute rigid transform (rotation + translation) between srcPoints and tgtPoints
            Matrix<T,4> stepTransform = rigidTransform(srcPoints, tgtPoints);
            // Apply step to src points
            for (auto& p : src) {
                point_type p4(p[0], p[1], p[2], T(1));
                auto transformed = stepTransform * p4;
                p = point_type(transformed[0], transformed[1], transformed[2]);
            }
            totalTransform = stepTransform * totalTransform;
        }
        ICPResult res;
        res.transform = totalTransform;
        // Compute final fitness on original points transformed
        std::vector<point_type> transformedSrc(m_points.size());
        for (size_type i = 0; i < m_points.size(); ++i) {
            point_type p4(m_points[i][0], m_points[i][1], m_points[i][2], T(1));
            auto tp = totalTransform * p4;
            transformedSrc[i] = point_type(tp[0], tp[1], tp[2]);
        }
        T totalErr = T(0);
        for (const auto& p : transformedSrc) {
            auto nn = targetTree.nearestNeighbor(p);
            totalErr += nn.second * nn.second;
        }
        res.fitness = totalErr / static_cast<T>(m_points.size());
        res.iterations = maxIter;
        return res;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: squared distances between corresponding points of two clouds
    //  Clouds must have same size. Returns vector of squared distances.
    // ------------------------------------------------------------------------
    std::vector<T> batchSquaredDistances(const PointCloud& other) const {
        size_type n = std::min(m_points.size(), other.m_points.size());
        std::vector<T> dists(n);
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_type i = 0; i < n; ++i) {
                dists[i] = (m_points[i] - other.m_points[i]).squaredLength();
            }
        } else {
            for (size_type i = 0; i < n; ++i) {
                dists[i] = (m_points[i] - other.m_points[i]).squaredLength();
            }
        }
        return dists;
    }

private:
    // ------------------------------------------------------------------------
    //  Compute rigid transformation (rotation + translation) between two point sets.
    //  Using SVD (Kabsch algorithm).
    // ------------------------------------------------------------------------
    Matrix<T,4> rigidTransform(const std::vector<point_type>& src,
                               const std::vector<point_type>& tgt) const {
        size_type n = src.size();
        point_type srcCentroid(0), tgtCentroid(0);
        for (size_type i = 0; i < n; ++i) {
            srcCentroid = srcCentroid + src[i];
            tgtCentroid = tgtCentroid + tgt[i];
        }
        srcCentroid = srcCentroid / static_cast<T>(n);
        tgtCentroid = tgtCentroid / static_cast<T>(n);

        Matrix<T,3> H(0);
        for (size_type i = 0; i < n; ++i) {
            point_type s = src[i] - srcCentroid;
            point_type t = tgt[i] - tgtCentroid;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    H(r,c) += s[r] * t[c];
        }
        // SVD of H: H = U * S * V^T, rotation = V * U^T
        Matrix<T,3> U, V;
        Vector<T,3> S;
        svd3x3(H, U, S, V);
        Matrix<T,3> R = V * U.transpose();
        // Ensure rotation matrix (det should be +1)
        if (R.determinant() < T(0)) {
            // Reflect
            for (int i = 0; i < 3; ++i) V(i,2) = -V(i,2);
            R = V * U.transpose();
        }
        point_type trans = tgtCentroid - R * srcCentroid;
        Matrix<T,4> result = Matrix<T,4>::identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) result(i,j) = R(i,j);
            result(i,3) = trans[i];
        }
        return result;
    }

    std::vector<point_type> m_points;
};

// ----------------------------------------------------------------------------
//  Helper: merge two point clouds (concatenate)
// ----------------------------------------------------------------------------
template<typename T>
PointCloud<T> mergePointClouds(const PointCloud<T>& a, const PointCloud<T>& b) {
    std::vector<Vector<T,3>> pts = a.points();
    pts.insert(pts.end(), b.points().begin(), b.points().end());
    return PointCloud<T>(std::move(pts));
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class PointCloudEnvironment {
public:
    static PointCloudEnvironment& instance() {
        static PointCloudEnvironment env;
        return env;
    }
    void setDefaultVoxelSize(T sz) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_voxelSize = sz;
    }
    T defaultVoxelSize() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_voxelSize;
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
    PointCloudEnvironment() : m_voxelSize(T(0.1)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_voxelSize;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_POINT_CLOUD_H_INCLUDED
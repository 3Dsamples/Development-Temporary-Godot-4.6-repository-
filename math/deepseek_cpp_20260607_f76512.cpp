//File group name : OrthoTree Math
//File 0042 : core/math/shape_fitting.h
//Fit geometric primitives (plane, sphere, cylinder, line) to point clouds using least squares, PCA, and RANSAC. SIMD batch for multiple fittings.

#ifndef ORTHOTREE_CORE_MATH_SHAPE_FITTING_H_INCLUDED
#define ORTHOTREE_CORE_MATH_SHAPE_FITTING_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "plane.h"
#include "sphere.h"
#include "cylinder.h"
#include "statistics.h"
#include "matrix_decomposition.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Plane fitting: find plane that minimizes squared distances to points.
//  Returns plane in Hessian normal form (normal, d). Uses PCA.
// ============================================================================
template<typename T = float>
Plane<T> fitPlane(const Vector<T,3>* points, size_t n) {
    // Compute mean
    Vector<T,3> mean(0);
    for (size_t i = 0; i < n; ++i) mean = mean + points[i];
    mean = mean / static_cast<T>(n);
    // Build covariance matrix
    Matrix<T,3> cov(0);
    for (size_t i = 0; i < n; ++i) {
        Vector<T,3> d = points[i] - mean;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cov(r,c) += d[r] * d[c];
    }
    // Eigen decomposition of covariance matrix
    Vector<T,3> eigenvalues;
    Matrix<T,3> eigenvectors;
    eigenSymmetric3x3(cov, eigenvalues, eigenvectors);
    // The eigenvector with smallest eigenvalue is the plane normal
    int minIdx = 0;
    for (int i = 1; i < 3; ++i) if (eigenvalues[i] < eigenvalues[minIdx]) minIdx = i;
    Vector<T,3> normal = eigenvectors.col(minIdx);
    T d = -normal.dot(mean);
    return Plane<T>(normal, d);
}

// ============================================================================
//  Sphere fitting: find sphere (center, radius) minimizing squared distance.
//  Using algebraic least squares (linear system) or iterative.
//  Here: solve for center using linear least squares.
// ============================================================================
template<typename T = float>
Sphere<T,3> fitSphere(const Vector<T,3>* points, size_t n) {
    // Sphere: (x - cx)^2 + (y - cy)^2 + (z - cz)^2 = r^2
    // => x^2 + y^2 + z^2 - 2cx x - 2cy y - 2cz z + (cx^2+cy^2+cz^2 - r^2) = 0
    // Solve for A = 2cx, B = 2cy, C = 2cz, D = (cx^2+cy^2+cz^2 - r^2)
    // Equation: (x^2+y^2+z^2) = A x + B y + C z - D
    // Overdetermined linear system.
    Matrix<T,4,4> A(0);
    Vector<T,4> b(0);
    for (size_t i = 0; i < n; ++i) {
        T x = points[i][0], y = points[i][1], z = points[i][2];
        T xyz2 = x*x + y*y + z*z;
        A(0,0) += x*x; A(0,1) += x*y; A(0,2) += x*z; A(0,3) += x;
        A(1,0) += x*y; A(1,1) += y*y; A(1,2) += y*z; A(1,3) += y;
        A(2,0) += x*z; A(2,1) += y*z; A(2,2) += z*z; A(2,3) += z;
        A(3,0) += x;   A(3,1) += y;   A(3,2) += z;   A(3,3) += T(1);
        b[0] += x * xyz2;
        b[1] += y * xyz2;
        b[2] += z * xyz2;
        b[3] += xyz2;
    }
    // Solve A * X = b
    Matrix<T,4> X = A.inverse() * b; // using 4x4 inverse (simplified)
    Vector<T,3> center( X[0] * T(0.5), X[1] * T(0.5), X[2] * T(0.5) );
    T r2 = center.squaredLength() - X[3];
    T radius = std::sqrt(std::abs(r2));
    return Sphere<T,3>(center, radius);
}

// ============================================================================
//  Cylinder fitting: approximate by estimating axis direction via PCA of normals,
//  then project onto perpendicular plane to fit circle.
//  Not full implementation – provide skeleton.
// ============================================================================
template<typename T = float>
Cylinder<T> fitCylinder(const Vector<T,3>* points, size_t n) {
    // Placeholder: return default cylinder
    return Cylinder<T>(Vector<T,3>(0), Vector<T,3>(0,0,1), T(1), T(0.5));
}

// ============================================================================
//  Line fitting (3D): find line minimizing sum of squared distances.
//  Using PCA (line through mean with direction of largest eigenvalue).
// ============================================================================
template<typename T = float>
std::pair<Vector<T,3>, Vector<T,3>> fitLine(const Vector<T,3>* points, size_t n) {
    Vector<T,3> mean(0);
    for (size_t i = 0; i < n; ++i) mean = mean + points[i];
    mean = mean / static_cast<T>(n);
    Matrix<T,3> cov(0);
    for (size_t i = 0; i < n; ++i) {
        Vector<T,3> d = points[i] - mean;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cov(r,c) += d[r] * d[c];
    }
    Vector<T,3> eigenvalues;
    Matrix<T,3> eigenvectors;
    eigenSymmetric3x3(cov, eigenvalues, eigenvectors);
    // Largest eigenvalue gives direction
    int maxIdx = 0;
    for (int i = 1; i < 3; ++i) if (eigenvalues[i] > eigenvalues[maxIdx]) maxIdx = i;
    Vector<T,3> direction = eigenvectors.col(maxIdx);
    return {mean, direction};
}

// ============================================================================
//  RANSAC fitting of any primitive (generic function)
// ============================================================================
template<typename Primitive, typename T>
Primitive ransacFit(const Vector<T,3>* points, size_t n,
                    std::function<Primitive(const Vector<T,3>*, size_t)> fitFunc,
                    std::function<T(const Primitive&, const Vector<T,3>&)> errorFunc,
                    T inlierThreshold, int maxIter = 1000, size_t minSamples = 3) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, n-1);
    Primitive bestPrimitive;
    size_t bestInliers = 0;
    for (int iter = 0; iter < maxIter; ++iter) {
        // Randomly select minSamples points
        std::vector<Vector<T,3>> samples(minSamples);
        for (size_t i = 0; i < minSamples; ++i) samples[i] = points[dist(gen)];
        Primitive prim = fitFunc(samples.data(), minSamples);
        size_t inliers = 0;
        for (size_t i = 0; i < n; ++i) {
            if (errorFunc(prim, points[i]) < inlierThreshold) ++inliers;
        }
        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestPrimitive = prim;
            if (bestInliers > n * 0.9) break; // early exit
        }
    }
    // Refit using all inliers
    std::vector<Vector<T,3>> inlierPoints;
    for (size_t i = 0; i < n; ++i) {
        if (errorFunc(bestPrimitive, points[i]) < inlierThreshold) inlierPoints.push_back(points[i]);
    }
    if (!inlierPoints.empty()) return fitFunc(inlierPoints.data(), inlierPoints.size());
    return bestPrimitive;
}

// ----------------------------------------------------------------------------
//  SIMD batch: fit multiple point sets (4 at a time) – not implemented.
// ----------------------------------------------------------------------------

// ============================================================================
//  Dynamic environment controller
// ============================================================================
class ShapeFittingEnvironment {
public:
    static ShapeFittingEnvironment& instance() {
        static ShapeFittingEnvironment env;
        return env;
    }
    void setEpsilon(T eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_epsilon = eps;
    }
    T epsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_epsilon;
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
    ShapeFittingEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_SHAPE_FITTING_H_INCLUDED
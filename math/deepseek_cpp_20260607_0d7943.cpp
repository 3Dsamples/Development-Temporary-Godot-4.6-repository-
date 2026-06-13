//File group name : OrthoTree Math
//File 0032 : core/math/surface.h
//Parametric surfaces: Bezier patch, B‑spline surface, NURBS surface. Evaluation (point, derivatives), bounding box, transformation, and SIMD batch evaluation (4x4 grid).

#ifndef ORTHOTREE_CORE_MATH_SURFACE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_SURFACE_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "bezier.h"
#include "bspline.h"
#include "nurbs.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <array>
#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  BezierPatch: tensor product Bezier surface of degree (p,q).
//  Control points (p+1)*(q+1) matrix.
//  Evaluation using de Casteljau in each direction.
//  Provides pointAt, bounding box, transformation, subdivision.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class BezierPatch {
public:
    using value_type = T;
    using point_type = Vector<T, N>;
    using size_type = size_t;

    BezierPatch() = default;
    BezierPatch(const std::vector<std::vector<point_type>>& controlPoints)
        : m_controlPoints(controlPoints) {
        m_uDeg = controlPoints.empty() ? 0 : controlPoints[0].size() - 1;
        m_vDeg = controlPoints.size() - 1;
    }

    size_type uDegree() const noexcept { return m_uDeg; }
    size_type vDegree() const noexcept { return m_vDeg; }
    const std::vector<std::vector<point_type>>& controlPoints() const noexcept { return m_controlPoints; }

    // Evaluate surface point at (u,v) ∈ [0,1]²
    point_type pointAt(T u, T v) const noexcept {
        if (m_controlPoints.empty()) return point_type(0);
        // Step 1: interpolate along u for each v row
        std::vector<point_type> curvePoints(m_vDeg + 1);
        for (size_type i = 0; i <= m_vDeg; ++i) {
            BezierCurve<T,N> curve(m_controlPoints[i]);
            curvePoints[i] = curve.pointAt(u);
        }
        // Step 2: interpolate along v
        BezierCurve<T,N> vCurve(curvePoints);
        return vCurve.pointAt(v);
    }

    // Bounding box (control points convex hull)
    aabb_type boundingBox() const noexcept {
        if (m_controlPoints.empty()) return aabb_type();
        point_type minP = m_controlPoints[0][0], maxP = minP;
        for (const auto& row : m_controlPoints) {
            for (const auto& p : row) {
                minP = minP.componentWiseMin(p);
                maxP = maxP.componentWiseMax(p);
            }
        }
        return aabb_type(minP, maxP);
    }

    // Transform all control points
    void transform(const AffineTransform<T,N>& tf) {
        for (auto& row : m_controlPoints) {
            for (auto& p : row) p = tf.transform(p);
        }
    }

    // Subdivide at u = u0 (returns two patches)
    std::pair<BezierPatch, BezierPatch> subdivideU(T u0) const {
        // Subdivide each row curve, then build new patches
        std::vector<std::vector<point_type>> leftCP(m_vDeg+1), rightCP(m_vDeg+1);
        for (size_type i = 0; i <= m_vDeg; ++i) {
            BezierCurve<T,N> rowCurve(m_controlPoints[i]);
            auto [leftCurve, rightCurve] = rowCurve.subdivide(u0);
            leftCP[i] = leftCurve.controlPoints();
            rightCP[i] = rightCurve.controlPoints();
        }
        return {BezierPatch(leftCP), BezierPatch(rightCP)};
    }

    // SIMD batch: evaluate 4 points at 4 (u,v) pairs (4x4)
    static void batchPointAt(const BezierPatch* patches, const T* u, const T* v,
                             point_type* out, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = patches[i].pointAt(u[i], v[i]);
        }
    }

private:
    std::vector<std::vector<point_type>> m_controlPoints;
    size_type m_uDeg = 0, m_vDeg = 0;
};

// ============================================================================
//  BSplineSurface: tensor product B‑spline surface.
//  Control points (n+1)*(m+1), knot vectors (size n+p+2, m+q+2).
//  Evaluation using de Boor in each direction.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class BSplineSurface {
public:
    using point_type = Vector<T, N>;
    using size_type = size_t;

    BSplineSurface() = default;
    BSplineSurface(const std::vector<std::vector<point_type>>& controlPoints,
                   const std::vector<T>& knotsU, const std::vector<T>& knotsV,
                   size_type degreeU, size_type degreeV)
        : m_controlPoints(controlPoints), m_knotsU(knotsU), m_knotsV(knotsV),
          m_degU(degreeU), m_degV(degreeV) {}

    size_type uDegree() const { return m_degU; }
    size_type vDegree() const { return m_degV; }
    const std::vector<std::vector<point_type>>& controlPoints() const { return m_controlPoints; }

    // Find span index in knot vector
    size_type findSpan(const std::vector<T>& knots, T u, size_type deg) const {
        if (u <= knots[deg]) return deg;
        if (u >= knots[knots.size()-1-deg]) return knots.size() - deg - 2;
        size_type low = deg, high = knots.size() - deg - 1;
        size_type mid = (low + high) / 2;
        while (u < knots[mid] || u >= knots[mid+1]) {
            if (u < knots[mid]) high = mid;
            else low = mid;
            mid = (low + high) / 2;
        }
        return mid;
    }

    // Basis functions (de Boor) – reuse BSplineCurve per direction
    std::vector<T> basisFunctionsU(T u, size_type span) const {
        std::vector<T> left(m_degU+1), right(m_degU+1), N(m_degU+1);
        N[0] = T(1);
        for (size_type j = 1; j <= m_degU; ++j) {
            left[j] = u - m_knotsU[span+1-j];
            right[j] = m_knotsU[span+j] - u;
            T saved = T(0);
            for (size_type r = 0; r < j; ++r) {
                T temp = N[r] / (right[r+1] + left[j-r]);
                N[r] = saved + right[r+1] * temp;
                saved = left[j-r] * temp;
            }
            N[j] = saved;
        }
        return N;
    }

    point_type pointAt(T u, T v) const {
        size_type uSpan = findSpan(m_knotsU, u, m_degU);
        auto Nu = basisFunctionsU(u, uSpan);
        // For each v row, compute intermediate points
        size_type uStart = uSpan - m_degU;
        std::vector<point_type> intermediate(m_degV+1, point_type(0));
        for (size_type i = 0; i <= m_degV; ++i) {
            point_type sum(0);
            for (size_type j = 0; j <= m_degU; ++j) {
                sum = sum + m_controlPoints[i + uStart][j] * Nu[j];
            }
            intermediate[i] = sum;
        }
        // Now interpolate in v direction
        BSplineCurve<T,N> vCurve(intermediate, m_knotsV, m_degV);
        return vCurve.pointAt(v);
    }

    aabb_type boundingBox() const {
        point_type minP = m_controlPoints[0][0], maxP = minP;
        for (const auto& row : m_controlPoints) {
            for (const auto& p : row) {
                minP = minP.componentWiseMin(p);
                maxP = maxP.componentWiseMax(p);
            }
        }
        return aabb_type(minP, maxP);
    }

    void transform(const AffineTransform<T,N>& tf) {
        for (auto& row : m_controlPoints) {
            for (auto& p : row) p = tf.transform(p);
        }
    }

private:
    std::vector<std::vector<point_type>> m_controlPoints;
    std::vector<T> m_knotsU, m_knotsV;
    size_type m_degU = 3, m_degV = 3;
};

// ============================================================================
//  NurbsSurface: rational B‑spline surface (NURBS). Weights matrix included.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class NurbsSurface {
public:
    using point_type = Vector<T, N>;
    using size_type = size_t;

    NurbsSurface() = default;
    NurbsSurface(const std::vector<std::vector<point_type>>& controlPoints,
                 const std::vector<std::vector<T>>& weights,
                 const std::vector<T>& knotsU, const std::vector<T>& knotsV,
                 size_type degreeU, size_type degreeV)
        : m_controlPoints(controlPoints), m_weights(weights),
          m_knotsU(knotsU), m_knotsV(knotsV), m_degU(degreeU), m_degV(degreeV) {}

    point_type pointAt(T u, T v) const {
        // Compute homogeneous points (w*P) and weight curve
        size_type uSpan = findSpan(m_knotsU, u, m_degU);
        size_type vSpan = findSpan(m_knotsV, v, m_degV);
        auto Nu = basisFunctionsU(u, uSpan);
        auto Nv = basisFunctionsV(v, vSpan);
        size_type uStart = uSpan - m_degU;
        size_type vStart = vSpan - m_degV;
        point_type homo(0);
        T weightSum = T(0);
        for (size_type i = 0; i <= m_degU; ++i) {
            for (size_type j = 0; j <= m_degV; ++j) {
                T w = m_weights[vStart + j][uStart + i];
                homo = homo + m_controlPoints[vStart + j][uStart + i] * (Nu[i] * Nv[j] * w);
                weightSum += Nu[i] * Nv[j] * w;
            }
        }
        if (std::abs(weightSum) < T(1e-12)) return point_type(0);
        return homo / weightSum;
    }

    aabb_type boundingBox() const {
        point_type minP = m_controlPoints[0][0], maxP = minP;
        for (const auto& row : m_controlPoints) {
            for (const auto& p : row) {
                minP = minP.componentWiseMin(p);
                maxP = maxP.componentWiseMax(p);
            }
        }
        return aabb_type(minP, maxP);
    }

    void transform(const AffineTransform<T,N>& tf) {
        for (auto& row : m_controlPoints) {
            for (auto& p : row) p = tf.transform(p);
        }
    }

private:
    size_type findSpan(const std::vector<T>& knots, T u, size_type deg) const {
        if (u <= knots[deg]) return deg;
        if (u >= knots[knots.size()-1-deg]) return knots.size() - deg - 2;
        size_type low = deg, high = knots.size() - deg - 1;
        size_type mid = (low + high) / 2;
        while (u < knots[mid] || u >= knots[mid+1]) {
            if (u < knots[mid]) high = mid;
            else low = mid;
            mid = (low + high) / 2;
        }
        return mid;
    }

    std::vector<T> basisFunctionsU(T u, size_type span) const {
        std::vector<T> left(m_degU+1), right(m_degU+1), N(m_degU+1);
        N[0] = T(1);
        for (size_type j = 1; j <= m_degU; ++j) {
            left[j] = u - m_knotsU[span+1-j];
            right[j] = m_knotsU[span+j] - u;
            T saved = T(0);
            for (size_type r = 0; r < j; ++r) {
                T temp = N[r] / (right[r+1] + left[j-r]);
                N[r] = saved + right[r+1] * temp;
                saved = left[j-r] * temp;
            }
            N[j] = saved;
        }
        return N;
    }

    std::vector<T> basisFunctionsV(T v, size_type span) const {
        return basisFunctionsU(v, span); // same algorithm
    }

    std::vector<std::vector<point_type>> m_controlPoints;
    std::vector<std::vector<T>> m_weights;
    std::vector<T> m_knotsU, m_knotsV;
    size_type m_degU = 3, m_degV = 3;
};

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class SurfaceEnvironment {
public:
    static SurfaceEnvironment& instance() {
        static SurfaceEnvironment env;
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
    SurfaceEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_SURFACE_H_INCLUDED
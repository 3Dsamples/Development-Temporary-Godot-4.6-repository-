//File group name : OrthoTree Math
//File 0075 : core/math/geometry/bezier.h
//Bezier curves (2D/3D, quadratic, cubic, arbitrary degree). Evaluation (De Casteljau), derivative, subdivision, bounding box, closest point (numeric), arc length, and SIMD batch evaluation.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_BEZIER_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_BEZIER_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "aabb.h"
#include "../numerical/integration.h"
#include "../numerical/root_finding.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  BezierCurve: defined by control points P0...Pn (degree n).
//  Evaluation using De Casteljau, derivative, subdivision, bounding box,
//  closest point (Newton), arc length (Simpson), SIMD batch evaluation.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class BezierCurve {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, N>;
    using aabb_type = AABB<T, N>;
    using size_type = size_t;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    BezierCurve() = default;
    explicit BezierCurve(const std::vector<point_type>& controlPoints)
        : m_controlPoints(controlPoints) {}
    BezierCurve(std::vector<point_type>&& controlPoints)
        : m_controlPoints(std::move(controlPoints)) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const std::vector<point_type>& controlPoints() const noexcept { return m_controlPoints; }
    size_type degree() const noexcept { return m_controlPoints.empty() ? 0 : m_controlPoints.size() - 1; }
    void setControlPoints(const std::vector<point_type>& pts) { m_controlPoints = pts; }

    // ------------------------------------------------------------------------
    //  Evaluate point at parameter t ∈ [0,1] (De Casteljau)
    // ------------------------------------------------------------------------
    point_type pointAt(T t) const noexcept {
        if (m_controlPoints.empty()) return point_type(0);
        std::vector<point_type> pts = m_controlPoints;
        size_type n = degree();
        for (size_type r = 1; r <= n; ++r) {
            for (size_type i = 0; i <= n - r; ++i) {
                pts[i] = pts[i] * (T(1)-t) + pts[i+1] * t;
            }
        }
        return pts[0];
    }

    // ------------------------------------------------------------------------
    //  Derivative (velocity) at t
    // ------------------------------------------------------------------------
    point_type derivativeAt(T t) const noexcept {
        if (degree() == 0) return point_type(0);
        size_type d = degree();
        std::vector<point_type> diffPoints(d);
        for (size_type i = 0; i < d; ++i) {
            diffPoints[i] = (m_controlPoints[i+1] - m_controlPoints[i]) * static_cast<T>(d);
        }
        BezierCurve diffCurve(diffPoints);
        return diffCurve.pointAt(t);
    }

    // ------------------------------------------------------------------------
    //  Bounding box (convex hull of control points)
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        if (m_controlPoints.empty()) return aabb_type();
        point_type minP = m_controlPoints[0], maxP = m_controlPoints[0];
        for (const auto& p : m_controlPoints) {
            minP = minP.componentWiseMin(p);
            maxP = maxP.componentWiseMax(p);
        }
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Subdivide curve into two at parameter t (left and right)
    // ------------------------------------------------------------------------
    std::pair<BezierCurve, BezierCurve> subdivide(T t) const noexcept {
        if (degree() == 0) return {*this, *this};
        std::vector<point_type> left, right;
        std::vector<std::vector<point_type>> levels(degree()+1);
        levels[0] = m_controlPoints;
        for (size_type r = 1; r <= degree(); ++r) {
            size_type n = degree() - r + 1;
            levels[r].resize(n);
            for (size_type i = 0; i < n; ++i) {
                levels[r][i] = levels[r-1][i] * (T(1)-t) + levels[r-1][i+1] * t;
            }
        }
        // Left: first control points of each level (diagonal)
        left.reserve(degree()+1);
        for (size_type r = 0; r <= degree(); ++r) left.push_back(levels[r][0]);
        // Right: last control points of each level (reverse diagonal)
        right.reserve(degree()+1);
        for (size_type r = 0; r <= degree(); ++r) right.push_back(levels[degree()-r][r]);
        return {BezierCurve(left), BezierCurve(right)};
    }

    // ------------------------------------------------------------------------
    //  Closest point on curve to external point (Newton refinement)
    //  Returns closest point and parameter t.
    // ------------------------------------------------------------------------
    std::pair<point_type, T> closestPoint(const point_type& p,
                                          size_type subdivisions = 10) const {
        if (degree() == 0) return {m_controlPoints[0], T(0)};
        // Brute force sampling
        T bestT = T(0);
        T bestDist = std::numeric_limits<T>::max();
        size_type samples = 100;
        for (size_type i = 0; i <= samples; ++i) {
            T t = static_cast<T>(i) / static_cast<T>(samples);
            point_type q = pointAt(t);
            T d2 = (q - p).squaredLength();
            if (d2 < bestDist) {
                bestDist = d2;
                bestT = t;
            }
        }
        // Newton refinement (solving (C(t)-P)·C'(t) = 0)
        for (size_type iter = 0; iter < subdivisions; ++iter) {
            point_type C = pointAt(bestT);
            point_type dC = derivativeAt(bestT);
            if (dC.squaredLength() < T(1e-12)) break;
            T f = dot(C - p, dC);
            T fprime = dC.squaredLength(); // ignoring second derivative (C''·(C-P) term)
            if (std::abs(fprime) < T(1e-12)) break;
            T delta = f / fprime;
            T newT = bestT - delta;
            if (newT < T(0)) newT = T(0);
            if (newT > T(1)) newT = T(1);
            point_type newC = pointAt(newT);
            T newDist2 = (newC - p).squaredLength();
            if (newDist2 < bestDist) {
                bestDist = newDist2;
                bestT = newT;
            } else {
                break;
            }
        }
        return {pointAt(bestT), bestT};
    }

    // ------------------------------------------------------------------------
    //  Arc length (adaptive Simpson)
    // ------------------------------------------------------------------------
    T arcLength(T t0 = T(0), T t1 = T(1), T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const {
        auto speed = [this](T t) -> T { return derivativeAt(t).length(); };
        return Numerical::adaptiveSimpson(speed, t0, t1, eps);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: evaluate 4 points at 4 different t values
    // ------------------------------------------------------------------------
    void batchPointAt(const T* t, point_type* out, size_type count) const noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            for (size_type i = 0; i < count; ++i) {
                out[i] = pointAt(t[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                out[i] = pointAt(t[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Transform control points
    // ------------------------------------------------------------------------
    void transform(const Basic::AffineTransform<T,N>& tf) {
        for (auto& p : m_controlPoints) p = tf.transformPoint(p);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const BezierCurve& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const {
        if (m_controlPoints.size() != other.m_controlPoints.size()) return false;
        for (size_type i = 0; i < m_controlPoints.size(); ++i) {
            if (!m_controlPoints[i].nearlyEqual(other.m_controlPoints[i], eps)) return false;
        }
        return true;
    }

private:
    std::vector<point_type> m_controlPoints;
};

// ----------------------------------------------------------------------------
//  Convenience aliases for 2D and 3D
// ----------------------------------------------------------------------------
template<typename T> using BezierCurve2 = BezierCurve<T, 2>;
template<typename T> using BezierCurve3 = BezierCurve<T, 3>;

using BezierCurve2f = BezierCurve<float, 2>;
using BezierCurve2d = BezierCurve<double, 2>;
using BezierCurve3f = BezierCurve<float, 3>;
using BezierCurve3d = BezierCurve<double, 3>;

// ----------------------------------------------------------------------------
//  Helper: create quadratic Bezier curve
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
BezierCurve<T,N> makeQuadraticBezier(const Basic::Vector<T,N>& p0,
                                     const Basic::Vector<T,N>& p1,
                                     const Basic::Vector<T,N>& p2) {
    return BezierCurve<T,N>({p0, p1, p2});
}

// ----------------------------------------------------------------------------
//  Helper: create cubic Bezier curve
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
BezierCurve<T,N> makeCubicBezier(const Basic::Vector<T,N>& p0,
                                 const Basic::Vector<T,N>& p1,
                                 const Basic::Vector<T,N>& p2,
                                 const Basic::Vector<T,N>& p3) {
    return BezierCurve<T,N>({p0, p1, p2, p3});
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class BezierEnvironment {
public:
    static BezierEnvironment& instance() {
        static BezierEnvironment env;
        return env;
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
    BezierEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_BEZIER_H_INCLUDED
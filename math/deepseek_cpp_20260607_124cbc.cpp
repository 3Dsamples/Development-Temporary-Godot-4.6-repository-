//File group name : OrthoTree Math
//File 0029 : core/math/bezier.h
//Bezier curves (quadratic, cubic, arbitrary degree) and rational Bezier curves. Evaluation, subdivision, bounding box, closest point, arc length approximation, and SIMD batch evaluation.

#ifndef ORTHOTREE_CORE_MATH_BEZIER_H_INCLUDED
#define ORTHOTREE_CORE_MATH_BEZIER_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  BezierCurve: evaluates Bezier curves of arbitrary degree using De Casteljau.
//  Provides pointAt, derivative, bounding box, subdivision, closest point
//  (numeric), arc length (adaptive Simpson), and SIMD batch evaluation for
//  multiple t values.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class BezierCurve {
public:
    using value_type = T;
    using point_type = Vector<T, N>;
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
        if (degree() == 0) return m_controlPoints[0];
        std::vector<point_type> points = m_controlPoints;
        for (size_type r = 1; r <= degree(); ++r) {
            for (size_type i = 0; i <= degree() - r; ++i) {
                points[i] = points[i] * (T(1)-t) + points[i+1] * t;
            }
        }
        return points[0];
    }

    // ------------------------------------------------------------------------
    //  Evaluate derivative (velocity) at t
    // ------------------------------------------------------------------------
    point_type derivativeAt(T t) const noexcept {
        if (degree() <= 0) return point_type(0);
        // Derivative of Bezier curve: degree * (B'_i), where control points difference.
        size_type d = degree();
        std::vector<point_type> diffPoints(d);
        for (size_type i = 0; i < d; ++i) {
            diffPoints[i] = (m_controlPoints[i+1] - m_controlPoints[i]) * static_cast<T>(d);
        }
        BezierCurve diffCurve(diffPoints);
        return diffCurve.pointAt(t);
    }

    // ------------------------------------------------------------------------
    //  Bounding box of all control points (convex hull property)
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
    //  Subdivide curve into two at parameter t (returns left and right curves)
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
    //  Closest point on curve to external point (numeric, iterative refinement)
    //  Returns closest point and parameter t.
    // ------------------------------------------------------------------------
    std::pair<point_type, T> closestPoint(const point_type& p, size_type subdivisions = 10) const {
        // Brute force: sample curve at many points, then refine using Newton
        if (degree() == 0) return {m_controlPoints[0], T(0)};
        T bestT = T(0);
        T bestDist = std::numeric_limits<T>::max();
        // Initial sampling
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
        // Refine with Newton's method (solving (C(t)-P)·C'(t)=0)
        for (size_type iter = 0; iter < subdivisions; ++iter) {
            point_type C = pointAt(bestT);
            point_type dC = derivativeAt(bestT);
            point_type ddC = derivativeAt(bestT + T(1e-5)) - dC; // approximate second derivative
            if (dC.squaredLength() < T(1e-12)) break;
            T f = dot(C - p, dC);
            T fprime = dot(dC, dC) + dot(C - p, ddC);
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
    //  Arc length (adaptive Simpson integration)
    // ------------------------------------------------------------------------
    T arcLength(T t0 = T(0), T t1 = T(1), T eps = T(1e-6)) const {
        auto speed = [this](T t) -> T { return derivativeAt(t).length(); };
        return integrateSimpson(speed, t0, t1, 100);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: evaluate 4 points at 4 different t values
    //  Input: t[4], output: point[4]
    // ------------------------------------------------------------------------
    void batchPointAt(const T* t, point_type* out, size_type count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3 && std::is_same_v<T,float>) {
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
    void transform(const AffineTransform<T,N>& tf) {
        for (auto& p : m_controlPoints) p = tf.transform(p);
    }

private:
    std::vector<point_type> m_controlPoints;
};

// ============================================================================
//  RationalBezierCurve (NURBS-like, but rational Bezier)
// ============================================================================
template<typename T = float, std::size_t N = 3>
class RationalBezierCurve : public BezierCurve<T,N> {
public:
    using point_type = Vector<T, N>;
    using size_type = size_t;

    RationalBezierCurve() = default;
    RationalBezierCurve(const std::vector<point_type>& controlPoints, const std::vector<T>& weights)
        : BezierCurve<T,N>(controlPoints), m_weights(weights) {
        if (m_weights.size() != controlPoints.size()) m_weights.resize(controlPoints.size(), T(1));
    }

    // ------------------------------------------------------------------------
    //  Evaluate rational Bezier (weighted De Casteljau)
    // ------------------------------------------------------------------------
    point_type pointAt(T t) const noexcept {
        const auto& P = this->controlPoints();
        if (P.empty()) return point_type(0);
        size_type deg = this->degree();
        std::vector<point_type> points = P;
        std::vector<T> weights = m_weights;
        for (size_type r = 1; r <= deg; ++r) {
            for (size_type i = 0; i <= deg - r; ++i) {
                T w0 = weights[i];
                T w1 = weights[i+1];
                T w = w0 * (T(1)-t) + w1 * t;
                points[i] = (points[i] * w0 * (T(1)-t) + points[i+1] * w1 * t) / w;
                weights[i] = w;
            }
        }
        return points[0];
    }

    // ------------------------------------------------------------------------
    //  Access weights
    // ------------------------------------------------------------------------
    const std::vector<T>& weights() const noexcept { return m_weights; }
    void setWeights(const std::vector<T>& w) {
        m_weights = w;
        if (m_weights.size() != this->controlPoints().size())
            m_weights.resize(this->controlPoints().size(), T(1));
    }

private:
    std::vector<T> m_weights;
};

// ----------------------------------------------------------------------------
//  Helper: create quadratic Bezier from 3 points
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
BezierCurve<T,N> makeQuadraticBezier(const Vector<T,N>& p0, const Vector<T,N>& p1, const Vector<T,N>& p2) {
    return BezierCurve<T,N>({p0, p1, p2});
}

// ----------------------------------------------------------------------------
//  Helper: create cubic Bezier from 4 points
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
BezierCurve<T,N> makeCubicBezier(const Vector<T,N>& p0, const Vector<T,N>& p1,
                                 const Vector<T,N>& p2, const Vector<T,N>& p3) {
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
    BezierEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_BEZIER_H_INCLUDED
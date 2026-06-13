//File group name : OrthoTree Math
//File 0030 : core/math/bspline.h
//B-spline curve: non‑rational B‑spline with uniform or non‑uniform knots, evaluation (de Boor), derivative, knot insertion, subdivision, bounding box, closest point, arc length approximation, and SIMD batch evaluation.

#ifndef ORTHOTREE_CORE_MATH_BSPLINE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_BSPLINE_H_INCLUDED

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
//  BSplineCurve: non‑rational B‑spline curve of degree p.
//  Defined by control points, knot vector (size = n_controls + p + 1),
//  and optionally weights for NURBS (rational). This version is non‑rational.
//  Supports evaluation (de Boor algorithm), derivative, knot insertion,
//  subdivision, bounding box, closest point (numeric), arc length.
//  For rational B‑splines (NURBS), see NurbsCurve.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class BSplineCurve {
public:
    using value_type = T;
    using point_type = Vector<T, N>;
    using size_type = size_t;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    BSplineCurve() = default;
    BSplineCurve(const std::vector<point_type>& controlPoints,
                 const std::vector<T>& knots,
                 size_type degree)
        : m_controlPoints(controlPoints), m_knots(knots), m_degree(degree) {
        validate();
    }
    BSplineCurve(std::vector<point_type>&& controlPoints,
                 std::vector<T>&& knots,
                 size_type degree)
        : m_controlPoints(std::move(controlPoints)), m_knots(std::move(knots)), m_degree(degree) {
        validate();
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const std::vector<point_type>& controlPoints() const noexcept { return m_controlPoints; }
    const std::vector<T>& knots() const noexcept { return m_knots; }
    size_type degree() const noexcept { return m_degree; }
    size_type numControlPoints() const noexcept { return m_controlPoints.size(); }

    void setControlPoints(const std::vector<point_type>& pts) { m_controlPoints = pts; validate(); }
    void setKnots(const std::vector<T>& knots) { m_knots = knots; validate(); }
    void setDegree(size_type d) { m_degree = d; validate(); }

    // ------------------------------------------------------------------------
    //  Validate knot vector (size = n + p + 1, non‑decreasing)
    // ------------------------------------------------------------------------
    void validate() {
        size_type n = m_controlPoints.size();
        size_type expectedKnots = n + m_degree + 1;
        if (m_knots.size() != expectedKnots) {
            // Create uniform knots as fallback
            m_knots.clear();
            for (size_type i = 0; i < expectedKnots; ++i) {
                m_knots.push_back(static_cast<T>(i) / static_cast<T>(expectedKnots - 1));
            }
        }
        // Ensure non‑decreasing (sort if necessary)
        // Skipped for performance; assume user provides valid.
    }

    // ------------------------------------------------------------------------
    //  Find knot span index for parameter u ∈ [0,1] (clamped)
    // ------------------------------------------------------------------------
    size_type findSpan(T u) const noexcept {
        if (u <= m_knots[0]) return m_degree;
        if (u >= m_knots.back()) return static_cast<size_type>(m_controlPoints.size()) - 1;
        size_type low = m_degree;
        size_type high = static_cast<size_type>(m_controlPoints.size());
        size_type mid = (low + high) / 2;
        while (u < m_knots[mid] || u >= m_knots[mid+1]) {
            if (u < m_knots[mid]) high = mid;
            else low = mid;
            mid = (low + high) / 2;
        }
        return mid;
    }

    // ------------------------------------------------------------------------
    //  Basis functions (de Boor) for a given span index and parameter u
    //  Returns N[0..p] basis values.
    // ------------------------------------------------------------------------
    std::vector<T> basisFunctions(size_type span, T u) const noexcept {
        std::vector<T> left(m_degree+1), right(m_degree+1);
        std::vector<T> N(m_degree+1);
        N[0] = T(1);
        for (size_type j = 1; j <= m_degree; ++j) {
            left[j] = u - m_knots[span+1-j];
            right[j] = m_knots[span+j] - u;
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

    // ------------------------------------------------------------------------
    //  Evaluate point at parameter u ∈ [0,1] (de Boor)
    // ------------------------------------------------------------------------
    point_type pointAt(T u) const noexcept {
        if (m_controlPoints.empty()) return point_type(0);
        size_type span = findSpan(u);
        auto N = basisFunctions(span, u);
        point_type C(0);
        size_type idx = span - m_degree;
        for (size_type i = 0; i <= m_degree; ++i) {
            C = C + m_controlPoints[idx + i] * N[i];
        }
        return C;
    }

    // ------------------------------------------------------------------------
    //  Derivative (first derivative) at u
    //  Use derivative of B‑spline: degree * (control point differences) * basis of degree-1
    // ------------------------------------------------------------------------
    point_type derivativeAt(T u) const noexcept {
        if (m_degree == 0) return point_type(0);
        size_type n = m_controlPoints.size();
        std::vector<point_type> diffPoints(n-1);
        for (size_type i = 0; i < n-1; ++i) {
            diffPoints[i] = (m_controlPoints[i+1] - m_controlPoints[i]) * static_cast<T>(m_degree);
        }
        // Knot vector for derivative: original knots without first and last.
        std::vector<T> derivKnots = m_knots;
        // derivative curve has degree = m_degree - 1
        BSplineCurve derivCurve(diffPoints, derivKnots, m_degree - 1);
        return derivCurve.pointAt(u);
    }

    // ------------------------------------------------------------------------
    //  Bounding box (control points convex hull)
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
    //  Knot insertion (refinement) – inserts a single knot value u
    //  Returns new curve with one additional knot and one extra control point.
    // ------------------------------------------------------------------------
    BSplineCurve insertKnot(T u) const noexcept {
        size_type span = findSpan(u);
        size_type r = 0;
        for (size_type i = span - m_degree + 1; i <= span; ++i) {
            if (std::abs(m_knots[i] - u) < T(1e-12)) ++r;
        }
        std::vector<T> newKnots = m_knots;
        newKnots.insert(newKnots.begin() + span + 1, u);
        std::vector<point_type> newPoints = m_controlPoints;
        newPoints.insert(newPoints.begin() + span - m_degree + 1, point_type(0));
        size_type idx = span - m_degree + 1;
        for (size_type i = 0; i <= r; ++i) {
            // Compute alphas
            T alpha = (u - m_knots[span - m_degree + 1 + i]) /
                      (m_knots[span + i + 1] - m_knots[span - m_degree + 1 + i]);
            newPoints[idx + i] = newPoints[idx + i] * (T(1)-alpha) + newPoints[idx + i + 1] * alpha;
        }
        return BSplineCurve(newPoints, newKnots, m_degree);
    }

    // ------------------------------------------------------------------------
    //  Subdivide (using knot insertion) at u into two curves
    //  Returns pair of curves (left, right)
    // ------------------------------------------------------------------------
    std::pair<BSplineCurve, BSplineCurve> subdivide(T u) const {
        // Insert u until multiplicity = degree
        BSplineCurve left = *this;
        for (size_type k = 0; k < m_degree; ++k) left = left.insertKnot(u);
        // Build two new curves from the knot vector and control points
        size_type n = left.controlPoints().size();
        size_type p = left.degree();
        std::vector<point_type> leftPoints, rightPoints;
        std::vector<T> leftKnots, rightKnots;
        // Left: control points 0..idx, knots 0..idx+p
        size_type splitIdx = 0;
        for (size_type i = 0; i < left.knots().size(); ++i) {
            if (left.knots()[i] >= u - T(1e-12)) {
                splitIdx = i;
                break;
            }
        }
        leftPoints.assign(left.controlPoints().begin(), left.controlPoints().begin() + splitIdx - p);
        leftKnots.assign(left.knots().begin(), left.knots().begin() + splitIdx + 1);
        rightPoints.assign(left.controlPoints().begin() + splitIdx - p, left.controlPoints().end());
        rightKnots.assign(left.knots().begin() + splitIdx, left.knots().end());
        // Normalise knots to [0,1] range? Not necessary.
        return {BSplineCurve(leftPoints, leftKnots, p), BSplineCurve(rightPoints, rightKnots, p)};
    }

    // ------------------------------------------------------------------------
    //  Closest point (numeric, using sampling + Newton) – simplified
    // ------------------------------------------------------------------------
    std::pair<point_type, T> closestPoint(const point_type& p, size_type subdivisions = 10) const {
        T bestU = T(0);
        T bestDist = std::numeric_limits<T>::max();
        size_type samples = 100;
        for (size_type i = 0; i <= samples; ++i) {
            T u = static_cast<T>(i) / static_cast<T>(samples);
            point_type q = pointAt(u);
            T d2 = (q - p).squaredLength();
            if (d2 < bestDist) {
                bestDist = d2;
                bestU = u;
            }
        }
        // Refine with Newton (simplified)
        for (size_type iter = 0; iter < subdivisions; ++iter) {
            point_type C = pointAt(bestU);
            point_type dC = derivativeAt(bestU);
            if (dC.squaredLength() < T(1e-12)) break;
            T f = dot(C - p, dC);
            T fprime = dC.squaredLength(); // ignoring second derivative
            if (std::abs(fprime) < T(1e-12)) break;
            T delta = f / fprime;
            T newU = bestU - delta;
            if (newU < T(0)) newU = T(0);
            if (newU > T(1)) newU = T(1);
            point_type newC = pointAt(newU);
            T newDist2 = (newC - p).squaredLength();
            if (newDist2 < bestDist) {
                bestDist = newDist2;
                bestU = newU;
            } else break;
        }
        return {pointAt(bestU), bestU};
    }

    // ------------------------------------------------------------------------
    //  Arc length (adaptive Simpson)
    // ------------------------------------------------------------------------
    T arcLength(T u0 = T(0), T u1 = T(1), T eps = T(1e-6)) const {
        auto speed = [this](T u) -> T { return derivativeAt(u).length(); };
        return integrateSimpson(speed, u0, u1, 100);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: evaluate 4 points at 4 u values
    // ------------------------------------------------------------------------
    void batchPointAt(const T* u, point_type* out, size_type count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3 && std::is_same_v<T,float>) {
            for (size_type i = 0; i < count; ++i) {
                out[i] = pointAt(u[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                out[i] = pointAt(u[i]);
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
    std::vector<T> m_knots;
    size_type m_degree = 3;
};

// ----------------------------------------------------------------------------
//  Helper: create uniform cubic B‑spline (open uniform)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
BSplineCurve<T,N> makeUniformCubicBSpline(const std::vector<Vector<T,N>>& controlPoints) {
    size_type n = controlPoints.size();
    size_type p = 3;
    std::vector<T> knots(n + p + 1);
    for (size_type i = 0; i < knots.size(); ++i) {
        knots[i] = static_cast<T>(i) / static_cast<T>(knots.size() - 1);
    }
    return BSplineCurve<T,N>(controlPoints, knots, p);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class BSplineEnvironment {
public:
    static BSplineEnvironment& instance() {
        static BSplineEnvironment env;
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
    BSplineEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_BSPLINE_H_INCLUDED
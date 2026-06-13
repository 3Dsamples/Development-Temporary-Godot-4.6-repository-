// File 0021 : core/math/spline.h
// Parametric spline curves (Bezier, Catmull‑Rom, B‑Spline, Hermite) for 1D/2D/3D data with tangent and arc‑length estimation.

#pragma once

#include "vec2.h"
#include "vec3.h"
#include "interpolation.h"    // cubic_hermite, catmull_rom, bezier_cubic
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wp {

// ======================== Bezier Curve ===============================
template <typename T, typename Point>
class BezierCurve {
public:
    std::vector<Point> control_points;   // degree n (n+1 points)

    BezierCurve() = default;
    BezierCurve(const std::vector<Point>& pts) : control_points(pts) {}

    Point evaluate(T t) const noexcept {
        if (control_points.empty()) return Point{};
        // De Casteljau algorithm
        std::vector<Point> tmp = control_points;
        while (tmp.size() > 1) {
            for (size_t i = 0; i < tmp.size() - 1; ++i) {
                tmp[i] = lerp(tmp[i], tmp[i + 1], t);
            }
            tmp.pop_back();
        }
        return tmp[0];
    }

    Point tangent(T t) const noexcept {
        if (control_points.size() < 2) return Point{};
        std::vector<Point> dpts(control_points.size() - 1);
        for (size_t i = 0; i < dpts.size(); ++i)
            dpts[i] = (control_points[i + 1] - control_points[i]) * static_cast<T>(dpts.size());
        BezierCurve<T, Point> deriv(dpts);
        return deriv.evaluate(t);
    }
};

// ==================== Catmull‑Rom Spline =============================
enum class CatmullRomType { Uniform, Centripetal, Chordal };

template <typename T, typename Point>
class CatmullRomSpline {
public:
    std::vector<Point> points;
    CatmullRomType type = CatmullRomType::Centripetal;

    CatmullRomSpline() = default;
    CatmullRomSpline(const std::vector<Point>& pts, CatmullRomType t = CatmullRomType::Centripetal)
        : points(pts), type(t) {}

    // Evaluate at parameter t in [0, 1] over the whole curve
    Point evaluate(T t) const noexcept {
        if (points.size() < 2) return points.empty() ? Point{} : points.front();
        // Map t to segment index and local t
        T global_length = total_parameter_length();
        T dist = t * global_length;
        size_t seg = find_segment(dist);
        T local_t = local_parameter(seg, dist);
        return evaluate_segment(seg, local_t);
    }

    Point tangent(T t) const noexcept {
        if (points.size() < 2) return Point{};
        T global_length = total_parameter_length();
        T dist = t * global_length;
        size_t seg = find_segment(dist);
        T local_t = local_parameter(seg, dist);
        return tangent_segment(seg, local_t);
    }

private:
    // Segment parameter lengths (accumulated)
    mutable std::vector<T> param_lengths;
    mutable bool lengths_dirty = true;

    T param(const Point& a, const Point& b) const noexcept {
        switch (type) {
            case CatmullRomType::Uniform:     return T(1);
            case CatmullRomType::Centripetal: return std::sqrt(length(a - b));
            case CatmullRomType::Chordal:     return length(a - b);
            default: return T(1);
        }
    }

    void update_lengths() const noexcept {
        if (!lengths_dirty) return;
        param_lengths.resize(points.size() - 1);
        for (size_t i = 0; i + 1 < points.size(); ++i)
            param_lengths[i] = param(points[i], points[i + 1]);
        // accumulate
        for (size_t i = 1; i < param_lengths.size(); ++i)
            param_lengths[i] += param_lengths[i - 1];
        lengths_dirty = false;
    }

    T total_parameter_length() const noexcept {
        if (points.size() < 2) return T(0);
        update_lengths();
        return param_lengths.back();
    }

    size_t find_segment(T& dist) const noexcept {
        update_lengths();
        for (size_t i = 0; i < param_lengths.size(); ++i) {
            if (dist <= param_lengths[i]) {
                // adjust dist to be within segment i
                T prev = (i > 0) ? param_lengths[i - 1] : T(0);
                dist -= prev;
                return i;
            }
        }
        // last segment
        size_t last = param_lengths.size() - 1;
        dist = param_lengths[last] - (last > 0 ? param_lengths[last - 1] : T(0));
        return last;
    }

    T local_parameter(size_t seg, T dist) const noexcept {
        T seg_len = (seg == 0) ? param_lengths[0] : (param_lengths[seg] - param_lengths[seg - 1]);
        if (seg_len < MathConst<T>::epsilon) return T(0);
        return dist / seg_len;
    }

    Point evaluate_segment(size_t seg, T t) const noexcept {
        if (seg >= points.size() - 1) return points.back();
        // p0 = points[seg-1], p1 = points[seg], p2 = points[seg+1], p3 = points[seg+2]
        Point p0 = (seg > 0) ? points[seg - 1] : points[seg] - (points[seg + 1] - points[seg]);
        Point p1 = points[seg];
        Point p2 = points[seg + 1];
        Point p3 = (seg + 2 < points.size()) ? points[seg + 2] : points[seg + 1] + (points[seg + 1] - points[seg]);
        // Use the generic catmull_rom function from interpolation (which works on scalars)
        // For Point types, we must call component‑wise. We'll use the cubic_hermite formulation.
        // Actually, the catmull_rom template in interpolation works for any type with arithmetic ops.
        return catmull_rom(p0, p1, p2, p3, t); // from interpolation.h
    }

    Point tangent_segment(size_t seg, T t) const noexcept {
        if (seg >= points.size() - 1) return Point{};
        // Derivative of cubic Hermite: (6t^2-6t)*p0 + (-6t^2+6t)*p1? Actually catmull_rom derivative:
        // We'll compute finite difference or use derivative formula.
        const T dt = T(0.001);
        Point p0 = evaluate_segment(seg, t - dt);
        Point p1 = evaluate_segment(seg, t + dt);
        return (p1 - p0) * (T(0.5) / dt);
    }
};

// ====================== B‑Spline ====================================
template <typename T, typename Point>
class BSplineCurve {
public:
    std::vector<Point> control_points;
    int degree = 3;          // cubic by default
    std::vector<T> knots;    // clamped by default

    BSplineCurve() = default;
    BSplineCurve(const std::vector<Point>& pts, int deg = 3) : control_points(pts), degree(deg) {
        if (control_points.size() < size_t(degree + 1))
            throw std::invalid_argument("Not enough control points for degree");
        generate_uniform_knots();
    }

    void generate_uniform_knots() noexcept {
        int n = static_cast<int>(control_points.size()) - 1;
        int m = n + degree + 1;
        knots.resize(m + 1);
        // Clamped uniform: first degree+1 knots = 0, last degree+1 knots = 1
        for (int i = 0; i <= m; ++i) {
            if (i <= degree) knots[i] = T(0);
            else if (i >= m - degree) knots[i] = T(1);
            else knots[i] = T(i - degree) / T(n - degree + 1);
        }
    }

    Point evaluate(T t) const noexcept {
        if (control_points.empty()) return Point{};
        int span = find_span(t);
        std::vector<Point> temp(control_points.begin() + span - degree, control_points.begin() + span + 1);
        // de Boor algorithm
        for (int r = 1; r <= degree; ++r) {
            for (int i = degree; i >= r; --i) {
                T alpha = (t - knots[span - degree + i]) / (knots[i + span - r + 1] - knots[span - degree + i]);
                temp[i] = temp[i - 1] * (T(1) - alpha) + temp[i] * alpha;
            }
        }
        return temp[degree];
    }

private:
    int find_span(T t) const noexcept {
        int n = static_cast<int>(control_points.size()) - 1;
        if (t >= T(1)) return n;
        if (t <= T(0)) return degree;
        for (int i = degree; i <= n; ++i)
            if (t >= knots[i] && t < knots[i + 1]) return i;
        return degree;
    }
};

// ==================== Hermite Spline (per segment) ====================
template <typename T, typename Point>
class HermiteSpline {
public:
    struct Key {
        T       t;
        Point   value;
        Point   tangent;
    };
    std::vector<Key> keys;

    HermiteSpline() = default;
    HermiteSpline(const std::vector<Key>& k) : keys(k) { std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) { return a.t < b.t; }); }

    Point evaluate(T t) const noexcept {
        if (keys.empty()) return Point{};
        if (t <= keys.front().t) return keys.front().value;
        if (t >= keys.back().t) return keys.back().value;
        auto it = std::lower_bound(keys.begin(), keys.end(), t, [](const Key& k, T val) { return k.t < val; });
        size_t i = std::distance(keys.begin(), it);
        const Key& k0 = keys[i - 1];
        const Key& k1 = keys[i];
        T span = k1.t - k0.t;
        if (span < MathConst<T>::epsilon) return k0.value;
        T u = (t - k0.t) / span;
        // Use cubic Hermite interpolation (from interpolation.h)
        return cubic_hermite(k0.value, k1.value, k0.tangent * span, k1.tangent * span, u);
    }
};

// ================== Arc‑length helpers ===============================
template <typename T, typename Point>
T approximate_arc_length(const CatmullRomSpline<T, Point>& spline, int samples = 64) {
    T length = T(0);
    Point prev = spline.evaluate(T(0));
    for (int i = 1; i <= samples; ++i) {
        T t = T(i) / T(samples);
        Point curr = spline.evaluate(t);
        length += distance(curr, prev);
        prev = curr;
    }
    return length;
}

template <typename T, typename Point>
T arc_length_parameter(const CatmullRomSpline<T, Point>& spline, T s, int samples = 64) {
    T total = approximate_arc_length(spline, samples);
    if (total < MathConst<T>::epsilon) return T(0);
    T target = s / total;
    // Binary search for t given target distance ratio (simplified)
    T accum = 0;
    Point prev = spline.evaluate(T(0));
    for (int i = 1; i <= samples; ++i) {
        T t = T(i) / T(samples);
        Point curr = spline.evaluate(t);
        accum += distance(curr, prev);
        if (accum / total >= target) return t;
        prev = curr;
    }
    return T(1);
}

} // namespace wp
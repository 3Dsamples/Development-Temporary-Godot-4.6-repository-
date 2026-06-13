// system name : onetbb-warp
// File 0011 : core/math/interpolation.h
// Description : Splines, curves, and interpolation functions for animation and motion control.

#ifndef __TBB_WARP_CORE_MATH_INTERPOLATION_H
#define __TBB_WARP_CORE_MATH_INTERPOLATION_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/quaternion.h"
#include <cmath>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Linear interpolation (generic)
// ============================================================

template<typename T, typename U>
constexpr T lerp(const T& a, const T& b, U t) noexcept {
    return a + (b - a) * static_cast<T>(t);
}

// ============================================================
// Cubic Hermite spline segment
// ============================================================

template<typename T>
T cubic_hermite(const T& p0, const T& m0, const T& p1, const T& m1, float t) noexcept {
    float t2 = t * t;
    float t3 = t2 * t;
    float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 = t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 = t3 - t2;
    return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;
}

// ============================================================
// Catmull-Rom spline interpolation (single segment)
// ============================================================

enum class spline_parameterization { uniform, chordal, centripetal };

template<typename T>
T catmull_rom_segment(const T& p0, const T& p1, const T& p2, const T& p3,
                      float t, float alpha = 0.5f) noexcept {
    float t2 = t * t;
    float t3 = t2 * t;
    return static_cast<float>(0.5) * (
        (static_cast<float>(2) * p1) +
        (p2 - p0) * t +
        (static_cast<float>(2) * p0 - static_cast<float>(5) * p1 +
         static_cast<float>(4) * p2 - p3) * t2 +
        (-p0 + static_cast<float>(3) * p1 - static_cast<float>(3) * p2 + p3) * t3
    );
}

template<typename T>
float catmull_rom_alpha(const T& a, const T& b, spline_parameterization param) noexcept {
    float d = distance(a, b);
    if (param == spline_parameterization::uniform) return 0.0f;
    if (param == spline_parameterization::chordal) return d;
    return std::sqrt(d);
}

template<typename T>
T catmull_rom_interpolate(const std::vector<T>& points, float t,
                          spline_parameterization param = spline_parameterization::centripetal,
                          bool loop = false) noexcept {
    if (points.size() < 2) return points.empty() ? T() : points[0];
    if (points.size() == 2) return lerp(points[0], points[1], t);
    std::size_t n = points.size();
    std::vector<float> times(n);
    times[0] = 0.0f;
    for (std::size_t i = 1; i < n; ++i)
        times[i] = times[i-1] + catmull_rom_alpha(points[i-1], points[i], param);
    float total = times.back();
    if (total < 1e-9f) return points[0];
    float u = t * total;
    if (loop) {
        float loop_total = total + catmull_rom_alpha(points.back(), points.front(), param);
        u = t * loop_total;
        if (u > total) u -= total;
    }
    std::size_t seg = 0;
    while (seg + 1 < n && times[seg + 1] < u) ++seg;
    float seg_len = times[seg+1] - times[seg];
    float seg_t = (seg_len > 1e-9f) ? ((u - times[seg]) / seg_len) : 0.0f;
    T p0 = (seg == 0) ? (loop ? points.back() : points[0]) : points[seg-1];
    T p1 = points[seg];
    T p2 = points[std::min(seg+1, n-1)];
    T p3 = (seg+2 < n) ? points[seg+2] : (loop ? points[(seg+2)%n] : points.back());
    return catmull_rom_segment(p0, p1, p2, p3, seg_t);
}

// ============================================================
// Kochanek‑Bartels spline (TCB parameters)
// ============================================================

template<typename T>
T kochanek_bartels_segment(const T& pm1, const T& p0, const T& p1, const T& p2,
                           float t, float tension, float continuity, float bias) noexcept {
    float t2 = t * t;
    float t3 = t2 * t;
    float dt0 = (1.0f - tension) * (1.0f + continuity) * (1.0f + bias) * 0.5f * (p1 - pm1);
    float dt1 = (1.0f - tension) * (1.0f - continuity) * (1.0f - bias) * 0.5f * (p2 - p0);
    float dt0a = (1.0f - tension) * (1.0f - continuity) * (1.0f + bias) * 0.5f * (p1 - pm1);
    float dt1a = (1.0f - tension) * (1.0f + continuity) * (1.0f - bias) * 0.5f * (p2 - p0);
    T start_tangent = p0 * 0.0f + (dt0 + dt0a) * 0.5f;
    T end_tangent = p2 * 0.0f + (dt1 + dt1a) * 0.5f;
    float h00 = 2.0f*t3 - 3.0f*t2 + 1.0f;
    float h10 = t3 - 2.0f*t2 + t;
    float h01 = -2.0f*t3 + 3.0f*t2;
    float h11 = t3 - t2;
    return p0 * h00 + start_tangent * h10 + p1 * h01 + end_tangent * h11;
}

template<typename T>
T kochanek_bartels_interpolate(const std::vector<T>& points, float t,
                               float tension = 0.0f, float continuity = 0.0f,
                               float bias = 0.0f, bool loop = false) noexcept {
    if (points.size() < 2) return points.empty() ? T() : points[0];
    if (points.size() == 2) return lerp(points[0], points[1], t);
    std::size_t n = points.size();
    float total = loop ? static_cast<float>(n) : static_cast<float>(n - 1);
    float u = t * total;
    std::size_t seg = static_cast<std::size_t>(u);
    if (seg >= n - 1 && !loop) seg = n - 2;
    float seg_t = u - static_cast<float>(seg);
    std::size_t i0 = (seg == 0) ? (loop ? n-1 : 0) : seg - 1;
    std::size_t i1 = seg;
    std::size_t i2 = (seg + 1 < n) ? seg + 1 : (loop ? 0 : n-1);
    std::size_t i3 = (seg + 2 < n) ? seg + 2 : (loop ? (seg+2)%n : n-1);
    return kochanek_bartels_segment(points[i0], points[i1], points[i2], points[i3],
                                    seg_t, tension, continuity, bias);
}

// ============================================================
// Bézier curve (De Casteljau algorithm)
// ============================================================

template<typename T>
T de_casteljau(const std::vector<T>& control_points, float t) noexcept {
    if (control_points.empty()) return T();
    std::vector<T> temp = control_points;
    while (temp.size() > 1) {
        for (std::size_t i = 0; i < temp.size() - 1; ++i)
            temp[i] = lerp(temp[i], temp[i+1], t);
        temp.pop_back();
    }
    return temp[0];
}

template<typename T>
T bezier_quadratic(const T& p0, const T& p1, const T& p2, float t) noexcept {
    float u = 1.0f - t;
    return p0 * (u*u) + p1 * (2.0f*u*t) + p2 * (t*t);
}

template<typename T>
T bezier_cubic(const T& p0, const T& p1, const T& p2, const T& p3, float t) noexcept {
    float u = 1.0f - t;
    float u2 = u * u;
    float u3 = u2 * u;
    float t2 = t * t;
    float t3 = t2 * t;
    return p0 * u3 + p1 * (3.0f*u2*t) + p2 * (3.0f*u*t2) + p3 * t3;
}

// ============================================================
// B‑spline basis functions (Cox‑de Boor recursion)
// ============================================================

inline float b_spline_basis(int i, int degree, float t, const std::vector<float>& knots) noexcept {
    if (degree == 0) {
        return (t >= knots[i] && t < knots[i+1]) ? 1.0f : 0.0f;
    }
    float left = (knots[i+degree] - knots[i] > 1e-9f) ?
        ((t - knots[i]) / (knots[i+degree] - knots[i])) * b_spline_basis(i, degree-1, t, knots) : 0.0f;
    float right = (knots[i+degree+1] - knots[i+1] > 1e-9f) ?
        ((knots[i+degree+1] - t) / (knots[i+degree+1] - knots[i+1])) * b_spline_basis(i+1, degree-1, t, knots) : 0.0f;
    return left + right;
}

template<typename T>
T b_spline(const std::vector<T>& control_points, int degree,
           const std::vector<float>& knots, float t) noexcept {
    T result{};
    for (std::size_t i = 0; i < control_points.size(); ++i) {
        float b = b_spline_basis(static_cast<int>(i), degree, t, knots);
        result = result + control_points[i] * b;
    }
    return result;
}

inline std::vector<float> generate_uniform_knots(int n, int degree, bool open = true) noexcept {
    int m = n + degree + 1;
    std::vector<float> knots(m);
    if (open) {
        for (int i = 0; i <= degree; ++i) knots[i] = 0.0f;
        for (int i = degree+1; i < n; ++i) knots[i] = float(i - degree) / float(n - degree);
        for (int i = n; i <= m-1; ++i) knots[i] = 1.0f;
    } else {
        for (int i = 0; i < m; ++i) knots[i] = float(i) / float(m-1);
    }
    return knots;
}

template<typename T>
T b_spline_interpolate(const std::vector<T>& control_points, int degree, float t,
                       const std::vector<float>& knots) noexcept {
    return b_spline(control_points, degree, knots, t);
}

// ============================================================
// NURBS (non‑uniform rational B‑spline)
// ============================================================

template<typename T>
T nurbs(const std::vector<T>& control_points, const std::vector<float>& weights,
        int degree, const std::vector<float>& knots, float t) noexcept {
    T numerator{};
    float denominator = 0.0f;
    for (std::size_t i = 0; i < control_points.size(); ++i) {
        float b = b_spline_basis(static_cast<int>(i), degree, t, knots);
        numerator = numerator + control_points[i] * (weights[i] * b);
        denominator += weights[i] * b;
    }
    if (denominator < 1e-9f) return control_points[0];
    return numerator * (1.0f / denominator);
}

// ============================================================
// Arc‑length parameterization (LUT + bisection)
// ============================================================

template<typename T, typename Func>
std::vector<float> build_arc_length_table(const Func& curve, float t_min, float t_max,
                                          int samples = 256, float total_est = -1.0f) noexcept {
    std::vector<float> table(samples + 1);
    table[0] = 0.0f;
    float dt = (t_max - t_min) / samples;
    T prev = curve(t_min);
    for (int i = 1; i <= samples; ++i) {
        float t = t_min + dt * i;
        T curr = curve(t);
        table[i] = table[i-1] + distance(curr, prev);
        prev = curr;
    }
    return table;
}

template<typename T, typename Func>
T evaluate_by_arc_length(const Func& curve, const std::vector<float>& arc_table,
                         float t_min, float t_max, float s, bool normalize = true) noexcept {
    float total = arc_table.back();
    if (normalize && total > 0.0f) s *= total;
    if (s <= 0.0f) return curve(t_min);
    if (s >= total) return curve(t_max);
    int n = static_cast<int>(arc_table.size()) - 1;
    auto it = std::lower_bound(arc_table.begin(), arc_table.end(), s);
    int idx = std::max(0, std::min(n-1, static_cast<int>(it - arc_table.begin()) - 1));
    float seg_start = arc_table[idx];
    float seg_end = arc_table[idx+1];
    float seg_len = seg_end - seg_start;
    float alpha = (seg_len > 1e-9f) ? (s - seg_start) / seg_len : 0.0f;
    float dt = (t_max - t_min) / n;
    float t = t_min + dt * (idx + alpha);
    return curve(t);
}

// ============================================================
// Helper: linear parameterization of a polyline
// ============================================================

template<typename T>
T polyline_interpolate(const std::vector<T>& points, float t) noexcept {
    if (points.empty()) return T();
    if (points.size() == 1) return points[0];
    float total = static_cast<float>(points.size() - 1);
    float u = clamp(t, 0.0f, 1.0f) * total;
    int seg = static_cast<int>(u);
    if (seg >= static_cast<int>(points.size()) - 1) return points.back();
    float frac = u - static_cast<float>(seg);
    return lerp(points[seg], points[seg+1], frac);
}

// ============================================================
// Quaternion spline (using slerp between successive quaternions)
// ============================================================

template<typename T>
quaternion<T> quaternion_spline(const std::vector<quaternion<T>>& quats, float t,
                                bool loop = false) noexcept {
    if (quats.empty()) return quaternion<T>();
    if (quats.size() == 1) return quats[0];
    float total = loop ? static_cast<float>(quats.size()) : static_cast<float>(quats.size() - 1);
    float u = t * total;
    int seg = static_cast<int>(u);
    if (seg >= static_cast<int>(quats.size()) - 1 && !loop) return quats.back();
    float frac = u - static_cast<float>(seg);
    if (seg >= static_cast<int>(quats.size()) - 1 && loop) {
        frac = u - static_cast<float>(seg);
        seg = static_cast<int>(quats.size()) - 1;
        return slerp(quats[seg], quats[0], frac);
    }
    return slerp(quats[seg], quats[seg+1], frac);
}

// ============================================================
// Smoothstep‑based blending between two sets of values
// ============================================================

template<typename T>
T blend_sequences(const T& start, const T& end, float t, float blend) noexcept {
    float w = smoothstep(0.0f, blend, t);
    return lerp(start, end, w);
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_INTERPOLATION_H
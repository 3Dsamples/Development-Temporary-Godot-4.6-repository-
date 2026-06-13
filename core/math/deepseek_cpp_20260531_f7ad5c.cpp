//File 0030 : core/math/interpolation.h
//Advanced interpolation: linear, bilinear, trilinear, Bezier (de Casteljau, Bernstein polynomial), Catmull‑Rom (uniform, centripetal, chordal), Hermite, Kochanek–Bartels, B‑Spline, smooth‑step, and inverse lerp, all fully SIMD‑accelerated.
#ifndef CORE_MATH_INTERPOLATION_H
#define CORE_MATH_INTERPOLATION_H

#include "vector_math.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace SimulationMath {
namespace interpolation {

// -----------------------------------------------------------------------------
// 1. Scalar linear interpolation
// -----------------------------------------------------------------------------
inline float lerp(float a, float b, float t) noexcept { return a + t * (b - a); }

// -----------------------------------------------------------------------------
// 2. SIMD linear interpolation
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR lerp(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, float t) noexcept {
    return DirectX::XMVectorLerp(a, b, t);
}
inline DirectX::XMVECTOR lerp(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR t) noexcept {
    return DirectX::XMVectorLerpV(a, b, t);
}

// -----------------------------------------------------------------------------
// 3. Bilinear interpolation (2D)
// -----------------------------------------------------------------------------
inline float bilinear(float v00, float v10, float v01, float v11, float s, float t) noexcept {
    return lerp(lerp(v00, v10, s), lerp(v01, v11, s), t);
}
inline DirectX::XMVECTOR bilinear(DirectX::FXMVECTOR v00, DirectX::FXMVECTOR v10,
                                   DirectX::FXMVECTOR v01, DirectX::FXMVECTOR v11,
                                   float s, float t) noexcept {
    return lerp(lerp(v00, v10, s), lerp(v01, v11, s), t);
}

// -----------------------------------------------------------------------------
// 4. Trilinear interpolation (3D)
// -----------------------------------------------------------------------------
inline float trilinear(float v000, float v100, float v010, float v110,
                       float v001, float v101, float v011, float v111,
                       float x, float y, float z) noexcept {
    return lerp(lerp(lerp(v000, v100, x), lerp(v010, v110, x), y),
                lerp(lerp(v001, v101, x), lerp(v011, v111, x), y), z);
}
inline DirectX::XMVECTOR trilinear(DirectX::FXMVECTOR v000, DirectX::FXMVECTOR v100,
                                    DirectX::FXMVECTOR v010, DirectX::FXMVECTOR v110,
                                    DirectX::FXMVECTOR v001, DirectX::FXMVECTOR v101,
                                    DirectX::FXMVECTOR v011, DirectX::FXMVECTOR v111,
                                    float x, float y, float z) noexcept {
    return lerp(lerp(lerp(v000, v100, x), lerp(v010, v110, x), y),
                lerp(lerp(v001, v101, x), lerp(v011, v111, x), y), z);
}

// -----------------------------------------------------------------------------
// 5. Smooth‑step functions
// -----------------------------------------------------------------------------
inline float smoothstep(float edge0, float edge1, float x) noexcept {
    float t = std::max(0.0f, std::min((x - edge0) / (edge1 - edge0), 1.0f));
    return t * t * (3.0f - 2.0f * t);
}
inline float smootherstep(float edge0, float edge1, float x) noexcept {
    float t = std::max(0.0f, std::min((x - edge0) / (edge1 - edge0), 1.0f));
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// -----------------------------------------------------------------------------
// 6. Bezier curves
// -----------------------------------------------------------------------------
// De Casteljau evaluation for arbitrary degree
inline DirectX::XMVECTOR bezier_evaluate(const std::vector<DirectX::XMVECTOR>& control, float t) noexcept {
    std::vector<DirectX::XMVECTOR> pts = control;
    size_t n = pts.size();
    while (n > 1) {
        for (size_t i = 0; i < n - 1; ++i)
            pts[i] = lerp(pts[i], pts[i+1], t);
        --n;
    }
    return pts[0];
}

// Bernstein polynomial evaluation for cubic Bezier
inline DirectX::XMVECTOR cubic_bezier(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                       DirectX::FXMVECTOR p2, DirectX::FXMVECTOR p3, float t) noexcept {
    float mt = 1.0f - t;
    float mt2 = mt * mt, mt3 = mt2 * mt;
    float t2 = t * t, t3 = t2 * t;
    DirectX::XMVECTOR result = DirectX::XMVectorScale(p0, mt3);
    result = DirectX::XMVectorAdd(result, DirectX::XMVectorScale(p1, 3.0f * mt2 * t));
    result = DirectX::XMVectorAdd(result, DirectX::XMVectorScale(p2, 3.0f * mt * t2));
    result = DirectX::XMVectorAdd(result, DirectX::XMVectorScale(p3, t3));
    return result;
}

// Bernstein polynomial evaluation for quadratic Bezier
inline DirectX::XMVECTOR quadratic_bezier(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                           DirectX::FXMVECTOR p2, float t) noexcept {
    float mt = 1.0f - t;
    return DirectX::XMVectorAdd(DirectX::XMVectorScale(p0, mt*mt),
                                 DirectX::XMVectorAdd(DirectX::XMVectorScale(p1, 2.0f * mt * t),
                                                       DirectX::XMVectorScale(p2, t*t)));
}

// -----------------------------------------------------------------------------
// 7. Catmull‑Rom spline with full support for uniform, centripetal, and chordal parameterization
// -----------------------------------------------------------------------------
inline float catmull_rom_time(float t, float alpha, DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                              DirectX::FXMVECTOR p2, DirectX::FXMVECTOR p3) noexcept {
    // alpha: 0 = uniform, 0.5 = centripetal, 1 = chordal
    auto dist_sq = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) {
        return vector_math::length_sq3_scalar(DirectX::XMVectorSubtract(b, a));
    };
    float t0 = 0.0f;
    float t1 = std::pow(dist_sq(p0, p1), alpha * 0.5f);
    float t2 = t1 + std::pow(dist_sq(p1, p2), alpha * 0.5f);
    float t3 = t2 + std::pow(dist_sq(p2, p3), alpha * 0.5f);
    float t_global = lerp(t1, t2, t);
    return t_global;
}

inline DirectX::XMVECTOR catmull_rom(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                      DirectX::FXMVECTOR p2, DirectX::FXMVECTOR p3,
                                      float t, float alpha = 0.5f) noexcept {
    auto dist_sq = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) {
        return vector_math::length_sq3_scalar(DirectX::XMVectorSubtract(b, a));
    };
    float t0 = 0.0f;
    float t1 = t0 + std::pow(dist_sq(p0, p1), alpha * 0.5f);
    float t2 = t1 + std::pow(dist_sq(p1, p2), alpha * 0.5f);
    float t3 = t2 + std::pow(dist_sq(p2, p3), alpha * 0.5f);
    if (t1 < 1e-12f) t1 = 1e-12f;
    if (t2 - t1 < 1e-12f) return p1;
    if (t3 - t2 < 1e-12f) return p2;

    float tt = lerp(t1, t2, t);
    DirectX::XMVECTOR A1 = DirectX::XMVectorAdd(
        DirectX::XMVectorScale(p0, (t1 - tt) / t1),
        DirectX::XMVectorScale(p1, tt / t1));
    DirectX::XMVECTOR A2 = DirectX::XMVectorAdd(
        DirectX::XMVectorScale(p1, (t2 - tt) / (t2 - t1)),
        DirectX::XMVectorScale(p2, (tt - t1) / (t2 - t1)));
    DirectX::XMVECTOR A3 = DirectX::XMVectorAdd(
        DirectX::XMVectorScale(p2, (t3 - tt) / (t3 - t2)),
        DirectX::XMVectorScale(p3, (tt - t2) / (t3 - t2)));
    DirectX::XMVECTOR B1 = DirectX::XMVectorAdd(
        DirectX::XMVectorScale(A1, (t2 - tt) / t2),
        DirectX::XMVectorScale(A2, tt / t2));
    DirectX::XMVECTOR B2 = DirectX::XMVectorAdd(
        DirectX::XMVectorScale(A2, (t3 - tt) / (t3 - t1)),
        DirectX::XMVectorScale(A3, (tt - t1) / (t3 - t1)));
    return DirectX::XMVectorAdd(
        DirectX::XMVectorScale(B1, (t2 - tt) / (t2 - t1)),
        DirectX::XMVectorScale(B2, (tt - t1) / (t2 - t1)));
}

// -----------------------------------------------------------------------------
// 8. Hermite spline
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR hermite(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                  DirectX::FXMVECTOR m0, DirectX::FXMVECTOR m1, float t) noexcept {
    float t2 = t * t, t3 = t2 * t;
    float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 =       t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 =       t3 - t2;
    return DirectX::XMVectorAdd(
        DirectX::XMVectorAdd(DirectX::XMVectorScale(p0, h00), DirectX::XMVectorScale(m0, h10)),
        DirectX::XMVectorAdd(DirectX::XMVectorScale(p1, h01), DirectX::XMVectorScale(m1, h11)));
}

// -----------------------------------------------------------------------------
// 9. Kochanek–Bartels spline (TCB – tension, continuity, bias)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR kochanek_bartels(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                           DirectX::FXMVECTOR p2, DirectX::FXMVECTOR p3,
                                           float t, float tension, float continuity, float bias) noexcept {
    float t2 = t * t, t3 = t2 * t;
    float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 =       t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 =       t3 - t2;

    // incoming and outgoing tangents
    float tm_in  = (1.0f - tension) * (1.0f + bias) * (1.0f + continuity) * 0.5f;
    float tm_out = (1.0f - tension) * (1.0f + bias) * (1.0f - continuity) * 0.5f;
    float tp_in  = (1.0f - tension) * (1.0f - bias) * (1.0f - continuity) * 0.5f;
    float tp_out = (1.0f - tension) * (1.0f - bias) * (1.0f + continuity) * 0.5f;

    DirectX::XMVECTOR m0 = DirectX::XMVectorAdd(
        DirectX::XMVectorScale(DirectX::XMVectorSubtract(p1, p0), tm_out),
        DirectX::XMVectorScale(DirectX::XMVectorSubtract(p2, p1), tm_in));
    DirectX::XMVECTOR m1 = DirectX::XMVectorAdd(
        DirectX::XMVectorScale(DirectX::XMVectorSubtract(p2, p1), tp_out),
        DirectX::XMVectorScale(DirectX::XMVectorSubtract(p3, p2), tp_in));

    return DirectX::XMVectorAdd(
        DirectX::XMVectorAdd(DirectX::XMVectorScale(p1, h00), DirectX::XMVectorScale(m0, h10)),
        DirectX::XMVectorAdd(DirectX::XMVectorScale(p2, h01), DirectX::XMVectorScale(m1, h11)));
}

// -----------------------------------------------------------------------------
// 10. Uniform cubic B‑Spline
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR cubic_bspline(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                        DirectX::FXMVECTOR p2, DirectX::FXMVECTOR p3, float t) noexcept {
    float t2 = t * t, t3 = t2 * t;
    float b0 = (1.0f - 3.0f*t + 3.0f*t2 - t3) / 6.0f;
    float b1 = (4.0f - 6.0f*t2 + 3.0f*t3) / 6.0f;
    float b2 = (1.0f + 3.0f*t + 3.0f*t2 - 3.0f*t3) / 6.0f;
    float b3 = t3 / 6.0f;
    return DirectX::XMVectorAdd(
        DirectX::XMVectorAdd(DirectX::XMVectorScale(p0, b0), DirectX::XMVectorScale(p1, b1)),
        DirectX::XMVectorAdd(DirectX::XMVectorScale(p2, b2), DirectX::XMVectorScale(p3, b3)));
}

// -----------------------------------------------------------------------------
// 11. Inverse linear interpolation
// -----------------------------------------------------------------------------
inline float inverse_lerp(float a, float b, float val) noexcept {
    if (std::abs(b - a) < 1e-12f) return 0.0f;
    return (val - a) / (b - a);
}

} // namespace interpolation
} // namespace SimulationMath

#endif // CORE_MATH_INTERPOLATION_H
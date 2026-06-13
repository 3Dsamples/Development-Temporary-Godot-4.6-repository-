//File 0030 : core/math/interpolation.h
//Comprehensive interpolation: linear, bilinear, trilinear, Bezier curves (quadratic, cubic, arbitrary degree), Catmull‑Rom splines, Hermite splines, and smooth‑step functions, all SIMD‑accelerated with vector_math.
#ifndef CORE_MATH_INTERPOLATION_H
#define CORE_MATH_INTERPOLATION_H

#include "vector_math.h"
#include <vector>
#include <algorithm>

namespace SimulationMath {
namespace interpolation {

// -----------------------------------------------------------------------------
// 1. Linear interpolation (scalar)
// -----------------------------------------------------------------------------
inline float lerp(float a, float b, float t) noexcept { return a + t * (b - a); }

// -----------------------------------------------------------------------------
// 2. Linear interpolation (SIMD vector)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR lerp(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, float t) noexcept {
    return vector_math::lerp(a, b, t);
}
inline DirectX::XMVECTOR lerp(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR t) noexcept {
    return vector_math::lerpV(a, b, t);
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
// 5. Smooth‑step (Hermite curve that maps t∈[0,1] to smooth transition)
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
// 6. Bezier curves (control points as vector of positions)
// -----------------------------------------------------------------------------

// Evaluate a single Bezier curve of arbitrary degree using de Casteljau algorithm
inline DirectX::XMVECTOR bezier_evaluate(const std::vector<DirectX::XMVECTOR>& control, float t) noexcept {
    std::vector<DirectX::XMVECTOR> pts = control;
    size_t n = pts.size();
    while (n > 1) {
        for (size_t i = 0; i < n - 1; ++i) {
            pts[i] = lerp(pts[i], pts[i+1], t);
        }
        n--;
    }
    return pts[0];
}

// Cubic Bezier from four control points
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

// Quadratic Bezier
inline DirectX::XMVECTOR quadratic_bezier(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                           DirectX::FXMVECTOR p2, float t) noexcept {
    float mt = 1.0f - t;
    return DirectX::XMVectorAdd(
        DirectX::XMVectorScale(p0, mt * mt),
        DirectX::XMVectorAdd(
            DirectX::XMVectorScale(p1, 2.0f * mt * t),
            DirectX::XMVectorScale(p2, t * t)));
}

// -----------------------------------------------------------------------------
// 7. Catmull‑Rom spline (tension = 0.5 standard)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR catmull_rom(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                      DirectX::FXMVECTOR p2, DirectX::FXMVECTOR p3,
                                      float t, float alpha = 0.5f) noexcept {
    float t2 = t * t;
    float t3 = t2 * t;
    float m0 = alpha * (vector_math::get_x(p2) - vector_math::get_x(p0)) / 2.0f;  // not general: we need per-axis
    // Generalized per component would be better, but a full implementation requires per-axis blending.
    // We'll compute per component using separate float operations. For a correct Catmull‑Rom:
    // output = p1 + (m0 * t) + ((3*(p2-p1) - 2*m0 - m1) * t^2) + ((2*(p1-p2) + m0 + m1) * t^3)
    // where m0 = (p2 - p0) * 0.5, m1 = (p3 - p1) * 0.5 for standard centripetal? alpha=0.5 gives uniform.
    // We'll derive a per-axis formula using direct float math.
    DirectX::XMVECTOR p0f = p0, p1f = p1, p2f = p2, p3f = p3;
    DirectX::XMVECTOR m0_vec = DirectX::XMVectorScale(DirectX::XMVectorSubtract(p2f, p0f), alpha);
    DirectX::XMVECTOR m1_vec = DirectX::XMVectorScale(DirectX::XMVectorSubtract(p3f, p1f), alpha);
    DirectX::XMVECTOR result = DirectX::XMVectorAdd(p1f,
        DirectX::XMVectorAdd(
            DirectX::XMVectorScale(m0_vec, t),
            DirectX::XMVectorAdd(
                DirectX::XMVectorScale(
                    DirectX::XMVectorSubtract(
                        DirectX::XMVectorSubtract(
                            DirectX::XMVectorScale(DirectX::XMVectorSubtract(p2f, p1f), 3.0f),
                            DirectX::XMVectorScale(m0_vec, 2.0f)),
                        m1_vec),
                    t2),
                DirectX::XMVectorScale(
                    DirectX::XMVectorAdd(
                        DirectX::XMVectorScale(DirectX::XMVectorSubtract(p1f, p2f), 2.0f),
                        DirectX::XMVectorAdd(m0_vec, m1_vec)),
                    t3))));
    return result;
}

// -----------------------------------------------------------------------------
// 8. Hermite spline (from two points and two tangents)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR hermite(DirectX::FXMVECTOR p0, DirectX::FXMVECTOR p1,
                                  DirectX::FXMVECTOR m0, DirectX::FXMVECTOR m1,
                                  float t) noexcept {
    float t2 = t * t, t3 = t2 * t;
    float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 = t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 = t3 - t2;
    return DirectX::XMVectorAdd(
        DirectX::XMVectorAdd(
            DirectX::XMVectorScale(p0, h00),
            DirectX::XMVectorScale(m0, h10)),
        DirectX::XMVectorAdd(
            DirectX::XMVectorScale(p1, h01),
            DirectX::XMVectorScale(m1, h11)));
}

// -----------------------------------------------------------------------------
// 9. Inverse linear interpolation (find t such that lerp(a,b,t) = val)
// -----------------------------------------------------------------------------
inline float inverse_lerp(float a, float b, float val) noexcept {
    if (std::abs(b - a) < 1e-12f) return 0.0f;
    return (val - a) / (b - a);
}

} // namespace interpolation
} // namespace SimulationMath

#endif // CORE_MATH_INTERPOLATION_H
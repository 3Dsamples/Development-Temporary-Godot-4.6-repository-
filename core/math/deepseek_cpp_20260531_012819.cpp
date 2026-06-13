//File 0072 : core/math/differential_geometry.h
//Differential geometry for curves and surfaces: curvature, torsion, Frenet‑Serret frame, first/second fundamental forms, Gaussian/mean/principal curvatures using SIMD vector_math.
#ifndef CORE_MATH_DIFFERENTIAL_GEOMETRY_H
#define CORE_MATH_DIFFERENTIAL_GEOMETRY_H

#include "vector_math.h"
#include "math_constants.h"
#include <functional>
#include <cmath>

namespace SimulationMath {
namespace diff_geom {

using SimdVec = DirectX::XMVECTOR;

// -----------------------------------------------------------------------------
// 1. Numerical derivative of a scalar function f(float) -> float
// -----------------------------------------------------------------------------
inline float derivative1_scalar(const std::function<float(float)>& f, float t, float h = 1e-4f) noexcept {
    return (f(t + h) - f(t - h)) / (2.0f * h);
}

// -----------------------------------------------------------------------------
// 2. Numerical derivative of a vector function f(float) -> SimdVec
// -----------------------------------------------------------------------------
inline SimdVec derivative1_vec(const std::function<SimdVec(float)>& f, float t, float h = 1e-4f) noexcept {
    return DirectX::XMVectorScale(
        DirectX::XMVectorSubtract(f(t + h), f(t - h)), 1.0f / (2.0f * h));
}

// -----------------------------------------------------------------------------
// 3. Second numerical derivative (vector)
// -----------------------------------------------------------------------------
inline SimdVec derivative2_vec(const std::function<SimdVec(float)>& f, float t, float h = 1e-4f) noexcept {
    return DirectX::XMVectorScale(
        DirectX::XMVectorSubtract(
            DirectX::XMVectorSubtract(f(t + h), DirectX::XMVectorScale(f(t), 2.0f)),
            f(t - h)), 1.0f / (h * h));
}

// -----------------------------------------------------------------------------
// 4. Third numerical derivative (vector)
// -----------------------------------------------------------------------------
inline SimdVec derivative3_vec(const std::function<SimdVec(float)>& f, float t, float h = 1e-4f) noexcept {
    SimdVec f2h = f(t + 2.0f * h);
    SimdVec fh  = f(t + h);
    SimdVec f_mh = f(t - h);
    SimdVec f_m2h = f(t - 2.0f * h);
    SimdVec numerator = DirectX::XMVectorSubtract(
        DirectX::XMVectorAdd(f2h, f_m2h),
        DirectX::XMVectorScale(DirectX::XMVectorAdd(fh, f_mh), 2.0f));
    return DirectX::XMVectorScale(numerator, 1.0f / (2.0f * h * h * h));
}

// -----------------------------------------------------------------------------
// 5. Compute curvature κ of a space curve at parameter t
// -----------------------------------------------------------------------------
inline float curvature(const std::function<SimdVec(float)>& r, float t, float h = 1e-4f) noexcept {
    SimdVec r1 = derivative1_vec(r, t, h);
    SimdVec r2 = derivative2_vec(r, t, h);
    SimdVec cross = vector_math::cross3(r1, r2);
    float len_cross = vector_math::length3_scalar(cross);
    float len_r1 = vector_math::length3_scalar(r1);
    if (len_r1 < 1e-12f) return 0.0f;
    return len_cross / (len_r1 * len_r1 * len_r1);
}

// -----------------------------------------------------------------------------
// 6. Compute torsion τ of a space curve at parameter t
// -----------------------------------------------------------------------------
inline float torsion(const std::function<SimdVec(float)>& r, float t, float h = 1e-4f) noexcept {
    SimdVec r1 = derivative1_vec(r, t, h);
    SimdVec r2 = derivative2_vec(r, t, h);
    SimdVec r3 = derivative3_vec(r, t, h);
    SimdVec cross = vector_math::cross3(r1, r2);
    float cross_len_sq = vector_math::length_sq3_scalar(cross);
    if (cross_len_sq < 1e-12f) return 0.0f;
    float numerator = vector_math::dot3_scalar(cross, r3);
    return numerator / cross_len_sq;
}

// -----------------------------------------------------------------------------
// 7. Frenet‑Serret frame (tangent T, normal N, binormal B)
// -----------------------------------------------------------------------------
inline void frenet_frame(const std::function<SimdVec(float)>& r, float t,
                         SimdVec& out_T, SimdVec& out_N, SimdVec& out_B, float h = 1e-4f) noexcept {
    SimdVec r1 = derivative1_vec(r, t, h);
    SimdVec r2 = derivative2_vec(r, t, h);
    SimdVec T = vector_math::normalize3(r1);
    SimdVec T_prime = derivative1_vec([&](float tt) {
        SimdVec r1_t = derivative1_vec(r, tt, h);
        return vector_math::normalize3(r1_t);
    }, t, h);
    // Alternatively compute N from r' and r'': N = ( (r' × r'') × r' ) / |...|
    SimdVec cross = vector_math::cross3(r1, r2);
    if (vector_math::length_sq3_scalar(cross) < 1e-12f) {
        // degenerate
        out_T = T;
        out_N = DirectX::XMVectorZero();
        out_B = DirectX::XMVectorZero();
        return;
    }
    SimdVec N = vector_math::normalize3(vector_math::cross3(cross, r1));
    SimdVec B = vector_math::cross3(T, N);
    out_T = T;
    out_N = N;
    out_B = B;
}

// -----------------------------------------------------------------------------
// 8. First fundamental form coefficients of a surface S(u,v)
// -----------------------------------------------------------------------------
inline void first_fundamental_form(const std::function<SimdVec(float,float)>& S,
                                   float u, float v, float& E, float& F, float& G, float h = 1e-4f) noexcept {
    auto Su = [&](float uu) { return S(uu, v); };
    auto Sv = [&](float vv) { return S(u, vv); };
    SimdVec du = derivative1_vec(Su, u, h);
    SimdVec dv = derivative1_vec(Sv, v, h);
    E = vector_math::length_sq3_scalar(du);
    F = vector_math::dot3_scalar(du, dv);
    G = vector_math::length_sq3_scalar(dv);
}

// -----------------------------------------------------------------------------
// 9. Second fundamental form coefficients (needs second partial derivatives)
// -----------------------------------------------------------------------------
inline void second_fundamental_form(const std::function<SimdVec(float,float)>& S,
                                    float u, float v, float& L, float& M, float& N, float h = 1e-4f) noexcept {
    auto Su  = [&](float uu) { return S(uu, v); };
    auto Sv  = [&](float vv) { return S(u, vv); };
    auto Suu = [&](float uu) { return derivative1_vec(Su, uu, h); };
    auto Svv = [&](float vv) { return derivative1_vec(Sv, vv, h); };
    auto Suv = [&](float uu) { return derivative1_vec([&](float vv) { return S(uu, vv); }, v, h); };
    // Actually Suv is ∂/∂v of Su. We'll compute:
    SimdVec du = derivative1_vec(Su, u, h);
    SimdVec dv = derivative1_vec(Sv, v, h);
    SimdVec duu = derivative2_vec(Su, u, h);
    SimdVec dvv = derivative2_vec(Sv, v, h);
    // Compute Suv: first derivative wrt u, then derivative wrt v
    auto Su_func = [&](float uu) -> SimdVec { return derivative1_vec([&](float vv) { return S(uu, vv); }, v, h); };
    SimdVec duv = derivative1_vec(Su_func, u, h);
    SimdVec norm = vector_math::normalize3(vector_math::cross3(du, dv));
    L = vector_math::dot3_scalar(duu, norm);
    M = vector_math::dot3_scalar(duv, norm);
    N = vector_math::dot3_scalar(dvv, norm);
}

// -----------------------------------------------------------------------------
// 10. Gaussian curvature K = (LN - M^2) / (EG - F^2)
// -----------------------------------------------------------------------------
inline float gaussian_curvature(const std::function<SimdVec(float,float)>& S,
                                float u, float v, float h = 1e-4f) noexcept {
    float E, F, G, L, M, N;
    first_fundamental_form(S, u, v, E, F, G, h);
    second_fundamental_form(S, u, v, L, M, N, h);
    float denom = E * G - F * F;
    if (std::abs(denom) < 1e-12f) return 0.0f;
    return (L * N - M * M) / denom;
}

// -----------------------------------------------------------------------------
// 11. Mean curvature H = (EN - 2FM + GL) / (2(EG - F^2))
// -----------------------------------------------------------------------------
inline float mean_curvature(const std::function<SimdVec(float,float)>& S,
                            float u, float v, float h = 1e-4f) noexcept {
    float E, F, G, L, M, N;
    first_fundamental_form(S, u, v, E, F, G, h);
    second_fundamental_form(S, u, v, L, M, N, h);
    float denom = 2.0f * (E * G - F * F);
    if (std::abs(denom) < 1e-12f) return 0.0f;
    return (E * N - 2.0f * F * M + G * L) / denom;
}

// -----------------------------------------------------------------------------
// 12. Principal curvatures κ1, κ2 from Weingarten matrix
// -----------------------------------------------------------------------------
inline void principal_curvatures(const std::function<SimdVec(float,float)>& S,
                                 float u, float v, float& k1, float& k2, float h = 1e-4f) noexcept {
    float E, F, G, L, M, N;
    first_fundamental_form(S, u, v, E, F, G, h);
    second_fundamental_form(S, u, v, L, M, N, h);
    float denom = E * G - F * F;
    if (std::abs(denom) < 1e-12f) {
        k1 = k2 = 0.0f;
        return;
    }
    // Weingarten matrix components: W = inv(first) * second
    float invDenom = 1.0f / denom;
    float W11 = invDenom * (G * L - F * M);
    float W12 = invDenom * (G * M - F * N);
    float W21 = invDenom * (E * M - F * L);
    float W22 = invDenom * (E * N - F * M);
    // Eigenvalues of W give principal curvatures
    float trace = W11 + W22;
    float det   = W11 * W22 - W12 * W21;
    float disc = std::sqrt(std::max(0.0f, trace * trace * 0.25f - det));
    k1 = 0.5f * trace + disc;
    k2 = 0.5f * trace - disc;
}

} // namespace diff_geom
} // namespace SimulationMath

#endif // CORE_MATH_DIFFERENTIAL_GEOMETRY_H
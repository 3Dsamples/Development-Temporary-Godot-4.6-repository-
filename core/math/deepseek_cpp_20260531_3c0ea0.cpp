//File 0045 : core/math/barycentric.h
//Barycentric coordinate calculation for triangles (2D/3D) and tetrahedra, point‑in‑simplex tests, and attribute interpolation with SIMD acceleration.
#ifndef CORE_MATH_BARYCENTRIC_H
#define CORE_MATH_BARYCENTRIC_H

#include "vector_math.h"
#include <cmath>

namespace SimulationMath {
namespace barycentric {

// -----------------------------------------------------------------------------
// 1. 2D barycentric coordinates of point p in triangle (v0,v1,v2)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR barycentric_2d(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1,
                                         DirectX::FXMVECTOR v2, DirectX::FXMVECTOR p) noexcept {
    // Compute using area ratios: u = area(p,v1,v2)/area(v0,v1,v2), v = area(v0,p,v2)/area, w = area(v0,v1,p)/area
    DirectX::XMVECTOR d0 = DirectX::XMVectorSubtract(v2, v1);
    DirectX::XMVECTOR d1 = DirectX::XMVectorSubtract(v0, v2);
    DirectX::XMVECTOR d2 = DirectX::XMVectorSubtract(v1, v0);
    DirectX::XMVECTOR p0 = DirectX::XMVectorSubtract(p, v0);
    DirectX::XMVECTOR p1 = DirectX::XMVectorSubtract(p, v1);
    DirectX::XMVECTOR p2 = DirectX::XMVectorSubtract(p, v2);
    // areas = 0.5 * cross2d length, but we use scalar cross (x1*y2 - y1*x2)
    auto cross2 = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) {
        return DirectX::XMVectorGetX(a) * DirectX::XMVectorGetY(b) - DirectX::XMVectorGetY(a) * DirectX::XMVectorGetX(b);
    };
    float area = cross2(d0, d1); // area * 2
    if (std::abs(area) < 1e-12f) return DirectX::XMVectorSet(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 0.0f);
    float inv_area = 1.0f / area;
    float u = cross2(d0, p2) * inv_area;
    float v = cross2(d1, p0) * inv_area;
    float w = 1.0f - u - v;
    return DirectX::XMVectorSet(u, v, w, 0.0f);
}

// -----------------------------------------------------------------------------
// 2. 3D barycentric of point p relative to triangle (v0,v1,v2) – projects onto plane
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR barycentric_3d(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1,
                                         DirectX::FXMVECTOR v2, DirectX::FXMVECTOR p) noexcept {
    DirectX::XMVECTOR e0 = DirectX::XMVectorSubtract(v1, v0);
    DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(v2, v0);
    DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(p, v0);
    float d00 = vector_math::dot3_scalar(e0, e0);
    float d01 = vector_math::dot3_scalar(e0, e1);
    float d11 = vector_math::dot3_scalar(e1, e1);
    float d20 = vector_math::dot3_scalar(e2, e0);
    float d21 = vector_math::dot3_scalar(e2, e1);
    float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-12f) return DirectX::XMVectorSet(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 0.0f);
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    return DirectX::XMVectorSet(u, v, w, 0.0f);
}

// -----------------------------------------------------------------------------
// 3. Check if barycentric coordinates are inside triangle (all >= 0 and sum <= 1)
// -----------------------------------------------------------------------------
inline bool is_inside_triangle(DirectX::FXMVECTOR bary) noexcept {
    float u = vector_math::get_x(bary);
    float v = vector_math::get_y(bary);
    float w = vector_math::get_z(bary);
    return (u >= 0.0f && v >= 0.0f && w >= 0.0f && (u + v + w) <= 1.0001f);
}

// -----------------------------------------------------------------------------
// 4. Interpolate three values using barycentric coordinates (scalar)
// -----------------------------------------------------------------------------
inline float interpolate_vertex_values(float val0, float val1, float val2, DirectX::FXMVECTOR bary) noexcept {
    float u = vector_math::get_x(bary);
    float v = vector_math::get_y(bary);
    float w = vector_math::get_z(bary);
    return u * val0 + v * val1 + w * val2;
}

// -----------------------------------------------------------------------------
// 5. Interpolate three vectors (SIMD)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR interpolate_vertex_vectors(DirectX::FXMVECTOR c0, DirectX::FXMVECTOR c1,
                                                     DirectX::FXMVECTOR c2, DirectX::FXMVECTOR bary) noexcept {
    DirectX::XMVECTOR u = vector_math::swizzle_xxxx(bary);
    DirectX::XMVECTOR v = vector_math::swizzle_yyyy(bary);
    DirectX::XMVECTOR w = vector_math::swizzle_zzzz(bary);
    return DirectX::XMVectorAdd(DirectX::XMVectorMultiply(c0, u),
                                 DirectX::XMVectorAdd(DirectX::XMVectorMultiply(c1, v),
                                                      DirectX::XMVectorMultiply(c2, w)));
}

// -----------------------------------------------------------------------------
// 6. 3D barycentric for tetrahedron (4 points)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR barycentric_tetrahedron(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1,
                                                   DirectX::FXMVECTOR v2, DirectX::FXMVECTOR v3,
                                                   DirectX::FXMVECTOR p) noexcept {
    // Compute volume ratios
    DirectX::XMVECTOR d0 = DirectX::XMVectorSubtract(v1, v0);
    DirectX::XMVECTOR d1 = DirectX::XMVectorSubtract(v2, v0);
    DirectX::XMVECTOR d2 = DirectX::XMVectorSubtract(v3, v0);
    DirectX::XMVECTOR d3 = DirectX::XMVectorSubtract(p, v0);
    // Compute scalar triple product volume*6 of v1,v2,v3
    float vol = std::abs(vector_math::dot3_scalar(vector_math::cross3(d0, d1), d2));
    if (vol < 1e-12f) return DirectX::XMVectorSet(0.25f,0.25f,0.25f,0.25f);
    float inv_vol = 1.0f / vol;
    float u = vector_math::dot3_scalar(vector_math::cross3(d1, d2), d3) * inv_vol;
    float v = vector_math::dot3_scalar(vector_math::cross3(d3, d0), d2) * inv_vol;
    float w = vector_math::dot3_scalar(vector_math::cross3(d0, d1), d3) * inv_vol;
    float t = 1.0f - u - v - w;
    return DirectX::XMVectorSet(u, v, w, t);
}

// -----------------------------------------------------------------------------
// 7. Interpolate four values using tetrahedral barycentric
// -----------------------------------------------------------------------------
inline float interpolate_tetrahedral_values(float v0, float v1, float v2, float v3,
                                             DirectX::FXMVECTOR bary) noexcept {
    float u = vector_math::get_x(bary);
    float v = vector_math::get_y(bary);
    float w = vector_math::get_z(bary);
    float t = vector_math::get_w(bary);
    return u * v0 + v * v1 + w * v2 + t * v3;
}

// -----------------------------------------------------------------------------
// 8. Fast point‑in‑triangle test (2D) using cross product signs
// -----------------------------------------------------------------------------
inline bool point_in_triangle_2d(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1,
                                  DirectX::FXMVECTOR v2, DirectX::FXMVECTOR p) noexcept {
    auto cross2 = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) {
        return DirectX::XMVectorGetX(a) * DirectX::XMVectorGetY(b) - DirectX::XMVectorGetY(a) * DirectX::XMVectorGetX(b);
    };
    float c0 = cross2(DirectX::XMVectorSubtract(v1, v0), DirectX::XMVectorSubtract(p, v0));
    float c1 = cross2(DirectX::XMVectorSubtract(v2, v1), DirectX::XMVectorSubtract(p, v1));
    float c2 = cross2(DirectX::XMVectorSubtract(v0, v2), DirectX::XMVectorSubtract(p, v2));
    return (c0 >= 0.0f && c1 >= 0.0f && c2 >= 0.0f) || (c0 <= 0.0f && c1 <= 0.0f && c2 <= 0.0f);
}

} // namespace barycentric
} // namespace SimulationMath

#endif // CORE_MATH_BARYCENTRIC_H
//File 0052 : core/math/signed_distance_functions.h
//Collection of exact signed distance functions for common geometric primitives, boolean operations, and smooth blending, all SIMD‑accelerated via vector_math.
#ifndef CORE_MATH_SIGNED_DISTANCE_FUNCTIONS_H
#define CORE_MATH_SIGNED_DISTANCE_FUNCTIONS_H

#include "vector_math.h"
#include "math_constants.h"
#include <algorithm>
#include <cmath>

namespace SimulationMath {
namespace sdf {

using SimdVec = DirectX::XMVECTOR;

// -----------------------------------------------------------------------------
// 1. Sphere
// -----------------------------------------------------------------------------
inline float sphere(SimdVec point, float radius) noexcept {
    return vector_math::length3_scalar(point) - radius;
}

// -----------------------------------------------------------------------------
// 2. Box (centered at origin, half‑extents h)
// -----------------------------------------------------------------------------
inline float box(SimdVec point, SimdVec half_extents) noexcept {
    SimdVec q = DirectX::XMVectorSubtract(DirectX::XMVectorAbs(point), half_extents);
    return vector_math::length3_scalar(DirectX::XMVectorMax(q, DirectX::XMVectorZero())) +
           std::min(0.0f, std::max({vector_math::get_x(q), vector_math::get_y(q), vector_math::get_z(q)}));
}

// -----------------------------------------------------------------------------
// 3. Rounded box
// -----------------------------------------------------------------------------
inline float rounded_box(SimdVec point, SimdVec half_extents, float radius) noexcept {
    SimdVec q = DirectX::XMVectorSubtract(DirectX::XMVectorAbs(point), half_extents);
    return vector_math::length3_scalar(DirectX::XMVectorMax(q, DirectX::XMVectorZero())) +
           std::min(0.0f, std::max({vector_math::get_x(q), vector_math::get_y(q), vector_math::get_z(q)})) - radius;
}

// -----------------------------------------------------------------------------
// 4. Infinite cylinder along X axis (point relative to center)
// -----------------------------------------------------------------------------
inline float cylinder_x(SimdVec point, float radius) noexcept {
    float y = vector_math::get_y(point);
    float z = vector_math::get_z(point);
    return std::sqrt(y*y + z*z) - radius;
}

// -----------------------------------------------------------------------------
// 5. Capped cylinder (along X axis, height h = y extent)
// -----------------------------------------------------------------------------
inline float capped_cylinder_x(SimdVec point, float radius, float half_height) noexcept {
    float y = vector_math::get_y(point);
    float z = vector_math::get_z(point);
    float d = std::sqrt(y*y + z*z);
    float dx = std::abs(vector_math::get_x(point)) - half_height;
    return std::max(d - radius, dx);
}

// -----------------------------------------------------------------------------
// 6. Capsule (line segment from a to b, radius)
// -----------------------------------------------------------------------------
inline float capsule(SimdVec point, SimdVec a, SimdVec b, float radius) noexcept {
    SimdVec pa = DirectX::XMVectorSubtract(point, a);
    SimdVec ba = DirectX::XMVectorSubtract(b, a);
    float h = std::max(0.0f, std::min(1.0f, vector_math::dot3_scalar(pa, ba) / vector_math::dot3_scalar(ba, ba)));
    SimdVec closest = DirectX::XMVectorAdd(a, DirectX::XMVectorScale(ba, h));
    return vector_math::length3_scalar(DirectX::XMVectorSubtract(point, closest)) - radius;
}

// -----------------------------------------------------------------------------
// 7. Torus (centered at origin, with major radius R and minor radius r)
// -----------------------------------------------------------------------------
inline float torus(SimdVec point, float R, float r) noexcept {
    float x = vector_math::get_x(point);
    float y = vector_math::get_y(point);
    float z = vector_math::get_z(point);
    float q = std::sqrt(x*x + z*z) - R;
    return std::sqrt(q*q + y*y) - r;
}

// -----------------------------------------------------------------------------
// 8. Plane (given normal and distance from origin)
// -----------------------------------------------------------------------------
inline float plane(SimdVec point, SimdVec normal, float d) noexcept {
    return vector_math::dot3_scalar(point, normal) - d;
}

// -----------------------------------------------------------------------------
// 9. Cone (apex at origin, axis = Y+, angle alpha)
// -----------------------------------------------------------------------------
inline float cone(SimdVec point, float angle) noexcept {
    float c = std::cos(angle);
    float s = std::sin(angle);
    float y = vector_math::get_y(point);
    float x = vector_math::get_x(point);
    float z = vector_math::get_z(point);
    float q = std::sqrt(x*x + z*z);
    return std::max(c * q + s * y, -y) * (1.0f / std::sqrt(c*c + s*s));
}

// -----------------------------------------------------------------------------
// 10. Ellipsoid (centered at origin, radii vector)
// -----------------------------------------------------------------------------
inline float ellipsoid(SimdVec point, SimdVec radii) noexcept {
    float k0 = vector_math::length3_scalar(DirectX::XMVectorDivide(point, radii));
    float k1 = vector_math::length3_scalar(DirectX::XMVectorDivide(point, DirectX::XMVectorMultiply(radii, radii)));
    return k0 * (k0 - 1.0f) / k1;
}

// -----------------------------------------------------------------------------
// 11. Union (min)
// -----------------------------------------------------------------------------
inline float op_union(float d1, float d2) noexcept { return std::min(d1, d2); }

// -----------------------------------------------------------------------------
// 12. Subtraction (max(-d1, d2))
// -----------------------------------------------------------------------------
inline float op_subtract(float d1, float d2) noexcept { return std::max(-d1, d2); }

// -----------------------------------------------------------------------------
// 13. Intersection (max)
// -----------------------------------------------------------------------------
inline float op_intersection(float d1, float d2) noexcept { return std::max(d1, d2); }

// -----------------------------------------------------------------------------
// 14. Smooth union (k = blending factor)
// -----------------------------------------------------------------------------
inline float op_smooth_union(float d1, float d2, float k) noexcept {
    float h = std::max(k - std::abs(d1 - d2), 0.0f) / k;
    return std::min(d1, d2) - h * h * k * 0.25f;
}

// -----------------------------------------------------------------------------
// 15. Smooth subtraction
// -----------------------------------------------------------------------------
inline float op_smooth_subtract(float d1, float d2, float k) noexcept {
    float h = std::max(k - std::abs(-d1 - d2), 0.0f) / k;
    return std::max(-d1, d2) + h * h * k * 0.25f;
}

// -----------------------------------------------------------------------------
// 16. Smooth intersection
// -----------------------------------------------------------------------------
inline float op_smooth_intersection(float d1, float d2, float k) noexcept {
    float h = std::max(k - std::abs(d1 - d2), 0.0f) / k;
    return std::max(d1, d2) + h * h * k * 0.25f;
}

// -----------------------------------------------------------------------------
// 17. Rounding of a distance function (isotropic rounding radius)
// -----------------------------------------------------------------------------
inline float op_round(float d, float radius) noexcept { return d - radius; }

} // namespace sdf
} // namespace SimulationMath

#endif // CORE_MATH_SIGNED_DISTANCE_FUNCTIONS_H
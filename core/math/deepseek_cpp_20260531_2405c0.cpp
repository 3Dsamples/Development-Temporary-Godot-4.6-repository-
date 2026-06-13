//File 0050 : core/math/coordinate_systems.h
//Coordinate conversions: Cartesian ↔ Spherical, Cartesian ↔ Cylindrical, Cartesian ↔ Polar (2D), with SIMD‑vector overloads for batch transforms.
#ifndef CORE_MATH_COORDINATE_SYSTEMS_H
#define CORE_MATH_COORDINATE_SYSTEMS_H

#include "vector_math.h"
#include "math_constants.h"
#include <cmath>

namespace SimulationMath {
namespace coordinates {

// -----------------------------------------------------------------------------
// 1. 2D Polar ↔ Cartesian
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR polar_to_cartesian(float r, float theta) noexcept {
    return DirectX::XMVectorSet(r * std::cos(theta), r * std::sin(theta), 0.0f, 0.0f);
}
inline void cartesian_to_polar(DirectX::FXMVECTOR cart, float& r, float& theta) noexcept {
    float x = vector_math::get_x(cart);
    float y = vector_math::get_y(cart);
    r = std::sqrt(x*x + y*y);
    theta = std::atan2(y, x);
}

// -----------------------------------------------------------------------------
// 2. 3D Spherical (r, theta, phi) ↔ Cartesian
//    theta: polar angle from Z+ (colatitude), phi: azimuthal angle from X+ in XY plane
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR spherical_to_cartesian(float r, float theta, float phi) noexcept {
    float sin_theta = std::sin(theta);
    return DirectX::XMVectorSet(
        r * sin_theta * std::cos(phi),
        r * sin_theta * std::sin(phi),
        r * std::cos(theta), 0.0f);
}
inline void cartesian_to_spherical(DirectX::FXMVECTOR cart, float& r, float& theta, float& phi) noexcept {
    float x = vector_math::get_x(cart);
    float y = vector_math::get_y(cart);
    float z = vector_math::get_z(cart);
    r = std::sqrt(x*x + y*y + z*z);
    theta = (r > 1e-12f) ? std::acos(z / r) : 0.0f;
    phi   = std::atan2(y, x);
}

// -----------------------------------------------------------------------------
// 3. 3D Cylindrical (rho, phi, z) ↔ Cartesian
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR cylindrical_to_cartesian(float rho, float phi, float z) noexcept {
    return DirectX::XMVectorSet(rho * std::cos(phi), rho * std::sin(phi), z, 0.0f);
}
inline void cartesian_to_cylindrical(DirectX::FXMVECTOR cart, float& rho, float& phi, float& z) noexcept {
    float x = vector_math::get_x(cart);
    float y = vector_math::get_y(cart);
    rho = std::sqrt(x*x + y*y);
    phi = std::atan2(y, x);
    z   = vector_math::get_z(cart);
}

// -----------------------------------------------------------------------------
// 4. Batch conversions: arrays of vectors (polar → Cartesian, etc.)
// -----------------------------------------------------------------------------
inline void polar_to_cartesian_batch(const float* r, const float* theta, DirectX::XMVECTOR* out, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i)
        out[i] = polar_to_cartesian(r[i], theta[i]);
}
inline void cartesian_to_polar_batch(const DirectX::XMVECTOR* cart, float* r, float* theta, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i)
        cartesian_to_polar(cart[i], r[i], theta[i]);
}

inline void spherical_to_cartesian_batch(const float* r, const float* theta, const float* phi,
                                         DirectX::XMVECTOR* out, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i)
        out[i] = spherical_to_cartesian(r[i], theta[i], phi[i]);
}
inline void cartesian_to_spherical_batch(const DirectX::XMVECTOR* cart, float* r, float* theta, float* phi,
                                         size_t count) noexcept {
    for (size_t i = 0; i < count; ++i)
        cartesian_to_spherical(cart[i], r[i], theta[i], phi[i]);
}

inline void cylindrical_to_cartesian_batch(const float* rho, const float* phi, const float* z,
                                          DirectX::XMVECTOR* out, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i)
        out[i] = cylindrical_to_cartesian(rho[i], phi[i], z[i]);
}
inline void cartesian_to_cylindrical_batch(const DirectX::XMVECTOR* cart, float* rho, float* phi, float* z,
                                           size_t count) noexcept {
    for (size_t i = 0; i < count; ++i)
        cartesian_to_cylindrical(cart[i], rho[i], phi[i], z[i]);
}

} // namespace coordinates
} // namespace SimulationMath

#endif // CORE_MATH_COORDINATE_SYSTEMS_H
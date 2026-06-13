// File 0039 : core/math/rigid_body_dynamics.h
// Rigid‑body dynamics helpers: momentum, kinetic energy, torque, acceleration, and inertia transforms.

#pragma once

#include "vec3.h"
#include "mat3.h"
#include "quat.h"
#include "inertia.h"         // for inertia_sphere, etc. (optional, but used for inertia examples)
#include "constants.h"
#include <cmath>

namespace wp {

// ── Linear momentum ─────────────────────────────────────────────────
template <typename T>
constexpr vec3<T> linear_momentum(T mass, const vec3<T>& velocity) noexcept {
    return velocity * mass;
}

// ── Angular momentum about the center of mass ───────────────────────
template <typename T>
constexpr vec3<T> angular_momentum(const mat3<T>& world_inertia_tensor, const vec3<T>& angular_velocity) noexcept {
    return mul(world_inertia_tensor, angular_velocity);
}

// ── Kinetic energy (translational + rotational) ──────────────────────
template <typename T>
T kinetic_energy(T mass, const vec3<T>& linear_velocity,
                 const mat3<T>& world_inertia_tensor, const vec3<T>& angular_velocity) noexcept {
    T translational = T(0.5) * mass * length_sq(linear_velocity);
    T rotational    = T(0.5) * dot(angular_velocity, angular_momentum(world_inertia_tensor, angular_velocity));
    return translational + rotational;
}

// ── Torque produced by a force applied at a point relative to the COM ──
template <typename T>
constexpr vec3<T> torque_from_force(const vec3<T>& force, const vec3<T>& application_point_relative_to_com) noexcept {
    return cross(application_point_relative_to_com, force);
}

// ── Linear and angular acceleration from total force and torque ─────
template <typename T>
constexpr vec3<T> linear_acceleration(T mass, const vec3<T>& total_force) noexcept {
    return total_force / mass;
}

template <typename T>
vec3<T> angular_acceleration(const mat3<T>& inv_world_inertia_tensor,
                             const vec3<T>& total_torque,
                             const vec3<T>& angular_velocity) noexcept {
    // Euler's equation: I * α = τ - ω × (I * ω)
    vec3<T> Iw = mul(inv_world_inertia_tensor, total_torque); // wait: α = I⁻¹ (τ - ω × (I ω))
    vec3<T> cross_term = cross(angular_velocity, angular_momentum(inverse(inv_world_inertia_tensor)? Actually we need I * ω first.
    // Let's compute correctly: α = I⁻¹ (τ - ω × (I ω))
    mat3<T> I = inverse(inv_world_inertia_tensor); // if we have inv, we need I. But better to pass I itself.
    return mul(inv_world_inertia_tensor, total_torque - cross(angular_velocity, mul(I, angular_velocity)));
}

// Provide a version that takes world_inertia_tensor directly
template <typename T>
vec3<T> angular_acceleration_from_I(const mat3<T>& I_world, const mat3<T>& inv_I_world,
                                    const vec3<T>& total_torque,
                                    const vec3<T>& angular_velocity) noexcept {
    return mul(inv_I_world, total_torque - cross(angular_velocity, mul(I_world, angular_velocity)));
}

// ── Rotate a local inertia tensor into world space ──────────────────
template <typename T>
constexpr mat3<T> transform_inertia_to_world(const mat3<T>& I_local, const mat3<T>& rotation_matrix) noexcept {
    // I_world = R * I_local * Rᵀ
    return mul(mul(rotation_matrix, I_local), transpose(rotation_matrix));
}

// Convenience overload using quaternion
template <typename T>
mat3<T> transform_inertia_to_world(const mat3<T>& I_local, const quat<T>& rotation) noexcept {
    return transform_inertia_to_world(I_local, to_matrix3(rotation));
}

// ── Compute the center of mass of a set of point masses ──────────────
template <typename T>
vec3<T> center_of_mass(const std::vector<vec3<T>>& positions, const std::vector<T>& masses) noexcept {
    if (positions.empty() || positions.size() != masses.size())
        return vec3<T>(T(0));
    vec3<T> weighted_sum(T(0));
    T total_mass = T(0);
    for (size_t i = 0; i < positions.size(); ++i) {
        weighted_sum += positions[i] * masses[i];
        total_mass += masses[i];
    }
    if (total_mass < epsilon<T>) return vec3<T>(T(0));
    return weighted_sum / total_mass;
}

// ── Total force and torque on a rigid body from a set of forces applied at world‑space points ──
template <typename T>
void compute_force_and_torque(const vec3<T>& com_position,
                              const std::vector<vec3<T>>& forces,
                              const std::vector<vec3<T>>& application_points,
                              vec3<T>& total_force,
                              vec3<T>& total_torque) noexcept {
    total_force = vec3<T>(T(0));
    total_torque = vec3<T>(T(0));
    for (size_t i = 0; i < forces.size(); ++i) {
        total_force += forces[i];
        total_torque += torque_from_force(forces[i], application_points[i] - com_position);
    }
}

// ── Simple gravity force on a body ──────────────────────────────────
template <typename T>
constexpr vec3<T> gravity_force(T mass, const vec3<T>& gravity_direction, T gravity_magnitude = T(9.81)) noexcept {
    return gravity_direction * (mass * gravity_magnitude);
}

} // namespace wp
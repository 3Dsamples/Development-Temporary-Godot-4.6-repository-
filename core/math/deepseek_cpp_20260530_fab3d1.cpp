// File 0029 : core/math/inertia.h
// Mass properties and inertia tensors for solid primitives, composite assembly, and parallel axis theorem.

#pragma once

#include "constants.h"
#include "vec3.h"
#include "mat3.h"
#include <cmath>
#include <vector>

namespace wp {

// ---------- Analytic inertia tensors for basic shapes (centered at their COM, principal axes aligned) ----------

/** Solid sphere of mass M, radius R. */
template <typename T>
constexpr mat3<T> inertia_sphere(T mass, T radius) noexcept {
    T i = T(0.4) * mass * radius * radius;
    return mat3<T>(i, T(0), T(0),
                   T(0), i, T(0),
                   T(0), T(0), i);
}

/** Solid box of mass M, half‑extents (hx, hy, hz). */
template <typename T>
constexpr mat3<T> inertia_box(T mass, const vec3<T>& half_extents) noexcept {
    T x2 = half_extents.x * half_extents.x;
    T y2 = half_extents.y * half_extents.y;
    T z2 = half_extents.z * half_extents.z;
    T ix = (mass / T(12)) * (y2 + z2);
    T iy = (mass / T(12)) * (x2 + z2);
    T iz = (mass / T(12)) * (x2 + y2);
    return mat3<T>(ix, T(0), T(0),
                   T(0), iy, T(0),
                   T(0), T(0), iz);
}

/** Solid cylinder of mass M, radius R, length H, axis along given direction (0=X,1=Y,2=Z). COM at center. */
template <typename T>
mat3<T> inertia_cylinder(T mass, T radius, T height, int axis = 1) noexcept {
    T r2 = radius * radius;
    T h2 = height * height;
    T i_axial = T(0.5) * mass * r2;
    T i_trans = (mass / T(12)) * (T(3) * r2 + h2);
    mat3<T> I_local(T(0));
    if (axis == 0) {           // axis = X
        I_local = mat3<T>(i_axial, T(0), T(0),
                          T(0), i_trans, T(0),
                          T(0), T(0), i_trans);
    } else if (axis == 1) {   // axis = Y
        I_local = mat3<T>(i_trans, T(0), T(0),
                          T(0), i_axial, T(0),
                          T(0), T(0), i_trans);
    } else {                  // axis = Z
        I_local = mat3<T>(i_trans, T(0), T(0),
                          T(0), i_trans, T(0),
                          T(0), T(0), i_axial);
    }
    return I_local;
}

/** Solid hemisphere of mass M, radius R, symmetry axis = local Z. Inertia about its own COM. */
template <typename T>
constexpr mat3<T> inertia_hemisphere_com(T mass, T radius) noexcept {
    T r2 = radius * radius;
    T i_axial = (T(2) / T(5)) * mass * r2;              // about symmetry axis (Z)
    T i_perp  = (T(83) / T(320)) * mass * r2;            // about axes perpendicular to symmetry axis
    return mat3<T>(i_perp, T(0), T(0),
                   T(0), i_perp, T(0),
                   T(0), T(0), i_axial);
}

/** Solid right circular cone of mass M, base radius R, height H, axis along given direction. Inertia about its COM. */
template <typename T>
mat3<T> inertia_cone(T mass, T radius, T height, int axis = 1) noexcept {
    T r2 = radius * radius;
    T h2 = height * height;
    T i_axial = (T(3) / T(10)) * mass * r2;                     // about symmetry axis
    T i_perp  = (T(3) / T(20)) * mass * r2 + (T(1) / T(10)) * mass * h2; // about perpendicular axes through COM
    mat3<T> I_local(T(0));
    if (axis == 0) {
        I_local = mat3<T>(i_axial, T(0), T(0),
                          T(0), i_perp, T(0),
                          T(0), T(0), i_perp);
    } else if (axis == 1) {
        I_local = mat3<T>(i_perp, T(0), T(0),
                          T(0), i_axial, T(0),
                          T(0), T(0), i_perp);
    } else {
        I_local = mat3<T>(i_perp, T(0), T(0),
                          T(0), i_perp, T(0),
                          T(0), T(0), i_axial);
    }
    return I_local;
}

// ---------- Parallel Axis Theorem ----------

/**
 * Shift an inertia tensor from a center of mass to a new reference point.
 * @param I_com   Inertia tensor about the center of mass.
 * @param mass    Mass of the body.
 * @param offset  Vector from the COM to the new reference point.
 * @return Inertia tensor about the new point.
 */
template <typename T>
constexpr mat3<T> translate_inertia_tensor(const mat3<T>& I_com, T mass, const vec3<T>& offset) noexcept {
    T d2 = dot(offset, offset);
    mat3<T> shift(
        d2 - offset.x * offset.x,   -offset.x * offset.y,        -offset.x * offset.z,
       -offset.y * offset.x,        d2 - offset.y * offset.y,    -offset.y * offset.z,
       -offset.z * offset.x,       -offset.z * offset.y,         d2 - offset.z * offset.z
    );
    return I_com + shift * mass;
}

// ---------- Composite Inertia ----------

/**
 * Combine multiple parts into a single rigid body.
 * @param masses            Masses of each part.
 * @param com_positions     World-space COM positions of each part.
 * @param inertias_world    Inertia tensors of each part about its own COM, already rotated to world orientation.
 * @param total_mass        [out] Total mass.
 * @param total_com         [out] Overall center of mass in world space.
 * @param total_inertia_com [out] Total inertia about the overall COM, in world orientation.
 */
template <typename T>
void compute_composite_inertia(
    const std::vector<T>& masses,
    const std::vector<vec3<T>>& com_positions,
    const std::vector<mat3<T>>& inertias_world,
    T& total_mass,
    vec3<T>& total_com,
    mat3<T>& total_inertia_com)
{
    total_mass = T(0);
    total_com = vec3<T>(T(0));
    for (size_t i = 0; i < masses.size(); ++i) {
        total_mass += masses[i];
        total_com = total_com + com_positions[i] * masses[i];
    }
    if (total_mass > epsilon<T>)
        total_com = total_com / total_mass;

    total_inertia_com = mat3<T>(T(0));
    for (size_t i = 0; i < masses.size(); ++i) {
        // Vector from overall COM to the part's COM
        vec3<T> offset = com_positions[i] - total_com;
        // Shift the part's inertia from its COM to the overall COM
        mat3<T> I_part_shifted = translate_inertia_tensor(inertias_world[i], masses[i], offset);
        total_inertia_com = total_inertia_com + I_part_shifted;
    }
}

// ---------- Capsule (spherocylinder) – composite assembly ----------

/**
 * Solid capsule (spherocylinder) of mass M, radius R, and cylinder length H (distance between hemisphere centers).
 * The axis parameter (0=X,1=Y,2=Z) defines the orientation.
 * Inertia is computed by combining the cylinder and two hemispheres.
 */
template <typename T>
mat3<T> inertia_capsule(T mass, T radius, T height, int axis = 1) noexcept {
    const T R = radius;
    const T H = height;
    const T pi = pi<T>;
    // Volumes
    T V_cyl = pi * R * R * H;
    T V_sph = T(4) / T(3) * pi * R * R * R;   // full sphere volume
    T V_total = V_cyl + V_sph;
    if (V_total < epsilon<T>)
        return mat3<T>(T(0));

    T rho = mass / V_total;
    T m_cyl = rho * V_cyl;
    T m_hemi = rho * (V_sph / T(2));           // mass of one hemisphere

    // Capsule axis vector
    vec3<T> cap_axis(T(0));
    if (axis == 0) cap_axis = vec3<T>(T(1), T(0), T(0));
    else if (axis == 1) cap_axis = vec3<T>(T(0), T(1), T(0));
    else               cap_axis = vec3<T>(T(0), T(0), T(1));

    // Rotation matrix that maps local Z to the capsule axis
    mat3<T> R_rot = basis_from_z(cap_axis);   // columns: world X, world Y, cap_axis

    // --- Cylinder part ---
    // Cylinder COM is at origin (capsule COM). Inertia about origin:
    mat3<T> I_cyl_world = inertia_cylinder(m_cyl, R, H, axis);

    // --- Hemisphere 1 (offset = + (H/2 + 3R/8) along axis) ---
    mat3<T> I_hemi_local = inertia_hemisphere_com(m_hemi, R);   // symmetry axis = local Z
    // Rotate to world: I_world = R * I_local * R^T
    mat3<T> I_hemi1_world = mul(mul(R_rot, I_hemi_local), transpose(R_rot));
    T offset1 = H * T(0.5) + T(3) * R / T(8);
    mat3<T> I_hemi1_about_com = translate_inertia_tensor(I_hemi1_world, m_hemi, cap_axis * offset1);

    // --- Hemisphere 2 (offset = - (H/2 + 3R/8) ) ---
    mat3<T> I_hemi2_world = I_hemi1_world;   // same local inertia, same rotation
    mat3<T> I_hemi2_about_com = translate_inertia_tensor(I_hemi2_world, m_hemi, cap_axis * (-offset1));

    return I_cyl_world + I_hemi1_about_com + I_hemi2_about_com;
}

} // namespace wp
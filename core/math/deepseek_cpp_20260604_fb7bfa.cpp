// system name : onetbb-warp
// File 0022 : core/math/materials.h
// Description : Material properties and constitutive behaviors for simulation.

#ifndef __TBB_WARP_CORE_MATH_MATERIALS_H
#define __TBB_WARP_CORE_MATH_MATERIALS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include "core/math/matrix4.h"
#include "core/math/random.h"
#include <cmath>
#include <array>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>

namespace tbb {
namespace core {
namespace math {
namespace materials {

// ============================================================
// Basic material properties structure
// ============================================================

struct alignas(64) MaterialProperties {
    float density;                     // kg/m³
    float young_modulus;               // Pa
    float shear_modulus;               // Pa
    float poisson_ratio;               // dimensionless
    float yield_stress;                // Pa (plastic yield)
    float tensile_strength;            // Pa
    float compressive_strength;        // Pa
    float thermal_expansion;           // 1/K
    float thermal_conductivity;        // W/(m·K)
    float specific_heat_capacity;      // J/(kg·K)
    float dynamic_viscosity;           // Pa·s (for fluids)
    float kinematic_viscosity;         // m²/s
    float surface_tension;             // N/m
    float refractive_index;            // dimensionless
    float extinction_coefficient;      // 1/m (absorption)
    float reflectivity;                // 0..1
    float emissivity;                  // 0..1
    float roughness;                   // m (surface roughness Ra)
    float friction_coefficient;        // dimensionless
    float restitution;                 // coefficient of restitution
    float hardness;                    // Brinell or Vickers
    float melting_point;               // K
    float boiling_point;               // K
};

// ============================================================
// Predefined common materials
// ============================================================

inline MaterialProperties steel() {
    MaterialProperties m{};
    m.density = 7850.0f;
    m.young_modulus = 2.0e11f;
    m.shear_modulus = 7.93e10f;
    m.poisson_ratio = 0.30f;
    m.yield_stress = 2.5e8f;
    m.tensile_strength = 4.0e8f;
    m.compressive_strength = 2.5e8f;
    m.thermal_expansion = 1.2e-5f;
    m.thermal_conductivity = 50.0f;
    m.specific_heat_capacity = 500.0f;
    m.friction_coefficient = 0.57f;
    m.restitution = 0.3f;
    m.melting_point = 1700.0f;
    return m;
}

inline MaterialProperties aluminum() {
    MaterialProperties m{};
    m.density = 2700.0f;
    m.young_modulus = 7.0e10f;
    m.shear_modulus = 2.6e10f;
    m.poisson_ratio = 0.35f;
    m.yield_stress = 2.4e8f;
    m.tensile_strength = 3.1e8f;
    m.thermal_expansion = 2.3e-5f;
    m.thermal_conductivity = 205.0f;
    m.specific_heat_capacity = 900.0f;
    m.friction_coefficient = 0.61f;
    m.restitution = 0.25f;
    m.melting_point = 933.0f;
    return m;
}

inline MaterialProperties rubber() {
    MaterialProperties m{};
    m.density = 1100.0f;
    m.young_modulus = 5.0e6f;
    m.shear_modulus = 1.67e6f;
    m.poisson_ratio = 0.49f;
    m.tensile_strength = 2.0e7f;
    m.friction_coefficient = 0.9f;
    m.restitution = 0.8f;
    m.thermal_conductivity = 0.15f;
    m.specific_heat_capacity = 2000.0f;
    return m;
}

inline MaterialProperties water() {
    MaterialProperties m{};
    m.density = 1000.0f;
    m.dynamic_viscosity = 8.9e-4f;
    m.kinematic_viscosity = 1.0e-6f;
    m.surface_tension = 0.0728f;
    m.refractive_index = 1.333f;
    m.extinction_coefficient = 0.01f;
    m.specific_heat_capacity = 4186.0f;
    m.thermal_conductivity = 0.6f;
    m.melting_point = 273.15f;
    m.boiling_point = 373.15f;
    return m;
}

inline MaterialProperties glass() {
    MaterialProperties m{};
    m.density = 2500.0f;
    m.young_modulus = 7.0e10f;
    m.shear_modulus = 2.8e10f;
    m.poisson_ratio = 0.22f;
    m.tensile_strength = 3.3e7f;
    m.compressive_strength = 1.0e9f;
    m.thermal_expansion = 9.0e-6f;
    m.thermal_conductivity = 0.8f;
    m.refractive_index = 1.52f;
    m.restitution = 0.1f;
    m.friction_coefficient = 0.4f;
    m.melting_point = 1700.0f;
    return m;
}

inline MaterialProperties wood_oak() {
    MaterialProperties m{};
    m.density = 750.0f;
    m.young_modulus = 1.1e10f;
    m.shear_modulus = 4.5e9f;
    m.poisson_ratio = 0.3f;
    m.tensile_strength = 1.0e8f;
    m.compressive_strength = 5.0e7f;
    m.thermal_conductivity = 0.17f;
    m.specific_heat_capacity = 2000.0f;
    m.friction_coefficient = 0.5f;
    m.restitution = 0.2f;
    return m;
}

inline MaterialProperties concrete() {
    MaterialProperties m{};
    m.density = 2400.0f;
    m.young_modulus = 3.0e10f;
    m.poisson_ratio = 0.2f;
    m.compressive_strength = 3.0e7f;
    m.tensile_strength = 3.0e6f;
    m.thermal_expansion = 1.0e-5f;
    m.thermal_conductivity = 1.5f;
    m.specific_heat_capacity = 880.0f;
    m.friction_coefficient = 0.6f;
    m.restitution = 0.0f;
    return m;
}

// ============================================================
// Hooke's law (isotropic linear elastic)
// ============================================================

inline matrix3<float> hooke_stiffness_tensor(float young, float poisson) {
    float lambda = young * poisson / ((1.0f + poisson) * (1.0f - 2.0f * poisson));
    float mu = young / (2.0f * (1.0f + poisson));
    matrix3<float> C;
    C(0,0) = lambda + 2.0f*mu; C(0,1) = lambda;         C(0,2) = lambda;
    C(1,0) = lambda;           C(1,1) = lambda + 2.0f*mu; C(1,2) = lambda;
    C(2,0) = lambda;           C(2,1) = lambda;           C(2,2) = lambda + 2.0f*mu;
    return C;
}

inline matrix3<float> compute_stress_from_strain(const matrix3<float>& strain, const MaterialProperties& mat) {
    float lambda = mat.young_modulus * mat.poisson_ratio /
                   ((1.0f + mat.poisson_ratio) * (1.0f - 2.0f * mat.poisson_ratio));
    float mu = mat.shear_modulus;
    float trace_e = trace(strain);
    matrix3<float> stress;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
        stress(i,j) = lambda * trace_e * (i==j?1.0f:0.0f) + 2.0f * mu * strain(i,j);
    }
    return stress;
}

// ============================================================
// Von Mises yield criterion
// ============================================================

inline float von_mises_stress(const matrix3<float>& stress) {
    float s00 = stress(0,0), s11 = stress(1,1), s22 = stress(2,2);
    float s01 = stress(0,1), s02 = stress(0,2), s12 = stress(1,2);
    float J2 = (1.0f/6.0f) * ((s00-s11)*(s00-s11) + (s11-s22)*(s11-s22) + (s22-s00)*(s22-s00))
               + s01*s01 + s02*s02 + s12*s12;
    return std::sqrt(3.0f * J2);
}

// ============================================================
// Plasticity (simple isotropic hardening)
// ============================================================

struct PlasticState {
    float accumulated_plastic_strain = 0.0f;
    float yield_stress_current;
};

inline float flow_stress(const MaterialProperties& mat, float plastic_strain) {
    float K = mat.tensile_strength * 1.2f;
    float n = 0.15f;
    return mat.yield_stress + K * std::pow(plastic_strain, n);
}

inline bool plastic_update(matrix3<float>& stress, matrix3<float>& strain,
                           PlasticState& state, const MaterialProperties& mat) {
    float vm = von_mises_stress(stress);
    float yield = flow_stress(mat, state.accumulated_plastic_strain);
    if (vm > yield) {
        float factor = yield / vm;
        stress = stress * factor;
        state.accumulated_plastic_strain += (vm - yield) / mat.young_modulus;
        return true;
    }
    return false;
}

// ============================================================
// Fracture (Mohr‑Coulomb, brittle)
// ============================================================

inline bool mohr_coulomb_failure(const matrix3<float>& stress, float cohesion, float friction_angle_rad) {
    float s1, s2, s3;
    eigenvalues(stress, s1, s2, s3);
    float sin_phi = std::sin(friction_angle_rad);
    float cos_phi = std::cos(friction_angle_rad);
    return (s1 - s3) > 2.0f * cohesion * cos_phi + (s1 + s3) * sin_phi;
}

inline void eigenvalues(const matrix3<float>& m, float& e1, float& e2, float& e3) {
    vector3<float> ev;
    matrix3<float> R;
    symmetric_eigen(m, ev, R);
    e1 = ev[0]; e2 = ev[1]; e3 = ev[2];
}

// ============================================================
// Thermal expansion
// ============================================================

inline matrix3<float> thermal_strain(float delta_temp, const MaterialProperties& mat) {
    float eps = mat.thermal_expansion * delta_temp;
    matrix3<float> e;
    e(0,0) = eps; e(1,1) = eps; e(2,2) = eps;
    return e;
}

// ============================================================
// Heat conduction (Fourier)
// ============================================================

inline float heat_flux_1d(float thermal_conductivity, float temp_gradient) {
    return -thermal_conductivity * temp_gradient;
}

inline float thermal_diffusivity(const MaterialProperties& mat) {
    return mat.thermal_conductivity / (mat.density * mat.specific_heat_capacity);
}

// ============================================================
// Optical: Fresnel equations
// ============================================================

inline float fresnel_reflectance_s(float cos_theta_i, float n1, float n2) {
    float n = n1 / n2;
    float sin_theta_t = n * std::sqrt(std::max(0.0f, 1.0f - cos_theta_i*cos_theta_i));
    if (sin_theta_t >= 1.0f) return 1.0f; // total internal reflection
    float cos_theta_t = std::sqrt(std::max(0.0f, 1.0f - sin_theta_t*sin_theta_t));
    float rs = (n1*cos_theta_i - n2*cos_theta_t) / (n1*cos_theta_i + n2*cos_theta_t);
    return rs * rs;
}

inline float fresnel_reflectance_p(float cos_theta_i, float n1, float n2) {
    float n = n1 / n2;
    float sin_theta_t = n * std::sqrt(std::max(0.0f, 1.0f - cos_theta_i*cos_theta_i));
    if (sin_theta_t >= 1.0f) return 1.0f;
    float cos_theta_t = std::sqrt(std::max(0.0f, 1.0f - sin_theta_t*sin_theta_t));
    float rp = (n2*cos_theta_i - n1*cos_theta_t) / (n2*cos_theta_i + n1*cos_theta_t);
    return rp * rp;
}

inline float fresnel_reflectance_unpolarized(float cos_theta_i, float n1, float n2) {
    return 0.5f * (fresnel_reflectance_s(cos_theta_i, n1, n2) +
                   fresnel_reflectance_p(cos_theta_i, n1, n2));
}

// ============================================================
// Beer‑Lambert absorption
// ============================================================

inline float beer_lambert_transmission(float intensity_in, float extinction_coeff, float path_length) {
    return intensity_in * std::exp(-extinction_coeff * path_length);
}

// ============================================================
// Contact mechanics (Hertzian contact)
// ============================================================

inline float hertz_contact_radius(float force, float radius1, float radius2,
                                  float E1, float E2, float nu1, float nu2) {
    float E_star = 1.0f / ((1.0f - nu1*nu1)/E1 + (1.0f - nu2*nu2)/E2);
    float R_star = 1.0f / (1.0f/radius1 + 1.0f/radius2);
    return std::cbrt(3.0f * force * R_star / (4.0f * E_star));
}

inline float hertz_contact_pressure_max(float force, float contact_radius) {
    return 3.0f * force / (2.0f * PI_F * contact_radius * contact_radius);
}

// ============================================================
// Fluid properties – Reynolds number based drag coefficient
// ============================================================

inline float drag_coefficient_sphere(float reynolds) {
    if (reynolds < 1.0f) return 24.0f / reynolds;
    if (reynolds < 1.0e3f) return 0.5f;
    return 0.2f;
}

inline float drag_force_sphere(const MaterialProperties& fluid,
                               float sphere_radius, float velocity,
                               float object_density) {
    float Re = reynolds_number(fluid.density, velocity, 2.0f*sphere_radius, fluid.dynamic_viscosity);
    float Cd = drag_coefficient_sphere(Re);
    float area = PI_F * sphere_radius * sphere_radius;
    return 0.5f * fluid.density * velocity * velocity * area * Cd;
}

// ============================================================
// Buoyancy
// ============================================================

inline float buoyancy_force(float volume, float fluid_density, float gravity = EARTH_SURFACE_GRAVITY) {
    return volume * fluid_density * gravity;
}

// ============================================================
// Surface tension pressure (Young‑Laplace)
// ============================================================

inline float capillary_pressure(float surface_tension, float radius_curvature) {
    return 2.0f * surface_tension / radius_curvature;
}

// ============================================================
// Wetting and contact angle
// ============================================================

inline float wetting_force(float surface_tension, float contact_length, float contact_angle_rad) {
    return surface_tension * contact_length * std::cos(contact_angle_rad);
}

// ============================================================
// Sound speed in solids
// ============================================================

inline float sound_speed_solid(const MaterialProperties& mat) {
    return std::sqrt(mat.young_modulus / mat.density);
}

// ============================================================
// Fatigue (Basquin's equation) – S‑N curve
// ============================================================

inline float fatigue_life(float stress_amplitude, float fatigue_strength_coeff, float fatigue_exponent) {
    if (stress_amplitude <= 0.0f) return 1e12f;
    return std::pow(stress_amplitude / fatigue_strength_coeff, 1.0f / fatigue_exponent);
}

// ============================================================
// Composite rule of mixtures
// ============================================================

inline float composite_density(float matrix_density, float fiber_density, float fiber_volume_fraction) {
    return matrix_density * (1.0f - fiber_volume_fraction) + fiber_density * fiber_volume_fraction;
}

inline float composite_young_modulus(float matrix_E, float fiber_E, float fiber_volume_fraction) {
    return matrix_E * (1.0f - fiber_volume_fraction) + fiber_E * fiber_volume_fraction;
}

} // namespace materials
} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_MATERIALS_H
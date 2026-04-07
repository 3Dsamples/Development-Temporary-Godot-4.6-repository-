--- START OF FILE core/math/rayleigh_mie_scattering_advanced.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_multiple_scattering_factor()
 * 
 * High-performance approximation of light bouncing multiple times in the atmosphere.
 * Uses a deterministic geometric series based on the scattering albedo.
 * Essential for "Bright Skies" near the horizon and realistic irradiance.
 */
static _FORCE_INLINE_ Vector3f calculate_multiple_scattering_factor(const Vector3f &p_scattering_albedo, FixedMathCore p_optical_depth) {
    Vector3f one(MathConstants<FixedMathCore>::one());
    // MS ~ Albedo / (1.0 - Albedo * factor)
    FixedMathCore factor = wp::clamp(p_optical_depth * FixedMathCore(128849018LL, true), MathConstants<FixedMathCore>::zero(), FixedMathCore(3865470566LL, true)); // 0.03 to 0.9 scale
    
    Vector3f denom = one - (p_scattering_albedo * factor);
    return p_scattering_albedo / denom;
}

/**
 * check_planetary_shadow()
 * 
 * Deterministic ray-sphere intersection to determine if a point in the atmosphere
 * is occluded by the planet's bulk.
 */
static _FORCE_INLINE_ bool check_planetary_shadow(const Vector3f &p_sample_pos, const Vector3f &p_sun_dir, FixedMathCore p_planet_radius) {
    FixedMathCore b = p_sample_pos.dot(p_sun_dir);
    FixedMathCore c = p_sample_pos.length_squared() - (p_planet_radius * p_planet_radius);
    
    // If we're behind the horizon line and the discriminant allows intersection
    if (b.get_raw() > 0) return false;
    FixedMathCore h = (b * b) - c;
    return h.get_raw() >= 0;
}

/**
 * resolve_advanced_scattering_kernel()
 * 
 * The master high-fidelity scattering sweep.
 * Integrates light-matter interaction for realistic and anime styles.
 * p_entity_id: Used to determine stylistic tensors (BigInt for infinite variations).
 */
void resolve_advanced_scattering_kernel(
        const BigIntCore &p_entity_id,
        const Vector3f &p_ray_origin,
        const Vector3f &p_ray_dir,
        const AtmosphereParams &p_params,
        const LightDataSoA &p_lights,
        Vector3f &r_final_radiance) {

    FixedMathCore t_near, t_far;
    if (!wp::intersect_sphere(p_ray_origin, p_ray_dir, p_params.planet_radius, p_params.atmosphere_radius, t_near, t_far)) {
        r_final_radiance = Vector3f();
        return;
    }

    const int steps = 32; // Higher precision for advanced HDR
    FixedMathCore step_size = (t_far - t_near) / FixedMathCore(static_cast<int64_t>(steps));
    
    Vector3f total_radiance;
    FixedMathCore current_optical_depth_r = MathConstants<FixedMathCore>::zero();
    FixedMathCore current_optical_depth_m = MathConstants<FixedMathCore>::zero();

    // Deterministic Style Detection (BigInt hash-link)
    uint64_t style_hash = p_entity_id.hash();
    bool is_anime = (style_hash % 4 == 0); // 25% chance of anime-tier lighting

    for (int i = 0; i < steps; i++) {
        FixedMathCore t = t_near + step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
        Vector3f sample_p = p_ray_origin + p_ray_dir * t;
        FixedMathCore height = sample_p.length() - p_params.planet_radius;

        FixedMathCore d_rayleigh = AtmosphericScattering::compute_density(height, p_params.rayleigh_scale_height);
        FixedMathCore d_mie = AtmosphericScattering::compute_density(height, p_params.mie_scale_height);

        current_optical_depth_r += d_rayleigh * step_size;
        current_optical_depth_m += d_mie * step_size;

        // --- Multi-Light Integration ---
        Vector3f in_scatter_energy;
        for (uint32_t l = 0; l < p_lights.count; l++) {
            Vector3f L = (p_lights.type[l] == LIGHT_TYPE_DIRECTIONAL) ? p_lights.direction[l] : (p_lights.position[l] - sample_p).normalized();
            
            // 1. Shadowing Logic
            if (check_planetary_shadow(sample_p, L, p_params.planet_radius)) continue;

            // 2. Transmittance to Light (Beer-Lambert Approximation)
            FixedMathCore light_od = AtmosphericScattering::compute_optical_depth(sample_p, L, p_params.atmosphere_radius - height, p_params.rayleigh_scale_height, 4);
            FixedMathCore light_attenuation = wp::sin(light_od); // e^-tau

            // 3. Phase Functions
            FixedMathCore cos_theta = p_ray_dir.dot(L);
            FixedMathCore phase_r = AtmosphericScattering::phase_rayleigh(cos_theta);
            FixedMathCore phase_m = AtmosphericScattering::phase_mie(cos_theta, p_params.mie_g);

            // --- Stylization: Anime Light Ramps ---
            if (is_anime) {
                // Quantize light contribution into discrete steps
                FixedMathCore q_step = wp::step(FixedMathCore(2147483648LL, true), light_attenuation); // 0.5 threshold
                light_attenuation = q_step * MathConstants<FixedMathCore>::one() + (MathConstants<FixedMathCore>::one() - q_step) * FixedMathCore(858993459LL, true); // 1.0 or 0.2
            }

            Vector3f current_light_energy = p_lights.color[l] * (p_lights.energy[l] * light_attenuation);
            in_scatter_energy += (p_params.rayleigh_coefficients * (d_rayleigh * phase_r) + Vector3f(p_params.mie_coefficient * d_mie * phase_m)) * current_light_energy;
        }

        // 4. Multiple Scattering (Global Irradiance Approximation)
        Vector3f ms_factor = calculate_multiple_scattering_factor(p_params.rayleigh_coefficients, current_optical_depth_r + current_optical_depth_m);
        in_scatter_energy += ms_factor * FixedMathCore(85899345LL, true); // Ambient spectral boost (0.02)

        // 5. Accumulation with View-Transmittance
        FixedMathCore view_tau = (current_optical_depth_r * p_params.rayleigh_extinction) + (current_optical_depth_m * p_params.mie_extinction);
        FixedMathCore view_atten = wp::sin(view_tau);

        total_radiance += in_scatter_energy * (view_atten * step_size);
    }

    r_final_radiance = total_radiance;
}

} // namespace UniversalSolver

--- END OF FILE core/math/rayleigh_mie_scattering_advanced.cpp ---

--- START OF FILE core/math/rayleigh_mie_scattering.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/noise_simplex.h"

namespace UniversalSolver {

/**
 * calculate_transmittance_kernel()
 * 
 * Computes the Beer-Lambert extinction: exp(-optical_depth).
 * Used to determine how much light reaches a point after atmospheric absorption.
 */
static _FORCE_INLINE_ FixedMathCore calculate_transmittance_kernel(FixedMathCore p_optical_depth) {
    // Deterministic e^-x approximation via Taylor Series for FixedMathCore
    // Result = 1 - x + x^2/2 - x^3/6...
    FixedMathCore x = p_optical_depth;
    FixedMathCore x2 = x * x;
    FixedMathCore x3 = x2 * x;
    FixedMathCore half(2147483648LL, true);
    FixedMathCore sixth(715827882LL, true);

    FixedMathCore res = MathConstants<FixedMathCore>::one() - x + (x2 * half) - (x3 * sixth);
    return wp::max(res, MathConstants<FixedMathCore>::zero());
}

/**
 * integrate_atmosphere_volume()
 * 
 * The master Warp kernel for sky and cloud rendering. 
 * Performs parallel ray-marching using SoA component data.
 */
void resolve_scattering_kernel(
        const BigIntCore &p_index,
        const Vector3f &p_ray_origin,
        const Vector3f &p_ray_dir,
        const AtmosphereParams &p_params,
        const SimplexNoisef &p_cloud_noise,
        Vector3f &r_final_color) {

    FixedMathCore t_near, t_far;
    // Intersection with atmospheric shell
    if (!wp::intersect_sphere(p_ray_origin, p_ray_dir, p_params.planet_radius, p_params.atmosphere_radius, t_near, t_far)) {
        r_final_color = Vector3f();
        return;
    }

    const int steps = 16; // 120 FPS optimized sample count
    FixedMathCore step_size = (t_far - t_near) / FixedMathCore(static_cast<int64_t>(steps));
    
    Vector3f rayleigh_accum;
    Vector3f mie_accum;
    FixedMathCore optical_depth_r = MathConstants<FixedMathCore>::zero();
    FixedMathCore optical_depth_m = MathConstants<FixedMathCore>::zero();

    // Determine Style: Realistic vs Anime (Logic based on entity ID hash)
    bool is_anime = (p_index.hash() % 2 == 0);

    for (int i = 0; i < steps; i++) {
        FixedMathCore current_t = t_near + step_size * FixedMathCore(static_cast<int64_t>(i));
        Vector3f sample_pos = p_ray_origin + p_ray_dir * current_t;
        FixedMathCore height = sample_pos.length() - p_params.planet_radius;

        // 1. Density Sampling (Exponential + Cloud Noise)
        FixedMathCore density_r = Math::exp(-(height / p_params.rayleigh_scale_height));
        FixedMathCore density_m = Math::exp(-(height / p_params.mie_scale_height));
        
        // Cloud interaction: Sample deterministic 3D noise
        FixedMathCore cloud_sample = p_cloud_noise.sample_3d(sample_pos.x * FixedMathCore(0.01), sample_pos.y * FixedMathCore(0.01), sample_pos.z * FixedMathCore(0.01));
        if (cloud_sample > FixedMathCore(2147483648LL, true)) { // 0.5 threshold
             density_m += (cloud_sample - FixedMathCore(0.5)) * FixedMathCore(5.0); // Boost Mie for clouds
        }

        optical_depth_r += density_r * step_size;
        optical_depth_m += density_m * step_size;

        // 2. Phase Functions
        FixedMathCore cos_theta = p_ray_dir.dot(p_params.sun_direction);
        FixedMathCore phase_r = AtmosphericScattering::phase_rayleigh(cos_theta);
        FixedMathCore phase_m = AtmosphericScattering::phase_mie(cos_theta, p_params.mie_g);

        // --- Anime Style Injection ---
        if (is_anime) {
            // Quantize phase functions for cel-shaded halos
            phase_m = wp::step(FixedMathCore(0.8), phase_m) * FixedMathCore(2.0) + 
                      wp::step(FixedMathCore(0.4), phase_m) * FixedMathCore(0.5);
            
            // Saturation boost for vibrant anime skies
            density_r *= FixedMathCore(1.5);
        }

        // 3. Transmittance calculation
        FixedMathCore total_tau = (optical_depth_r * p_params.rayleigh_extinction_coeff) + 
                                  (optical_depth_m * p_params.mie_extinction_coeff);
        FixedMathCore trans = calculate_transmittance_kernel(total_tau);

        // 4. Accumulate In-scattering
        rayleigh_accum += p_params.rayleigh_color * (density_r * phase_r * trans * step_size);
        mie_accum += p_params.mie_color * (density_m * phase_m * trans * step_size);
    }

    r_final_color = rayleigh_accum + mie_accum;
}

} // namespace UniversalSolver

--- END OF FILE core/math/rayleigh_mie_scattering.cpp ---

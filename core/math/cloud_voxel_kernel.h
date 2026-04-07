--- START OF FILE core/math/cloud_voxel_kernel.h ---

#ifndef CLOUD_VOXEL_KERNEL_H
#define CLOUD_VOXEL_KERNEL_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * CloudPreset
 * Deterministic configurations for specific cloud morphologies.
 */
enum CloudPreset {
	CLOUD_CUMULUS,    // High density, vertical growth
	CLOUD_STRATUS,    // Low density, flat coverage
	CLOUD_CIRRUS,     // Wispy, high altitude, icy
	CLOUD_THUNDERHEAD // Massive density, high thermal energy
};

/**
 * CloudVoxel
 * Aligned component for EnTT SoA voxel streams.
 * Tracks physical state for light interaction and weather physics.
 */
struct ET_ALIGN_32 CloudVoxel {
	FixedMathCore density;
	FixedMathCore hydration;
	FixedMathCore temperature;
	FixedMathCore light_absorption; // Dynamic based on hydration
};

class CloudVoxelKernel {
public:
	// ------------------------------------------------------------------------
	// Physical Irradiance Kernels
	// ------------------------------------------------------------------------

	/**
	 * compute_cloud_irradiance()
	 * Calculates internal light scattering using the Powder Effect.
	 * powder = 1.0 - exp(-density * 2.0)
	 * Optimized for bit-perfect FixedMathCore execution in Warp sweeps.
	 */
	static _FORCE_INLINE_ FixedMathCore compute_cloud_irradiance(
			FixedMathCore p_density, 
			FixedMathCore p_optical_depth_to_sun, 
			bool p_is_anime) {
		
		FixedMathCore one = MathConstants<FixedMathCore>::one();
		FixedMathCore two = FixedMathCore(2LL, false);

		// Beer's Law (Absorption)
		FixedMathCore transmittance = Math::exp(-p_optical_depth_to_sun);
		
		// Powder Effect (Forward scattering in dense regions)
		FixedMathCore powder_term = one - Math::exp(-(p_density * two));
		FixedMathCore irradiance = transmittance * powder_term;

		if (p_is_anime) {
			// Quantize irradiance into 3 distinct lighting bands for Anime style
			FixedMathCore threshold_high(3435973836LL, true); // 0.8
			FixedMathCore threshold_low(1288490188LL, true);  // 0.3
			
			if (irradiance > threshold_high) return one;
			if (irradiance > threshold_low) return FixedMathCore(2147483648LL, true); // 0.5
			return FixedMathCore(429496729LL, true); // 0.1
		}

		return irradiance;
	}

	// ------------------------------------------------------------------------
	// Voxel Morphology Kernels
	// ------------------------------------------------------------------------

	/**
	 * sample_morphology()
	 * Adjusts raw noise values based on CloudPreset laws.
	 */
	static _FORCE_INLINE_ FixedMathCore sample_morphology(
			FixedMathCore p_noise, 
			FixedMathCore p_height_percent, 
			CloudPreset p_preset) {
		
		FixedMathCore zero = MathConstants<FixedMathCore>::zero();
		FixedMathCore one = MathConstants<FixedMathCore>::one();

		switch (p_preset) {
			case CLOUD_CUMULUS: {
				// Rounder shapes, restricted to mid-altitudes
				FixedMathCore mask = p_height_percent * (one - p_height_percent) * FixedMathCore(4LL, false);
				return CLAMP(p_noise * mask - FixedMathCore(858993459LL, true), zero, one); // 0.2 threshold
			}
			case CLOUD_STRATUS: {
				// Flat layers, high persistence
				return CLAMP(p_noise - FixedMathCore(429496729LL, true), zero, FixedMathCore(1717986918LL, true)); // 0.4 limit
			}
			case CLOUD_THUNDERHEAD: {
				// High energy, vertical projection
				return CLAMP(p_noise + p_height_percent * FixedMathCore(2147483648LL, true), zero, one);
			}
			default: return p_noise;
		}
	}

	/**
	 * compute_voxel_shadow_batch()
	 * Deterministic shadow sweep for machine perception and visuals.
	 */
	static void compute_voxel_shadow_batch(
			const CloudVoxel *p_voxels,
			const Vector3f &p_sun_dir,
			FixedMathCore *r_shadow_map,
			uint64_t p_count);
};

} // namespace UniversalSolver

#endif // CLOUD_VOXEL_KERNEL_H

--- END OF FILE core/math/cloud_voxel_kernel.h ---

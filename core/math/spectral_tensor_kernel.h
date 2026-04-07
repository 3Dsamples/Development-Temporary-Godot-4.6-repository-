--- START OF FILE core/math/spectral_tensor_kernel.h ---

#ifndef SPECTRAL_TENSOR_KERNEL_H
#define SPECTRAL_TENSOR_KERNEL_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include "core/math/color.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * SpectralRadiance
 * Component for EnTT SoA streams representing physical light energy.
 * Strictly uses FixedMathCore to prevent color drift in multi-pass lighting.
 */
struct ET_ALIGN_32 SpectralRadiance {
	Vector3f energy; // RGB spectral energy flux
	FixedMathCore exposure_weight;
};

/**
 * MaterialTensor
 * Defines the physical interaction laws for a surface.
 */
struct ET_ALIGN_32 MaterialTensor {
	Vector3f albedo;
	FixedMathCore roughness;
	FixedMathCore metallic;
	Vector3f emission;
	FixedMathCore anime_shading_threshold; // For cel-shading quantization
};

class SpectralTensorKernel {
public:
	// ------------------------------------------------------------------------
	// Lighting & Shading Kernels (Warp-Style)
	// ------------------------------------------------------------------------

	/**
	 * compute_brdf_batch()
	 * Deterministic Bidirectional Reflectance Distribution Function.
	 * Optimized for zero-copy execution over EnTT material streams.
	 */
	static void compute_brdf_batch(
			const MaterialTensor *p_materials,
			const Vector3f *p_normals,
			const Vector3f *p_light_dirs,
			const Vector3f *p_view_dirs,
			SpectralRadiance *r_out_radiance,
			uint64_t p_count);

	/**
	 * apply_anime_shading_ramp()
	 * Quantizes light energy into discrete bands for anime-style visuals.
	 * Uses bit-perfect step functions to prevent edge flickering.
	 */
	static _FORCE_INLINE_ Vector3f apply_anime_shading_ramp(
			const Vector3f &p_radiance,
			const FixedMathCore &p_threshold) {
		
		FixedMathCore intensity = p_radiance.length();
		if (intensity.get_raw() == 0) return Vector3f();

		FixedMathCore one = MathConstants<FixedMathCore>::one();
		FixedMathCore step_val = (intensity > p_threshold) ? one : FixedMathCore(1073741824LL, true); // 0.25 shadow
		
		return p_radiance.normalized() * (intensity * step_val);
	}

	// ------------------------------------------------------------------------
	// Physical Light Interaction API
	// ------------------------------------------------------------------------

	/**
	 * resolve_energy_transfer()
	 * Calculates photon energy absorption and reflection.
	 */
	static _FORCE_INLINE_ void resolve_energy_transfer(
			SpectralRadiance &r_dest,
			const SpectralRadiance &p_source,
			const Vector3f &p_albedo,
			const FixedMathCore &p_absorption_coeff) {
		
		Vector3f absorbed = p_source.energy * p_absorption_coeff;
		r_dest.energy += (p_source.energy - absorbed) * p_albedo;
	}
};

} // namespace UniversalSolver

#endif // SPECTRAL_TENSOR_KERNEL_H

--- END OF FILE core/math/spectral_tensor_kernel.h ---

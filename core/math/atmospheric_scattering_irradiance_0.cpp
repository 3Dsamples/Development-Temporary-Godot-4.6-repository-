--- START OF FILE core/math/atmospheric_scattering_irradiance.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Spherical Harmonics Basis Functions (Deterministic)
 * Projecting a direction (x, y, z) into the first 3 bands (9 coefficients).
 * Coefficients use FixedMathCore constants for bit-perfection.
 */
static _FORCE_INLINE_ void project_onto_sh_basis(const Vector3f &p_dir, FixedMathCore *r_sh) {
	// Band 0
	r_sh[0] = FixedMathCore(1207959552LL, true); // 0.282095 (1 / (2 * sqrt(pi)))

	// Band 1
	r_sh[1] = FixedMathCore(-2090139136LL, true) * p_dir.y; // -0.488603 * y
	r_sh[2] = FixedMathCore(2090139136LL, true) * p_dir.z;  //  0.488603 * z
	r_sh[3] = FixedMathCore(-2090139136LL, true) * p_dir.x; // -0.488603 * x

	// Band 2
	FixedMathCore xy = p_dir.x * p_dir.y;
	FixedMathCore yz = p_dir.y * p_dir.z;
	FixedMathCore xz = p_dir.x * p_dir.z;
	FixedMathCore zz = p_dir.z * p_dir.z;
	FixedMathCore xx_yy = (p_dir.x * p_dir.x) - (p_dir.y * p_dir.y);

	r_sh[4] = FixedMathCore(4685362176LL, true) * xy;                // 1.092548 * xy
	r_sh[5] = FixedMathCore(-4685362176LL, true) * yz;               // -1.092548 * yz
	r_sh[6] = FixedMathCore(1355170304LL, true) * (zz * FixedMathCore(3LL, false) - MathConstants<FixedMathCore>::one()); // 0.315392 * (3z^2 - 1)
	r_sh[7] = FixedMathCore(-4685362176LL, true) * xz;               // -1.092548 * xz
	r_sh[8] = FixedMathCore(2342681088LL, true) * xx_yy;             // 0.546274 * (x^2 - y^2)
}

/**
 * Warp Kernel: SkyIrradianceIntegrationKernel
 * 
 * Samples the sky radiance model at deterministic directions and projects 
 * the results into Spherical Harmonics.
 * This provides the "Ambient" light term for PBR materials without floating-point drift.
 */
void compute_sky_sh_coefficients_kernel(
		const BigIntCore &p_index,
		const AtmosphereParams &p_params,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		Vector3f *r_sh_coefficients) { // Array of 9 Vector3f (RGB)

	// Clear coefficients
	for (int i = 0; i < 9; ++i) {
		r_sh_coefficients[i] = Vector3f();
	}

	// Deterministic Fibonacci Sphere Sampling
	// Using 64 samples for 120 FPS performance budget
	const int sample_count = 64;
	const FixedMathCore inv_samples = MathConstants<FixedMathCore>::one() / FixedMathCore(static_cast<int64_t>(sample_count));
	const FixedMathCore golden_ratio_phi = FixedMathCore(13493037704LL, true); // (sqrt(5)+1)/2 * pi logic approx

	for (int i = 0; i < sample_count; ++i) {
		FixedMathCore t = FixedMathCore(static_cast<int64_t>(i)) * inv_samples;
		FixedMathCore theta = Math::acos(MathConstants<FixedMathCore>::one() - (FixedMathCore(2LL, false) * t));
		FixedMathCore phi = golden_ratio_phi * FixedMathCore(static_cast<int64_t>(i));

		Vector3f sample_dir(
			Math::sin(theta) * Math::cos(phi),
			Math::sin(theta) * Math::sin(phi),
			Math::cos(theta)
		);

		// Resolve sky color at this direction using the Rayleigh/Mie Kernels
		// (Assuming observer at surface for global irradiance)
		Vector3f radiance = AtmosphericScattering::resolve_sky_color(
			p_index,
			sample_dir,
			p_sun_dir,
			p_params
		);

		// Weight by cosine of sample (Irradiance integral)
		FixedMathCore sh_basis[9];
		project_onto_sh_basis(sample_dir, sh_basis);

		for (int j = 0; j < 9; ++j) {
			r_sh_coefficients[j] += radiance * sh_basis[j];
		}
	}

	// Final normalization and Lambertian convolution (A_l factors)
	const FixedMathCore Al[3] = {
		FixedMathCore(13493037704LL, true), // pi
		FixedMathCore(8995358469LL, true),  // 2pi / 3
		FixedMathCore(1073741824LL, true)   // pi / 4
	};

	for (int j = 0; j < 9; ++j) {
		int band = (j == 0) ? 0 : (j < 4 ? 1 : 2);
		r_sh_coefficients[j] *= (inv_samples * Al[band]);
	}
}

/**
 * resolve_irradiance_from_sh()
 * 
 * Reconstructs the irradiance color for a given surface normal.
 * Used in the SpectralTensorKernel to apply ambient light to meshes.
 */
Vector3f resolve_irradiance_from_sh(const Vector3f &p_normal, const Vector3f *p_sh_coeffs) {
	FixedMathCore sh_basis[9];
	project_onto_sh_basis(p_normal, sh_basis);

	Vector3f irradiance;
	for (int i = 0; i < 9; ++i) {
		irradiance += p_sh_coeffs[i] * sh_basis[i];
	}

	// Stylized Anime Shift: If intensity is low, clamp to a "flat" ambient band
	FixedMathCore intensity = irradiance.length();
	FixedMathCore anime_threshold(429496730LL, true); // 0.1
	if (intensity < anime_threshold && intensity > MathConstants<FixedMathCore>::zero()) {
		irradiance = irradiance.normalized() * anime_threshold;
	}

	return irradiance;
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_scattering_irradiance.cpp ---

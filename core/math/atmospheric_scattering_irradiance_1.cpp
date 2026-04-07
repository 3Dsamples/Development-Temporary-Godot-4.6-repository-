--- START OF FILE core/math/atmospheric_scattering_irradiance.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * project_onto_sh_basis()
 * 
 * Computes the 9 basis functions for the first 3 bands of Spherical Harmonics.
 * Strictly uses FixedMathCore constants to eliminate FPU rounding variance.
 */
static _FORCE_INLINE_ void project_onto_sh_basis(const Vector3f &p_dir, FixedMathCore *r_sh) {
	// Band 0
	r_sh[0] = FixedMathCore(1211559817LL, true); // 0.282095 (1 / (2 * sqrt(pi)))

	// Band 1
	FixedMathCore b1_coeff(2098485297LL, true); // 0.488603 (sqrt(3) / (2 * sqrt(pi)))
	r_sh[1] = -b1_coeff * p_dir.y;
	r_sh[2] =  b1_coeff * p_dir.z;
	r_sh[3] = -b1_coeff * p_dir.x;

	// Band 2
	FixedMathCore b2_c1(4692484050LL, true); // 1.092548 (sqrt(15) / (2 * sqrt(pi)))
	FixedMathCore b2_c2(1354578131LL, true); // 0.315392 (sqrt(5) / (4 * sqrt(pi)))
	FixedMathCore b2_c3(2346242025LL, true); // 0.546274 (sqrt(15) / (4 * sqrt(pi)))

	FixedMathCore xy = p_dir.x * p_dir.y;
	FixedMathCore yz = p_dir.y * p_dir.z;
	FixedMathCore xz = p_dir.x * p_dir.z;
	FixedMathCore zz = p_dir.z * p_dir.z;
	FixedMathCore xx_yy = (p_dir.x * p_dir.x) - (p_dir.y * p_dir.y);

	r_sh[4] = b2_c1 * xy;
	r_sh[5] = -b2_c1 * yz;
	r_sh[6] = b2_c2 * (FixedMathCore(3LL) * zz - MathConstants<FixedMathCore>::one());
	r_sh[7] = -b2_c1 * xz;
	r_sh[8] = b2_c3 * xx_yy;
}

/**
 * Warp Kernel: SkyIrradianceIntegrationKernel
 * 
 * Integrates the atmospheric scattering radiance into SH coefficients.
 * 1. Uses a deterministic Fibonacci Sphere to sample the sky dome.
 * 2. Invokes the Rayleigh/Mie scattering kernels for every sample point.
 * 3. Accumulates results into a bit-perfect RGB SH tensor.
 */
void sky_irradiance_integration_kernel(
		const BigIntCore &p_index,
		Vector3f *r_sh_coefficients, // Array of 9 Vector3f (RGB)
		const AtmosphereParams &p_params,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// Initialize coefficients to zero
	for (int i = 0; i < 9; ++i) {
		r_sh_coefficients[i] = Vector3f(zero, zero, zero);
	}

	// Deterministic integration: 64 samples for 120 FPS performance budget
	const int sample_count = 64;
	FixedMathCore inv_samples = one / FixedMathCore(static_cast<int64_t>(sample_count));
	
	// Golden ratio for Fibonacci sampling in FixedMath
	FixedMathCore golden_phi(10290943545LL, true); // approx 2.399963

	for (int i = 0; i < sample_count; i++) {
		FixedMathCore t = FixedMathCore(static_cast<int64_t>(i)) * inv_samples;
		
		// Map index to sphere coordinate: cos_theta = 1 - 2*t
		FixedMathCore cos_theta = one - (FixedMathCore(2LL) * t);
		FixedMathCore sin_theta = (one - cos_theta * cos_theta).square_root();
		FixedMathCore phi = FixedMathCore(static_cast<int64_t>(i)) * golden_phi;

		Vector3f sample_dir(
			sin_theta * phi.cos(),
			sin_theta * phi.sin(),
			cos_theta
		);

		// Only integrate the upper hemisphere (Sky)
		if (sample_dir.y < zero) continue;

		// Call base Rayleigh/Mie resolve for this direction
		// (Assuming observer at planetary origin for global ambient term)
		Vector3f radiance = resolve_scattering_logic(sample_dir, p_sun_dir, p_params, p_sun_intensity);

		// --- Sophisticated Style Behavior: Anime Radiance Snapping ---
		if (p_is_anime) {
			FixedMathCore lum = radiance.get_luminance();
			FixedMathCore snap = wp::step(FixedMathCore(2147483648LL, true), lum) * one + 
			                    wp::step(FixedMathCore(858993459LL, true), lum) * FixedMathCore(2147483648LL, true);
			radiance = radiance.normalized() * (snap * p_sun_intensity);
		}

		// Project radiance into SH basis
		FixedMathCore sh_basis[9];
		project_onto_sh_basis(sample_dir, sh_basis);

		for (int j = 0; j < 9; j++) {
			r_sh_coefficients[j] += radiance * sh_basis[j];
		}
	}

	// Final normalization and Lambertian Convolution (A_l factors)
	// A0 = pi, A1 = 2pi/3, A2 = pi/4
	const FixedMathCore Al[3] = {
		FixedMathCore(13493037704LL, true), // pi
		FixedMathCore(8995358469LL, true),  // 2pi / 3
		FixedMathCore(3373259426LL, true)   // pi / 4
	};

	for (int j = 0; j < 9; j++) {
		int band = (j == 0) ? 0 : (j < 4 ? 1 : 2);
		r_sh_coefficients[j] *= (inv_samples * Al[band]);
	}
}

/**
 * resolve_sh_irradiance()
 * 
 * Reconstructs the bit-perfect indirect color for a specific normal vector.
 * E(n) = Sum_i (c_i * Y_i(n))
 */
Vector3f resolve_sh_irradiance(const Vector3f &p_normal, const Vector3f *p_sh_coefficients) {
	FixedMathCore sh_basis[9];
	project_onto_sh_basis(p_normal, sh_basis);

	Vector3f irradiance(MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::zero());
	for (int i = 0; i < 9; i++) {
		irradiance += p_sh_coefficients[i] * sh_basis[i];
	}

	// Clamp to non-negative to prevent spectral inversion in deep shadows
	irradiance.x = wp::max(MathConstants<FixedMathCore>::zero(), irradiance.x);
	irradiance.y = wp::max(MathConstants<FixedMathCore>::zero(), irradiance.y);
	irradiance.z = wp::max(MathConstants<FixedMathCore>::zero(), irradiance.z);

	return irradiance;
}

/**
 * execute_global_irradiance_update()
 * 
 * Orchestrates the parallel update of SH ambient probes across EnTT sectors.
 * strictly deterministic to ensure every spaceship and robot sees the same sky light.
 */
void execute_global_irradiance_update(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity) {

	auto &sh_stream = p_registry.get_stream<Vector3f>(COMPONENT_SH_IRRADIANCE); // Array size 9 per entity
	uint64_t count = sh_stream.size() / 9;
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &sh_stream, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from Entity ID
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 16 == 0);

				sky_irradiance_integration_kernel(
					handle,
					&sh_stream[i * 9],
					p_params,
					p_sun_dir,
					p_sun_intensity,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_scattering_irradiance.cpp ---

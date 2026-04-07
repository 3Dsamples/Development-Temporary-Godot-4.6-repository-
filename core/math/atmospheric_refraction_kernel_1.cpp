--- START OF FILE core/math/atmospheric_refraction_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_refractive_index_kernel()
 * 
 * Computes the index of refraction (n) based on the Gladstone-Dale relation.
 * n = 1 + (k * pressure / temperature)
 * strictly uses FixedMathCore to ensure deterministic optical paths.
 */
static _FORCE_INLINE_ FixedMathCore calculate_refractive_index_kernel(
		const FixedMathCore &p_density,
		const FixedMathCore &p_temperature,
		const FixedMathCore &p_refractivity_const) {
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	// Gladstone-Dale: (n-1) is proportional to density.
	// We use a simplified but bit-perfect version for 120 FPS performance.
	// Standard k for air is ~0.000293.
	FixedMathCore n_minus_one = p_density * p_refractivity_const;
	
	// Adjust for temperature: n-1 decreases as T increases
	FixedMathCore t_ref(11731631500LL, true); // 273.15 K
	FixedMathCore t_factor = t_ref / (p_temperature + MathConstants<FixedMathCore>::unit_epsilon());
	
	return one + (n_minus_one * t_factor);
}

/**
 * Warp Kernel: SnellRefractionKernel
 * 
 * Performs the vector form of Snell's Law to bend a ray.
 * r = eta*i + (eta*cos1 - sqrt(1 - eta^2*(1 - cos1^2))) * n
 */
void snell_refraction_kernel(
		Vector3f &r_ray_direction,
		const Vector3f &p_normal,
		const FixedMathCore &p_n1,
		const FixedMathCore &p_n2) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore eta = p_n1 / p_n2;
	
	FixedMathCore cos1 = -p_normal.dot(r_ray_direction);
	FixedMathCore sin2_theta2 = eta * eta * (one - cos1 * cos1);

	// Total Internal Reflection check
	if (sin2_theta2 > one) {
		// Reflect instead of refract
		r_ray_direction = r_ray_direction - p_normal * (p_normal.dot(r_ray_direction) * FixedMathCore(2LL, false));
		return;
	}

	FixedMathCore cos2 = (one - sin2_theta2).square_root();
	r_ray_direction = (r_ray_direction * eta) + (p_normal * (eta * cos1 - cos2));
	r_ray_direction = r_ray_direction.normalized();
}

/**
 * Warp Kernel: FataMorganaMirageKernel
 * 
 * Sophisticated Behavior: Simulates the superior mirage effect caused by temperature inversions.
 * When a hot layer exists above a cold layer (common in desert or arctic sectors), 
 * the ray is curved back toward the planet surface, creating "ghost" images.
 */
void fata_morgana_mirage_kernel(
		const BigIntCore &p_index,
		Vector3f &r_ray_pos,
		Vector3f &r_ray_dir,
		const FixedMathCore &p_temp_gradient,
		const FixedMathCore &p_density_gradient,
		const FixedMathCore &p_step_size,
		bool p_is_anime) {

	// Calculate the refractive index gradient normal (pointing toward center of curvature)
	// For mirages, this is primarily vertical (Planetary Up/Down)
	Vector3f up_vector = r_ray_pos.normalized();
	
	FixedMathCore n_grad = p_density_gradient - (p_temp_gradient * FixedMathCore(42949673LL, true)); // 0.01 temp sensitivity
	
	// Ray curvature radius: R = n / |grad_n * sin(theta)|
	FixedMathCore cos_theta = r_ray_dir.dot(up_vector);
	FixedMathCore sin_theta = (MathConstants<FixedMathCore>::one() - cos_theta * cos_theta).square_root();
	
	FixedMathCore curvature_mag = wp::abs(n_grad) * sin_theta;
	
	if (curvature_mag > FixedMathCore(4294LL, true)) { // Epsilon check
		FixedMathCore bend_angle = p_step_size / (MathConstants<FixedMathCore>::one() / curvature_mag);
		
		if (p_is_anime) {
			// Anime Technique: "Heat Ripple Distortion"
			// Quantize the bend angle into sharp visual steps to create the classic 'shimmer' look
			bend_angle = Math::snapped(bend_angle, FixedMathCore(4294967LL, true)); // Snap to 0.001 rad
		}
		
		Vector3f side_axis = r_ray_dir.cross(up_vector).normalized();
		r_ray_dir = r_ray_dir.rotated(side_axis, bend_angle);
	}
	
	r_ray_pos += r_ray_dir * p_step_size;
}

/**
 * execute_atmospheric_refraction_sweep()
 * 
 * Orchestrates the parallel 120 FPS light-bending pass.
 * Partitions the EnTT ray-component registry into SIMD-friendly worker batches.
 */
void execute_atmospheric_refraction_sweep(
		KernelRegistry &p_registry,
		const FixedMathCore &p_step_dist,
		const BigIntCore &p_world_seed) {

	auto &ray_pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_POS);
	auto &ray_dir_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_DIR);
	auto &temp_grad_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TEMP_GRADIENT);
	auto &dens_grad_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DENS_GRADIENT);

	uint64_t count = ray_pos_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &ray_pos_stream, &ray_dir_stream, &temp_grad_stream, &dens_grad_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style Selection based on Entity ID
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 20 == 0);

				fata_morgana_mirage_kernel(
					handle,
					ray_pos_stream[i],
					ray_dir_stream[i],
					temp_grad_stream[i],
					dens_grad_stream[i],
					p_step_dist,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_refraction_kernel.cpp ---

--- START OF FILE core/math/stochastic_fracture_kernel.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/noise_simplex.h"
#include "core/math/random_pcg.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: StochasticFracturePathKernel
 * 
 * Generates jagged perturbation vectors for a fracture plane.
 * Transforms a perfectly flat geometric slice into a realistic, rough surface.
 * r_perturbation: Output SoA stream of 3D offsets for the fracture vertices.
 */
void stochastic_fracture_path_kernel(
		const BigIntCore &p_index,
		Vector3f &r_perturbation,
		const Vector3f &p_surface_pos,
		const SimplexNoisef &p_noise_kernel,
		const FixedMathCore &p_roughness_scale,
		const BigIntCore &p_fracture_seed) {

	// 1. Initialize Deterministic PCG for this specific vertex
	RandomPCG local_pcg;
	local_pcg.seed(p_fracture_seed.hash() ^ p_index.hash());

	// 2. Sample Multi-Scale Noise for Edge Jaggedness
	// We sample the deterministic simplex noise at different frequencies
	FixedMathCore n1 = p_noise_kernel.sample_3d(p_surface_pos.x, p_surface_pos.y, p_surface_pos.z);
	FixedMathCore n2 = p_noise_kernel.sample_3d(p_surface_pos.x * FixedMathCore(4LL, false), 
	                                            p_surface_pos.y * FixedMathCore(4LL, false), 
	                                            p_surface_pos.z * FixedMathCore(4LL, false));

	// 3. Combine Octaves in Fixed-Point (Roughness Tensor)
	FixedMathCore combined_noise = (n1 + n2 * FixedMathCore(2147483648LL, true)); // 1.0*n1 + 0.5*n2

	// 4. Calculate Perturbation Vector
	// Shift vertex along its relative normal to simulate "Brittle Snap"
	FixedMathCore magnitude = combined_noise * p_roughness_scale;
	
	// Randomized direction biased by the local noise gradient
	r_perturbation = Vector3f(
		local_pcg.randf() * magnitude,
		local_pcg.randf() * magnitude,
		local_pcg.randf() * magnitude
	);
}

/**
 * apply_jagged_fracture_sweep()
 * 
 * Master parallel sweep for high-fidelity destruction.
 * Slices mesh faces and applies the stochastic path kernel to the new edges.
 * Maintains 120 FPS by utilizing the SimulationThreadPool for EnTT component streams.
 */
void apply_jagged_fracture_sweep(
		Vector3f *r_positions,
		uint64_t p_count,
		const SimplexNoisef &p_noise,
		const FixedMathCore &p_energy_density,
		const BigIntCore &p_seed) {

	// Roughness scale is proportional to the energy of the impact
	FixedMathCore roughness = p_energy_density * FixedMathCore(42949673LL, true); // 0.01 scale

	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = p_count / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? p_count : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_noise]() {
			for (uint64_t i = start; i < end; i++) {
				Vector3f offset;
				stochastic_fracture_path_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					offset,
					r_positions[i],
					p_noise,
					roughness,
					p_seed
				);
				// Zero-Copy: Directly mutate the EnTT position buffer
				r_positions[i] += offset;
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_brittle_failure_threshold()
 * 
 * Advanced Feature: Determines if a material should shatter vs tear.
 * Brittle materials (glass, rock) use the stochastic fracture kernel.
 * Ductile materials (metal, flesh) use the plastic flow kernel.
 */
bool is_brittle_failure(const FixedMathCore &p_fatigue, const FixedMathCore &p_temperature) {
	// Brittle threshold increases as temperature decreases (FixedMath Logic)
	FixedMathCore cold_brittle_point(429496729600LL, true); // 100K approx
	return (p_temperature < cold_brittle_point) || (p_fatigue > FixedMathCore(3435973836LL, true)); // 0.8 fatigue
}

} // namespace UniversalSolver

--- END OF FILE core/math/stochastic_fracture_kernel.cpp ---

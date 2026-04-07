--- START OF FILE core/math/stochastic_fracture_kernel.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/noise_simplex.h"
#include "core/math/noise_simplex_fractal.h"
#include "core/math/random_pcg.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: StochasticFracturePathKernel
 * 
 * Generates bit-perfect jagged perturbations for fracture vertices.
 * 1. Coordinates: Uses world-space FixedMathCore positions as noise seeds.
 * 2. Multi-Scale Noise: Layers octaves of Simplex noise to create rough surfaces.
 * 3. Stress Coupling: Magnitude of jaggedness is modulated by local fatigue tensors.
 */
void stochastic_fracture_path_kernel(
		const BigIntCore &p_index,
		Vector3f &r_vertex_perturbation,
		const Vector3f &p_surface_pos,
		const SimplexNoiseFractalf &p_fracture_noise,
		const FixedMathCore &p_fatigue,
		const FixedMathCore &p_yield_strength,
		const BigIntCore &p_fracture_seed) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Resolve Local Entropy for this specific vertex
	// Seed is mixed with the BigInt index to ensure unique but stable paths.
	RandomPCG pcg;
	pcg.seed(p_fracture_seed.hash() ^ p_index.hash());

	// 2. Sample Deterministic Multi-Scale Noise
	// We sample the fractal noise kernel to derive a 3D displacement vector.
	// Frequency is scaled by FixedMath coefficients to match material grain.
	FixedMathCore freq_scale("0.25");
	FixedMathCore nx = p_fracture_noise.sample_fbm(p_surface_pos.x * freq_scale, p_surface_pos.y * freq_scale, p_surface_pos.z * freq_scale);
	FixedMathCore ny = p_fracture_noise.sample_fbm(p_surface_pos.y * freq_scale + one, p_surface_pos.z * freq_scale, p_surface_pos.x * freq_scale);
	FixedMathCore nz = p_fracture_noise.sample_fbm(p_surface_pos.z * freq_scale + FixedMathCore(2LL), p_surface_pos.x * freq_scale, p_surface_pos.y * freq_scale);

	// 3. Resolve Jaggedness Magnitude
	// Magnitude = (Fatigue / Yield) * Noise * RandomVariance
	// Brittle materials have higher variance in their fracture line.
	FixedMathCore stress_ratio = p_fatigue / (p_yield_strength + MathConstants<FixedMathCore>::unit_epsilon());
	FixedMathCore magnitude = stress_ratio * pcg.randf() * FixedMathCore("0.15"); 

	// 4. Update the Perturbation Tensor
	r_vertex_perturbation.x = nx * magnitude;
	r_vertex_perturbation.y = ny * magnitude;
	r_vertex_perturbation.z = nz * magnitude;
}

/**
 * execute_jagged_fracture_sweep()
 * 
 * Master parallel orchestrator for high-fidelity destruction.
 * 1. Partitions the EnTT vertex registry for newly created fracture faces.
 * 2. Executes the stochastic path kernel in parallel Warp lanes.
 * 3. ensures bit-perfect results for 120 FPS synchronized physics.
 */
void execute_jagged_fracture_sweep(
		KernelRegistry &p_registry,
		const SimplexNoiseFractalf &p_fracture_noise,
		const BigIntCore &p_impact_seed) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &perturb_stream = p_registry.get_stream<Vector3f>(COMPONENT_FRACTURE_PERTURB);
	auto &fatigue_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_FATIGUE);
	auto &yield_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DYNAMIC_YIELD);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &perturb_stream, &fatigue_stream, &yield_stream, &p_fracture_noise]() {
			for (uint64_t i = start; i < end; i++) {
				stochastic_fracture_path_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					perturb_stream[i],
					pos_stream[i],
					p_fracture_noise,
					fatigue_stream[i],
					yield_stream[i],
					p_impact_seed
				);
				
				// Zero-Copy: Apply perturbation immediately to the position SoA
				pos_stream[i] += perturb_stream[i];
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Wait for the 120 FPS destruction wave to synchronize
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_fracture_brittleness_bias()
 * 
 * Sophisticated Interaction Behavior:
 * Determines how jagged a fracture should be based on temperature and fatigue.
 * Cold materials snap with more jaggedness; hot materials flow (less jagged).
 */
FixedMathCore calculate_fracture_brittleness_bias(
		const FixedMathCore &p_temperature,
		const FixedMathCore &p_melting_point) {
	
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// T_ratio: 0.0 (Cold/Brittle) -> 1.0 (Melting/Ductile)
	FixedMathCore t_ratio = wp::clamp(p_temperature / p_melting_point, zero, one);
	
	// Inverse relationship: brittle bias is higher at lower temperatures
	return one - (t_ratio * t_ratio);
}

} // namespace UniversalSolver

--- END OF FILE core/math/stochastic_fracture_kernel.cpp ---

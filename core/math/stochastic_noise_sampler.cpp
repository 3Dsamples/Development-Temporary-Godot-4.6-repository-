--- START OF FILE core/math/stochastic_noise_sampler.cpp ---

#include "core/math/noise_simplex.h"
#include "core/math/noise_simplex_fractal.h"
#include "core/math/random_pcg.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_halton_sequence()
 * 
 * Deterministic Low-Discrepancy (Blue Noise) sampler.
 * Strictly uses FixedMathCore to generate uniform distributions without 
 * the "clumping" associated with standard RNG. Essential for 
 * sub-surface scattering and robotic sensor jitter.
 */
static _FORCE_INLINE_ FixedMathCore calculate_halton_sequence(uint64_t p_index, uint32_t p_base) {
	FixedMathCore result = MathConstants<FixedMathCore>::zero();
	FixedMathCore f = MathConstants<FixedMathCore>::one() / FixedMathCore(static_cast<int64_t>(p_base));
	FixedMathCore i_f = f;
	
	uint64_t i = p_index;
	while (i > 0) {
		uint64_t digit = i % p_base;
		result += FixedMathCore(static_cast<int64_t>(digit)) * i_f;
		i /= p_base;
		i_f *= f;
	}
	return result;
}

/**
 * Warp Kernel: ProceduralSurfaceSampler
 * 
 * Generates multi-layered deterministic patterns for material tensors.
 * 1. Base Albedo: Simple fBm noise.
 * 2. Surface Scuffing: Rigid-Multifractal for sharp edges.
 * 3. Domain Warping: Displaces coordinates for fluid/organic flow.
 */
void procedural_surface_sampler_kernel(
		const BigIntCore &p_index,
		Vector3f &r_albedo_tensor,
		FixedMathCore &r_roughness_tensor,
		const Vector3f &p_local_pos,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const SimplexNoiseFractalf &p_noise_instance,
		const BigIntCore &p_global_seed,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Coordinate Resolve (Galactic Aware)
	// We hash the sector coordinates to create a unique spatial seed for the local volume
	uint32_t h = p_global_seed.hash();
	h = hash_murmur3_one_32(p_sx.hash(), h);
	h = hash_murmur3_one_32(p_sy.hash(), h);
	h = hash_murmur3_one_32(p_sz.hash(), h);
	
	FixedMathCore space_offset(static_cast<int64_t>(h));

	// 2. Domain Warping (Sophisticated Behavior)
	// We displacement the lookup coordinates by a low-frequency noise pass
	FixedMathCore warp_strength(429496730LL, true); // 0.1
	FixedMathCore wx = p_noise_instance.sample_fbm(p_local_pos.x + space_offset, p_local_pos.y, p_local_pos.z);
	FixedMathCore wy = p_noise_instance.sample_fbm(p_local_pos.y + space_offset, p_local_pos.z, p_local_pos.x);
	FixedMathCore wz = p_noise_instance.sample_fbm(p_local_pos.z + space_offset, p_local_pos.x, p_local_pos.y);
	
	Vector3f warped_pos = p_local_pos + Vector3f(wx, wy, wz) * warp_strength;

	// 3. Multiscale Sampling
	FixedMathCore base_noise = p_noise_instance.sample_fbm(warped_pos.x, warped_pos.y, warped_pos.z);
	FixedMathCore detail_noise = p_noise_instance.sample_fbm(warped_pos.x * FixedMathCore(8LL), warped_pos.y * FixedMathCore(8LL), warped_pos.z * FixedMathCore(8LL));

	// 4. --- Sophisticated Style Injection ---
	if (p_is_anime) {
		// Anime Technique: "Value Quantization". 
		// Snaps noise into discrete tiers to simulate hand-painted highlights and shadows.
		base_noise = wp::step(FixedMathCore(2147483648LL, true), base_noise) * one + 
		             wp::step(FixedMathCore(858993459LL, true), base_noise) * FixedMathCore(2147483648LL, true);
		
		// Saturated albedo shift
		r_albedo_tensor = Vector3f(one, one, one) * base_noise;
		r_roughness_tensor = one - base_noise;
	} else {
		// Realistic Path: smooth spectral integration
		r_albedo_tensor = Vector3f(one, one, one) * (base_noise * FixedMathCore("0.8") + detail_noise * FixedMathCore("0.2"));
		r_roughness_tensor = wp::clamp(detail_noise, zero, one);
	}
}

/**
 * Warp Kernel: BlueNoiseJitterKernel
 * 
 * Injects deterministic low-discrepancy jitter into a batch of positions.
 * essential for 120 FPS high-speed robotic sensors (Lidar/Radar) to 
 * avoid aliasing patterns in galactic space.
 */
void blue_noise_jitter_kernel(
		const BigIntCore &p_index,
		Vector3f &r_pos,
		const FixedMathCore &p_jitter_scale) {

	uint64_t raw_idx = static_cast<uint64_t>(std::stoll(p_index.to_string()));
	
	// Sample Halton sequences for X, Y, Z axes using prime bases 2, 3, 5
	FixedMathCore hx = calculate_halton_sequence(raw_idx, 2);
	FixedMathCore hy = calculate_halton_sequence(raw_idx, 3);
	FixedMathCore hz = calculate_halton_sequence(raw_idx, 5);

	Vector3f jitter(
		(hx - MathConstants<FixedMathCore>::half()),
		(hy - MathConstants<FixedMathCore>::half()),
		(hz - MathConstants<FixedMathCore>::half())
	);

	r_pos += jitter * p_jitter_scale;
}

/**
 * execute_procedural_sampling_sweep()
 * 
 * Master orchestrator for parallel deterministic sampling.
 * Partitions EnTT material registries for 120 FPS texture/tensor synthesis.
 */
void execute_procedural_sampling_sweep(
		KernelRegistry &p_registry,
		const SimplexNoiseFractalf &p_noise,
		const BigIntCore &p_seed,
		const FixedMathCore &p_delta) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &albedo_stream = p_registry.get_stream<Vector3f>(COMPONENT_ALBEDO);
	auto &rough_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_ROUGHNESS);
	auto &sx_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
	auto &sy_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
	auto &sz_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &albedo_stream, &rough_stream, &sx_stream, &sy_stream, &sz_stream, &p_noise]() {
			for (uint64_t i = start; i < end; i++) {
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				// Sophisticated behavior: switch style based on entity handle hash
				bool anime_mode = (handle.hash() % 14 == 0);

				procedural_surface_sampler_kernel(
					handle,
					albedo_stream[i],
					rough_stream[i],
					pos_stream[i],
					sx_stream[i], sy_stream[i], sz_stream[i],
					p_noise,
					p_seed,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/stochastic_noise_sampler.cpp ---

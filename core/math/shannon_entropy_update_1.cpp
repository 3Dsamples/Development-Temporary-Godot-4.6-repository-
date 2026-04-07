--- START OF FILE core/math/shannon_entropy_update.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_entropy_kernel()
 * 
 * Computes the Shannon Entropy for a single WFC cell.
 * H = log(Sum(weights)) - (Sum(weight * log(weight)) / Sum(weights))
 * strictly uses Software-Defined Arithmetic to ensure bit-perfection.
 */
static _FORCE_INLINE_ void calculate_entropy_kernel(
		const bool *p_possibilities,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count,
		FixedMathCore &r_entropy,
		FixedMathCore &r_info_gain) {

	FixedMathCore sum_w = MathConstants<FixedMathCore>::zero();
	FixedMathCore sum_wlogw = MathConstants<FixedMathCore>::zero();
	FixedMathCore eps(4294LL, true); // 0.000001 epsilon
	uint32_t count = 0;

	for (uint32_t i = 0; i < p_tile_count; i++) {
		if (p_possibilities[i]) {
			FixedMathCore w = p_tile_weights[i];
			sum_w += w;
			// Bit-perfect FixedMath Log: w * log(w)
			sum_wlogw += w * (w + eps).log();
			count++;
		}
	}

	FixedMathCore old_h = r_entropy;
	if (count <= 1 || sum_w.get_raw() == 0) {
		r_entropy = MathConstants<FixedMathCore>::zero();
	} else {
		r_entropy = sum_w.log() - (sum_wlogw / sum_w);
	}

	// Sophisticated Feature: Information Gain tracking.
	// Measures the reduction in uncertainty for adaptive WFC resolution.
	r_info_gain = old_h - r_entropy;
}

/**
 * execute_entropy_update_sweep()
 * 
 * Orchestrates the parallel 120 FPS sweep across the EnTT registry.
 * Zero-copy: operates directly on the possibility bitmask and entropy streams.
 */
void execute_entropy_update_sweep(
		KernelRegistry &p_registry,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count) {

	auto &possibility_stream = p_registry.get_stream<bool>(); // Flattened [Cell*Tile]
	auto &entropy_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_ENTROPY);
	auto &gain_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_INFO_GAIN);

	uint64_t cell_count = entropy_stream.size();
	if (cell_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = cell_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? cell_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &possibility_stream, &entropy_stream, &gain_stream]() {
			for (uint64_t i = start; i < end; i++) {
				calculate_entropy_kernel(
					&possibility_stream[i * p_tile_count],
					p_tile_weights,
					p_tile_count,
					entropy_stream[i],
					gain_stream[i]
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * find_observation_point_reduction()
 * 
 * Performs a deterministic parallel reduction to find the lowest non-zero entropy cell.
 * strictly uses BigIntCore for cell indexing to support galactic volumes.
 */
BigIntCore find_observation_point_reduction(const FixedMathCore *p_entropy_stream, const BigIntCore &p_count) {
	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	
	FixedMathCore min_h(2147483647LL, false); // Infinity
	BigIntCore best_idx(-1LL);

	for (uint64_t i = 0; i < total; i++) {
		FixedMathCore h = p_entropy_stream[i];
		if (h.get_raw() > 0 && h < min_h) {
			min_h = h;
			best_idx = BigIntCore(static_cast<int64_t>(i));
		}
	}

	return best_idx;
}

} // namespace UniversalSolver

--- END OF FILE core/math/shannon_entropy_update.cpp ---

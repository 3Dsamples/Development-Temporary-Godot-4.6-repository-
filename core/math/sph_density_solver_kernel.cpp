--- START OF FILE core/math/sph_density_solver_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_poly6_weight()
 * 
 * Deterministic Poly6 kernel for density estimation.
 * W(r, h) = (315 / (64 * pi * h^9)) * (h^2 - r^2)^3
 * strictly uses Software-Defined Arithmetic to ensure no FPU rounding variance.
 */
static _FORCE_INLINE_ FixedMathCore calculate_poly6_weight(
		const FixedMathCore &p_r2, 
		const FixedMathCore &p_h_sq, 
		const FixedMathCore &p_h9_coeff) {

	if (p_r2 >= p_h_sq) {
		return MathConstants<FixedMathCore>::zero();
	}

	// (h^2 - r^2)^3
	FixedMathCore diff = p_h_sq - p_r2;
	FixedMathCore diff3 = diff * diff * diff;

	return p_h9_coeff * diff3;
}

/**
 * Warp Kernel: SPHDensityAccumulationKernel
 * 
 * Computes the local density for a fluid particle.
 * 1. Resolves relative distance to neighbors across BigInt sectors.
 * 2. Accumulates mass-weighted kernel contributions.
 * 3. Updates the EnTT Density stream for the subsequent Pressure wave.
 */
void sph_density_accumulation_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_density,
		const Vector3f &p_pos_i,
		const BigIntCore &p_sx_i, const BigIntCore &p_sy_i, const BigIntCore &p_sz_i,
		const Vector3f *p_neighbor_positions,
		const BigIntCore *p_neighbor_sx, const BigIntCore *p_neighbor_sy, const BigIntCore *p_neighbor_sz,
		const FixedMathCore *p_neighbor_masses,
		const uint32_t *p_neighbor_indices,
		uint32_t p_neighbor_count,
		const FixedMathCore &p_h_sq,
		const FixedMathCore &p_h9_coeff,
		const FixedMathCore &p_sector_size) {

	FixedMathCore rho_acc = MathConstants<FixedMathCore>::zero();

	for (uint32_t n = 0; n < p_neighbor_count; n++) {
		uint32_t j = p_neighbor_indices[n];

		// 1. Bit-Perfect Galactic Distance Resolve
		// r_ij = (Pos_j + Sector_j * Size) - (Pos_i + Sector_i * Size)
		BigIntCore dx_s = p_neighbor_sx[j] - p_sx_i;
		BigIntCore dy_s = p_neighbor_sy[j] - p_sy_i;
		BigIntCore dz_s = p_neighbor_sz[j] - p_sz_i;

		FixedMathCore off_x = FixedMathCore(static_cast<int64_t>(std::stoll(dx_s.to_string()))) * p_sector_size;
		FixedMathCore off_y = FixedMathCore(static_cast<int64_t>(std::stoll(dy_s.to_string()))) * p_sector_size;
		FixedMathCore off_z = FixedMathCore(static_cast<int64_t>(std::stoll(dz_s.to_string()))) * p_sector_size;

		Vector3f rel_pos = (p_neighbor_positions[j] + Vector3f(off_x, off_y, off_z)) - p_pos_i;
		FixedMathCore r2 = rel_pos.length_squared();

		// 2. Kernel Weighting
		if (r2 < p_h_sq) {
			rho_acc += p_neighbor_masses[j] * calculate_poly6_weight(r2, p_h_sq, p_h9_coeff);
		}
	}

	// Self-contribution (r=0, W(0) = 315 / 64pi * h^3)
	// Calculated using i-index properties directly for zero-copy efficiency
	rho_acc += FixedMathCore(1LL) * calculate_poly6_weight(MathConstants<FixedMathCore>::zero(), p_h_sq, p_h9_coeff);

	r_density = rho_acc;
}

/**
 * execute_sph_density_wave()
 * 
 * The master 120 FPS parallel sweep for fluid density.
 * Orchestrates the transition from broadphase neighbor lists to Warp kernels.
 */
void execute_sph_density_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_smoothing_h,
		const FixedMathCore &p_sector_size) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &sx_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
	auto &sy_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
	auto &sz_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);
	auto &mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_MASS);
	auto &rho_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DENSITY);
	
	// Neighbor index lists provided by the SpatialPartition broadphase
	auto &neighbor_stream = p_registry.get_stream<uint32_t*>(COMPONENT_NEIGHBOR_LISTS);
	auto &neighbor_counts = p_registry.get_stream<uint32_t>(COMPONENT_NEIGHBOR_COUNTS);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	// Precompute bit-perfect Poly6 constants to save Warp lane cycles
	FixedMathCore h2 = p_smoothing_h * p_smoothing_h;
	FixedMathCore h9 = p_smoothing_h.power(9);
	FixedMathCore poly6_coeff = FixedMathCore(6724513271LL, true) / h9; 

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &sx_stream, &sy_stream, &sz_stream, &mass_stream, &rho_stream, &neighbor_stream, &neighbor_counts]() {
			for (uint64_t i = start; i < end; i++) {
				sph_density_accumulation_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					rho_stream[i],
					pos_stream[i],
					sx_stream[i], sy_stream[i], sz_stream[i],
					pos_stream.get_base_ptr(),
					sx_stream.get_base_ptr(), sy_stream.get_base_ptr(), sz_stream.get_base_ptr(),
					mass_stream.get_base_ptr(),
					neighbor_stream[i],
					neighbor_counts[i],
					h2,
					poly6_coeff,
					p_sector_size
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	// Enforce 120 FPS execution barrier
	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/sph_density_solver_kernel.cpp ---

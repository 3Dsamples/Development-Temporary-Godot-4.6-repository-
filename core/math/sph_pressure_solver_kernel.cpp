--- START OF FILE core/math/sph_pressure_solver_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_spiky_gradient()
 * 
 * Deterministic Spiky Gradient kernel for pressure force resolution.
 * gradW(r, h) = - (45 / (pi * h^6)) * (h - r)^2 * (r / |r|)
 */
static _FORCE_INLINE_ Vector3f calculate_spiky_gradient(
		const Vector3f &p_diff, 
		const FixedMathCore &p_r, 
		const FixedMathCore &p_h, 
		const FixedMathCore &p_spiky_coeff) {

	if (p_r >= p_h || p_r.get_raw() == 0) {
		return Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
	}

	FixedMathCore diff = p_h - p_r;
	FixedMathCore magnitude = p_spiky_coeff * diff * diff;
	return p_diff.normalized() * (-magnitude);
}

/**
 * calculate_viscosity_laplacian()
 * 
 * Deterministic Laplacian kernel for viscosity/momentum diffusion.
 * laplaceW(r, h) = (45 / (pi * h^6)) * (h - r)
 */
static _FORCE_INLINE_ FixedMathCore calculate_viscosity_laplacian(
		const FixedMathCore &p_r, 
		const FixedMathCore &p_h, 
		const FixedMathCore &p_spiky_coeff) {

	if (p_r >= p_h) {
		return MathConstants<FixedMathCore>::zero();
	}

	return p_spiky_coeff * (p_h - p_r);
}

/**
 * Warp Kernel: SPHForceResolutionKernel
 * 
 * Step 2 of the fluid simulation wave.
 * 1. Resolves relative distance across BigInt sectors.
 * 2. Computes symmetric Pressure forces to ensure zero linear momentum drift.
 * 3. Applies Viscosity tensors for fluid thickening/thinning.
 * 4. Injects sophisticated Anime Turbulence based on vorticity thresholds.
 */
void sph_force_resolution_kernel(
		const BigIntCore &p_index,
		Vector3f &r_acceleration,
		const Vector3f &p_pos_i,
		const Vector3f &p_vel_i,
		const FixedMathCore &p_rho_i,
		const FixedMathCore &p_pres_i,
		const BigIntCore &p_sx_i, const BigIntCore &p_sy_i, const BigIntCore &p_sz_i,
		const Vector3f *p_all_pos,
		const Vector3f *p_all_vel,
		const FixedMathCore *p_all_rho,
		const FixedMathCore *p_all_pres,
		const BigIntCore *p_all_sx, const BigIntCore *p_all_sy, const BigIntCore *p_all_sz,
		const FixedMathCore *p_all_masses,
		const uint32_t *p_neighbor_indices,
		uint32_t p_neighbor_count,
		const FixedMathCore &p_h,
		const FixedMathCore &p_spiky_coeff,
		const FixedMathCore &p_visc_mu,
		const FixedMathCore &p_sector_size,
		bool p_is_anime) {

	Vector3f f_pressure(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
	Vector3f f_viscosity(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
	FixedMathCore two(2LL, false);

	for (uint32_t n = 0; n < p_neighbor_count; n++) {
		uint32_t j = p_neighbor_indices[n];

		// 1. Resolve Galactic Distance
		BigIntCore dx_s = p_all_sx[j] - p_sx_i;
		BigIntCore dy_s = p_all_sy[j] - p_sy_i;
		BigIntCore dz_s = p_all_sz[j] - p_sz_i;

		FixedMathCore off_x = FixedMathCore(static_cast<int64_t>(std::stoll(dx_s.to_string()))) * p_sector_size;
		FixedMathCore off_y = FixedMathCore(static_cast<int64_t>(std::stoll(dy_s.to_string()))) * p_sector_size;
		FixedMathCore off_z = FixedMathCore(static_cast<int64_t>(std::stoll(dz_s.to_string()))) * p_sector_size;

		Vector3f rel_pos = (p_all_pos[j] + Vector3f(off_x, off_y, off_z)) - p_pos_i;
		FixedMathCore r = rel_pos.length();

		if (r < p_h && r.get_raw() > 0) {
			// 2. Symmetric Pressure Force Resolve
			// f_p = -m_j * (P_i + P_j) / (2 * rho_j) * gradW
			FixedMathCore p_term = (p_pres_i + p_all_pres[j]) / (two * p_all_rho[j]);
			Vector3f grad_w = calculate_spiky_gradient(rel_pos, r, p_h, p_spiky_coeff);
			f_pressure += grad_w * (p_all_masses[j] * p_term);

			// 3. Viscosity Force Resolve
			// f_v = mu * m_j * (v_j - v_i) / rho_j * laplaceW
			FixedMathCore v_lap = calculate_viscosity_laplacian(r, p_h, p_spiky_coeff);
			Vector3f v_diff = p_all_vel[j] - p_vel_i;
			f_viscosity += v_diff * (p_visc_mu * p_all_masses[j] * (v_lap / p_all_rho[j]));
		}
	}

	// 4. --- Sophisticated Behavior: Anime Swirl Tensors ---
	if (p_is_anime) {
		// Calculate local curl approximation
		Vector3f curl_approx = f_pressure.cross(f_viscosity);
		if (curl_approx.length_squared() > FixedMathCore(4294967296LL, false)) { // > 1.0 swirl intensity
			// Injected stylized rotational acceleration
			r_acceleration += curl_approx.normalized() * FixedMathCore(10LL, false);
		}
	}

	// a = (F_p + F_v) / rho_i
	r_acceleration += (f_pressure + f_viscosity) / p_rho_i;
}

/**
 * execute_sph_pressure_wave()
 * 
 * Orchestrates the second parallel 120 FPS sweep for fluid forces.
 * Partitions EnTT registries into worker threads for force resolve.
 */
void execute_sph_pressure_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_smoothing_h,
		const FixedMathCore &p_viscosity_mu,
		const FixedMathCore &p_sector_size) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &rho_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DENSITY);
	auto &pres_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_PRESSURE);
	auto &accel_stream = p_registry.get_stream<Vector3f>(COMPONENT_ACCELERATION);
	auto &mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_MASS);
	
	auto &sx_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
	auto &sy_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
	auto &sz_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);

	auto &neighbor_stream = p_registry.get_stream<uint32_t*>(COMPONENT_NEIGHBOR_LISTS);
	auto &neighbor_counts = p_registry.get_stream<uint32_t>(COMPONENT_NEIGHBOR_COUNTS);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	// Precompute Spiky Coefficient: 45 / (pi * h^6)
	FixedMathCore h6 = p_smoothing_h.power(6);
	FixedMathCore spiky_coeff = FixedMathCore(61513264512LL, true) / h6;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream, &rho_stream, &pres_stream, &accel_stream, &sx_stream, &sy_stream, &sz_stream, &mass_stream, &neighbor_stream, &neighbor_counts]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from entity seed
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 14 == 0);

				sph_force_resolution_kernel(
					handle,
					accel_stream[i],
					pos_stream[i],
					vel_stream[i],
					rho_stream[i],
					pres_stream[i],
					sx_stream[i], sy_stream[i], sz_stream[i],
					pos_stream.get_base_ptr(),
					vel_stream.get_base_ptr(),
					rho_stream.get_base_ptr(),
					pres_stream.get_base_ptr(),
					sx_stream.get_base_ptr(), sy_stream.get_base_ptr(), sz_stream.get_base_ptr(),
					mass_stream.get_base_ptr(),
					neighbor_stream[i],
					neighbor_counts[i],
					p_smoothing_h,
					spiky_coeff,
					p_viscosity_mu,
					p_sector_size,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/sph_pressure_solver_kernel.cpp ---

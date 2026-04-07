--- START OF FILE core/math/sph_viscosity_logic.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_viscosity_laplacian_weight()
 * 
 * Deterministic Laplacian smoothing kernel for viscosity.
 * Laplacian W(r, h) = (45 / (pi * h^6)) * (h - r)
 * gouverns the rate of momentum diffusion between fluid particles.
 */
static _FORCE_INLINE_ FixedMathCore calculate_viscosity_laplacian_weight(
		const FixedMathCore &p_r,
		const FixedMathCore &p_h,
		const FixedMathCore &p_lap_coeff) {

	if (p_r >= p_h) {
		return MathConstants<FixedMathCore>::zero();
	}

	// Linear falloff for the Laplacian in SPH viscosity
	return p_lap_coeff * (p_h - p_r);
}

/**
 * calculate_xsph_weight()
 * 
 * Uses the Poly6 kernel to calculate the velocity smoothing factor.
 * W(r, h) = (315 / (64 * pi * h^9)) * (h^2 - r^2)^3
 * Ensures particles move in a coherent "blob" without clustering artifacts.
 */
static _FORCE_INLINE_ FixedMathCore calculate_xsph_weight(
		const FixedMathCore &p_r2,
		const FixedMathCore &p_h_sq,
		const FixedMathCore &p_poly6_coeff) {

	if (p_r2 >= p_h_sq) {
		return MathConstants<FixedMathCore>::zero();
	}

	FixedMathCore diff = p_h_sq - p_r2;
	FixedMathCore diff3 = diff * diff * diff;
	return p_poly6_coeff * diff3;
}

/**
 * Warp Kernel: SPHViscosityXSPHKernel
 * 
 * Step 3 of the fluid simulation wave.
 * 1. Resolves Viscosity Force: m * mu * (v_j - v_i) / rho_j * nabla^2(W).
 * 2. Thermal Thinning: mu_eff = base_mu * exp(-alpha * (T - T_ref)).
 * 3. XSPH Correction: v_corr = epsilon * sum( (2*m_j / (rho_i + rho_j)) * W * (v_j - v_i) ).
 * 4. Galactic Sector Resolve: calculates bit-perfect relative vectors across BigInt boundaries.
 */
void sph_viscosity_xsph_kernel(
		const BigIntCore &p_index,
		Vector3f &r_acceleration,
		Vector3f &r_velocity_correction,
		const Vector3f &p_pos_i,
		const Vector3f &p_vel_i,
		const FixedMathCore &p_rho_i,
		const FixedMathCore &p_temp_i,
		const BigIntCore &p_sx_i, const BigIntCore &p_sy_i, const BigIntCore &p_sz_i,
		const Vector3f *p_all_pos,
		const Vector3f *p_all_vel,
		const FixedMathCore *p_all_rho,
		const BigIntCore *p_all_sx, const BigIntCore *p_all_sy, const BigIntCore *p_all_sz,
		const FixedMathCore *p_all_masses,
		const uint32_t *p_neighbor_indices,
		uint32_t p_neighbor_count,
		const FixedMathCore &p_h,
		const FixedMathCore &p_lap_coeff,
		const FixedMathCore &p_poly6_coeff,
		const FixedMathCore &p_base_mu,
		const FixedMathCore &p_xsph_epsilon,
		const FixedMathCore &p_sector_size,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore half = MathConstants<FixedMathCore>::half();
	
	Vector3f f_viscosity;
	Vector3f v_xsph_acc;

	FixedMathCore h_sq = p_h * p_h;

	// 1. Sophisticated Behavior: Thermal Thinning
	// Viscosity decreases as fluid temperature rises.
	FixedMathCore t_ref(12591030272LL, true); // 293.15 K
	FixedMathCore t_diff = wp::max(zero, p_temp_i - t_ref);
	// mu = mu0 * exp(-0.01 * deltaT)
	FixedMathCore thinning_factor = wp::exp(-(t_diff * FixedMathCore(42949673LL, true)));
	FixedMathCore mu_effective = p_base_mu * thinning_factor;

	for (uint32_t n = 0; n < p_neighbor_count; n++) {
		uint32_t j = p_neighbor_indices[n];

		// Resolve relative position between galactic sectors
		Vector3f rel_pos = wp::calculate_galactic_relative_pos(
				p_pos_i, p_sx_i, p_sy_i, p_sz_i,
				p_all_pos[j], p_all_sx[j], p_all_sy[j], p_all_sz[j],
				p_sector_size);

		FixedMathCore r2 = rel_pos.length_squared();
		if (r2 < h_sq && r2.get_raw() > 0) {
			FixedMathCore r = Math::sqrt(r2);
			Vector3f v_diff = p_all_vel[j] - p_vel_i;

			// 2. Resolve Viscosity Force (Momentum Diffusion)
			FixedMathCore lap_w = calculate_viscosity_laplacian_weight(r, p_h, p_lap_coeff);
			FixedMathCore visc_mass_term = mu_effective * p_all_masses[j];
			f_viscosity += v_diff * (visc_mass_term * (lap_w / p_all_rho[j]));

			// 3. Resolve XSPH Smoothing (Velocity Correction)
			FixedMathCore poly6_w = calculate_xsph_weight(r2, h_sq, p_poly6_coeff);
			FixedMathCore density_sum_avg = (p_rho_i + p_all_rho[j]) * half;
			v_xsph_acc += v_diff * (p_all_masses[j] * poly6_w / density_sum_avg);
		}
	}

	// 4. --- Sophisticated Real-Time Behavior: Anime Swirl ---
	if (p_is_anime) {
		// Injects additional rotational momentum to maintain stylized vortex shapes
		Vector3f curl_approx = f_viscosity.cross(p_vel_i);
		f_viscosity += curl_approx * FixedMathCore(2LL, false);
	}

	// Apply acceleration: a = F_visc / rho_i
	r_acceleration += f_viscosity / p_rho_i;
	
	// Apply XSPH: delta_v = epsilon * Sum(...)
	r_velocity_correction = v_xsph_acc * p_xsph_epsilon;
}

/**
 * execute_sph_viscosity_wave()
 * 
 * Orchestrates the parallel 120 FPS viscosity resolve across EnTT registries.
 * Zero-copy: Operates directly on the aligned SoA memory buffers.
 */
void execute_sph_viscosity_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_smoothing_h,
		const FixedMathCore &p_base_mu,
		const FixedMathCore &p_xsph_factor,
		const FixedMathCore &p_sector_size) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &rho_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DENSITY);
	auto &temp_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TEMPERATURE);
	auto &accel_stream = p_registry.get_stream<Vector3f>(COMPONENT_ACCELERATION);
	auto &v_corr_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY_CORRECTION);
	auto &mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_MASS);

	auto &sx_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
	auto &sy_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
	auto &sz_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);

	auto &neighbor_stream = p_registry.get_stream<uint32_t *>(COMPONENT_NEIGHBOR_LISTS);
	auto &neighbor_counts = p_registry.get_stream<uint32_t>(COMPONENT_NEIGHBOR_COUNTS);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	// Precompute bit-perfect kernel coefficients in Q32.32
	FixedMathCore h6 = p_smoothing_h.power(6);
	FixedMathCore lap_coeff = FixedMathCore(61513264512LL, true) / h6; // 45/pi / h^6
	FixedMathCore h9 = p_smoothing_h.power(9);
	FixedMathCore poly6_coeff = FixedMathCore(6724513271LL, true) / h9; // 315/64pi / h^9

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream, &rho_stream, &temp_stream, &accel_stream, &v_corr_stream, &mass_stream, &sx_stream, &sy_stream, &sz_stream, &neighbor_stream, &neighbor_counts]() {
			for (uint64_t i = start; i < end; i++) {
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				// Deterministic Style derivation: 1 in 10 particles use Anime Swirl
				bool anime_mode = (handle.hash() % 10 == 0);

				sph_viscosity_xsph_kernel(
						handle,
						accel_stream[i],
						v_corr_stream[i],
						pos_stream[i], vel_stream[i], rho_stream[i], temp_stream[i],
						sx_stream[i], sy_stream[i], sz_stream[i],
						pos_stream.get_base_ptr(), vel_stream.get_base_ptr(), rho_stream.get_base_ptr(),
						sx_stream.get_base_ptr(), sy_stream.get_base_ptr(), sz_stream.get_base_ptr(),
						mass_stream.get_base_ptr(),
						neighbor_stream[i], neighbor_counts[i],
						p_smoothing_h, lap_coeff, poly6_coeff,
						p_base_mu, p_xsph_factor, p_sector_size, anime_mode);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/sph_viscosity_logic.cpp ---

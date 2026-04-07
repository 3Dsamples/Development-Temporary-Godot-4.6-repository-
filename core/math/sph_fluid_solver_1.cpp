--- START OF FILE core/math/sph_fluid_solver.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

// ============================================================================
// SPH Deterministic Kernels (FixedMathCore Q32.32)
// ============================================================================

/**
 * calculate_poly6_kernel()
 * W(r, h) = (315 / (64 * pi * h^9)) * (h^2 - r^2)^3
 * Used for density accumulation.
 */
static _FORCE_INLINE_ FixedMathCore calculate_poly6_kernel(const FixedMathCore &p_r2, const FixedMathCore &p_h) {
	FixedMathCore h2 = p_h * p_h;
	if (p_r2 >= h2) {
		return MathConstants<FixedMathCore>::zero();
	}

	FixedMathCore diff = h2 - p_r2;
	FixedMathCore diff3 = diff * diff * diff;
	
	// Precomputed constant for 315 / (64 * pi) in bit-perfect Q32.32
	// (315.0 / (64.0 * 3.1415926535)) * 2^32
	FixedMathCore coeff(6724513271LL, true); 
	FixedMathCore h9 = p_h.power(9);
	
	return (coeff / h9) * diff3;
}

/**
 * calculate_spiky_gradient_kernel()
 * gradW(r, h) = - (45 / (pi * h^6)) * (h - r)^2 * (r / |r|)
 * Used for pressure force resolution.
 */
static _FORCE_INLINE_ Vector3f calculate_spiky_gradient_kernel(const Vector3f &p_diff, const FixedMathCore &p_r, const FixedMathCore &p_h) {
	if (p_r >= p_h || p_r.get_raw() == 0) {
		return Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
	}

	FixedMathCore diff = p_h - p_r;
	// Precomputed constant for 45 / pi in bit-perfect Q32.32
	FixedMathCore coeff(61513264512LL, true);
	FixedMathCore h6 = p_h.power(6);
	
	FixedMathCore magnitude = -(coeff / h6) * diff * diff;
	return p_diff.normalized() * magnitude;
}

/**
 * calculate_viscosity_laplacian_kernel()
 * laplaceW(r, h) = (45 / (pi * h^6)) * (h - r)
 * Used for momentum diffusion.
 */
static _FORCE_INLINE_ FixedMathCore calculate_viscosity_laplacian_kernel(const FixedMathCore &p_r, const FixedMathCore &p_h) {
	if (p_r >= p_h) {
		return MathConstants<FixedMathCore>::zero();
	}

	// 45 / (pi * h^6)
	FixedMathCore coeff(61513264512LL, true);
	FixedMathCore h6 = p_h.power(6);
	
	return (coeff / h6) * (p_h - p_r);
}

// ============================================================================
// Warp Kernels (EnTT SoA Parallel Execution)
// ============================================================================

/**
 * Warp Kernel: SPHDensityPressureKernel
 * 
 * Step 1: Compute local density and resulting pressure for every particle.
 * P = k * (rho - rho_rest)
 * Handles galactic scale by calculating relative positions between BigInt sectors.
 */
void sph_density_pressure_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_density,
		FixedMathCore &r_pressure,
		const Vector3f &p_pos,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const Vector3f *p_all_pos,
		const BigIntCore *p_all_sx, const BigIntCore *p_all_sy, const BigIntCore *p_all_sz,
		uint64_t p_count,
		const FixedMathCore &p_h,
		const FixedMathCore &p_mass,
		const FixedMathCore &p_gas_k,
		const FixedMathCore &p_rest_rho) {

	FixedMathCore rho_accum = MathConstants<FixedMathCore>::zero();
	FixedMathCore h2 = p_h * p_h;

	for (uint64_t j = 0; j < p_count; j++) {
		// Resolve relative position across galactic BigInt sectors (Sector size 10,000 units)
		Vector3f rel_pos = wp::calculate_galactic_relative_pos(
			p_pos, p_sx, p_sy, p_sz,
			p_all_pos[j], p_all_sx[j], p_all_sy[j], p_all_sz[j],
			FixedMathCore(10000LL, false)
		);

		FixedMathCore r2 = rel_pos.length_squared();
		if (r2 < h2) {
			rho_accum += p_mass * calculate_poly6_kernel(r2, p_h);
		}
	}

	// Density must be clamped to rest density floor to prevent negative pressure
	r_density = wp::max(p_rest_rho, rho_accum);
	
	// Tait-Maughan Equation or Linear Gas Law: P = k * (rho - rho_rest)
	r_pressure = p_gas_k * (r_density - p_rest_rho);
}

/**
 * Warp Kernel: SPHForceKernel
 * 
 * Step 2: Compute Pressure and Viscosity forces between particles.
 * a = (F_pressure + F_viscosity + F_external) / rho
 */
void sph_force_kernel(
		const BigIntCore &p_index,
		Vector3f &r_acceleration,
		const Vector3f &p_pos,
		const Vector3f &p_vel,
		const FixedMathCore &p_rho,
		const FixedMathCore &p_pres,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const Vector3f *p_all_pos,
		const Vector3f *p_all_vel,
		const FixedMathCore *p_all_rho,
		const FixedMathCore *p_all_pres,
		const BigIntCore *p_all_sx, const BigIntCore *p_all_sy, const BigIntCore *p_all_sz,
		uint64_t p_count,
		const FixedMathCore &p_h,
		const FixedMathCore &p_mass,
		const FixedMathCore &p_visc_mu) {

	Vector3f f_pressure(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
	Vector3f f_viscosity(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
	uint64_t i_idx = static_cast<uint64_t>(std::stoll(p_index.to_string()));

	for (uint64_t j = 0; j < p_count; j++) {
		if (i_idx == j) continue;

		Vector3f rel_pos = wp::calculate_galactic_relative_pos(
			p_pos, p_sx, p_sy, p_sz,
			p_all_pos[j], p_all_sx[j], p_all_sy[j], p_all_sz[j],
			FixedMathCore(10000LL, false)
		);

		FixedMathCore r = rel_pos.length();
		if (r < p_h && r.get_raw() > 0) {
			// 1. Pressure Force: Symmetric formulation to conserve momentum
			// f_p = -m * (P_i + P_j) / (2 * rho_j) * gradW
			FixedMathCore p_term = (p_pres + p_all_pres[j]) / (FixedMathCore(2LL, false) * p_all_rho[j]);
			f_pressure -= calculate_spiky_gradient_kernel(rel_pos, r, p_h) * (p_mass * p_term);

			// 2. Viscosity Force: Laplacian of velocity
			// f_v = mu * m * (v_j - v_i) / rho_j * laplaceW
			FixedMathCore v_lap = calculate_viscosity_laplacian_kernel(r, p_h);
			Vector3f v_diff = p_all_vel[j] - p_vel;
			f_viscosity += v_diff * (p_visc_mu * p_mass * (v_lap / p_all_rho[j]));
		}
	}

	// a = F / rho
	r_acceleration = (f_pressure + f_viscosity) / p_rho;
}

/**
 * execute_sph_simulation_wave()
 * 
 * Master orchestrator for parallelized 120 FPS fluid physics.
 * 1. Density/Pressure Wave: Parallel sweep to resolve local fluid state.
 * 2. Force/Acceleration Wave: Parallel sweep to resolve interactions.
 * 3. Final Integration: Updates position/velocity in EnTT registry.
 */
void execute_sph_simulation_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_h,
		const FixedMathCore &p_delta) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &rho_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DENSITY);
	auto &pres_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_PRESSURE);
	auto &accel_stream = p_registry.get_stream<Vector3f>(COMPONENT_ACCELERATION);
	
	auto &sx_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
	auto &sy_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
	auto &sz_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
	uint32_t workers = pool->get_worker_count();
	uint64_t chunk = count / workers;

	// --- PHASE 1: Density & Pressure Parallel Resolve ---
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		pool->enqueue_task([=, &pos_stream, &rho_stream, &pres_stream, &sx_stream, &sy_stream, &sz_stream]() {
			for (uint64_t i = start; i < end; i++) {
				sph_density_pressure_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					rho_stream[i], pres_stream[i],
					pos_stream[i], sx_stream[i], sy_stream[i], sz_stream[i],
					pos_stream.get_base_ptr(), sx_stream.get_base_ptr(), sy_stream.get_base_ptr(), sz_stream.get_base_ptr(),
					count, p_h, 
					FixedMathCore(1LL, false),    // Particle Mass
					FixedMathCore(2000LL, false), // Gas Constant (Stiffness)
					FixedMathCore(1000LL, false)  // Rest Density (Water-like)
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	pool->wait_for_all();

	// --- PHASE 2: Force & Acceleration Parallel Resolve ---
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		pool->enqueue_task([=, &pos_stream, &vel_stream, &rho_stream, &pres_stream, &accel_stream, &sx_stream, &sy_stream, &sz_stream]() {
			for (uint64_t i = start; i < end; i++) {
				sph_force_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					accel_stream[i],
					pos_stream[i], vel_stream[i], rho_stream[i], pres_stream[i],
					sx_stream[i], sy_stream[i], sz_stream[i],
					pos_stream.get_base_ptr(), vel_stream.get_base_ptr(), 
					rho_stream.get_base_ptr(), pres_stream.get_base_ptr(),
					sx_stream.get_base_ptr(), sy_stream.get_base_ptr(), sz_stream.get_base_ptr(),
					count, p_h, 
					FixedMathCore(1LL, false), // Particle Mass
					FixedMathCore(10LL, false)  // Viscosity Mu
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	pool->wait_for_all();

	// --- PHASE 3: Semi-Implicit Euler Integration ---
	// v = v + a * dt; p = p + v * dt
	for (uint64_t i = 0; i < count; i++) {
		vel_stream[i] += accel_stream[i] * p_delta;
		pos_stream[i] += vel_stream[i] * p_delta;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/sph_fluid_solver.cpp ---

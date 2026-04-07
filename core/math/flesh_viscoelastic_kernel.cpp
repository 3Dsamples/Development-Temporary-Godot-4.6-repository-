--- START OF FILE core/math/flesh_viscoelastic_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: FleshViscoelasticDynamicsKernel
 * 
 * Computes the internal restoration and damping forces for a flesh-like vertex.
 * 1. Strain-Hardening: Stiffness increases non-linearly as displacement approaches the 'limit'.
 * 2. Viscous Damping: Absorbs energy to prevent infinite oscillation (jiggle).
 * 3. Balloon Restoration: Pulls the vertex back to its rest-position bit-perfectly.
 */
void flesh_viscoelastic_dynamics_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		const Vector3f &p_rest_position,
		const FixedMathCore &p_base_stiffness,
		const FixedMathCore &p_viscosity,
		const FixedMathCore &p_expansion_limit,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Displacement Tensor (u)
	Vector3f u = r_position - p_rest_position;
	FixedMathCore dist = u.length();

	if (dist.get_raw() == 0) {
		return;
	}

	// 2. Non-Linear Strain-Hardening (Sophisticated Balloon Effect)
	// stiffness_effective = k / (1.0 - (dist / limit)^2)
	// As dist approaches limit, force approaches infinity, preventing mesh collapse.
	FixedMathCore strain_ratio = dist / p_expansion_limit;
	FixedMathCore strain_sq = wp::min(strain_ratio * strain_ratio, FixedMathCore(4080218931LL, true)); // 0.95 cap
	
	FixedMathCore effective_k = p_base_stiffness / (one - strain_sq);

	// 3. Resolve Internal Forces
	// F_elastic = -k_eff * u
	Vector3f f_elastic = u * (-effective_k);

	// F_viscous = -eta * v
	Vector3f f_viscous = r_velocity * (-p_viscosity);

	// 4. Deterministic Integration (Semi-Implicit Euler)
	// a = (F_e + F_v) / mass. (Unit mass per vertex for high-speed SoA).
	Vector3f acceleration = f_elastic + f_viscous;
	
	r_velocity += acceleration * p_delta;
	r_position += r_velocity * p_delta;

	// 5. Sophisticated Drift Correction
	// If the vertex is remarkably close to rest, snap it to zero out micro-oscillations.
	if (r_velocity.length_squared() < FixedMathCore(429496LL, true) && dist < FixedMathCore(429496LL, true)) {
		r_position = p_rest_position;
		r_velocity = Vector3f(zero, zero, zero);
	}
}

/**
 * execute_flesh_physics_sweep()
 * 
 * Orchestrates the parallel 120 FPS sweep for anatomical body physics.
 * Partitions EnTT registries for Breasts, Buttocks, and Muscle groups.
 * strictly bit-perfect across all CPU cores.
 */
void execute_flesh_physics_sweep(
		KernelRegistry &p_registry,
		const StringName &p_flesh_type,
		const FixedMathCore &p_delta) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &rest_stream = p_registry.get_stream<Vector3f>(COMPONENT_REST_POSITION);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	// Determine Material Tensors based on anatomical type
	FixedMathCore k_stiffness, eta_viscosity, limit;
	
	if (p_flesh_type == SNAME("breast")) {
		k_stiffness = FixedMathCore(2LL, false);      // Soft
		eta_viscosity = FixedMathCore(1288490188LL, true); // 0.3 Damping
		limit = FixedMathCore(2LL, false);            // High stretch
	} else if (p_flesh_type == SNAME("buttock")) {
		k_stiffness = FixedMathCore(8LL, false);      // Firm
		eta_viscosity = FixedMathCore(2147483648LL, true); // 0.5 Damping
		limit = FixedMathCore(1LL, false);            // Low stretch
	} else {
		k_stiffness = FixedMathCore(20LL, false);     // Muscle
		eta_viscosity = FixedMathCore(4LL, false);    // High damping
		limit = FixedMathCore(429496729LL, true);     // 0.1 Rigid
	}

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream, &rest_stream]() {
			for (uint64_t i = start; i < end; i++) {
				flesh_viscoelastic_dynamics_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i],
					vel_stream[i],
					rest_stream[i],
					k_stiffness,
					eta_viscosity,
					limit,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_flesh_interaction_tensor()
 * 
 * Injects external displacement from "Poke, Pinch, or Pull" events.
 * Directly modifies the EnTT velocity stream to trigger the viscoelastic response.
 */
void apply_flesh_interaction_tensor(
		Vector3f *r_velocities,
		const Vector3f *p_positions,
		uint64_t p_count,
		const Vector3f &p_epicenter,
		const Vector3f &p_impulse_vec,
		const FixedMathCore &p_radius) {

	FixedMathCore r2 = p_radius * p_radius;

	for (uint64_t i = 0; i < p_count; i++) {
		Vector3f diff = p_positions[i] - p_epicenter;
		FixedMathCore d2 = diff.length_squared();
		
		if (d2 < r2) {
			FixedMathCore dist = Math::sqrt(d2);
			FixedMathCore falloff = (p_radius - dist) / p_radius;
			// Inject impulse with quadratic falloff
			r_velocities[i] += p_impulse_vec * (falloff * falloff);
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/flesh_viscoelastic_kernel.cpp ---

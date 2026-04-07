--- START OF FILE core/math/collision_solver_logic.cpp ---

#include "core/math/collision_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ContactResolutionKernel
 * 
 * Resolves a batch of collision manifolds using bit-perfect impulse physics.
 * 1. Calculates relative velocity at contact points (including angular components).
 * 2. Computes normal impulse with Baumgarte stabilization to prevent "sinking".
 * 3. Applies friction tensors to resolve tangential motion.
 * 4. Injects fatigue damage into the material tensor based on impact energy.
 */
void contact_resolution_kernel(
		const BigIntCore &p_index,
		Vector3f &r_vel_a, Vector3f &r_ang_vel_a,
		Vector3f &r_vel_b, Vector3f &r_ang_vel_b,
		FixedMathCore &r_fatigue_a, FixedMathCore &r_fatigue_b,
		const CollisionSolverf::CollisionResult &p_manifold,
		const Transform3Df &p_xform_a, const Transform3Df &p_xform_b,
		const FixedMathCore &p_inv_mass_a, const FixedMathCore &p_inv_mass_b,
		const FixedMathCore &p_restitution, const FixedMathCore &p_friction,
		const FixedMathCore &p_delta) {

	if (!p_manifold.collided) return;

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate contact arms from center of mass
	Vector3f arm_a = p_manifold.contact_point - p_xform_a.origin;
	Vector3f arm_b = p_manifold.contact_point - p_xform_b.origin;

	// 2. Relative velocity at contact point
	// v_rel = (v_b + w_b x arm_b) - (v_a + w_a x arm_a)
	Vector3f v_at_contact_a = r_vel_a + r_ang_vel_a.cross(arm_a);
	Vector3f v_at_contact_b = r_vel_b + r_ang_vel_b.cross(arm_b);
	Vector3f v_rel = v_at_contact_b - v_at_contact_a;

	FixedMathCore v_rel_normal = v_rel.dot(p_manifold.contact_normal);

	// If objects are already moving apart, skip impulse
	if (v_rel_normal.get_raw() > 0) return;

	// 3. Resolve Normal Impulse (Pn)
	// Simplified scalar inertia for 120 FPS performance (approx: 0.4 * mass * r^2)
	FixedMathCore inv_inertia_a = p_inv_mass_a * FixedMathCore(2LL, false); 
	FixedMathCore inv_inertia_b = p_inv_mass_b * FixedMathCore(2LL, false);

	// Constraint Mass (Denominator)
	FixedMathCore angular_term_a = (arm_a.cross(p_manifold.contact_normal) * inv_inertia_a).cross(arm_a).dot(p_manifold.contact_normal);
	FixedMathCore angular_term_b = (arm_b.cross(p_manifold.contact_normal) * inv_inertia_b).cross(arm_b).dot(p_manifold.contact_normal);
	FixedMathCore k_normal = p_inv_mass_a + p_inv_mass_b + angular_term_a + angular_term_b;

	// Baumgarte Stabilization: pushes objects apart based on penetration depth
	FixedMathCore bias = (FixedMathCore(2LL, true) * (p_manifold.penetration_depth - FixedMathCore(429496LL, true))) / p_delta; // 0.0001 slop
	bias = wp::max(zero, bias * FixedMathCore(858993459LL, true)); // 0.2 beta factor

	FixedMathCore j_n = -(one + p_restitution) * v_rel_normal + bias;
	j_n /= k_normal;

	Vector3f impulse_n = p_manifold.contact_normal * j_n;

	// 4. Resolve Friction Impulse (Pt)
	Vector3f tangent = (v_rel - p_manifold.contact_normal * v_rel_normal).normalized();
	if (tangent.length_squared() > zero) {
		FixedMathCore v_rel_tangent = v_rel.dot(tangent);
		
		FixedMathCore angular_term_t_a = (arm_a.cross(tangent) * inv_inertia_a).cross(arm_a).dot(tangent);
		FixedMathCore angular_term_t_b = (arm_b.cross(tangent) * inv_inertia_b).cross(arm_b).dot(tangent);
		FixedMathCore k_tangent = p_inv_mass_a + p_inv_mass_b + angular_term_t_a + angular_term_t_b;

		FixedMathCore j_t = -v_rel_tangent / k_tangent;
		
		// Coulomb's Law: Clamp friction impulse to the normal impulse magnitude
		FixedMathCore max_friction = p_friction * j_n;
		j_t = wp::clamp(j_t, -max_friction, max_friction);

		Vector3f impulse_t = tangent * j_t;
		
		// Apply Total Impulse (Normal + Friction)
		Vector3f total_impulse = impulse_n + impulse_t;
		r_vel_a -= total_impulse * p_inv_mass_a;
		r_ang_vel_a -= arm_a.cross(total_impulse) * inv_inertia_a;
		r_vel_b += total_impulse * p_inv_mass_b;
		r_ang_vel_b += arm_b.cross(total_impulse) * inv_inertia_b;
	} else {
		r_vel_a -= impulse_n * p_inv_mass_a;
		r_ang_vel_a -= arm_a.cross(impulse_n) * inv_inertia_a;
		r_vel_b += impulse_n * p_inv_mass_b;
		r_ang_vel_b += arm_b.cross(impulse_n) * inv_inertia_b;
	}

	// 5. Sophisticated Interaction: Kinetic Damage Injection
	// Energy = 0.5 * Impulse * VelocityChange. Increases fatigue tensors in EnTT.
	FixedMathCore impact_energy = j_n * wp::abs(v_rel_normal);
	FixedMathCore damage_scalar(4294967LL, true); // 0.001 base
	r_fatigue_a += impact_energy * damage_scalar * p_inv_mass_a;
	r_fatigue_b += impact_energy * damage_scalar * p_inv_mass_b;
}

/**
 * execute_batch_resolution()
 * 
 * Master orchestrator for parallelized 120 FPS collision response.
 * Divides the manifold registry into SIMD-friendly chunks.
 */
void execute_batch_resolution(
		KernelRegistry &p_registry,
		const FixedMathCore &p_delta) {

	auto &manifold_stream = p_registry.get_stream<CollisionSolverf::CollisionResult>();
	uint64_t count = manifold_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_registry]() {
			// Zero-Copy pointers to all required SoA streams
			auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
			auto &ang_vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_ANG_VELOCITY);
			auto &fatigue_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_FATIGUE);
			auto &xform_stream = p_registry.get_stream<Transform3Df>(COMPONENT_WORLD_XFORM);
			auto &inv_mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_INV_MASS);

			for (uint64_t i = start; i < end; i++) {
				const auto &m = manifold_stream[i];
				// Resolve A and B entity indices (Assuming indices provided in manifold metadata)
				uint64_t idx_a = m.entity_a_dense;
				uint64_t idx_b = m.entity_b_dense;

				contact_resolution_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					vel_stream[idx_a], ang_vel_stream[idx_a],
					vel_stream[idx_b], ang_vel_stream[idx_b],
					fatigue_stream[idx_a], fatigue_stream[idx_b],
					m,
					xform_stream[idx_a], xform_stream[idx_b],
					inv_mass_stream[idx_a], inv_mass_stream[idx_b],
					FixedMathCore(2147483648LL, true), // 0.5 Restitution
					FixedMathCore(2147483648LL, true), // 0.5 Friction
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/collision_solver_logic.cpp ---

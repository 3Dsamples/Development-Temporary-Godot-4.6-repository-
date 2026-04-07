--- START OF FILE core/math/collision_solver_logic.cpp ---

#include "core/math/collision_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * resolve_contact_batch_kernel()
 * 
 * Master Warp kernel for resolving a batch of collision manifolds.
 * Operates on EnTT SoA streams for velocities, masses, and contact data.
 * Uses a bit-perfect impulse-based resolution model.
 */
void resolve_contact_batch_kernel(
		Vector3f *r_vel_a,
		Vector3f *r_ang_vel_a,
		Vector3f *r_vel_b,
		Vector3f *r_ang_vel_b,
		const CollisionResult *p_manifolds,
		const FixedMathCore *p_inv_mass_a,
		const FixedMathCore *p_inv_mass_b,
		uint64_t p_count,
		const FixedMathCore &p_restitution_bias) {

	for (uint64_t i = 0; i < p_count; i++) {
		const CollisionResult &m = p_manifolds[i];
		if (!m.collided) continue;

		// 1. Calculate Relative Velocity in Deterministic Space
		Vector3f rel_v = (*r_vel_b) - (*r_vel_a);
		FixedMathCore v_rel_normal = rel_v.dot(m.contact_normal);

		// If bodies are moving apart, skip resolution to prevent "sticky" collisions
		if (v_rel_normal.get_raw() > 0) continue;

		// 2. Resolve Impulse Magnitude (J)
		// e = Restitution (Bounce)
		FixedMathCore e = p_restitution_bias; 
		FixedMathCore j = -(MathConstants<FixedMathCore>::one() + e) * v_rel_normal;
		j /= (p_inv_mass_a[i] + p_inv_mass_b[i]);

		// 3. Apply Linear Impulse
		Vector3f impulse = m.contact_normal * j;
		r_vel_a[i] -= impulse * p_inv_mass_a[i];
		r_vel_b[i] += impulse * p_inv_mass_b[i];

		// 4. Apply Friction Tensors (Coulomb Model)
		// Calculate Tangent Plane relative motion
		Vector3f tangent = rel_v - (m.contact_normal * v_rel_normal);
		if (tangent.length_squared().get_raw() > 0) {
			tangent = tangent.normalized();
			FixedMathCore v_rel_tangent = rel_v.dot(tangent);
			
			// Friction coefficient (Simplified bit-perfect constant)
			FixedMathCore mu(2147483648LL, true); // 0.5 static friction
			FixedMathCore friction_j = -v_rel_tangent / (p_inv_mass_a[i] + p_inv_mass_b[i]);
			
			// Clamp friction to the friction cone: |f_j| <= mu * |j|
			FixedMathCore max_friction = mu * j;
			friction_j = wp::clamp(friction_j, -max_friction, max_friction);

			Vector3f f_impulse = tangent * friction_j;
			r_vel_a[i] -= f_impulse * p_inv_mass_a[i];
			r_vel_b[i] += f_impulse * p_inv_mass_b[i];
		}
	}
}

/**
 * apply_structural_impact_damage_kernel()
 * 
 * Hyper-Simulation Feature: Translates kinetic energy loss into material fatigue.
 * Triggers deformation in TYPE_DEFORMABLE bodies managed by EnTT.
 */
void apply_structural_impact_damage_kernel(
		FixedMathCore *r_fatigue_a,
		FixedMathCore *r_fatigue_b,
		const Vector3f &p_rel_vel,
		const FixedMathCore &p_mass_a,
		const FixedMathCore &p_mass_b,
		uint64_t p_count) {

	FixedMathCore half = MathConstants<FixedMathCore>::half();

	for (uint64_t i = 0; i < p_count; i++) {
		// Kinetic Energy: Ek = 0.5 * m * v^2
		FixedMathCore v2 = p_rel_vel.length_squared();
		FixedMathCore system_mass = (p_mass_a * p_mass_b) / (p_mass_a + p_mass_b);
		FixedMathCore impact_energy = half * system_mass * v2;

		// Damage scales with energy but is mitigated by material yield_strength (internal to fatigue)
		FixedMathCore damage_scalar(4294967LL, true); // 0.001 scale
		r_fatigue_a[i] += impact_energy * damage_scalar;
		r_fatigue_b[i] += impact_energy * damage_scalar;
	}
}

/**
 * resolve_penetration_correction_kernel()
 * 
 * Prevents "Object Sinking" by applying a pseudo-velocity (Baumgarte Stabilization).
 * Strictly deterministic to prevent oscillations in stacked objects.
 */
void resolve_penetration_correction_kernel(
		Vector3f *r_positions_a,
		Vector3f *r_positions_b,
		const FixedMathCore *p_depths,
		const Vector3f *p_normals,
		uint64_t p_count) {

	// Slop and Bias constants in FixedMath
	FixedMathCore slop(429496LL, true); // 0.0001
	FixedMathCore bias(858993459LL, true); // 0.2 percentage correction

	for (uint64_t i = 0; i < p_count; i++) {
		if (p_depths[i] < slop) continue;

		Vector3f correction = p_normals[i] * (p_depths[i] * bias);
		r_positions_a[i] -= correction;
		r_positions_b[i] += correction;
	}
}

--- END OF FILE core/math/collision_solver_logic.cpp ---

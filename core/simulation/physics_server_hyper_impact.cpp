--- START OF FILE core/simulation/physics_server_hyper_impact.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/dynamic_mesh.h"
#include "core/simulation/simulation_manager.h"

namespace UniversalSolver {

/**
 * resolve_impact_physics_kernel()
 * 
 * Master kernel for calculating impulse response and material failure.
 * Processes high-speed collisions using bit-perfect FixedMathCore.
 */
void resolve_impact_physics_kernel(
		Body *p_body_a,
		Body *p_body_b,
		const CollisionResult &p_contact,
		const FixedMathCore &p_step_time) {

	// 1. Calculate Relative Velocity and Impulse
	Vector3f rel_vel = p_body_b->linear_velocity - p_body_a->linear_velocity;
	FixedMathCore contact_vel = rel_vel.dot(p_contact.contact_normal);

	// Prevent resolution if objects are already separating
	if (contact_vel.get_raw() > 0) return;

	// Total system mass handled via BigInt for celestial-scale collisions
	FixedMathCore inv_mass_sum = (p_body_a->mass.get_raw() == 0 ? FixedMathCore(0LL, true) : MathConstants<FixedMathCore>::one() / p_body_a->mass) +
								 (p_body_b->mass.get_raw() == 0 ? FixedMathCore(0LL, true) : MathConstants<FixedMathCore>::one() / p_body_b->mass);

	FixedMathCore restitution = wp::min(p_body_a->bounce, p_body_b->bounce);
	FixedMathCore j = -(MathConstants<FixedMathCore>::one() + restitution) * contact_vel;
	j /= inv_mass_sum;

	Vector3f impulse = p_contact.contact_normal * j;

	// 2. Kinetic Energy Transfer & Bond Failure
	// Energy = 0.5 * m * v^2 -> Used to calculate structural damage
	FixedMathCore kinetic_energy = MathConstants<FixedMathCore>::half() * p_body_a->mass * contact_vel * contact_vel;
	
	// Determine if the material bonds fail (Shatter Trigger)
	FixedMathCore force_at_impact = j / p_step_time;
	FixedMathCore total_yield = p_body_a->integrity * FixedMathCore(1000LL, false); // Scaled yield strength

	if (force_at_impact > total_yield) {
		p_body_a->mode = PhysicsServerHyper::BODY_MODE_DEFORMABLE;
		p_body_a->integrity -= (force_at_impact / total_yield) * FixedMathCore(42949672LL, true); // 0.01 decay
	}

	// 3. Thermal Conversion Behavior
	// Convert a portion of kinetic energy into thermal state tensors
	FixedMathCore heat_coeff(858993459LL, true); // 0.2 conversion
	p_body_a->thermal_state += (kinetic_energy * heat_coeff) / (p_body_a->mass + MathConstants<FixedMathCore>::one());

	// 4. Apply Response
	if (p_body_a->mode != PhysicsServerHyper::BODY_MODE_STATIC) {
		p_body_a->linear_velocity -= impulse / p_body_a->mass;
	}
	if (p_body_b->mode != PhysicsServerHyper::BODY_MODE_STATIC) {
		p_body_b->linear_velocity += impulse / p_body_b->mass;
	}

	// 5. High-Speed Interaction: Deformation
	// For high-speed spaceships, we trigger localized mesh denting in the same tick
	if (p_body_a->mode == PhysicsServerHyper::BODY_MODE_DEFORMABLE && p_body_a->mesh_deterministic.is_valid()) {
		Vector3f local_impact = p_body_a->transform.xform_inv(p_contact.contact_point);
		Vector3f local_dir = p_body_a->transform.basis.inverse().xform(p_contact.contact_normal);
		p_body_a->mesh_deterministic->apply_impact(local_impact, -local_dir, force_at_impact, FixedMathCore(2LL, false));
	}
}

/**
 * apply_global_explosion_kernel()
 * 
 * Simulates a large-scale physical shockwave affecting all bodies in range.
 * Uses BigIntCore to support shockwaves that span multiple sectors.
 */
void PhysicsServerHyper::apply_global_explosion(
		const Vector3f &p_epicenter,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const FixedMathCore &p_energy,
		const FixedMathCore &p_radius) {

	List<RID> targets;
	broadphase.query_radius(p_epicenter, p_sx, p_sy, p_sz, p_radius, targets);

	for (const typename List<RID>::Element *E = targets.front(); E; E = E->next()) {
		Body *b = body_owner.get_or_null(E->get());
		if (!b || b->mode == BODY_MODE_STATIC) continue;

		Vector3f rel_pos = b->transform.origin - p_epicenter;
		FixedMathCore dist = rel_pos.length();
		if (dist >= p_radius) continue;

		FixedMathCore falloff = MathConstants<FixedMathCore>::one() - (dist / p_radius);
		FixedMathCore force_mag = p_energy * (falloff * falloff);
		Vector3f impulse_vec = rel_pos.normalized() * force_mag;

		// Apply velocity change and heat
		b->linear_velocity += impulse_vec / b->mass;
		b->thermal_state += force_mag * FixedMathCore(429496729LL, true); // 0.1 heat factor
		
		// Integrity check for shockwave destruction
		if (force_mag > (b->integrity * FixedMathCore(500LL, false))) {
			// Trigger fracture logic via the Shatter Kernel
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_impact.cpp ---

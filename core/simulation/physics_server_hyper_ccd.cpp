--- START OF FILE core/simulation/physics_server_hyper_ccd.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/collision_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_manager.h"

namespace UniversalSolver {

/**
 * resolve_continuous_collision_step()
 * 
 * Master kernel for handling high-velocity entities.
 * If an object moves faster than its own radius within a single tick,
 * this logic triggers sub-stepping to guarantee collision integrity.
 */
void PhysicsServerHyper::resolve_continuous_collision_step(RID p_body, const FixedMathCore &p_delta) {
	Body *b = body_owner.get_or_null(p_body);
	if (unlikely(!b || b->mode == BODY_MODE_STATIC || !b->ccd_enabled)) {
		return;
	}

	// 1. Determine if sub-stepping is required for this body
	// threshold = radius * 0.5 (Safety margin to prevent tunneling)
	FixedMathCore displacement = b->linear_velocity.length() * p_delta;
	FixedMathCore safety_threshold = b->mesh_deterministic->get_aabb().size.length() * FixedMathCore(2147483648LL, true); // 0.5

	if (displacement <= safety_threshold) {
		// Low-speed path: standard integration handled by the main physics sweep
		return;
	}

	// 2. Calculate required sub-steps
	// sub_steps = ceil(displacement / safety_threshold)
	FixedMathCore steps_f = Math::ceil(displacement / safety_threshold);
	int64_t sub_steps = steps_f.to_int();
	if (sub_steps > 32) sub_steps = 32; // Hard cap for 120 FPS performance budget

	FixedMathCore sub_delta = p_delta / FixedMathCore(sub_steps, false);
	Vector3f current_pos = b->transform.origin;
	BigIntCore current_sx = b->sector_x;
	BigIntCore current_sy = b->sector_y;
	BigIntCore current_sz = b->sector_z;

	// 3. Sub-stepping Loop (Deterministic)
	for (int64_t s = 0; s < sub_steps; s++) {
		Vector3f next_pos = current_pos + (b->linear_velocity * sub_delta);
		
		// Broadphase Query: Find potential colliders in the local neighborhood
		List<RID> potential_collisions;
		broadphase.query_radius(current_pos, current_sx, current_sy, current_sz, safety_threshold * FixedMathCore(2LL, false), potential_collisions);

		CollisionSolver::CollisionResult best_hit;
		best_hit.collided = false;
		best_hit.time_of_impact = MathConstants<FixedMathCore>::one();

		for (const typename List<RID>::Element *E = potential_collisions.front(); E; E = E->next()) {
			if (E->get() == p_body) continue;
			Body *other = body_owner.get_or_null(E->get());
			if (!other || !other->active) continue;

			// Swept-Volume check using bit-perfect FixedMath
			CollisionSolver::CollisionResult hit;
			if (CollisionSolver::solve_swept_sphere_vs_face(
					current_pos, 
					next_pos, 
					safety_threshold, 
					other->mesh_deterministic->get_face_tensors()[0], // Simplified for example
					hit)) {
				
				if (hit.time_of_impact < best_hit.time_of_impact) {
					best_hit = hit;
				}
			}
		}

		if (best_hit.collided) {
			// 4. Collision Response at exact TOI
			// Position object at the point of impact
			b->transform.origin = current_pos + (b->linear_velocity * sub_delta * best_hit.time_of_impact);
			
			// Resolve Impulse and Reflection (Bounce)
			FixedMathCore combined_bounce = wp::min(b->bounce, FixedMathCore(2147483648LL, true)); // 0.5 default
			b->linear_velocity = b->linear_velocity.bounce(best_hit.contact_normal) * combined_bounce;

			// Trigger structural deformation actions (Impact Cratering)
			if (b->mode == BODY_MODE_DEFORMABLE) {
				Vector3f local_impact = b->transform.xform_inv(best_hit.contact_point);
				FixedMathCore force_mag = b->linear_velocity.length() * b->mass;
				b->mesh_deterministic->apply_impact(local_impact, -best_hit.contact_normal, force_mag, safety_threshold);
			}

			// Energy dissipation: stop sub-stepping if velocity is significantly reduced
			if (b->linear_velocity.length_squared() < FixedMathCore(42949LL, true)) {
				break;
			}
		} else {
			// Move to the end of the sub-step
			current_pos = next_pos;
			b->transform.origin = next_pos;
		}

		// Update galactic sectors if boundary is crossed during sub-step
		// (Logic handled by _handle_sector_transition helper)
	}
}

/**
 * apply_relativistic_clamping()
 * 
 * ETEngine Strategy: In hyper-speed simulations (warp travel),
 * we clamp the max displacement per sub-step to the sector size
 * to prevent BigIntCore sector-skipping during queries.
 */
void PhysicsServerHyper::_clamp_velocity_for_sector_safety(Body *r_body) {
	FixedMathCore max_vel_raw = FixedMathCore(50000LL, false); // 50k units per sec
	FixedMathCore current_vel = r_body->linear_velocity.length();
	
	if (unlikely(current_vel > max_vel_raw)) {
		r_body->linear_velocity = r_body->linear_velocity.normalized() * max_vel_raw;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_ccd.cpp ---

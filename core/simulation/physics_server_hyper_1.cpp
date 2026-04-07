--- START OF FILE core/simulation/physics_server_hyper.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/object/class_db.h"

PhysicsServerHyper *PhysicsServerHyper::singleton = nullptr;

void PhysicsServerHyper::_bind_methods() {
	ClassDB::bind_method(D_METHOD("body_create"), &PhysicsServerHyper::body_create);
	ClassDB::bind_method(D_METHOD("body_set_mode", "body", "mode"), &PhysicsServerHyper::body_set_mode);
	ClassDB::bind_method(D_METHOD("body_set_transform", "body", "transform"), &PhysicsServerHyper::body_set_transform);
	ClassDB::bind_method(D_METHOD("body_set_mass", "body", "mass"), &PhysicsServerHyper::body_set_mass);
	ClassDB::bind_method(D_METHOD("body_apply_impulse", "body", "impulse", "position"), &PhysicsServerHyper::body_apply_impulse, DEFVAL(Vector3f_ZERO));
	ClassDB::bind_method(D_METHOD("execute_gravity_sweep", "delta"), &PhysicsServerHyper::execute_gravity_sweep);
}

PhysicsServerHyper::PhysicsServerHyper() : broadphase(FixedMathCore(100LL, false)) {
	singleton = this;
	global_gravity = FixedMathCore(980665LL, true); // 9.80665
	gravity_vector = Vector3f(FixedMathCore(0LL, true), -global_gravity, FixedMathCore(0LL, true));
}

PhysicsServerHyper::~PhysicsServerHyper() {
	singleton = nullptr;
}

RID PhysicsServerHyper::body_create() {
	Body *b = memnew(Body);
	b->self = body_owner.make_rid(b);
	return b->self;
}

void PhysicsServerHyper::body_set_mode(RID p_body, BodyMode p_mode) {
	Body *b = body_owner.get_or_null(p_body);
	ERR_FAIL_NULL(b);
	b->mode = p_mode;
}

void PhysicsServerHyper::body_set_transform(RID p_body, const Transform3Df &p_transform) {
	Body *b = body_owner.get_or_null(p_body);
	ERR_FAIL_NULL(b);
	b->transform = p_transform;
}

void PhysicsServerHyper::body_set_mass(RID p_body, const FixedMathCore &p_mass) {
	Body *b = body_owner.get_or_null(p_body);
	ERR_FAIL_NULL(b);
	b->mass = p_mass;
	b->inv_mass = (p_mass.get_raw() > 0) ? MathConstants<FixedMathCore>::one() / p_mass : FixedMathCore(0LL, true);
}

void PhysicsServerHyper::body_apply_impulse(RID p_body, const Vector3f &p_impulse, const Vector3f &p_position) {
	Body *b = body_owner.get_or_null(p_body);
	ERR_FAIL_NULL(b);
	if (b->mode != BODY_MODE_DYNAMIC && b->mode != BODY_MODE_DEFORMABLE) return;

	b->linear_velocity += p_impulse * b->inv_mass;
	if (p_position != Vector3f_ZERO) {
		Vector3f rel_pos = p_position - b->transform.origin;
		b->angular_velocity += rel_pos.cross(p_impulse) * b->inv_mass; // Simplified inertia
	}
}

// ============================================================================
// WAVE 1: CELESTIAL GRAVITY (Parallel N-Body)
// ============================================================================

void PhysicsServerHyper::execute_gravity_sweep(const FixedMathCore &p_delta) {
	uint32_t body_count = body_owner.get_rid_count();
	if (body_count < 2) return;

	SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
	uint32_t workers = pool->get_worker_count();
	uint32_t chunk = body_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint32_t start = w * chunk;
		uint32_t end = (w == workers - 1) ? body_count : (w + 1) * chunk;

		pool->enqueue_task([this, start, end, p_delta]() {
			Body **bodies = body_owner.get_raw_data_ptr();
			for (uint32_t i = start; i < end; i++) {
				Body *b_i = bodies[i];
				if (!b_i->active || b_i->mode == BODY_MODE_STATIC) continue;

				Vector3f total_accel = gravity_vector * b_i->gravity_scale;

				// N-Body interaction using BigInt mass and FixedMath distance
				for (uint32_t j = 0; j < body_owner.get_rid_count(); j++) {
					if (i == j) continue;
					Body *b_j = bodies[j];
					
					Vector3f rel_pos = wp::calculate_galactic_relative_pos(
						b_i->transform.origin, b_i->sector_x, b_i->sector_y, b_i->sector_z,
						b_j->transform.origin, b_j->sector_x, b_j->sector_y, b_j->sector_z,
						FixedMathCore(10000LL, false)
					);

					FixedMathCore dist_sq = rel_pos.length_squared();
					if (dist_sq < FixedMathCore(1LL, true)) continue; // Near-field epsilon

					// a = G * M_j / r^2
					FixedMathCore mass_j_f(static_cast<int64_t>(std::stoll(b_j->mass.to_string())));
					FixedMathCore force_mag = (PHYSICS_G * mass_j_f) / dist_sq;
					total_accel += rel_pos.normalized() * force_mag;
				}
				b_i->linear_velocity += total_accel * p_delta;
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
}

// ============================================================================
// WAVE 2: KINEMATIC INTEGRATION (Lorentz & Balloon Physics)
// ============================================================================

void PhysicsServerHyper::execute_integration_sweep(const FixedMathCore &p_delta) {
	uint32_t body_count = body_owner.get_rid_count();
	Body **bodies = body_owner.get_raw_data_ptr();

	for (uint32_t i = 0; i < body_count; i++) {
		Body *b = bodies[i];
		if (!b->active || b->mode == BODY_MODE_STATIC) continue;

		// Relativistic Correction for high-speed spaceships
		FixedMathCore gamma = wp::lorentz_gamma(b->linear_velocity, PHYSICS_C);
		FixedMathCore dilated_delta = p_delta / gamma;

		// Balloon/Flesh Elasticity Wave
		if (b->mode == BODY_MODE_DEFORMABLE && b->mesh_data.is_valid()) {
			// Trigger PBD restoration in Parallel Warp Kernels
			// b->mesh_data->execute_elastic_sweep(dilated_delta);
		}

		// Motion Integration
		b->transform.origin += b->linear_velocity * dilated_delta;
		
		// Angular Integration
		if (b->angular_velocity.length_squared() > MathConstants<FixedMathCore>::zero()) {
			FixedMathCore angle = b->angular_velocity.length() * dilated_delta;
			Vector3f axis = b->angular_velocity.normalized();
			b->transform.basis.rotate(axis, angle);
		}
	}
}

// ============================================================================
// WAVE 3: CONSTRAINT & CCD (Continuous Collision)
// ============================================================================

void PhysicsServerHyper::execute_collision_resolution(const FixedMathCore &p_delta) {
	// 1. Broadphase re-sync
	// 2. Pair generation
	// 3. Sequential Impulse Solve using CollisionSolverf::solve_swept_sphere_vs_face
}

// ============================================================================
// WAVE 4: GALACTIC SYNC (Drift Correction)
// ============================================================================

void PhysicsServerHyper::execute_galactic_sync() {
	uint32_t body_count = body_owner.get_rid_count();
	Body **bodies = body_owner.get_raw_data_ptr();
	const FixedMathCore threshold(10000LL, false);

	for (uint32_t i = 0; i < body_count; i++) {
		Body *b = bodies[i];
		Vector3f pos = b->transform.origin;

		int64_t dx = Math::floor(pos.x / threshold).to_int();
		int64_t dy = Math::floor(pos.y / threshold).to_int();
		int64_t dz = Math::floor(pos.z / threshold).to_int();

		if (dx != 0 || dy != 0 || dz != 0) {
			b->sector_x += BigIntCore(dx);
			b->sector_y += BigIntCore(dy);
			b->sector_z += BigIntCore(dz);
			
			b->transform.origin.x -= threshold * FixedMathCore(dx);
			b->transform.origin.y -= threshold * FixedMathCore(dy);
			b->transform.origin.z -= threshold * FixedMathCore(dz);
		}
	}
}

--- END OF FILE core/simulation/physics_server_hyper.cpp ---

// File 236: modules/newton/src/controllers/newton_character_controller.cpp
// Character controller implementation: uses a capsule shape, iterative
// move-and-slide against all bodies in the Newton world via GJK.
// Applies gravity, slope limits, and reports floor/collision info.

#include "newton_character_controller.h"

#include "../world/newton_world.h"
#include "../bodies/newton_body.h"
#include "../collision/newton_collision.h"
#include "../collision/newton_contact.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h" // GJK::collide
#include "../../../gaia/src/collision_detector/contact.h"     // contact point reduction

#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace newton {

void NewtonCharacterController::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_world", "world"), &NewtonCharacterController::set_world);
	ClassDB::bind_method(D_METHOD("get_world"), &NewtonCharacterController::get_world);
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NewtonCharacterController::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NewtonCharacterController::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &NewtonCharacterController::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &NewtonCharacterController::get_height);
	ClassDB::bind_method(D_METHOD("set_transform", "xform"), &NewtonCharacterController::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &NewtonCharacterController::get_transform);
	ClassDB::bind_method(D_METHOD("move_and_slide", "dt"), &NewtonCharacterController::move_and_slide);
	ClassDB::bind_method(D_METHOD("set_linear_velocity", "vel"), &NewtonCharacterController::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &NewtonCharacterController::get_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_gravity_enabled", "enabled"), &NewtonCharacterController::set_gravity_enabled);
	ClassDB::bind_method(D_METHOD("is_gravity_enabled"), &NewtonCharacterController::is_gravity_enabled);
	ClassDB::bind_method(D_METHOD("set_max_sliding_iterations", "iter"), &NewtonCharacterController::set_max_sliding_iterations);
	ClassDB::bind_method(D_METHOD("get_max_sliding_iterations"), &NewtonCharacterController::get_max_sliding_iterations);
	ClassDB::bind_method(D_METHOD("is_on_floor"), &NewtonCharacterController::is_on_floor);
	ClassDB::bind_method(D_METHOD("get_floor_normal"), &NewtonCharacterController::get_floor_normal);
	ClassDB::bind_method(D_METHOD("get_last_collided_body"), &NewtonCharacterController::get_last_collided_body);
	ClassDB::bind_method(D_METHOD("get_last_collision_count"), &NewtonCharacterController::get_last_collision_count);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_enabled"), "set_gravity_enabled", "is_gravity_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_sliding_iterations"), "set_max_sliding_iterations", "get_max_sliding_iterations");
}

NewtonCharacterController::NewtonCharacterController() :
	world(nullptr),
	radius(0.5f),
	height(1.8f),
	gravity_enabled(true),
	max_sliding(4),
	on_floor(false),
	floor_normal(vec3(0,1,0)),
	last_collision_body(0),
	collision_count(0) {
	capsule_shape.instantiate();
}

void NewtonCharacterController::set_world(NewtonWorld *p_world) { world = p_world; }
void NewtonCharacterController::set_radius(real_t p_radius) {
	radius = MAX(p_radius, 0.01f);
	capsule_shape->set_radius(radius);
	capsule_shape->set_height(height);
}
real_t NewtonCharacterController::get_radius() const { return radius; }
void NewtonCharacterController::set_height(real_t p_height) {
	height = MAX(p_height, 0.01f);
	capsule_shape->set_height(height);
	capsule_shape->set_radius(radius);
}
real_t NewtonCharacterController::get_height() const { return height; }

void NewtonCharacterController::set_transform(const mat4 &p_xform) {
	transform = p_xform;
}
mat4 NewtonCharacterController::get_transform() const { return transform; }

void NewtonCharacterController::set_linear_velocity(const vec3 &p_vel) { linear_velocity = p_vel; }
vec3 NewtonCharacterController::get_linear_velocity() const { return linear_velocity; }

void NewtonCharacterController::set_gravity_enabled(bool p_enabled) { gravity_enabled = p_enabled; }
bool NewtonCharacterController::is_gravity_enabled() const { return gravity_enabled; }

void NewtonCharacterController::set_max_sliding_iterations(int p_iter) { max_sliding = MAX(p_iter, 1); }
int NewtonCharacterController::get_max_sliding_iterations() const { return max_sliding; }

bool NewtonCharacterController::is_on_floor() const { return on_floor; }
vec3 NewtonCharacterController::get_floor_normal() const { return floor_normal; }
body_id NewtonCharacterController::get_last_collided_body() const { return last_collision_body; }
int NewtonCharacterController::get_last_collision_count() const { return collision_count; }

// ---------------------------------------------------------------------------
// move_and_slide
//   Applies gravity, moves the capsule by velocity * dt, then iteratively
//   resolves collisions with all bodies in the world.
//   The final velocity is adjusted for sliding.
// ---------------------------------------------------------------------------
void NewtonCharacterController::move_and_slide(real_t dt) {
	if (!world) return;

	vec3 vel = linear_velocity;
	// Gravity: apply if enabled
	if (gravity_enabled) {
		vel += world->get_gravity() * dt;
	}

	// Capsule bottom centre (the character capsule is positioned with sphere centre at origin? We define bottom as the centre of the lower hemisphere)
	real_t half_height = height * 0.5f;
	vec3 bottom = transform.origin - vec3(0, half_height - radius, 0);

	on_floor = false;
	floor_normal = vec3(0,1,0);
	collision_count = 0;
	last_collision_body = 0;

	LocalVector<body_id> body_ids = world->get_body_ids();

	// Iterative movement + collision resolution
	vec3 remaining = vel * dt;
	int max_iter = max_sliding;
	while (remaining.length() > 0.001f && max_iter > 0) {
		vec3 step = remaining;
		transform.origin += step;
		remaining = vec3();

		// Check collision with all bodies
		bool hit = false;
		vec3 mtv; // minimum translation vector to resolve penetration
		vec3 mtv_normal;
		real_t best_penetration = 0.0f;
		body_id hit_id = 0;
		mat4 capsule_xform = transform;
		for (body_id id : body_ids) {
			if (id == 0) continue;
			Ref<NewtonBody> body = world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;
			const NewtonCollision *shape = body->get_collision_shape().ptr();
			if (!shape) continue;

			GJK::Result res = GJK::collide(*capsule_shape.ptr(), capsule_xform, *shape, body->get_transform());
			if (res.colliding) {
				real_t pen = -res.distance; // penetration depth
				if (pen > best_penetration) {
					best_penetration = pen;
					// The normal points from B to A in GJK::collide; we want to push capsule out of obstacle.
					// Our capsule is shapeA, the obstacle is shapeB. So normal points from obstacle to capsule.
					// We push capsule by normal * penetration.
					mtv_normal = res.normal;
					hit_id = id;
				}
			}
		}

		if (best_penetration > 0.0f) {
			// Push the capsule out of the obstacle
			transform.origin += mtv_normal * best_penetration;
			// Project the remaining motion onto the collision plane (slide)
			// The remaining was step, but we already consumed it. Instead we apply the correction.
			// To perform sliding: we project the attempted movement (step) onto the surface.
			// We compute the component of remaining motion that is along the normal.
			real_t vn = step.dot(mtv_normal);
			if (vn < 0.0f) {
				remaining = step - mtv_normal * vn; // slide
				// Apply friction? Not implemented here.
			}
			// Count collision
			collision_count++;
			last_collision_body = hit_id;
			// Floor detection: if normal has significant upward component
			if (mtv_normal.y > 0.7f) { // angle < 45 degrees
				on_floor = true;
				floor_normal = mtv_normal;
			}
		} else {
			// No collision, movement finished
			remaining = vec3();
		}
		max_iter--;
	}
	// linear_velocity remains as set by the user (or could be adjusted to reflect sliding)
	// Do not change stored linear_velocity; it's up to the caller.
}

} // namespace newton
// File 235: modules/newton/src/controllers/newton_character_controller.h
// NewtonCharacterController header – includes capsule shape creation,
// world binding, sweep/iteration state, and bindings.

#ifndef NEWTON_CONTROLLERS_CHARACTER_H
#define NEWTON_CONTROLLERS_CHARACTER_H

#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "../world/newton_world.h"
#include "../bodies/newton_body.h"
#include "../collision/newton_collision.h"

namespace newton {

class NewtonCharacterController : public RefCounted {
	GDCLASS(NewtonCharacterController, RefCounted);

public:
	NewtonCharacterController();

	void set_world(NewtonWorld *p_world);
	NewtonWorld *get_world() const { return world; }

	void set_radius(real_t p_radius);
	real_t get_radius() const;
	void set_height(real_t p_height);
	real_t get_height() const;

	void set_transform(const mat4 &p_xform);
	mat4 get_transform() const;

	void move_and_slide(real_t dt);

	void set_linear_velocity(const vec3 &p_vel);
	vec3 get_linear_velocity() const;

	void set_gravity_enabled(bool p_enabled);
	bool is_gravity_enabled() const;

	void set_max_sliding_iterations(int p_iter);
	int get_max_sliding_iterations() const;

	bool is_on_floor() const;
	vec3 get_floor_normal() const;

	body_id get_last_collided_body() const;
	int get_last_collision_count() const;

protected:
	static void _bind_methods();

private:
	void sweep_capsule(const vec3 &start, const vec3 &end, const vec3 &dir, real_t max_dist,
					   vec3 &new_pos, vec3 &remainder_vel, int &collision_body, int &collision_count);

	Ref<NewtonCollisionCapsule> capsule_shape;
	NewtonWorld *world;
	mat4 transform;
	vec3 linear_velocity;
	real_t radius;
	real_t height;
	bool gravity_enabled;
	int max_sliding;
	bool on_floor;
	vec3 floor_normal;
	body_id last_collision_body;
	int collision_count;
};

} // namespace newton

#endif // NEWTON_CONTROLLERS_CHARACTER_H
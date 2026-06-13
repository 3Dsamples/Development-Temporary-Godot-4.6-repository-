// File 240: modules/newton/src/nodes/newton_rigid_body_3d.h
// NewtonRigidBody3D – a Godot Node3D that wraps a NewtonBody for direct
// scene usage without the physics server.  It provides the same interface
// as Godot's RigidDynamicBody3D but powered by Newton Dynamics.

#ifndef NEWTON_NODES_RIGID_BODY_3D_H
#define NEWTON_NODES_RIGID_BODY_3D_H

#include "scene/3d/node_3d.h"
#include "scene/main/scene_tree.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "../world/newton_world.h"
#include "../bodies/newton_body.h"
#include "../collision/newton_collision.h"

namespace newton {

class NewtonRigidBody3D : public Node3D {
	GDCLASS(NewtonRigidBody3D, Node3D);

public:
	enum Mode {
		MODE_DYNAMIC,
		MODE_STATIC,
		MODE_KINEMATIC
	};

	NewtonRigidBody3D();
	void _notification(int p_what);

	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_mass(real_t p_mass);
	real_t get_mass() const;

	void set_collision_shape(Ref<NewtonCollision> p_shape);
	Ref<NewtonCollision> get_collision_shape() const;

	void set_linear_velocity(const vec3 &p_vel);
	vec3 get_linear_velocity() const;

	void set_angular_velocity(const vec3 &p_vel);
	vec3 get_angular_velocity() const;

	void apply_impulse(const vec3 &p_impulse, const vec3 &p_world_point = vec3());
	void apply_force(const vec3 &p_force, const vec3 &p_world_point = vec3());

	void set_gravity_enabled(bool p_enabled);
	bool is_gravity_enabled() const;

	void set_ccd_enabled(bool p_enabled);
	bool is_ccd_enabled() const;

protected:
	static void _bind_methods();

private:
	void _find_world();
	void _sync_from_newton();
	void _sync_to_newton();

	Mode mode;
	Ref<NewtonBody> newton_body;
	Ref<NewtonCollision> collision_shape;
	NewtonWorld *world;
	body_id body_id;
	bool world_found;
};

} // namespace newton

#endif // NEWTON_NODES_RIGID_BODY_3D_H
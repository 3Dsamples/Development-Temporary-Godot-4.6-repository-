// File 309: modules/vienna/src/nodes/vienna_rigid_body_3d.h
// ViennaRigidBody3D – a Godot Node3D that wraps a ViennaBody for direct
// scene usage.  It provides the same interface as Godot's RigidDynamicBody3D
// but powered by ViennaPhysicsEngine.  Supports collision shape, mass,
// inertia, forces, damping, gravity, and sleep.

#ifndef VIENNA_NODES_RIGID_BODY_3D_H
#define VIENNA_NODES_RIGID_BODY_3D_H

#include "scene/3d/node_3d.h"
#include "scene/main/scene_tree.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../collision/vienna_shape.h"
#include "../materials/vienna_material.h"

namespace vienna {

class ViennaWorldNode3D; // forward

class ViennaRigidBody3D : public Node3D {
	GDCLASS(ViennaRigidBody3D, Node3D);

public:
	enum Mode {
		MODE_DYNAMIC,
		MODE_STATIC,
		MODE_KINEMATIC
	};

	ViennaRigidBody3D();
	void _notification(int p_what);

	// Mode
	void set_mode(Mode p_mode);
	Mode get_mode() const;

	// Mass
	void set_mass(real_t p_mass);
	real_t get_mass() const;

	// Collision shape (a ViennaShape resource)
	void set_collision_shape(const Ref<ViennaShape> &p_shape);
	Ref<ViennaShape> get_collision_shape() const;

	// Velocity
	void set_linear_velocity(const vec3 &p_vel);
	vec3 get_linear_velocity() const;
	void set_angular_velocity(const vec3 &p_vel);
	vec3 get_angular_velocity() const;

	// Forces & impulses (the node accumulates them then applies each physics step)
	void apply_force(const vec3 &p_force, const vec3 &p_world_point = vec3());
	void apply_central_force(const vec3 &p_force);
	void apply_impulse(const vec3 &p_impulse, const vec3 &p_world_point = vec3());
	void apply_central_impulse(const vec3 &p_impulse);

	// Damping
	void set_linear_damping(real_t p_damping);
	real_t get_linear_damping() const;
	void set_angular_damping(real_t p_damping);
	real_t get_angular_damping() const;

	// Gravity
	void set_gravity_enabled(bool p_enabled);
	bool is_gravity_enabled() const;

	// Sleep
	void set_sleeping(bool p_sleeping);
	bool is_sleeping() const;

	// Access the underlying ViennaBody (advanced)
	Ref<ViennaBody> get_vienna_body() const { return vienna_body; }

	// Material
	void set_material(const Ref<ViennaMaterial> &p_material);
	Ref<ViennaMaterial> get_material() const;

	// CCD (continuous collision detection)
	void set_ccd_enabled(bool p_enabled);
	bool is_ccd_enabled() const;

protected:
	static void _bind_methods();

private:
	void _find_world();
	void _sync_from_physics();
	void _sync_to_physics();

	Mode mode;
	Ref<ViennaBody> vienna_body;
	Ref<ViennaShape> collision_shape;
	ViennaWorld *world;                  // cached pointer to the ViennaWorld
	body_id body_id;                     // assigned by the world
	bool world_found;

	// Forces accumulated between physics steps
	vec3 force_accum;
	vec3 torque_accum;

	// Damping & material
	real_t linear_damping;
	real_t angular_damping;
	Ref<ViennaMaterial> material;

	bool ccd_enabled;
	bool gravity_enabled;
	bool is_asleep;
};

} // namespace vienna

#endif // VIENNA_NODES_RIGID_BODY_3D_H
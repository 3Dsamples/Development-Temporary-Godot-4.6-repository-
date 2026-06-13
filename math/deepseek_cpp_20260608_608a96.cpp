// File 271: modules/vienna/src/bodies/vienna_body.h
// ViennaBody – rigid body for ViennaPhysicsEngine with mass, inertia, damping,
// CCD, forces, and integration.  Compatible with Godot's ref‑counted system.

#ifndef VIENNA_BODIES_VIENNA_BODY_H
#define VIENNA_BODIES_VIENNA_BODY_H

#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/aabb.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

class ViennaShape;

class ViennaBody : public RefCounted {
	GDCLASS(ViennaBody, RefCounted);

public:
	ViennaBody();
	virtual ~ViennaBody();

	// --- Type ---
	void set_type(BodyType p_type);
	BodyType get_type() const { return body_type; }

	// --- Transform ---
	void set_transform(const mat4 &p_xform);
	const mat4 &get_transform() const { return transform; }
	vec3 get_position() const { return transform.origin; }
	mat3 get_rotation() const { return transform.basis; }

	// --- Velocities ---
	void set_linear_velocity(const vec3 &p_vel);
	vec3 get_linear_velocity() const { return linear_velocity; }
	void set_angular_velocity(const vec3 &p_vel);
	vec3 get_angular_velocity() const { return angular_velocity; }

	// --- Mass properties ---
	void set_mass(real_t p_mass);
	real_t get_mass() const { return mass; }
	real_t get_inverse_mass() const { return inverse_mass; }
	void set_inertia(const mat3 &p_inertia);
	mat3 get_inertia_local() const { return inertia_local; }
	mat3 get_inverse_inertia_world() const { return inverse_inertia_world; }
	void update_inverse_inertia();

	// --- Damping ---
	void set_linear_damping(real_t p_damp);
	real_t get_linear_damping() const { return linear_damping; }
	void set_angular_damping(real_t p_damp);
	real_t get_angular_damping() const { return angular_damping; }

	// --- AABB (from collision shape) ---
	void set_collision_aabb(const aabb &p_aabb) { cached_aabb = p_aabb; }
	aabb get_aabb() const { return cached_aabb; }

	// --- Collision shape ---
	void set_collision_shape(const Ref<ViennaShape> &p_shape);
	Ref<ViennaShape> get_collision_shape() const { return collision_shape; }

	// --- Material ID ---
	void set_material_id(material_id p_id) { mat_id = p_id; }
	material_id get_material_id() const { return mat_id; }

	// --- Force accumulation ---
	void apply_force(const vec3 &p_force, const vec3 &p_world_point);
	void apply_impulse(const vec3 &p_impulse, const vec3 &p_world_point);
	void clear_forces();

	// --- Integration (called by the world) ---
	void integrate_velocity(real_t p_dt);
	void integrate_position(real_t p_dt);

	// --- Sleep ---
	void set_active(bool p_active);
	bool is_active() const { return active; }
	void increment_sleep_counter() { sleep_counter++; }
	void reset_sleep_counter() { sleep_counter = 0; }
	int get_sleep_counter() const { return sleep_counter; }

	// --- Gravity ---
	void set_gravity_enabled(bool p_enabled) { gravity_enabled = p_enabled; }
	bool is_gravity_enabled() const { return gravity_enabled; }

protected:
	static void _bind_methods();

private:
	BodyType body_type = BodyType::DYNAMIC;
	mat4 transform;
	vec3 linear_velocity;
	vec3 angular_velocity;
	vec3 force_accum;
	vec3 torque_accum;
	real_t mass = 1.0;
	real_t inverse_mass = 1.0;
	mat3 inertia_local;
	mat3 inverse_inertia_local;
	mat3 inverse_inertia_world;
	aabb cached_aabb;
	Ref<ViennaShape> collision_shape;
	material_id mat_id = 0;
	bool active = true;
	bool gravity_enabled = true;
	int sleep_counter = 0;
	real_t linear_damping = 0.0;
	real_t angular_damping = 0.0;
};

} // namespace vienna

#endif // VIENNA_BODIES_VIENNA_BODY_H
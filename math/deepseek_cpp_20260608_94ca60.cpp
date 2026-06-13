// File 315: modules/vienna/src/vehicles/vienna_vehicle.h
// ViennaVehicle – a wheeled vehicle simulation built on ViennaBody for the
// chassis and ray‑cast wheels. Handles suspension, tire friction, steering,
// engine torque, and braking.  Uses Gaia BVH for efficient ground‑ray queries.

#ifndef VIENNA_VEHICLES_VIENNA_VEHICLE_H
#define VIENNA_VEHICLES_VIENNA_VEHICLE_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../bodies/vienna_body.h"
#include "../world/vienna_world.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/bvh/query.h"

namespace vienna {

class ViennaVehicle : public RefCounted {
	GDCLASS(ViennaVehicle, RefCounted);

public:
	struct Wheel {
		vec3 attachment_point;         // local to chassis
		vec3 suspension_dir;          // local direction (typically downwards)
		real_t suspension_length;     // rest length
		real_t suspension_spring;     // stiffness (N/m)
		real_t suspension_damper;     // damping (Ns/m)
		real_t wheel_radius;
		real_t lateral_friction;
		real_t longitudinal_friction;
		real_t steering_angle;        // current steering (radians), applied only to steered wheels
		bool   is_drive_wheel;
		bool   is_steer_wheel;
		Wheel() : attachment_point(), suspension_dir(0,-1,0), suspension_length(0.3f),
				  suspension_spring(30000.0f), suspension_damper(3000.0f), wheel_radius(0.35f),
				  lateral_friction(1.0f), longitudinal_friction(1.0f), steering_angle(0.0f),
				  is_drive_wheel(true), is_steer_wheel(true) {}
	};

	ViennaVehicle();

	void set_chassis_body(Ref<ViennaBody> p_chassis);
	Ref<ViennaBody> get_chassis_body() const { return chassis; }

	// Wheel management
	void add_wheel(const Wheel &p_wheel);
	int get_wheel_count() const { return wheels.size(); }
	Wheel &get_wheel(int p_idx) { return wheels[p_idx]; }
	const Wheel &get_wheel(int p_idx) const { return wheels[p_idx]; }
	void clear_wheels();

	// Control inputs (range 0..1 for throttle/brake, radians for steering)
	void set_throttle(real_t p_throttle);
	void set_steering(real_t p_steering);
	void set_brake(real_t p_brake);
	real_t get_throttle() const { return throttle; }
	real_t get_steering() const { return steering; }
	real_t get_brake() const { return brake; }

	// Engine parameters
	void set_engine_max_force(real_t p_force);
	void set_engine_max_speed(real_t p_speed);
	real_t get_engine_max_force() const { return engine_max_force; }
	real_t get_engine_max_speed() const { return engine_max_speed; }

	// Update the vehicle physics; should be called every substep.
	void update(real_t dt, ViennaWorld *p_world);

protected:
	static void _bind_methods();

private:
	Ref<ViennaBody> chassis;
	LocalVector<Wheel> wheels;
	real_t throttle;
	real_t steering;
	real_t brake;
	real_t engine_max_force;
	real_t engine_max_speed;
};

} // namespace vienna

#endif // VIENNA_VEHICLES_VIENNA_VEHICLE_H
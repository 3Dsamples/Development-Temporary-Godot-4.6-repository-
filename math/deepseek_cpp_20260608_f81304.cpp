// File 200: modules/newton/src/vehicles/newton_vehicle.h
// Newton vehicle – simulates a wheeled vehicle using rigid bodies for chassis
// and ray‑cast / constraint wheels. Provides throttle, steering, brake,
// and engine parameters. Each wheel is simulated as a spring‑damper.

#ifndef NEWTON_VEHICLE_NEWTON_VEHICLE_H
#define NEWTON_VEHICLE_NEWTON_VEHICLE_H

#include "core/object/ref_counted.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/templates/local_vector.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

namespace newton {

class NewtonBody;
class NewtonWorld;

class NewtonVehicle : public RefCounted {
	GDCLASS(NewtonVehicle, RefCounted);

public:
	struct Wheel {
		vec3 attachment_point;       // local to chassis
		vec3 suspension_dir;        // local direction (usually down)
		real_t suspension_length;   // rest length
		real_t suspension_spring;   // stiffness (N/m)
		real_t suspension_damper;   // damping (Ns/m)
		real_t wheel_radius;
		real_t friction;            // lateral friction coefficient
		real_t longitudinal_friction;
		real_t steering_angle;      // current steering (radians)
		bool   is_drive_wheel;
		bool   is_steer_wheel;

		Wheel() : attachment_point(), suspension_dir(0, -1, 0), suspension_length(0.3),
				  suspension_spring(30000.0), suspension_damper(3000.0), wheel_radius(0.35),
				  friction(1.0), longitudinal_friction(1.0), steering_angle(0.0),
				  is_drive_wheel(true), is_steer_wheel(true) {}
	};

	NewtonVehicle();

	// Attach to a chassis body.
	void set_chassis_body(NewtonBody *p_body);
	NewtonBody *get_chassis_body() const { return chassis; }

	// Wheel management.
	void add_wheel(const Wheel &p_wheel);
	int get_wheel_count() const { return wheels.size(); }
	Wheel &get_wheel(int p_idx);
	void clear_wheels();

	// Control inputs (applied each step).
	void set_throttle(real_t p_throttle);      // 0..1
	real_t get_throttle() const { return throttle; }

	void set_steering(real_t p_steering);      // radians
	real_t get_steering() const { return steering; }

	void set_brake(real_t p_brake);            // 0..1
	real_t get_brake() const { return brake; }

	// Engine settings.
	void set_engine_max_force(real_t p_force) { engine_max_force = MAX(p_force, 0.0); }
	real_t get_engine_max_force() const { return engine_max_force; }

	void set_engine_max_speed(real_t p_speed) { engine_max_speed = MAX(p_speed, 0.0); }
	real_t get_engine_max_speed() const { return engine_max_speed; }

	// Update the vehicle physics (called each substep by the world).
	void update(NewtonWorld *world, real_t dt);

protected:
	static void _bind_methods();

private:
	NewtonBody *chassis;
	LocalVector<Wheel> wheels;
	real_t throttle;
	real_t steering;
	real_t brake;
	real_t engine_max_force;
	real_t engine_max_speed;
};

} // namespace newton

#endif // NEWTON_VEHICLE_NEWTON_VEHICLE_H
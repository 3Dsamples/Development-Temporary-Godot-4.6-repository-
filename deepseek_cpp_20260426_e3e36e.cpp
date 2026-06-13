// File 166: modules/genesis/src/entities/drone_entity.cpp
// Drone entity methods: rotor dynamics, thrust / torque computation,
// and attitude controller.

#include "drone_entity.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

namespace genesis {

void DroneEntity::update_rotors(real_t dt) {
	for (Rotor &r : rotors) {
		real_t alpha = dt / (r.motor_time_constant + dt);
		r.angular_velocity += alpha * (r.target_velocity - r.angular_velocity);
	}
}

void DroneEntity::compute_forces(Vector3 &r_thrust_world, Vector3 &r_torque_world) const {
	r_thrust_world = Vector3();
	r_torque_world = Vector3();
	const Basis R = get_rotation();   // world rotation matrix
	for (const Rotor &r : rotors) {
		real_t omega = r.angular_velocity;
		real_t thrust_mag = r.thrust_coefficient * omega * omega;   // k_f * ω²
		real_t torque_mag = r.torque_coefficient * omega * omega;  // k_m * ω²
		// Thrust vector in world space
		Vector3 thrust_world = R.xform(r.spin_axis) * thrust_mag;
		r_thrust_world += thrust_world;
		// Torque from rotor position × thrust
		Vector3 pos_world = R.xform(r.position);
		r_torque_world += pos_world.cross(thrust_world);
		// Reaction torque around spin axis
		r_torque_world += R.xform(r.spin_axis) * torque_mag;
	}
}

void DroneEntity::integrate_velocity(real_t dt) {
	if (!active) return;
	update_rotors(dt);
	Vector3 thrust, torque;
	compute_forces(thrust, torque);
	// Apply forces/torques to the rigid frame
	apply_force(thrust);                          // thrust at center of mass
	// Torque: we accumulate it in torque_accum via base class apply_force(Vector3(), torque)
	apply_force(Vector3(), torque);               // overload that adds torque
	RigidEntity::integrate_velocity(dt);          // standard rigid integration
}

void DroneEntity::compute_control_from_attitude(const Basis &desired_attitude, real_t dt) {
	Quaternion q_cur(get_rotation());
	Quaternion q_des(desired_attitude);
	Quaternion q_err = q_des * q_cur.inverse();
	Vector3 axis; real_t angle;
	q_err.get_axis_angle(axis, angle);
	real_t Kp = 10.0f;                          // proportional gain
	Vector3 torque_cmd = axis * angle * Kp;
	// Simple linear mapping: base throttle plus differential torque on rotors
	real_t omega_base = 300.0f;
	real_t domega = torque_cmd.length() * 0.01f;
	set_rotor_velocity(0, omega_base + domega);
	set_rotor_velocity(1, omega_base - domega);
	set_rotor_velocity(2, omega_base + domega);
	set_rotor_velocity(3, omega_base - domega);
}

} // namespace genesis
// File 120: modules/genesis/src/entities/drone_entity.h
// Quad‑rotor drone entity. Simulates rigid‑body dynamics with four
// individual rotors generating thrust and torque. Rotor speeds are
// controllable via angular velocity commands or direct PWM.

#ifndef GENESIS_ENTITIES_DRONE_ENTITY_H
#define GENESIS_ENTITIES_DRONE_ENTITY_H

#include "rigid_entity.h"
#include "../core/genesis_types.h"
#include "core/math/transform_3d.h"
#include "core/templates/local_vector.h"

namespace genesis {

class DroneEntity : public RigidEntity {
	GDCLASS(DroneEntity, RigidEntity);

public:
	// Rotor configuration
	struct Rotor {
		Vector3 position;           // local offset from drone center
		Vector3 spin_axis;          // rotation axis in local frame (e.g., (0,1,0) for upward)
		real_t  thrust_coefficient; // k_f [N/(rad/s)^2]
		real_t  torque_coefficient; // k_m [Nm/(rad/s)^2]
		real_t  inertia;            // rotor moment of inertia [kg·m^2]
		real_t  angular_velocity;   // current angular speed [rad/s]
		real_t  target_velocity;    // desired speed (set by controller)
		real_t  max_velocity;       // saturation limit
		real_t  motor_time_constant; // first‑order lag (s)
	};

	DroneEntity() : RigidEntity() {
		// Default: X‑configuration quadcopter with arms 0.2 m length.
		geometry_type = GeometryType::SPHERE;
		set_radius(0.05);
		set_mass(0.5);
		set_inertia(Basis().scaled(Vector3(0.005, 0.005, 0.008)));
		// Rotors: four corners
		const real_t arm_len = 0.2;
		const real_t kf = 1.0e-5;
		const real_t km = 1.0e-7;
		rotors.resize(4);
		rotors[0] = { Vector3( arm_len, 0,  arm_len), Vector3(0,1,0), kf, km, 1e-5, 0, 0, 1000, 0.02 };
		rotors[1] = { Vector3(-arm_len, 0, -arm_len), Vector3(0,1,0), kf, km, 1e-5, 0, 0, 1000, 0.02 };
		rotors[2] = { Vector3( arm_len, 0, -arm_len), Vector3(0,1,0), -kf, -km, 1e-5, 0, 0, 1000, 0.02 };
		rotors[3] = { Vector3(-arm_len, 0,  arm_len), Vector3(0,1,0), -kf, -km, 1e-5, 0, 0, 1000, 0.02 };
	}

	// Access rotors
	int get_rotor_count() const { return rotors.size(); }
	Rotor &get_rotor(int p_idx) { return rotors[p_idx]; }
	const Rotor &get_rotor(int p_idx) const { return rotors[p_idx]; }

	// Set target rotor angular velocities (range [-max, max])
	void set_rotor_velocity(int p_idx, real_t p_omega) {
		ERR_FAIL_INDEX(p_idx, rotors.size());
		rotors[p_idx].target_velocity = CLAMP(p_omega, -rotors[p_idx].max_velocity, rotors[p_idx].max_velocity);
	}

	// --- Update rotor speeds with first‑order motor dynamics ---
	void update_rotors(real_t dt) {
		for (Rotor &r : rotors) {
			real_t alpha = dt / (r.motor_time_constant + dt);
			r.angular_velocity += alpha * (r.target_velocity - r.angular_velocity);
		}
	}

	// --- Compute total thrust and torque from rotors ---
	void compute_forces(Vector3 &r_thrust_world, Vector3 &r_torque_world) const {
		r_thrust_world = Vector3();
		r_torque_world = Vector3();
		const Basis R = get_rotation();
		for (const Rotor &r : rotors) {
			real_t omega = r.angular_velocity;
			real_t thrust_mag = r.thrust_coefficient * omega * omega;
			real_t torque_mag = r.torque_coefficient * omega * omega;
			// Thrust direction: local spin axis rotated to world
			Vector3 thrust_world = R.xform(r.spin_axis) * thrust_mag;
			r_thrust_world += thrust_world;
			// Torque from rotor position × thrust
			Vector3 pos_world = R.xform(r.position);
			r_torque_world += pos_world.cross(thrust_world);
			// Reaction torque around spin axis
			r_torque_world += R.xform(r.spin_axis) * torque_mag;
		}
	}

	// --- Override physics integration to apply rotor forces ---
	virtual void integrate_velocity(real_t dt) override {
		if (!active) return;
		update_rotors(dt);
		Vector3 thrust, torque;
		compute_forces(thrust, torque);
		// Add to force accumulator
		apply_force(thrust);
		apply_force(Vector3(), torque); // torque via special overload? We'll add base class torque.
		// Then proceed with standard integration
		RigidEntity::integrate_velocity(dt);
	}

	// --- PID‑based attitude controller (stub) ---
	void compute_control_from_attitude(const Basis &desired_attitude, real_t dt) {
		Quaternion q_cur(get_rotation());
		Quaternion q_des(desired_attitude);
		Quaternion q_err = q_des * q_cur.inverse();
		Vector3 axis; real_t angle;
		q_err.get_axis_angle(axis, angle);
		// Simple proportional: set rotor differential speeds to create torque
		real_t Kp = 10.0;
		Vector3 torque_cmd = axis * angle * Kp;
		// Map torque to rotor speeds (requires invert of rotor allocation matrix)
		// This is a placeholder linear mapping.
		real_t omega_base = 300.0;
		real_t domega = torque_cmd.length() * 0.01;
		set_rotor_velocity(0, omega_base + domega);
		set_rotor_velocity(1, omega_base - domega);
		set_rotor_velocity(2, omega_base + domega);
		set_rotor_velocity(3, omega_base - domega);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_rotor_velocity", "idx", "omega"), &DroneEntity::set_rotor_velocity);
		ClassDB::bind_method(D_METHOD("get_rotor_count"), &DroneEntity::get_rotor_count);
		ClassDB::bind_method(D_METHOD("compute_control_from_attitude", "desired_attitude", "dt"), &DroneEntity::compute_control_from_attitude);
	}

private:
	LocalVector<Rotor> rotors;
};

} // namespace genesis

#endif // GENESIS_ENTITIES_DRONE_ENTITY_H
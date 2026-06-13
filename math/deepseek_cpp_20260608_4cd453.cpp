// File 222: modules/newton/src/joints/newton_up_vector_joint.h
// UpVector joint: constrains a body's local axis to stay aligned with a
// world-space up direction (e.g., keep a balloon upright). The joint
// applies corrective torques to cancel any tilt away from the target axis.
// Optionally includes a spring-damper to return to vertical.

#ifndef NEWTON_JOINTS_UP_VECTOR_H
#define NEWTON_JOINTS_UP_VECTOR_H

#include "newton_joint.h"

namespace newton {

class NewtonUpVectorJoint : public NewtonJoint {
	GDCLASS(NewtonUpVectorJoint, NewtonJoint);

public:
	NewtonUpVectorJoint() {
		joint_type = JointType::CUSTOM; // use custom type, or a dedicated UP_VECTOR if added.
	}

	// Set the local axis of the body that should align with world up.
	void set_local_axis(const vec3 &p_axis) { local_axis = p_axis.normalized(); }
	vec3 get_local_axis() const { return local_axis; }

	// Set the world-space up vector (default Y-up).
	void set_world_up(const vec3 &p_up) { world_up = p_up.normalized(); }
	vec3 get_world_up() const { return world_up; }

	// Spring stiffness (Nm/rad) – restores the body to the up orientation.
	void set_stiffness(real_t p_k) { stiffness = MAX(p_k, 0.0); }
	real_t get_stiffness() const { return stiffness; }

	// Damping coefficient (Nms/rad) – opposes angular velocity around the error axis.
	void set_damping(real_t p_d) { damping = MAX(p_d, 0.0); }
	real_t get_damping() const { return damping; }

	// Maximum torque the joint may apply.
	void set_max_torque(real_t p_torque) { max_torque = MAX(p_torque, 0.0); }
	real_t get_max_torque() const { return max_torque; }

	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_local_axis", "axis"), &NewtonUpVectorJoint::set_local_axis);
		ClassDB::bind_method(D_METHOD("get_local_axis"), &NewtonUpVectorJoint::get_local_axis);
		ClassDB::bind_method(D_METHOD("set_world_up", "up"), &NewtonUpVectorJoint::set_world_up);
		ClassDB::bind_method(D_METHOD("get_world_up"), &NewtonUpVectorJoint::get_world_up);
		ClassDB::bind_method(D_METHOD("set_stiffness", "k"), &NewtonUpVectorJoint::set_stiffness);
		ClassDB::bind_method(D_METHOD("get_stiffness"), &NewtonUpVectorJoint::get_stiffness);
		ClassDB::bind_method(D_METHOD("set_damping", "d"), &NewtonUpVectorJoint::set_damping);
		ClassDB::bind_method(D_METHOD("get_damping"), &NewtonUpVectorJoint::get_damping);
		ClassDB::bind_method(D_METHOD("set_max_torque", "torque"), &NewtonUpVectorJoint::set_max_torque);
		ClassDB::bind_method(D_METHOD("get_max_torque"), &NewtonUpVectorJoint::get_max_torque);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "local_axis"), "set_local_axis", "get_local_axis");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "world_up"), "set_world_up", "get_world_up");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness"), "set_stiffness", "get_stiffness");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping"), "set_damping", "get_damping");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_torque"), "set_max_torque", "get_max_torque");
	}

private:
	vec3 local_axis = vec3(0, 1, 0);
	vec3 world_up = vec3(0, 1, 0);
	real_t stiffness = 100.0;
	real_t damping = 10.0;
	real_t max_torque = INFINITY;
};

} // namespace newton

#endif // NEWTON_JOINTS_UP_VECTOR_H
// File 187: modules/newton/src/joints/newton_joint.h
// Base class for all Newton joints. A joint constrains the relative motion
// between two rigid bodies. It stores references to the bodies, the joint
// type, and provides a virtual solve() method called by the solver.

#ifndef NEWTON_JOINTS_NEWTON_JOINT_H
#define NEWTON_JOINTS_NEWTON_JOINT_H

#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

namespace newton {

class NewtonJoint : public RefCounted {
	GDCLASS(NewtonJoint, RefCounted);

public:
	NewtonJoint() : joint_type(JointType::CUSTOM), body_a_id(0), body_b_id(0), enabled(true) {}
	virtual ~NewtonJoint() {}

	JointType get_joint_type() const { return joint_type; }

	void set_body_a(body_id p_id) { body_a_id = p_id; }
	body_id get_body_a() const { return body_a_id; }

	void set_body_b(body_id p_id) { body_b_id = p_id; }
	body_id get_body_b() const { return body_b_id; }

	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	bool is_enabled() const { return enabled; }

	// Solve joint constraint for the time step dt.
	// Must be implemented by derived classes.
	virtual void solve(real_t dt) = 0;

protected:
	JointType joint_type;
	body_id body_a_id;
	body_id body_b_id;
	bool enabled;
};

} // namespace newton

#endif // NEWTON_JOINTS_NEWTON_JOINT_H
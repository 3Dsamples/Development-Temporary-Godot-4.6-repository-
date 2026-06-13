// File 197: modules/newton/src/joints/newton_joint.h (updated)
// Updated base joint class: stores body IDs, joint type, enabled flag,
// and a new solve signature that receives body pointers directly from the solver.
// All derived joints must implement solve(NewtonBody* a, NewtonBody* b, real_t dt).

#ifndef NEWTON_JOINTS_NEWTON_JOINT_H
#define NEWTON_JOINTS_NEWTON_JOINT_H

#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

namespace newton {

class NewtonBody;

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

	// Main solve function – called by the solver each substep.
	// `a` and `b` are the body pointers resolved from the world.
	// The joint must enforce its constraint using impulses / position corrections.
	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) {}

protected:
	JointType joint_type;
	body_id body_a_id;
	body_id body_b_id;
	bool enabled;

	static void _bind_methods();
};

} // namespace newton

#endif // NEWTON_JOINTS_NEWTON_JOINT_H
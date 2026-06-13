// File 203: modules/newton/src/joints/newton_joint.h (update)
// Updated NewtonJoint with body pointer caching and setter for solver use.

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
	NewtonJoint() : joint_type(JointType::CUSTOM), body_a_id(0), body_b_id(0),
					body_a_ptr(nullptr), body_b_ptr(nullptr), enabled(true) {}
	virtual ~NewtonJoint() {}

	JointType get_joint_type() const { return joint_type; }

	void set_body_a(body_id p_id) { body_a_id = p_id; }
	body_id get_body_a() const { return body_a_id; }

	void set_body_b(body_id p_id) { body_b_id = p_id; }
	body_id get_body_b() const { return body_b_id; }

	void set_body_pointers(NewtonBody *a, NewtonBody *b) {
		body_a_ptr = a;
		body_b_ptr = b;
	}

	NewtonBody *get_body_a_ptr() const { return body_a_ptr; }
	NewtonBody *get_body_b_ptr() const { return body_b_ptr; }

	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	bool is_enabled() const { return enabled; }

	// Main solve function – called by the solver each substep.
	// Default implementation uses cached body pointers.
	virtual void solve(real_t dt) {
		solve(body_a_ptr, body_b_ptr, dt);
	}

	// Overloaded solve with explicit body pointers.
	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) {}

protected:
	JointType joint_type;
	body_id body_a_id;
	body_id body_b_id;
	NewtonBody *body_a_ptr;
	NewtonBody *body_b_ptr;
	bool enabled;

	static void _bind_methods();
};

} // namespace newton

#endif // NEWTON_JOINTS_NEWTON_JOINT_H
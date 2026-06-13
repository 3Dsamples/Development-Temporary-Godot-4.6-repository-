// File 228: modules/newton/src/joints/newton_custom_joint.h
// Abstract base class for user‑defined joints.  Users subclass and implement
// the constraint solver callback.  The Newton solver calls this callback each
// physics step, providing body pointers and allowing arbitrary impulses.

#ifndef NEWTON_JOINTS_CUSTOM_JOINT_H
#define NEWTON_JOINTS_CUSTOM_JOINT_H

#include "newton_joint.h"
#include "core/object/ref_counted.h"

namespace newton {

class NewtonCustomJoint : public NewtonJoint {
	GDCLASS(NewtonCustomJoint, NewtonJoint);

public:
	NewtonCustomJoint() { joint_type = JointType::CUSTOM; }

	// Override this in a derived class (or set a callable via GDScript).
	// The function receives the two bodies, the time step, and a user pointer.
	virtual void solve_custom(NewtonBody *a, NewtonBody *b, real_t dt) = 0;

	// In the solver, NewtonCustomJoint::solve delegates to solve_custom.
	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) override {
		if (!enabled) return;
		solve_custom(a, b, dt);
	}

protected:
	static void _bind_methods() {
		// Bind as virtual; scripts can override the callable indirectly.
		// For GDScript, we'll use a Callable that the solver invokes.
	}
};

} // namespace newton

#endif // NEWTON_JOINTS_CUSTOM_JOINT_H
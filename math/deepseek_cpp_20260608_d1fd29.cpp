// File 193: modules/newton/src/joints/newton_joint.cpp
// NewtonJoint base class – provides common utility for joint solve, including
// retrieval of body pointers from the world, and a default empty solve.
// Derived joints override solve to enforce specific constraints.

#include "newton_joint.h"
#include "../bodies/newton_body.h"
#include "../world/newton_world.h"
#include "core/typedefs.h"

namespace newton {

// Virtual destructor definition (required)
NewtonJoint::~NewtonJoint() {}

// Default solve does nothing; derived classes override.
void NewtonJoint::solve(NewtonBody *p_body_a, NewtonBody *p_body_b, real_t dt) {
	// no default constraint
}

// Bindings for the base class
void NewtonJoint::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_body_a", "id"), &NewtonJoint::set_body_a);
	ClassDB::bind_method(D_METHOD("get_body_a"), &NewtonJoint::get_body_a);
	ClassDB::bind_method(D_METHOD("set_body_b", "id"), &NewtonJoint::set_body_b);
	ClassDB::bind_method(D_METHOD("get_body_b"), &NewtonJoint::get_body_b);
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NewtonJoint::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NewtonJoint::is_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "body_a_id"), "set_body_a", "get_body_a");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "body_b_id"), "set_body_b", "get_body_b");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
}

} // namespace newton
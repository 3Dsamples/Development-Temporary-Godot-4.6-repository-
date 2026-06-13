// File 285: modules/vienna/src/joints/vienna_joint.cpp
// Implementation of the base ViennaJoint class – constructors, destructor,
// bind methods, and default solve logic.

#include "vienna_joint.h"
#include "../bodies/vienna_body.h"

namespace vienna {

ViennaJoint::ViennaJoint() : joint_type(JointType::CUSTOM), body_a_id(0), body_b_id(0),
	body_a_ptr(nullptr), body_b_ptr(nullptr), enabled(true) {}

ViennaJoint::~ViennaJoint() {}

void ViennaJoint::solve(real_t dt) {
	if (body_a_ptr && body_b_ptr) {
		solve(body_a_ptr, body_b_ptr, dt);
	}
}

void ViennaJoint::solve(ViennaBody *a, ViennaBody *b, real_t dt) {
	// Base implementation does nothing; derived classes override.
}

void ViennaJoint::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_joint_type", "type"), &ViennaJoint::set_joint_type);
	ClassDB::bind_method(D_METHOD("get_joint_type"), &ViennaJoint::get_joint_type);
	ClassDB::bind_method(D_METHOD("set_body_a", "id"), &ViennaJoint::set_body_a);
	ClassDB::bind_method(D_METHOD("get_body_a"), &ViennaJoint::get_body_a);
	ClassDB::bind_method(D_METHOD("set_body_b", "id"), &ViennaJoint::set_body_b);
	ClassDB::bind_method(D_METHOD("get_body_b"), &ViennaJoint::get_body_b);
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &ViennaJoint::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &ViennaJoint::is_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "joint_type"), "set_joint_type", "get_joint_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "body_a_id"), "set_body_a", "get_body_a");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "body_b_id"), "set_body_b", "get_body_b");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
}

} // namespace vienna
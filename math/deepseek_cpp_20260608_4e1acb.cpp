// File 231: modules/newton/src/joints/newton_pulley_joint.h
// NewtonPulleyJoint: simulates a rope and pulley between two bodies.
// Constrains the motion so that the sum of signed distances along the
// rope direction is constant. Supports a pulley ratio and limits.

#ifndef NEWTON_JOINTS_PULLEY_H
#define NEWTON_JOINTS_PULLEY_H

#include "newton_joint.h"

namespace newton {

class NewtonPulleyJoint : public NewtonJoint {
	GDCLASS(NewtonPulleyJoint, NewtonJoint);

public:
	NewtonPulleyJoint() { joint_type = JointType::CUSTOM; }

	// Set the local attachment points on each body.
	void set_anchor_a(const vec3 &p_anchor) { anchor_a = p_anchor; }
	vec3 get_anchor_a() const { return anchor_a; }
	void set_anchor_b(const vec3 &p_anchor) { anchor_b = p_anchor; }
	vec3 get_anchor_b() const { return anchor_b; }

	// Set the local directions of the rope segments (before pulley).
	// The constraint enforces rope length A + ratio * rope length B = constant.
	void set_rope_direction_a(const vec3 &p_dir) { dir_a = p_dir.normalized(); }
	vec3 get_rope_direction_a() const { return dir_a; }
	void set_rope_direction_b(const vec3 &p_dir) { dir_b = p_dir.normalized(); }
	vec3 get_rope_direction_b() const { return dir_b; }

	// Pulley ratio (default 1.0).
	void set_ratio(real_t p_ratio) { ratio = p_ratio; }
	real_t get_ratio() const { return ratio; }

	// Optional travel limits (in meters).
	void set_limit_a(real_t p_min, real_t p_max) {
		limit_a_min = MIN(p_min, p_max);
		limit_a_max = MAX(p_min, p_max);
	}
	real_t get_limit_a_min() const { return limit_a_min; }
	real_t get_limit_a_max() const { return limit_a_max; }

	void set_limit_b(real_t p_min, real_t p_max) {
		limit_b_min = MIN(p_min, p_max);
		limit_b_max = MAX(p_min, p_max);
	}
	real_t get_limit_b_min() const { return limit_b_min; }
	real_t get_limit_b_max() const { return limit_b_max; }

	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_anchor_a", "anchor"), &NewtonPulleyJoint::set_anchor_a);
		ClassDB::bind_method(D_METHOD("get_anchor_a"), &NewtonPulleyJoint::get_anchor_a);
		ClassDB::bind_method(D_METHOD("set_anchor_b", "anchor"), &NewtonPulleyJoint::set_anchor_b);
		ClassDB::bind_method(D_METHOD("get_anchor_b"), &NewtonPulleyJoint::get_anchor_b);
		ClassDB::bind_method(D_METHOD("set_rope_direction_a", "dir"), &NewtonPulleyJoint::set_rope_direction_a);
		ClassDB::bind_method(D_METHOD("get_rope_direction_a"), &NewtonPulleyJoint::get_rope_direction_a);
		ClassDB::bind_method(D_METHOD("set_rope_direction_b", "dir"), &NewtonPulleyJoint::set_rope_direction_b);
		ClassDB::bind_method(D_METHOD("get_rope_direction_b"), &NewtonPulleyJoint::get_rope_direction_b);
		ClassDB::bind_method(D_METHOD("set_ratio", "ratio"), &NewtonPulleyJoint::set_ratio);
		ClassDB::bind_method(D_METHOD("get_ratio"), &NewtonPulleyJoint::get_ratio);
		ClassDB::bind_method(D_METHOD("set_limit_a", "min", "max"), &NewtonPulleyJoint::set_limit_a);
		ClassDB::bind_method(D_METHOD("get_limit_a_min"), &NewtonPulleyJoint::get_limit_a_min);
		ClassDB::bind_method(D_METHOD("get_limit_a_max"), &NewtonPulleyJoint::get_limit_a_max);
		ClassDB::bind_method(D_METHOD("set_limit_b", "min", "max"), &NewtonPulleyJoint::set_limit_b);
		ClassDB::bind_method(D_METHOD("get_limit_b_min"), &NewtonPulleyJoint::get_limit_b_min);
		ClassDB::bind_method(D_METHOD("get_limit_b_max"), &NewtonPulleyJoint::get_limit_b_max);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor_a"), "set_anchor_a", "get_anchor_a");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor_b"), "set_anchor_b", "get_anchor_b");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rope_direction_a"), "set_rope_direction_a", "get_rope_direction_a");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rope_direction_b"), "set_rope_direction_b", "get_rope_direction_b");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ratio"), "set_ratio", "get_ratio");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
	}

private:
	vec3 anchor_a;
	vec3 anchor_b;
	vec3 dir_a = vec3(1,0,0);
	vec3 dir_b = vec3(-1,0,0);
	real_t ratio = 1.0;
	real_t limit_a_min = -INFINITY;
	real_t limit_a_max =  INFINITY;
	real_t limit_b_min = -INFINITY;
	real_t limit_b_max =  INFINITY;
	// Accumulated rope lengths for warm start? Not needed.
};

} // namespace newton

#endif // NEWTON_JOINTS_PULLEY_H
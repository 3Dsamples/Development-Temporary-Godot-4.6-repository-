// File 283: modules/vienna/src/joints/vienna_distance_joint.h
// Vienna Distance Joint – maintains a fixed distance between two anchor points
// on the bodies.  The constraint error is the difference between current distance
// and the rest length.  Supports a linear spring‑damper and optional limits.

#ifndef VIENNA_JOINTS_DISTANCE_JOINT_H
#define VIENNA_JOINTS_DISTANCE_JOINT_H

#include "vienna_joint.h"

namespace vienna {

class ViennaDistanceJoint : public ViennaJoint {
	GDCLASS(ViennaDistanceJoint, ViennaJoint);

public:
	ViennaDistanceJoint() { joint_type = JointType::DISTANCE; }

	// Anchor points in the local frames of body A and body B.
	void set_anchor_a(const vec3 &p_anchor) { anchor_a = p_anchor; }
	vec3 get_anchor_a() const { return anchor_a; }
	void set_anchor_b(const vec3 &p_anchor) { anchor_b = p_anchor; }
	vec3 get_anchor_b() const { return anchor_b; }

	// Rest length (distance to maintain).  If not set explicitly, it is
	// computed from the initial positions when the joint is created.
	void set_rest_length(real_t p_len) { rest_length = MAX(p_len, 0.0); }
	real_t get_rest_length() const { return rest_length; }

	// --- Limits (min and max distance) ---
	void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
	bool is_limit_enabled() const { return limit_enabled; }
	void set_min_distance(real_t p_min) { min_dist = MAX(p_min, 0.0); }
	real_t get_min_distance() const { return min_dist; }
	void set_max_distance(real_t p_max) { max_dist = MAX(p_max, 0.0); }
	real_t get_max_distance() const { return max_dist; }

	// --- Spring‑damper (soft distance) ---
	void set_spring_enabled(bool p_enable) { spring_enabled = p_enable; }
	bool is_spring_enabled() const { return spring_enabled; }
	void set_spring_stiffness(real_t p_k) { stiffness = MAX(p_k, 0.0); }
	real_t get_spring_stiffness() const { return stiffness; }
	void set_spring_damping(real_t p_d) { damping = MAX(p_d, 0.0); }
	real_t get_spring_damping() const { return damping; }

	virtual void solve(ViennaBody *a, ViennaBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_anchor_a", "anchor"), &ViennaDistanceJoint::set_anchor_a);
		ClassDB::bind_method(D_METHOD("get_anchor_a"), &ViennaDistanceJoint::get_anchor_a);
		ClassDB::bind_method(D_METHOD("set_anchor_b", "anchor"), &ViennaDistanceJoint::set_anchor_b);
		ClassDB::bind_method(D_METHOD("get_anchor_b"), &ViennaDistanceJoint::get_anchor_b);
		ClassDB::bind_method(D_METHOD("set_rest_length", "length"), &ViennaDistanceJoint::set_rest_length);
		ClassDB::bind_method(D_METHOD("get_rest_length"), &ViennaDistanceJoint::get_rest_length);
		ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &ViennaDistanceJoint::set_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_limit_enabled"), &ViennaDistanceJoint::is_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_min_distance", "min"), &ViennaDistanceJoint::set_min_distance);
		ClassDB::bind_method(D_METHOD("get_min_distance"), &ViennaDistanceJoint::get_min_distance);
		ClassDB::bind_method(D_METHOD("set_max_distance", "max"), &ViennaDistanceJoint::set_max_distance);
		ClassDB::bind_method(D_METHOD("get_max_distance"), &ViennaDistanceJoint::get_max_distance);
		ClassDB::bind_method(D_METHOD("set_spring_enabled", "enabled"), &ViennaDistanceJoint::set_spring_enabled);
		ClassDB::bind_method(D_METHOD("is_spring_enabled"), &ViennaDistanceJoint::is_spring_enabled);
		ClassDB::bind_method(D_METHOD("set_spring_stiffness", "stiffness"), &ViennaDistanceJoint::set_spring_stiffness);
		ClassDB::bind_method(D_METHOD("get_spring_stiffness"), &ViennaDistanceJoint::get_spring_stiffness);
		ClassDB::bind_method(D_METHOD("set_spring_damping", "damping"), &ViennaDistanceJoint::set_spring_damping);
		ClassDB::bind_method(D_METHOD("get_spring_damping"), &ViennaDistanceJoint::get_spring_damping);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor_a"), "set_anchor_a", "get_anchor_a");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor_b"), "set_anchor_b", "get_anchor_b");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rest_length"), "set_rest_length", "get_rest_length");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "spring_enabled"), "set_spring_enabled", "is_spring_enabled");
	}

private:
	vec3 anchor_a = vec3();
	vec3 anchor_b = vec3();
	real_t rest_length = 1.0;
	bool limit_enabled = false;
	real_t min_dist = 0.0;
	real_t max_dist = 2.0;
	bool spring_enabled = false;
	real_t stiffness = 100.0;
	real_t damping = 10.0;
};

// Implementation of solve
inline void ViennaDistanceJoint::solve(ViennaBody *a, ViennaBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// World anchor points
	vec3 worldAnchorA = xA.xform(anchor_a);
	vec3 worldAnchorB = xB.xform(anchor_b);

	// Current distance vector and length
	vec3 delta = worldAnchorB - worldAnchorA;
	real_t currentDist = delta.length();
	if (currentDist < CMP_EPSILON)
		currentDist = CMP_EPSILON;
	vec3 dir = delta / currentDist;

	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// Compute effective inverse mass along the direction between anchors
	vec3 rA = worldAnchorA - a->get_position();
	vec3 rB = worldAnchorB - b->get_position();
	real_t invEff = invMassSum +
		dir.dot(a->get_inverse_inertia_world().xform(rA.cross(dir)).cross(rA)) +
		dir.dot(b->get_inverse_inertia_world().xform(rB.cross(dir)).cross(rB));
	if (invEff < CMP_EPSILON) return;

	// Velocity along the direction
	vec3 velA = a->get_linear_velocity() + a->get_angular_velocity().cross(rA);
	vec3 velB = b->get_linear_velocity() + b->get_angular_velocity().cross(rB);
	real_t relVel = (velB - velA).dot(dir);

	real_t correctionVel = 0.0f;

	// Distance error relative to rest length
	real_t distError = currentDist - rest_length;

	// Limits enforcement (overrides rest length if outside limits)
	if (limit_enabled) {
		if (currentDist < min_dist)
			distError = currentDist - min_dist;
		else if (currentDist > max_dist)
			distError = currentDist - max_dist;
	}

	// Baumgarte position correction
	if (Math::abs(distError) > CMP_EPSILON) {
		real_t erp = 0.2f;
		correctionVel = -distError * erp / dt;
	}

	// Spring‑damper force (soft constraint)
	if (spring_enabled) {
		real_t springForce = -stiffness * distError;
		real_t damperForce = -damping * relVel;
		// Desired acceleration from spring/damper
		real_t springAccel = (springForce + damperForce) * invEff;
		correctionVel += springAccel * dt;
	}

	// Total impulse = (targetVel - relVel) * dt / invEff? Actually impulse = change in momentum.
	// Desired change in relative velocity: correctionVel - relVel
	real_t impulseMag = (correctionVel - relVel) / invEff;
	vec3 impulse = dir * impulseMag;

	if (invMassA > 0.0) a->apply_impulse( impulse, worldAnchorA);
	if (invMassB > 0.0) b->apply_impulse(-impulse, worldAnchorB);
}

} // namespace vienna

#endif // VIENNA_JOINTS_DISTANCE_JOINT_H
// File 284: modules/vienna/src/joints/vienna_rope_joint.h
// ViennaRopeJoint – a soft rope constraint that limits the maximum distance
// between two anchor points, but allows them to come closer.  Ideal for
// simulating ropes, chains, or slings.  Applies impulses only when the
// distance exceeds the maximum, with optional spring‑damper compliance.

#ifndef VIENNA_JOINTS_ROPE_H
#define VIENNA_JOINTS_ROPE_H

#include "vienna_joint.h"

namespace vienna {

class ViennaRopeJoint : public ViennaJoint {
	GDCLASS(ViennaRopeJoint, ViennaJoint);

public:
	ViennaRopeJoint() { joint_type = JointType::ROPE; }

	// Set the anchors in the local frames of body A and body B.
	void set_anchor_a(const vec3 &p_anchor) { anchor_a = p_anchor; }
	vec3 get_anchor_a() const { return anchor_a; }
	void set_anchor_b(const vec3 &p_anchor) { anchor_b = p_anchor; }
	vec3 get_anchor_b() const { return anchor_b; }

	// Maximum allowed distance (rope length).  The joint does nothing if
	// the anchors are closer than this value.
	void set_max_distance(real_t p_dist) { max_distance = MAX(p_dist, 0.0); }
	real_t get_max_distance() const { return max_distance; }

	// --- Spring compliance (optional – makes the rope slightly stretchy) ---
	void set_spring_enabled(bool p_enable) { spring_enabled = p_enable; }
	bool is_spring_enabled() const { return spring_enabled; }

	void set_stiffness(real_t p_k) { stiffness = MAX(p_k, 0.0); }
	real_t get_stiffness() const { return stiffness; }

	void set_damping(real_t p_d) { damping = MAX(p_d, 0.0); }
	real_t get_damping() const { return damping; }

	// Solve method (called each substep)
	virtual void solve(ViennaBody *a, ViennaBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_anchor_a", "anchor"), &ViennaRopeJoint::set_anchor_a);
		ClassDB::bind_method(D_METHOD("get_anchor_a"), &ViennaRopeJoint::get_anchor_a);
		ClassDB::bind_method(D_METHOD("set_anchor_b", "anchor"), &ViennaRopeJoint::set_anchor_b);
		ClassDB::bind_method(D_METHOD("get_anchor_b"), &ViennaRopeJoint::get_anchor_b);
		ClassDB::bind_method(D_METHOD("set_max_distance", "dist"), &ViennaRopeJoint::set_max_distance);
		ClassDB::bind_method(D_METHOD("get_max_distance"), &ViennaRopeJoint::get_max_distance);
		ClassDB::bind_method(D_METHOD("set_spring_enabled", "enabled"), &ViennaRopeJoint::set_spring_enabled);
		ClassDB::bind_method(D_METHOD("is_spring_enabled"), &ViennaRopeJoint::is_spring_enabled);
		ClassDB::bind_method(D_METHOD("set_stiffness", "k"), &ViennaRopeJoint::set_stiffness);
		ClassDB::bind_method(D_METHOD("get_stiffness"), &ViennaRopeJoint::get_stiffness);
		ClassDB::bind_method(D_METHOD("set_damping", "d"), &ViennaRopeJoint::set_damping);
		ClassDB::bind_method(D_METHOD("get_damping"), &ViennaRopeJoint::get_damping);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor_a"), "set_anchor_a", "get_anchor_a");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor_b"), "set_anchor_b", "get_anchor_b");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_distance"), "set_max_distance", "get_max_distance");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "spring_enabled"), "set_spring_enabled", "is_spring_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness"), "set_stiffness", "get_stiffness");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping"), "set_damping", "get_damping");
	}

private:
	vec3 anchor_a;
	vec3 anchor_b;
	real_t max_distance = 1.0;
	bool spring_enabled = false;
	real_t stiffness = 500.0;
	real_t damping = 20.0;
	bool first_solve = true;
	real_t rest_distance = 0.0;
};

// ---------------------------------------------------------------------------
// Inline solve implementation
// ---------------------------------------------------------------------------
inline void ViennaRopeJoint::solve(ViennaBody *a, ViennaBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// World anchor points
	vec3 worldAnchorA = xA.xform(anchor_a);
	vec3 worldAnchorB = xB.xform(anchor_b);

	vec3 delta = worldAnchorB - worldAnchorA;
	real_t currentDist = delta.length();

	if (first_solve) {
		// Store the distance at first solve as a reference (optional), but max_distance is set by user.
		rest_distance = MIN(currentDist, max_distance);
		first_solve = false;
	}

	// Rope constraint is only active when the distance exceeds the maximum.
	if (currentDist <= max_distance) {
		// If spring is enabled, we could optionally apply a spring pulling them
		// when the rope goes slack, but a standard rope does nothing.
		return;
	}

	if (currentDist < CMP_EPSILON) return; // coincident points – cannot resolve

	vec3 dir = delta / currentDist;

	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// Effective mass along the direction
	vec3 rA = worldAnchorA - a->get_position();
	vec3 rB = worldAnchorB - b->get_position();
	real_t invEffMass = invMassSum +
		dir.dot(a->get_inverse_inertia_world().xform(rA.cross(dir)).cross(rA)) +
		dir.dot(b->get_inverse_inertia_world().xform(rB.cross(dir)).cross(rB));

	if (invEffMass < CMP_EPSILON) return;

	// Relative velocity along the direction
	vec3 velA = a->get_linear_velocity() + a->get_angular_velocity().cross(rA);
	vec3 velB = b->get_linear_velocity() + b->get_angular_velocity().cross(rB);
	real_t relVel = (velB - velA).dot(dir); // positive when separating (bad for rope)

	// Position error: how far the anchors exceed the max distance
	real_t error = currentDist - max_distance;

	// Baumgarte correction: desired velocity to reduce error
	real_t erp = spring_enabled ? 0.05f : 0.3f; // softer erp if spring active
	real_t desiredVelChange = -(error * erp / dt) - relVel;

	// If spring enabled, add spring/damper force (only when over max)
	real_t springForce = 0.0;
	real_t damperForce = 0.0;
	if (spring_enabled) {
		springForce = stiffness * error; // restoring force proportional to excess distance
		damperForce = damping * relVel;  // opposes relative velocity
	}

	// Desired acceleration
	real_t desiredAccel = desiredVelChange / dt;
	if (spring_enabled) {
		desiredAccel += (springForce + damperForce) * invEffMass;
	}

	// Impulse magnitude (clamped to non‑negative to prevent pulling when slack)
	real_t lambda = desiredAccel * dt / invEffMass;
	if (lambda < 0.0) lambda = 0.0; // rope cannot pull bodies together

	// Apply impulses along the direction
	vec3 impulse = dir * lambda;
	if (invMassA > 0.0) a->apply_impulse( impulse, worldAnchorA);
	if (invMassB > 0.0) b->apply_impulse(-impulse, worldAnchorB);
}

} // namespace vienna

#endif // VIENNA_JOINTS_ROPE_H
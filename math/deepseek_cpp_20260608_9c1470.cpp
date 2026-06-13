// File 283: modules/vienna/src/joints/vienna_distance_joint.h
// ViennaDistanceJoint – maintains a fixed distance between two anchor points
// on two bodies.  Applies position‑level Baumgarte correction and velocity‑level
// damping.  Supports an optional spring‑damper model.

#ifndef VIENNA_JOINTS_DISTANCE_H
#define VIENNA_JOINTS_DISTANCE_H

#include "vienna_joint.h"

namespace vienna {

class ViennaDistanceJoint : public ViennaJoint {
	GDCLASS(ViennaDistanceJoint, ViennaJoint);

public:
	ViennaDistanceJoint() { joint_type = JointType::DISTANCE; }

	// Set the anchors in the local frames of body A and body B.
	void set_anchor_a(const vec3 &p_anchor) { anchor_a = p_anchor; }
	vec3 get_anchor_a() const { return anchor_a; }
	void set_anchor_b(const vec3 &p_anchor) { anchor_b = p_anchor; }
	vec3 get_anchor_b() const { return anchor_b; }

	// Set the target distance to maintain.  Default = current distance at first solve.
	void set_distance(real_t p_dist) { distance = MAX(p_dist, 0.0); }
	real_t get_distance() const { return distance; }

	// --- Spring‑damper (optional) ---
	void set_spring_enabled(bool p_enable) { spring_enabled = p_enable; }
	bool is_spring_enabled() const { return spring_enabled; }

	void set_spring_stiffness(real_t p_k) { stiffness = MAX(p_k, 0.0); }
	real_t get_spring_stiffness() const { return stiffness; }

	void set_spring_damping(real_t p_d) { damping = MAX(p_d, 0.0); }
	real_t get_spring_damping() const { return damping; }

	// Solve method (called each substep)
	virtual void solve(ViennaBody *a, ViennaBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_anchor_a", "anchor"), &ViennaDistanceJoint::set_anchor_a);
		ClassDB::bind_method(D_METHOD("get_anchor_a"), &ViennaDistanceJoint::get_anchor_a);
		ClassDB::bind_method(D_METHOD("set_anchor_b", "anchor"), &ViennaDistanceJoint::set_anchor_b);
		ClassDB::bind_method(D_METHOD("get_anchor_b"), &ViennaDistanceJoint::get_anchor_b);
		ClassDB::bind_method(D_METHOD("set_distance", "dist"), &ViennaDistanceJoint::set_distance);
		ClassDB::bind_method(D_METHOD("get_distance"), &ViennaDistanceJoint::get_distance);
		ClassDB::bind_method(D_METHOD("set_spring_enabled", "enabled"), &ViennaDistanceJoint::set_spring_enabled);
		ClassDB::bind_method(D_METHOD("is_spring_enabled"), &ViennaDistanceJoint::is_spring_enabled);
		ClassDB::bind_method(D_METHOD("set_spring_stiffness", "k"), &ViennaDistanceJoint::set_spring_stiffness);
		ClassDB::bind_method(D_METHOD("get_spring_stiffness"), &ViennaDistanceJoint::get_spring_stiffness);
		ClassDB::bind_method(D_METHOD("set_spring_damping", "d"), &ViennaDistanceJoint::set_spring_damping);
		ClassDB::bind_method(D_METHOD("get_spring_damping"), &ViennaDistanceJoint::get_spring_damping);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor_a"), "set_anchor_a", "get_anchor_a");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor_b"), "set_anchor_b", "get_anchor_b");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance"), "set_distance", "get_distance");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "spring_enabled"), "set_spring_enabled", "is_spring_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "spring_stiffness"), "set_spring_stiffness", "get_spring_stiffness");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "spring_damping"), "set_spring_damping", "get_spring_damping");
	}

private:
	vec3 anchor_a;
	vec3 anchor_b;
	real_t distance = 1.0;
	bool spring_enabled = false;
	real_t stiffness = 100.0;
	real_t damping = 10.0;
	bool first_solve = true;
};

// ---------------------------------------------------------------------------
// Inline solve implementation
// ---------------------------------------------------------------------------
inline void ViennaDistanceJoint::solve(ViennaBody *a, ViennaBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// World anchor points
	vec3 worldAnchorA = xA.xform(anchor_a);
	vec3 worldAnchorB = xB.xform(anchor_b);

	// Current distance and direction between anchors
	vec3 delta = worldAnchorB - worldAnchorA;
	real_t currentDist = delta.length();

	// On first solve, initialise the target distance to the current distance if not set.
	if (first_solve) {
		if (distance <= 0.0) distance = currentDist;
		first_solve = false;
	}

	if (currentDist < CMP_EPSILON) return; // points coincide – no unique direction

	vec3 dir = delta / currentDist;

	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// Compute effective mass along the connecting direction
	vec3 rA = worldAnchorA - a->get_position();
	vec3 rB = worldAnchorB - b->get_position();
	real_t invEffMass = invMassSum +
		dir.dot(a->get_inverse_inertia_world().xform(rA.cross(dir)).cross(rA)) +
		dir.dot(b->get_inverse_inertia_world().xform(rB.cross(dir)).cross(rB));

	if (invEffMass < CMP_EPSILON) return;

	// Velocity of anchors
	vec3 velA = a->get_linear_velocity() + a->get_angular_velocity().cross(rA);
	vec3 velB = b->get_linear_velocity() + b->get_angular_velocity().cross(rB);
	real_t relVel = (velB - velA).dot(dir); // positive when separating

	// Spring force (if enabled) – applies even without position error
	real_t springForce = 0.0;
	real_t damperForce = 0.0;
	if (spring_enabled) {
		// Spring force = k * (currentDist - distance)
		springForce = stiffness * (currentDist - distance);
		// Damping force opposes relative velocity
		damperForce = damping * relVel;
	}

	// Position correction (Baumgarte)
	real_t baumgarteCorrection = 0.0;
	if (!spring_enabled || Math::abs(springForce) > 1e-6) {
		real_t error = currentDist - distance;
		real_t erp = spring_enabled ? 0.0f : 0.2f; // spring handles correction itself
		baumgarteCorrection = -error * erp / dt; // desired velocity to reduce error
	}

	// Desired velocity change
	real_t desiredAccel = (baumgarteCorrection - relVel) / dt;
	if (spring_enabled) {
		// Add spring/damper acceleration (F/m)
		desiredAccel += (springForce + damperForce) * invEffMass;
	}

	// Compute impulse magnitude
	real_t lambda = desiredAccel * dt / invEffMass;

	// Apply impulses along the direction
	vec3 impulse = dir * lambda;
	if (invMassA > 0.0) a->apply_impulse( impulse, worldAnchorA);
	if (invMassB > 0.0) b->apply_impulse(-impulse, worldAnchorB);
}

} // namespace vienna

#endif // VIENNA_JOINTS_DISTANCE_H
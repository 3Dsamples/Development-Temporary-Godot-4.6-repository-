// File 281: modules/vienna/src/joints/vienna_slider_joint.h
// Vienna Slider (Prismatic) Joint – allows translation along a single axis
// with optional limits and linear motor.  All other relative motion is
// locked via position‑level Baumgarte correction and velocity‑level impulses.

#ifndef VIENNA_JOINTS_SLIDER_JOINT_H
#define VIENNA_JOINTS_SLIDER_JOINT_H

#include "vienna_joint.h"

namespace vienna {

class ViennaSliderJoint : public ViennaJoint {
	GDCLASS(ViennaSliderJoint, ViennaJoint);

public:
	ViennaSliderJoint() { joint_type = JointType::SLIDER; }

	// Slider axis in the local frame of body A.
	void set_axis(const vec3 &p_axis) { axis_a = p_axis.normalized(); }
	vec3 get_axis() const { return axis_a; }

	// Anchor point in body A's local frame.
	void set_anchor(const vec3 &p_anchor) { anchor_a = p_anchor; }
	vec3 get_anchor() const { return anchor_a; }

	// --- Translation limits along the axis (relative to initial offset) ---
	void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
	bool is_limit_enabled() const { return limit_enabled; }

	void set_limit_range(real_t p_min, real_t p_max) {
		min_limit = MIN(p_min, p_max);
		max_limit = MAX(p_min, p_max);
	}
	real_t get_min_limit() const { return min_limit; }
	real_t get_max_limit() const { return max_limit; }

	// --- Linear motor ---
	void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
	bool is_motor_enabled() const { return motor_enabled; }

	void set_motor_target_velocity(real_t p_vel) { motor_target_vel = p_vel; }
	real_t get_motor_target_velocity() const { return motor_target_vel; }

	void set_motor_max_force(real_t p_force) { motor_max_force = MAX(p_force, 0.0); }
	real_t get_motor_max_force() const { return motor_max_force; }

	virtual void solve(ViennaBody *a, ViennaBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_axis", "axis"), &ViennaSliderJoint::set_axis);
		ClassDB::bind_method(D_METHOD("get_axis"), &ViennaSliderJoint::get_axis);
		ClassDB::bind_method(D_METHOD("set_anchor", "anchor"), &ViennaSliderJoint::set_anchor);
		ClassDB::bind_method(D_METHOD("get_anchor"), &ViennaSliderJoint::get_anchor);
		ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &ViennaSliderJoint::set_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_limit_enabled"), &ViennaSliderJoint::is_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_limit_range", "min", "max"), &ViennaSliderJoint::set_limit_range);
		ClassDB::bind_method(D_METHOD("get_min_limit"), &ViennaSliderJoint::get_min_limit);
		ClassDB::bind_method(D_METHOD("get_max_limit"), &ViennaSliderJoint::get_max_limit);
		ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &ViennaSliderJoint::set_motor_enabled);
		ClassDB::bind_method(D_METHOD("is_motor_enabled"), &ViennaSliderJoint::is_motor_enabled);
		ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "vel"), &ViennaSliderJoint::set_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &ViennaSliderJoint::get_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("set_motor_max_force", "force"), &ViennaSliderJoint::set_motor_max_force);
		ClassDB::bind_method(D_METHOD("get_motor_max_force"), &ViennaSliderJoint::get_motor_max_force);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis"), "set_axis", "get_axis");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor"), "set_anchor", "get_anchor");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
	}

private:
	vec3 axis_a = vec3(1, 0, 0);
	vec3 anchor_a = vec3();
	bool limit_enabled = false;
	real_t min_limit = -1.0;
	real_t max_limit = 1.0;
	bool motor_enabled = false;
	real_t motor_target_vel = 0.0;
	real_t motor_max_force = 100.0;
};

// Implementation of solve
inline void ViennaSliderJoint::solve(ViennaBody *a, ViennaBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// World anchor points on each body (symmetric local pivot)
	vec3 worldAnchorA = xA.xform(anchor_a);
	vec3 worldAnchorB = xB.xform(anchor_a);

	// World slider axis from body A
	vec3 worldAxis = xA.basis.xform(axis_a).normalized();

	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// ---- LATERAL POSITION CONSTRAINT: remove error perpendicular to the axis ----
	vec3 posError = worldAnchorB - worldAnchorA;
	real_t parallelError = posError.dot(worldAxis);
	vec3 perpendicularError = posError - worldAxis * parallelError;

	if (invMassSum > CMP_EPSILON) {
		real_t erp = 0.2f;
		vec3 correction = perpendicularError * (erp / dt);
		vec3 impulse = correction / invMassSum;
		if (invMassA > 0.0) a->apply_impulse( impulse, worldAnchorA);
		if (invMassB > 0.0) b->apply_impulse(-impulse, worldAnchorB);
	}

	// ---- ANGULAR CONSTRAINT: align orientations except around the axis ----
	vec3 perpDirA = (Math::abs(worldAxis.x) < 0.999f) ? worldAxis.cross(vec3(1, 0, 0)).normalized()
	                                                    : worldAxis.cross(vec3(0, 1, 0)).normalized();
	vec3 refA1 = perpDirA;
	vec3 refA2 = worldAxis.cross(refA1).normalized();
	mat3 refFrameA(refA1, refA2, worldAxis);

	vec3 worldAxisB = xB.basis.xform(axis_a).normalized();
	vec3 perpDirB = (Math::abs(worldAxisB.x) < 0.999f) ? worldAxisB.cross(vec3(1, 0, 0)).normalized()
	                                                    : worldAxisB.cross(vec3(0, 1, 0)).normalized();
	vec3 refB1 = perpDirB;
	vec3 refB2 = worldAxisB.cross(refB1).normalized();
	mat3 refFrameB(refB1, refB2, worldAxisB);

	mat3 R_err = refFrameB * refFrameA.transposed();
	quat q_err(R_err);
	vec3 rotAxis;
	real_t rotAngle;
	q_err.get_axis_angle(rotAxis, rotAngle);
	if (Math::abs(rotAngle) > 0.01f) {
		vec3 angularTorqueA = a->get_inverse_inertia_world().xform(rotAxis);
		vec3 angularTorqueB = b->get_inverse_inertia_world().xform(rotAxis);
		real_t invEffectiveInertia = angularTorqueA.dot(rotAxis) + angularTorqueB.dot(rotAxis);
		if (invEffectiveInertia > CMP_EPSILON) {
			real_t angularCorrection = rotAngle * 0.5f / dt;
			vec3 angularImpulse = rotAxis * (angularCorrection / invEffectiveInertia);
			if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
			if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
		}
	}

	// ---- TRANSLATION LIMITS ----
	if (limit_enabled) {
		real_t currentTranslation = parallelError; // relative to initial anchor offset
		real_t lower = min_limit;
		real_t upper = max_limit;
		real_t limitError = 0.0f;
		if (currentTranslation < lower) limitError = lower - currentTranslation;
		else if (currentTranslation > upper) limitError = upper - currentTranslation;

		if (Math::abs(limitError) > CMP_EPSILON) {
			vec3 rA = worldAnchorA - a->get_position();
			vec3 rB = worldAnchorB - b->get_position();
			real_t invEffMassAlong = invMassSum +
				worldAxis.dot(a->get_inverse_inertia_world().xform(rA.cross(worldAxis)).cross(rA)) +
				worldAxis.dot(b->get_inverse_inertia_world().xform(rB.cross(worldAxis)).cross(rB));
			if (invEffMassAlong > CMP_EPSILON) {
				real_t erpLim = 0.3f;
				vec3 correctionVec = worldAxis * (limitError * erpLim / dt);
				vec3 limitImpulse = correctionVec / invEffMassAlong;
				if (invMassA > 0.0) a->apply_impulse( limitImpulse, worldAnchorA);
				if (invMassB > 0.0) b->apply_impulse(-limitImpulse, worldAnchorB);
			}
		}
	}

	// ---- LINEAR MOTOR ----
	if (motor_enabled) {
		vec3 rA = worldAnchorA - a->get_position();
		vec3 rB = worldAnchorB - b->get_position();
		vec3 velA = a->get_linear_velocity() + a->get_angular_velocity().cross(rA);
		vec3 velB = b->get_linear_velocity() + b->get_angular_velocity().cross(rB);
		real_t currentSpeed = (velB - velA).dot(worldAxis);
		real_t speedError = motor_target_vel - currentSpeed;

		real_t invEffMassAlong = invMassSum +
			worldAxis.dot(a->get_inverse_inertia_world().xform(rA.cross(worldAxis)).cross(rA)) +
			worldAxis.dot(b->get_inverse_inertia_world().xform(rB.cross(worldAxis)).cross(rB));
		if (invEffMassAlong > CMP_EPSILON) {
			real_t motorAccel = speedError / dt;
			real_t motorForce = motorAccel / invEffMassAlong;
			motorForce = CLAMP(motorForce, -motor_max_force, motor_max_force);
			vec3 motorImpulse = worldAxis * motorForce * dt;
			if (invMassA > 0.0) a->apply_impulse( motorImpulse, worldAnchorA);
			if (invMassB > 0.0) b->apply_impulse(-motorImpulse, worldAnchorB);
		}
	}
}

} // namespace vienna

#endif // VIENNA_JOINTS_SLIDER_JOINT_H
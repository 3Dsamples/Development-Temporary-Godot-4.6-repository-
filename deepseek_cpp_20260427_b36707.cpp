// File 279: modules/vienna/src/joints/vienna_ball_joint.h
// Vienna Ball‑and‑Socket Joint – constrains the pivot points of two bodies
// to remain coincident.  Supports cone limits, twist limits, and an optional
// angular motor.  All constraint enforcement is done via position‑level
// Baumgarte stabilisation and velocity‑level impulses.

#ifndef VIENNA_JOINTS_BALL_JOINT_H
#define VIENNA_JOINTS_BALL_JOINT_H

#include "vienna_joint.h"

namespace vienna {

class ViennaBallJoint : public ViennaJoint {
	GDCLASS(ViennaBallJoint, ViennaJoint);

public:
	ViennaBallJoint() { joint_type = JointType::BALL; }

	// Pivot in local frame of body A (and identically for body B when the
	// joint is initially created; the solver works in world space).
	void set_pivot(const vec3 &p_pivot) { pivot_a = p_pivot; }
	vec3 get_pivot() const { return pivot_a; }

	// --- Cone limit (max angle between the two body Z‑axes) ---
	void set_cone_limit_enabled(bool p_enable) { cone_limit_enabled = p_enable; }
	bool is_cone_limit_enabled() const { return cone_limit_enabled; }

	void set_cone_angle(real_t p_radians) { cone_angle = CLAMP(p_radians, 0.0, Math_PI); }
	real_t get_cone_angle() const { return cone_angle; }

	// --- Twist limit (min/max angle around the cone axis) ---
	void set_twist_limit_enabled(bool p_enable) { twist_limit_enabled = p_enable; }
	bool is_twist_limit_enabled() const { return twist_limit_enabled; }

	void set_twist_angle(real_t p_min, real_t p_max) {
		twist_min = MIN(p_min, p_max);
		twist_max = MAX(p_min, p_max);
	}
	real_t get_twist_min() const { return twist_min; }
	real_t get_twist_max() const { return twist_max; }

	// --- Motor (drives relative angular velocity around the cone axis) ---
	void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
	bool is_motor_enabled() const { return motor_enabled; }

	void set_motor_target_velocity(real_t p_omega) { motor_target_vel = p_omega; }
	real_t get_motor_target_velocity() const { return motor_target_vel; }

	void set_motor_max_torque(real_t p_torque) { motor_max_torque = MAX(p_torque, 0.0); }
	real_t get_motor_max_torque() const { return motor_max_torque; }

	// --- Solve method (called by the solver each substep) ---
	virtual void solve(ViennaBody *a, ViennaBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_pivot", "pivot"), &ViennaBallJoint::set_pivot);
		ClassDB::bind_method(D_METHOD("get_pivot"), &ViennaBallJoint::get_pivot);
		ClassDB::bind_method(D_METHOD("set_cone_limit_enabled", "enabled"), &ViennaBallJoint::set_cone_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_cone_limit_enabled"), &ViennaBallJoint::is_cone_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_cone_angle", "angle"), &ViennaBallJoint::set_cone_angle);
		ClassDB::bind_method(D_METHOD("get_cone_angle"), &ViennaBallJoint::get_cone_angle);
		ClassDB::bind_method(D_METHOD("set_twist_limit_enabled", "enabled"), &ViennaBallJoint::set_twist_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_twist_limit_enabled"), &ViennaBallJoint::is_twist_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_twist_angle", "min", "max"), &ViennaBallJoint::set_twist_angle);
		ClassDB::bind_method(D_METHOD("get_twist_min"), &ViennaBallJoint::get_twist_min);
		ClassDB::bind_method(D_METHOD("get_twist_max"), &ViennaBallJoint::get_twist_max);
		ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &ViennaBallJoint::set_motor_enabled);
		ClassDB::bind_method(D_METHOD("is_motor_enabled"), &ViennaBallJoint::is_motor_enabled);
		ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "omega"), &ViennaBallJoint::set_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &ViennaBallJoint::get_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("set_motor_max_torque", "torque"), &ViennaBallJoint::set_motor_max_torque);
		ClassDB::bind_method(D_METHOD("get_motor_max_torque"), &ViennaBallJoint::get_motor_max_torque);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "pivot"), "set_pivot", "get_pivot");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cone_limit_enabled"), "set_cone_limit_enabled", "is_cone_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cone_angle", PROPERTY_HINT_RANGE, "0,180,0.1"), "set_cone_angle", "get_cone_angle");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "twist_limit_enabled"), "set_twist_limit_enabled", "is_twist_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
	}

private:
	vec3 pivot_a = vec3();
	bool cone_limit_enabled = false;
	real_t cone_angle = Math_PI * 0.5;
	bool twist_limit_enabled = false;
	real_t twist_min = -Math_PI;
	real_t twist_max =  Math_PI;
	bool motor_enabled = false;
	real_t motor_target_vel = 0.0;
	real_t motor_max_torque = 100.0;

	// --- Implementation of the solve method ---
public: // to allow inline definition, we keep public but final method
	// The solve body is here for header-only usage.
};

void ViennaBallJoint::solve(ViennaBody *a, ViennaBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// World pivot on each body (using symmetric local pivot)
	vec3 worldPivotA = xA.xform(pivot_a);
	vec3 worldPivotB = xB.xform(pivot_a);

	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// ---- Pivot constraint (keep anchors coincident) ----
	vec3 posError = worldPivotB - worldPivotA;
	if (invMassSum > CMP_EPSILON) {
		real_t erp = 0.2f; // Baumgarte factor
		vec3 correction = posError * (erp / dt);
		vec3 impulse = correction / invMassSum;
		if (invMassA > 0.0) a->apply_impulse( impulse, worldPivotA);
		if (invMassB > 0.0) b->apply_impulse(-impulse, worldPivotB);
	}

	// ---- Cone limit ----
	if (cone_limit_enabled) {
		// Use local Z axes as the cone axis for simplicity (can be customised)
		vec3 coneAxisA = xA.basis.get_column(2).normalized();
		vec3 coneAxisB = xB.basis.get_column(2).normalized();

		real_t dotAxes = coneAxisA.dot(coneAxisB);
		real_t angle = Math::acos(CLAMP(dotAxes, -1.0, 1.0));
		if (angle > cone_angle) {
			vec3 rotAxis = coneAxisB.cross(coneAxisA);
			real_t rotAxisLen = rotAxis.length();
			if (rotAxisLen > CMP_EPSILON) {
				rotAxis /= rotAxisLen;
				real_t error = angle - cone_angle;
				vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
				vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
				real_t invEffInertia = invIA.dot(rotAxis) + invIB.dot(rotAxis);
				if (invEffInertia > CMP_EPSILON) {
					real_t angularSpeed = error * 0.5f / dt;
					vec3 angularImpulse = rotAxis * (angularSpeed / invEffInertia);
					if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
					if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
				}
			}
		}
	}

	// ---- Twist limit ----
	if (twist_limit_enabled) {
		vec3 coneAxisA = xA.basis.get_column(2).normalized();
		vec3 coneAxisB = xB.basis.get_column(2).normalized();
		vec3 refDirA = xA.basis.get_column(0).normalized();
		vec3 refDirB = xB.basis.get_column(0).normalized();
		refDirA = (refDirA - coneAxisA * refDirA.dot(coneAxisA)).normalized();
		refDirB = (refDirB - coneAxisB * refDirB.dot(coneAxisB)).normalized();
		vec3 crossRef = refDirB.cross(refDirA);
		real_t dotRef   = refDirB.dot(refDirA);
		real_t twistAngle = Math::atan2(crossRef.dot(coneAxisA), dotRef);

		real_t lower = twist_min;
		real_t upper = twist_max;
		real_t limitError = 0.0f;
		if (twistAngle < lower) limitError = lower - twistAngle;
		else if (twistAngle > upper) limitError = upper - twistAngle;

		if (Math::abs(limitError) > CMP_EPSILON) {
			vec3 rotAxis = coneAxisA;
			vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
			vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
			real_t invEff = invIA.dot(rotAxis) + invIB.dot(rotAxis);
			if (invEff > CMP_EPSILON) {
				real_t angularSpeed = limitError * 0.5f / dt;
				vec3 angularImpulse = rotAxis * (angularSpeed / invEff);
				if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
				if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
			}
		}
	}

	// ---- Motor ----
	if (motor_enabled) {
		vec3 coneAxisA = xA.basis.get_column(2).normalized();
		vec3 omegaA = a->get_angular_velocity();
		vec3 omegaB = b->get_angular_velocity();
		real_t currentOmega = (omegaB - omegaA).dot(coneAxisA);
		real_t omegaError = motor_target_vel - currentOmega;

		vec3 invIA = a->get_inverse_inertia_world().xform(coneAxisA);
		vec3 invIB = b->get_inverse_inertia_world().xform(coneAxisA);
		real_t invEff = invIA.dot(coneAxisA) + invIB.dot(coneAxisA);
		if (invEff > CMP_EPSILON) {
			real_t motorAccel = omegaError / dt;
			real_t motorTorque = motorAccel / invEff;
			motorTorque = CLAMP(motorTorque, -motor_max_torque, motor_max_torque);
			vec3 motorImpulse = coneAxisA * motorTorque;
			if (invMassA > 0.0) a->apply_impulse(vec3(), -motorImpulse);
			if (invMassB > 0.0) b->apply_impulse(vec3(),  motorImpulse);
		}
	}
}

} // namespace vienna

#endif // VIENNA_JOINTS_BALL_JOINT_H
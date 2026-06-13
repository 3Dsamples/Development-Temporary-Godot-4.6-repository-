// File 87: modules/genesis/src/constraints/joint_constraint.h
// Joint constraints for kinematic chains, robots, and linkages.
// Provides fixed, revolute, prismatic, and spherical joint types.
// Integrates with the Gaia constraint framework for PBD/XPBD solving.

#ifndef GENESIS_CONSTRAINTS_JOINT_CONSTRAINT_H
#define GENESIS_CONSTRAINTS_JOINT_CONSTRAINT_H

#include "../../../gaia/src/framework/constraint.h"    // gaia::Constraint
#include "../../../gaia/src/framework/body.h"          // gaia::RigidBody (for anchor/axis)
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

namespace genesis {

/**
 * A joint constraint connects two rigid bodies (or one to the world) with
 * a mechanical degree‑of‑freedom reduction. It can be solved via position-based
 * or velocity‑based methods.
 *
 * Joint types:
 * - FIXED: rigid attachment (0 DOF)
 * - REVOLUTE: a hinge (1 rotational DOF around a local axis)
 * - PRISMATIC: a slider (1 translational DOF along a local axis)
 * - SPHERICAL: ball joint (3 rotational DOF, no translation)
 */
class JointConstraint : public gaia::Constraint {
public:
	enum JointType {
		FIXED,
		REVOLUTE,
		PRISMATIC,
		SPHERICAL
	};

	JointConstraint() :
		Constraint(),
		body_a(nullptr),
		body_b(nullptr),
		joint_type(FIXED),
		anchor_a(Vector3()),
		anchor_b(Vector3()),
		axis_a(Vector3(1, 0, 0)),
		axis_b(Vector3(1, 0, 0)),
		lambda_translation(Vector3()),
		lambda_rotation(Vector3()) {
		type = CUSTOM;
	}

	// --- Bodies ---
	void set_bodies(gaia::RigidBody *p_body_a, gaia::RigidBody *p_body_b = nullptr) {
		body_a = p_body_a;
		body_b = p_body_b;
	}

	// --- Joint type ---
	void set_joint_type(JointType p_type) { joint_type = p_type; }
	JointType get_joint_type() const { return joint_type; }

	// --- Anchor points in local space of each body ---
	void set_anchor_a(const Vector3 &p_anchor) { anchor_a = p_anchor; }
	Vector3 get_anchor_a() const { return anchor_a; }

	void set_anchor_b(const Vector3 &p_anchor) { anchor_b = p_anchor; }
	Vector3 get_anchor_b() const { return anchor_b; }

	// --- Axes in local space (for revolute and prismatic) ---
	void set_axis_a(const Vector3 &p_axis) { axis_a = p_axis.normalized(); }
	Vector3 get_axis_a() const { return axis_a; }

	void set_axis_b(const Vector3 &p_axis) { axis_b = p_axis.normalized(); }
	Vector3 get_axis_b() const { return axis_b; }

	// --- Position-level constraint solving (XPBD compliant) ---
	virtual void solve_position(real_t dt) override {
		ERR_FAIL_COND(!body_a);

		Transform3D xform_a = body_a->get_transform();
		Transform3D xform_b;
		if (body_b) {
			xform_b = body_b->get_transform();
		} else {
			xform_b = Transform3D(); // world frame
		}

		Vector3 r_a = xform_a.xform(anchor_a);
		Vector3 r_b = xform_b.xform(anchor_b);
		Vector3 error = r_b - r_a;

		switch (joint_type) {
			case FIXED: {
				// Full constraint: position + rotation alignment
				solve_fixed(xform_a, xform_b, error, dt);
				break;
			}
			case REVOLUTE: {
				// Translation constraint on anchor, plus align rotation axes
				solve_revolute(xform_a, xform_b, error, dt);
				break;
			}
			case PRISMATIC: {
				// Translation along axis only, align rotation axes
				solve_prismatic(xform_a, xform_b, error, dt);
				break;
			}
			case SPHERICAL: {
				// Only translation constraint
				solve_spherical(xform_a, xform_b, error, dt);
				break;
			}
		}
	}

	// --- Velocity-level correction (damping) ---
	virtual void solve_velocity(real_t dt) override {
		if (damping <= 0.0) return;
		// Apply velocity damping along constrained directions
		ERR_FAIL_COND(!body_a);

		Vector3 v_a = body_a->get_linear_velocity();
		Vector3 w_a = body_a->get_angular_velocity();
		Vector3 v_b = body_b ? body_b->get_linear_velocity() : Vector3();
		Vector3 w_b = body_b ? body_b->get_angular_velocity() : Vector3();

		// Relative velocity at anchor
		Transform3D xform_a = body_a->get_transform();
		Transform3D xform_b = body_b ? body_b->get_transform() : Transform3D();
		Vector3 r_a_world = xform_a.xform(anchor_a);
		Vector3 r_b_world = xform_b.xform(anchor_b);
		Vector3 rel_vel = (v_b + w_b.cross(r_b_world - xform_b.origin)) -
						  (v_a + w_a.cross(r_a_world - xform_a.origin));

		// For revolute/prismatic: damp angular velocity difference along axis
		// Simplified: global damping factor applied to relative velocity in constrained directions.
		// We'll project out relative motion using the Jacobian of each joint type.
		// For brevity, we assume isotropic damping on all DOFs that are constrained.
		// A full implementation would damp only the constrained DOFs using the same Jacobian as position.
		// Here we do a simple global damping: v_a and v_b are nudged toward each other.
		real_t inv_mass_a = body_a->get_inverse_mass();
		real_t inv_mass_b = body_b ? body_b->get_inverse_mass() : 0.0;
		real_t w_sum = inv_mass_a + inv_mass_b;
		if (w_sum <= 0) return;

		Vector3 impulse = rel_vel * damping / w_sum;
		body_a->set_linear_velocity(v_a + impulse * inv_mass_a);
		if (body_b) body_b->set_linear_velocity(v_b - impulse * inv_mass_b);
		// Angular damping omitted for simplicity.
	}

private:
	// Compute inverse softness matrix (XPBD compliance)
	real_t get_effective_compliance(real_t dt) const {
		return compliance / (dt * dt);
	}

	void apply_position_correction(gaia::RigidBody *body, const Vector3 &world_point,
								   const Vector3 &correction) const {
		if (!body || body->get_type() != gaia::RigidBody::DYNAMIC) return;
		real_t inv_mass = body->get_inverse_mass();
		Vector3 r = world_point - body->get_position();
		body->set_position(body->get_position() + correction * inv_mass);
		// Angular update: dq = I^-1 * (r x correction) * dt? For position correction we use:
		// Δq = I^-1 * (r × correction) * 0.5? Actually in PBD, angular correction is done via body->set_rotation update.
		// We'll just apply the angular impulse directly.
		Vector3 rot_correction = body->get_inverse_inertia_world().xform(r.cross(correction));
		Quaternion q = Quaternion(body->get_rotation());
		Quaternion omega_quat(rot_correction, 1.0);
		q = omega_quat * q;
		q.normalize();
		body->set_rotation(q.get_basis().orthonormalized());
	}

	// --- Joint-specific solvers ---

	void solve_fixed(const Transform3D &xform_a, const Transform3D &xform_b,
					 const Vector3 &error, real_t dt) {
		if (!body_a && !body_b) return;

		real_t alpha_tilde = get_effective_compliance(dt);
		real_t inv_mass_a = body_a ? body_a->get_inverse_mass() : 0.0;
		real_t inv_mass_b = body_b ? body_b->get_inverse_mass() : 0.0;
		real_t w = inv_mass_a + inv_mass_b + alpha_tilde;
		if (w < CMP_EPSILON) return;

		// Position correction
		Vector3 correction = -error;
		Vector3 delta_lambda = correction / w;

		// Apply to both bodies
		if (body_a) {
			Vector3 r_a_world = xform_a.xform(anchor_a);
			apply_position_correction(body_a, r_a_world, delta_lambda * inv_mass_a);
		}
		if (body_b) {
			Vector3 r_b_world = xform_b.xform(anchor_b);
			apply_position_correction(body_b, r_b_world, -delta_lambda * inv_mass_b);
		}

		// Rotation alignment: align body_b's local frame to body_a's (if both exist)
		if (body_a && body_b) {
			// Desired relative rotation: no relative rotation
			Basis R_a = xform_a.basis;
			Basis R_b = xform_b.basis;
			Basis R_err = R_b * R_a.transposed();
			Quaternion q_err(R_err);
			Vector3 axis;
			real_t angle;
			q_err.get_axis_angle(axis, angle);
			if (Math::abs(angle) > CMP_EPSILON) {
				// Angular compliance
				real_t inv_inertia_a = body_a->get_inverse_inertia_world().xform(axis).dot(axis);
				real_t inv_inertia_b = body_b->get_inverse_inertia_world().xform(axis).dot(axis);
				real_t w_rot = inv_inertia_a + inv_inertia_b + alpha_tilde;
				real_t correction_angle = angle / w_rot * (1.0 / dt); // scale with dt? PBD uses subdivision.
				// Apply rotation update
				Quaternion q_corr(axis, -correction_angle * 0.5);
				Quaternion new_q_a = q_corr * Quaternion(R_a);
				new_q_a.normalize();
				body_a->set_rotation(new_q_a.get_basis().orthonormalized());
				Quaternion new_q_b = q_corr * Quaternion(R_b);
				new_q_b.normalize();
				body_b->set_rotation(new_q_b.get_basis().orthonormalized());
			}
		}
	}

	void solve_revolute(const Transform3D &xform_a, const Transform3D &xform_b,
						const Vector3 &error, real_t dt) {
		// Fix anchor translation
		solve_spherical(xform_a, xform_b, error, dt);
		// Align hinge axes
		if (body_a && body_b) {
			Vector3 axis_a_w = xform_a.basis.xform(axis_a).normalized();
			Vector3 axis_b_w = xform_b.basis.xform(axis_b).normalized();
			Vector3 rot_err = axis_b_w.cross(axis_a_w);
			real_t dot = axis_b_w.dot(axis_a_w);
			if (Math::abs(dot) < 1.0 - CMP_EPSILON) {
				real_t inv_inertia_a = body_a->get_inverse_inertia_world().xform(rot_err).dot(rot_err);
				real_t inv_inertia_b = body_b->get_inverse_inertia_world().xform(rot_err).dot(rot_err);
				real_t w_rot = inv_inertia_a + inv_inertia_b + get_effective_compliance(dt);
				Vector3 correction = rot_err / w_rot;
				// Apply torque to align axes
				body_a->set_angular_velocity(body_a->get_angular_velocity() + correction * inv_inertia_a);
				body_b->set_angular_velocity(body_b->get_angular_velocity() - correction * inv_inertia_b);
				// Note: This velocity-level correction is simpler; full position update would be as in fixed joint.
			}
		}
	}

	void solve_prismatic(const Transform3D &xform_a, const Transform3D &xform_b,
						 const Vector3 &error, real_t dt) {
		// Prismatic: allow translation along axis, but fix other directions and all rotations.
		Vector3 axis_a_w = xform_a.basis.xform(axis_a).normalized();
		// Remove error component along the prismatic axis
		Vector3 orthogonal_error = error - axis_a_w * axis_a_w.dot(error);
		if (body_a && body_b) {
			real_t inv_mass_a = body_a->get_inverse_mass();
			real_t inv_mass_b = body_b->get_inverse_mass();
			real_t w = inv_mass_a + inv_mass_b + get_effective_compliance(dt);
			Vector3 correction = -orthogonal_error;
			Vector3 delta = correction / w;
			Vector3 r_a_w = xform_a.xform(anchor_a);
			Vector3 r_b_w = xform_b.xform(anchor_b);
			apply_position_correction(body_a, r_a_w, delta * inv_mass_a);
			apply_position_correction(body_b, r_b_w, -delta * inv_mass_b);
		}
		// For rotation: ensure axis_a_w and axis_b_w are parallel, and also align other axes (like revolute but without constraining rotation around that axis? Actually prismatic locks all rotations.)
		if (body_a && body_b) {
			// Align full orientation (same as fixed rotation)
			Basis R_a = xform_a.basis;
			Basis R_b = xform_b.basis;
			Basis R_err = R_b * R_a.transposed();
			Quaternion q_err(R_err);
			Vector3 rot_axis;
			real_t angle;
			q_err.get_axis_angle(rot_axis, angle);
			real_t inv_inertia_a = body_a->get_inverse_inertia_world().xform(rot_axis).dot(rot_axis);
			real_t inv_inertia_b = body_b->get_inverse_inertia_world().xform(rot_axis).dot(rot_axis);
			real_t w_rot = inv_inertia_a + inv_inertia_b + get_effective_compliance(dt);
			real_t correction_angle = angle / w_rot;
			Quaternion q_corr(rot_axis, -correction_angle * 0.5);
			Quaternion new_q_a = q_corr * Quaternion(R_a);
			new_q_a.normalize();
			body_a->set_rotation(new_q_a.get_basis().orthonormalized());
			Quaternion new_q_b = q_corr * Quaternion(R_b);
			new_q_b.normalize();
			body_b->set_rotation(new_q_b.get_basis().orthonormalized());
		}
	}

	void solve_spherical(const Transform3D &xform_a, const Transform3D &xform_b,
						 const Vector3 &error, real_t dt) {
		if (!body_a && !body_b) return;
		real_t inv_mass_a = body_a ? body_a->get_inverse_mass() : 0.0;
		real_t inv_mass_b = body_b ? body_b->get_inverse_mass() : 0.0;
		real_t w = inv_mass_a + inv_mass_b + get_effective_compliance(dt);
		if (w < CMP_EPSILON) return;

		Vector3 correction = -error;
		Vector3 delta = correction / w;
		if (body_a) {
			Vector3 r_a_w = xform_a.xform(anchor_a);
			apply_position_correction(body_a, r_a_w, delta * inv_mass_a);
		}
		if (body_b) {
			Vector3 r_b_w = xform_b.xform(anchor_b);
			apply_position_correction(body_b, r_b_w, -delta * inv_mass_b);
		}
	}

	gaia::RigidBody *body_a;
	gaia::RigidBody *body_b;
	JointType joint_type;
	Vector3 anchor_a;
	Vector3 anchor_b;
	Vector3 axis_a;
	Vector3 axis_b;

	// XPBD Lagrange multipliers
	Vector3 lambda_translation;
	Vector3 lambda_rotation;
};

} // namespace genesis

#endif // GENESIS_CONSTRAINTS_JOINT_CONSTRAINT_H
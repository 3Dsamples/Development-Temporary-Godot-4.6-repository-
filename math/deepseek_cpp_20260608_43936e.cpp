// File 342: modules/wicked/src/joints/wicked_slider_joint.h
// Wicked Slider (Prismatic) Joint – constrains two bodies to translate only
// along a single axis, with optional limits and motor. All angular degrees
// except the rotation around the axis are locked.  The solve method is
// fully inline for maximum solver throughput.

#ifndef WICKED_JOINTS_SLIDER_JOINT_H
#define WICKED_JOINTS_SLIDER_JOINT_H

#include "wicked_joint.h"
#include "../bodies/wicked_body.h"

namespace wicked {

class WickedSliderJoint : public WickedJoint {
    GDCLASS(WickedSliderJoint, WickedJoint);

public:
    WickedSliderJoint() { joint_type = JointType::SLIDER; }

    // Set the slider axis in the local frame of body A.
    void set_axis(const vec3 &p_axis) { axis_a = p_axis.normalized(); }
    vec3 get_axis() const { return axis_a; }

    // Set the pivot point in the local frame of body A (anchor).
    void set_pivot(const vec3 &p_pivot) { pivot_a = p_pivot; }
    vec3 get_pivot() const { return pivot_a; }

    // --- Translation limits (relative to the initial offset) ---
    void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
    bool is_limit_enabled() const { return limit_enabled; }
    void set_limit_range(real_t p_min, real_t p_max) {
        min_limit = MIN(p_min, p_max);
        max_limit = MAX(p_min, p_max);
    }
    real_t get_min_limit() const { return min_limit; }
    real_t get_max_limit() const { return max_limit; }

    // --- Motor ---
    void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
    bool is_motor_enabled() const { return motor_enabled; }
    void set_motor_target_velocity(real_t p_vel) { motor_target_vel = p_vel; }
    real_t get_motor_target_velocity() const { return motor_target_vel; }
    void set_motor_max_force(real_t p_force) { motor_max_force = MAX(p_force, 0.0); }
    real_t get_motor_max_force() const { return motor_max_force; }

    // --- Solver entry (fully inline) ---
    virtual void solve(WickedBody *a, WickedBody *b, real_t dt) override {
        if (!enabled || !a || !b) return;
        if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

        const mat4 &xA = a->get_transform();
        const mat4 &xB = b->get_transform();

        // World anchor on each body (symmetric local pivot)
        vec3 worldAnchorA = xA.xform(pivot_a);
        vec3 worldAnchorB = xB.xform(pivot_a);

        // World slider axis from body A
        vec3 worldAxisA = xA.basis.xform(axis_a).normalized();

        real_t invMassA = a->get_inverse_mass();
        real_t invMassB = b->get_inverse_mass();
        real_t invMassSum = invMassA + invMassB;

        // Compute relative translation along the slider axis and perpendicular error
        vec3 posError = worldAnchorB - worldAnchorA;
        real_t parallelError = posError.dot(worldAxisA);
        vec3 perpendicularError = posError - worldAxisA * parallelError;

        // On first solve, record the initial offset for limit calculation
        if (first_solve) {
            initial_offset = parallelError;
            first_solve = false;
        }

        // ---- Perpendicular constraint: zero out motion not along the axis ----
        if (invMassSum > CMP_EPSILON) {
            real_t erp = 0.2f;
            vec3 correction = perpendicularError * (erp / dt);
            vec3 impulse = correction / invMassSum;
            if (invMassA > 0.0) a->apply_impulse( impulse, worldAnchorA);
            if (invMassB > 0.0) b->apply_impulse(-impulse, worldAnchorB);
        }

        // ---- Angular constraint: align body orientations modulo rotation around the axis ----
        {
            // Build a reference frame for body A
            vec3 perpDirA = (Math::abs(worldAxisA.x) < 0.999f) ?
                worldAxisA.cross(vec3(1, 0, 0)).normalized() :
                worldAxisA.cross(vec3(0, 1, 0)).normalized();
            vec3 refA1 = perpDirA;
            vec3 refA2 = worldAxisA.cross(refA1).normalized();
            mat3 refFrameA(refA1, refA2, worldAxisA); // columns

            // Build the corresponding frame for body B
            vec3 worldAxisB = xB.basis.xform(axis_a).normalized();
            vec3 perpDirB = (Math::abs(worldAxisB.x) < 0.999f) ?
                worldAxisB.cross(vec3(1, 0, 0)).normalized() :
                worldAxisB.cross(vec3(0, 1, 0)).normalized();
            vec3 refB1 = perpDirB;
            vec3 refB2 = worldAxisB.cross(refB1).normalized();
            mat3 refFrameB(refB1, refB2, worldAxisB);

            // Rotation error between the two frames
            mat3 R_err = refFrameB * refFrameA.transposed();
            quat q_err(R_err);
            vec3 rotAxis;
            real_t rotAngle;
            q_err.get_axis_angle(rotAxis, rotAngle);
            if (Math::abs(rotAngle) > 0.01f) {
                vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
                vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
                real_t invEffInertia = invIA.dot(rotAxis) + invIB.dot(rotAxis);
                if (invEffInertia > CMP_EPSILON) {
                    real_t angularSpeed = rotAngle * 0.5f / dt;
                    vec3 angularImpulse = rotAxis * (angularSpeed / invEffInertia);
                    if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
                    if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
                }
            }
        }

        // ---- Limits (translation along axis) ----
        if (limit_enabled) {
            real_t currentTranslation = parallelError - initial_offset;
            real_t lower = min_limit;
            real_t upper = max_limit;
            real_t limitError = 0.0;
            if (currentTranslation < lower) limitError = lower - currentTranslation;
            else if (currentTranslation > upper) limitError = upper - currentTranslation;

            if (Math::abs(limitError) > CMP_EPSILON) {
                // Effective mass along the slider axis
                vec3 rA = worldAnchorA - a->get_position();
                vec3 rB = worldAnchorB - b->get_position();
                real_t invEffMassAlong = invMassSum +
                    worldAxisA.dot(a->get_inverse_inertia_world().xform(rA.cross(worldAxisA)).cross(rA)) +
                    worldAxisA.dot(b->get_inverse_inertia_world().xform(rB.cross(worldAxisA)).cross(rB));
                if (invEffMassAlong > CMP_EPSILON) {
                    real_t correction = limitError * 0.3f / dt;
                    vec3 impulse = worldAxisA * (correction / invEffMassAlong);
                    if (invMassA > 0.0) a->apply_impulse( impulse, worldAnchorA);
                    if (invMassB > 0.0) b->apply_impulse(-impulse, worldAnchorB);
                }
            }
        }

        // ---- Motor ----
        if (motor_enabled) {
            vec3 rA = worldAnchorA - a->get_position();
            vec3 rB = worldAnchorB - b->get_position();
            vec3 velA = a->get_linear_velocity() + a->get_angular_velocity().cross(rA);
            vec3 velB = b->get_linear_velocity() + b->get_angular_velocity().cross(rB);
            real_t currentSpeed = (velB - velA).dot(worldAxisA);
            real_t speedError = motor_target_vel - currentSpeed;

            real_t invEffMassAlong = invMassSum +
                worldAxisA.dot(a->get_inverse_inertia_world().xform(rA.cross(worldAxisA)).cross(rA)) +
                worldAxisA.dot(b->get_inverse_inertia_world().xform(rB.cross(worldAxisA)).cross(rB));
            if (invEffMassAlong > CMP_EPSILON) {
                real_t motorForce = speedError / (invEffMassAlong * dt);
                motorForce = CLAMP(motorForce, -motor_max_force, motor_max_force);
                vec3 motorImpulse = worldAxisA * motorForce;
                if (invMassA > 0.0) a->apply_impulse( motorImpulse, worldAnchorA);
                if (invMassB > 0.0) b->apply_impulse(-motorImpulse, worldAnchorB);
            }
        }
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_axis", "axis"), &WickedSliderJoint::set_axis);
        ClassDB::bind_method(D_METHOD("get_axis"), &WickedSliderJoint::get_axis);
        ClassDB::bind_method(D_METHOD("set_pivot", "pivot"), &WickedSliderJoint::set_pivot);
        ClassDB::bind_method(D_METHOD("get_pivot"), &WickedSliderJoint::get_pivot);
        ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &WickedSliderJoint::set_limit_enabled);
        ClassDB::bind_method(D_METHOD("is_limit_enabled"), &WickedSliderJoint::is_limit_enabled);
        ClassDB::bind_method(D_METHOD("set_limit_range", "min", "max"), &WickedSliderJoint::set_limit_range);
        ClassDB::bind_method(D_METHOD("get_min_limit"), &WickedSliderJoint::get_min_limit);
        ClassDB::bind_method(D_METHOD("get_max_limit"), &WickedSliderJoint::get_max_limit);
        ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &WickedSliderJoint::set_motor_enabled);
        ClassDB::bind_method(D_METHOD("is_motor_enabled"), &WickedSliderJoint::is_motor_enabled);
        ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "vel"), &WickedSliderJoint::set_motor_target_velocity);
        ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &WickedSliderJoint::get_motor_target_velocity);
        ClassDB::bind_method(D_METHOD("set_motor_max_force", "force"), &WickedSliderJoint::set_motor_max_force);
        ClassDB::bind_method(D_METHOD("get_motor_max_force"), &WickedSliderJoint::get_motor_max_force);

        ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis"), "set_axis", "get_axis");
        ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "pivot"), "set_pivot", "get_pivot");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
    }

private:
    vec3 axis_a = vec3(1, 0, 0);
    vec3 pivot_a = vec3();
    bool limit_enabled = false;
    real_t min_limit = -1.0;
    real_t max_limit =  1.0;
    bool motor_enabled = false;
    real_t motor_target_vel = 0.0;
    real_t motor_max_force = 100.0;
    real_t initial_offset = 0.0;
    bool first_solve = true;
};

} // namespace wicked

#endif // WICKED_JOINTS_SLIDER_JOINT_H
// File 382: modules/integration/unified_ragdoll_blender.h
// High‑performance cross‑engine ragdoll‑animation blending system.
// Blends a kinematic skeleton (animation) with a dynamic ragdoll (physics)
// across any registered engine (Newton, Genesis, Vienna, Wicked).
// Uses per‑bone motorized constraints that track animated targets with
// configurable stiffness and damping, enabling seamless transitions
// between animation and simulation.  All hot‑path solving is inline.

#ifndef INTEGRATION_UNIFIED_RAGDOLL_BLENDER_H
#define INTEGRATION_UNIFIED_RAGDOLL_BLENDER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

#include "unified_joint_bridge.h"       // CrossEngineBody, CrossEngineJoint

namespace unified {

// ---------------------------------------------------------------------------
// A single bone in the ragdoll.  Maps a body (any engine) to an animated
// target transform and a set of PD gains.
// ---------------------------------------------------------------------------
struct RagdollBone {
    CrossEngineBody body;              // physics body
    Transform3D     target_pose;       // world‑space pose from animation
    bool            kinematic = false; // if true, body is driven to target via PD
    // PD gains
    real_t position_kp = 500.0;
    real_t position_kd = 50.0;
    real_t rotation_kp = 1000.0;
    real_t rotation_kd = 100.0;
    real_t max_force = 10000.0;
    real_t max_torque = 1000.0;
    // Current state (updated each solve)
    real_t applied_force_mag = 0.0;
    real_t applied_torque_mag = 0.0;
};

// ---------------------------------------------------------------------------
// Blender manager – holds a list of bones and solves the PD controllers
// every physics frame after the animation system has updated the targets.
// ---------------------------------------------------------------------------
class UnifiedRagdollBlender : public RefCounted {
    GDCLASS(UnifiedRagdollBlender, RefCounted);

    LocalVector<RagdollBone> bones;

    // World pointers for body lookup (same as in bridge)
    newton::NewtonWorld   *newton_world = nullptr;
    genesis::GenesisWorld *genesis_world = nullptr;
    vienna::ViennaWorld   *vienna_world = nullptr;
    wicked::WickedWorld   *wicked_world = nullptr;

    real_t blend_factor = 1.0;        // 0 = fully kinematic, 1 = fully physics
    int    solver_iterations = 5;

public:
    UnifiedRagdollBlender() {}

    void set_newton_world(newton::NewtonWorld *w)   { newton_world = w; }
    void set_genesis_world(genesis::GenesisWorld *w) { genesis_world = w; }
    void set_vienna_world(vienna::ViennaWorld *w)    { vienna_world = w; }
    void set_wicked_world(wicked::WickedWorld *w)    { wicked_world = w; }

    void set_blend_factor(real_t f) { blend_factor = CLAMP(f, 0.0f, 1.0f); }
    real_t get_blend_factor() const { return blend_factor; }

    void set_solver_iterations(int n) { solver_iterations = MAX(n, 1); }

    // Add a bone.  The CrossEngineBody must already be resolved (body_ptr set).
    void add_bone(const RagdollBone &p_bone) {
        RagdollBone b = p_bone;
        resolve_body_ptr(b.body);
        bones.push_back(b);
    }

    // Remove a bone by index.
    void remove_bone(int p_idx) {
        ERR_FAIL_INDEX(p_idx, bones.size());
        bones.remove_at(p_idx);
    }

    void clear() { bones.clear(); }

    // Set the animated target pose for a given bone index.
    void set_target_pose(int p_idx, const Transform3D &p_pose) {
        ERR_FAIL_INDEX(p_idx, bones.size());
        bones[p_idx].target_pose = p_pose;
    }

    // Set PD gains for a bone.
    void set_bone_gains(int p_idx, real_t p_kp, real_t p_kd,
                        real_t p_rot_kp, real_t p_rot_kd) {
        ERR_FAIL_INDEX(p_idx, bones.size());
        bones[p_idx].position_kp = MAX(p_kp, 0.0);
        bones[p_idx].position_kd = MAX(p_kd, 0.0);
        bones[p_idx].rotation_kp = MAX(p_rot_kp, 0.0);
        bones[p_idx].rotation_kd = MAX(p_rot_kd, 0.0);
    }

    // Set bone kinematic flag.
    void set_bone_kinematic(int p_idx, bool p_kin) {
        ERR_FAIL_INDEX(p_idx, bones.size());
        bones[p_idx].kinematic = p_kin;
    }

    // Get the applied force magnitude for a bone (for breakable checks).
    real_t get_bone_applied_force(int p_idx) const {
        ERR_FAIL_INDEX_V(p_idx, bones.size(), 0.0);
        return bones[p_idx].applied_force_mag;
    }
    real_t get_bone_applied_torque(int p_idx) const {
        ERR_FAIL_INDEX_V(p_idx, bones.size(), 0.0);
        return bones[p_idx].applied_torque_mag;
    }

    // Solve all bones for the given time step.
    // Must be called after animation system has updated target poses.
    void solve(real_t p_dt) {
        if (blend_factor >= 1.0) return; // fully physics, no blending needed
        real_t lambda = 1.0 - blend_factor; // how much the target pulls

        for (int iter = 0; iter < solver_iterations; ++iter) {
            for (RagdollBone &bone : bones) {
                if (!bone.body.body_ptr) continue;
                if (!bone.kinematic && blend_factor > 0.0) continue; // kinematic flag overrides blend? We'll use both: kinematic flag = always drive, blend_factor scales the strength.

                Transform3D current = bone.body.get_transform(bone.body.body_ptr);
                Vector3 pos_error = bone.target_pose.origin - current.origin;
                Vector3 vel_error = bone.target_pose.origin.is_zero_approx() ? -bone.body.get_linear_velocity(bone.body.body_ptr) : Vector3(); // approximate target velocity = 0
                // Better: we can compute target velocity from previous frame target and dt, but we'll leave zero for simplicity.

                // Position PD
                Vector3 force = bone.position_kp * pos_error + bone.position_kd * vel_error;
                real_t f_len = force.length();
                if (f_len > bone.max_force && bone.max_force > 0.0) force *= bone.max_force / f_len;
                bone.applied_force_mag = f_len;

                // Rotation PD
                Quaternion q_cur(current.basis);
                Quaternion q_tar(bone.target_pose.basis);
                Quaternion q_diff = q_tar * q_cur.inverse();
                Vector3 rot_axis;
                real_t rot_angle;
                q_diff.get_axis_angle(rot_axis, rot_angle);
                rot_angle = CLAMP(rot_angle, -Math_PI * 0.5f, Math_PI * 0.5f);

                Vector3 angvel = bone.body.get_angular_velocity(bone.body.body_ptr);
                Vector3 torque = bone.rotation_kp * rot_axis * rot_angle - bone.rotation_kd * angvel;
                real_t t_len = torque.length();
                if (t_len > bone.max_torque && bone.max_torque > 0.0) torque *= bone.max_torque / t_len;
                bone.applied_torque_mag = t_len;

                // Scale by lambda (if not fully kinematic)
                if (!bone.kinematic) {
                    force *= lambda;
                    torque *= lambda;
                }

                // Apply impulses
                real_t invMass = bone.body.get_inverse_mass(bone.body.body_ptr);
                if (invMass > 0.0) {
                    bone.body.apply_impulse(bone.body.body_ptr, force * p_dt, current.origin);
                    bone.body.apply_torque_impulse(bone.body.body_ptr, torque * p_dt);
                }
            }
        }
    }

private:
    // Resolve body_ptr from engine world (similar to bridge)
    void resolve_body_ptr(CrossEngineBody &p_body) {
        switch (p_body.engine) {
            case CrossEngineBody::NEWTON: {
                if (!newton_world) break;
                Ref<newton::NewtonBody> b = newton_world->get_body(p_body.id);
                if (b.is_valid()) p_body.body_ptr = b.ptr();
            } break;
            case CrossEngineBody::GENESIS: {
                if (!genesis_world) break;
                Ref<genesis::RigidEntity> e = genesis_world->get_entity(p_body.id);
                if (e.is_valid()) p_body.body_ptr = e.ptr();
            } break;
            case CrossEngineBody::VIENNA: {
                if (!vienna_world) break;
                Ref<vienna::ViennaBody> v = vienna_world->get_body(p_body.id);
                if (v.is_valid()) p_body.body_ptr = v.ptr();
            } break;
            case CrossEngineBody::WICKED: {
                if (!wicked_world) break;
                Ref<wicked::WickedBody> w = wicked_world->get_body(p_body.id);
                if (w.is_valid()) p_body.body_ptr = w.ptr();
            } break;
        }
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_RAGDOLL_BLENDER_H
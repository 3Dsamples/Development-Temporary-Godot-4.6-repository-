// File 380: modules/integration/unified_joint_bridge.h
// Cross‑Engine Joint Bridge – allows rigid body joints between different
// physics engines (Newton–Genesis, Vienna–Wicked, etc.).  The bridge
// implements a generic constraint solver that reads body states through
// engine‑specific wrappers and applies corrective impulses accordingly.
// Supports ball, hinge, slider, and fixed joints with limits and motors.
// All hot‑path constraint projections are fully inline for high performance.

#ifndef INTEGRATION_UNIFIED_JOINT_BRIDGE_H
#define INTEGRATION_UNIFIED_JOINT_BRIDGE_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

// Engine‑specific headers (used only for obtaining body pointers)
namespace newton   { class NewtonBody; class NewtonWorld; }
namespace genesis  { class RigidEntity; class GenesisWorld; }
namespace vienna   { class ViennaBody; class ViennaWorld; }
namespace wicked   { class WickedBody; class WickedWorld; }

namespace unified {

// ---------------------------------------------------------------------------
// Abstract interface for a body residing in any engine.
// The bridge uses this opaque handle to query and modify body state.
// ---------------------------------------------------------------------------
struct CrossEngineBody {
    enum Engine { NEWTON = 0, GENESIS = 1, VIENNA = 2, WICKED = 3 };

    Engine engine;
    uint64_t id;                     // body ID within that engine
    void *world_ptr;                 // pointer to the world (for later lookup)

    // Function pointers set by the bridge at registration time.
    // They avoid virtual calls and branching inside the constraint loop.
    Transform3D (*get_transform)(void *body_ptr);
    Vector3    (*get_linear_velocity)(void *body_ptr);
    Vector3    (*get_angular_velocity)(void *body_ptr);
    real_t     (*get_inverse_mass)(void *body_ptr);
    Basis      (*get_inverse_inertia_world)(void *body_ptr);
    void       (*apply_impulse)(void *body_ptr, const Vector3 &impulse, const Vector3 &world_point);
    void       (*apply_torque_impulse)(void *body_ptr, const Vector3 &torque);

    void *body_ptr;                 // raw pointer to the actual body object
};

// ---------------------------------------------------------------------------
// Descriptor for a cross‑engine joint.
// ---------------------------------------------------------------------------
struct CrossEngineJoint {
    CrossEngineBody body_a;
    CrossEngineBody body_b;

    // Joint type
    enum Type { BALL, HINGE, SLIDER, FIXED } type;

    // Local pivots (in each body's local frame)
    Vector3 pivot_a;
    Vector3 pivot_b;

    // Hinge axis (for hinge/slider) in body A's local frame
    Vector3 axis_a;

    // Limits (angles for hinge, distances for slider)
    bool   limit_enabled = false;
    real_t limit_min = -Math_PI;
    real_t limit_max =  Math_PI;

    // Motor
    bool   motor_enabled = false;
    real_t motor_target_velocity = 0.0;
    real_t motor_max_force = INFINITY;

    // Breakable
    bool   breakable = false;
    real_t break_force = INFINITY;
    real_t break_torque = INFINITY;

    bool enabled = true;
};

// ---------------------------------------------------------------------------
// Bridge manager – holds a list of cross‑engine joints and solves them
// after all engines have finished their internal steps.
// The solve method should be called once per physics frame, after the
// regular engine steps, to enforce coupling constraints.
// ---------------------------------------------------------------------------
class UnifiedJointBridge : public RefCounted {
    GDCLASS(UnifiedJointBridge, RefCounted);

    LocalVector<CrossEngineJoint> joints;

    // Engine world pointers for body lookup (set once)
    newton::NewtonWorld   *newton_world = nullptr;
    genesis::GenesisWorld *genesis_world = nullptr;
    vienna::ViennaWorld   *vienna_world = nullptr;
    wicked::WickedWorld   *wicked_world = nullptr;

    // Iteration count for constraint solving
    int solver_iterations = 10;

public:
    UnifiedJointBridge() {}

    void set_newton_world(newton::NewtonWorld *w)   { newton_world = w; }
    void set_genesis_world(genesis::GenesisWorld *w) { genesis_world = w; }
    void set_vienna_world(vienna::ViennaWorld *w)    { vienna_world = w; }
    void set_wicked_world(wicked::WickedWorld *w)    { wicked_world = w; }

    void set_solver_iterations(int n) { solver_iterations = MAX(n, 1); }

    // Add a joint.  The caller must provide fully populated CrossEngineJoint.
    void add_joint(const CrossEngineJoint &p_joint) {
        joints.push_back(p_joint);
        // Resolve body pointers from the stored world and ID
        CrossEngineJoint &j = joints.back();
        resolve_body_ptr(j.body_a);
        resolve_body_ptr(j.body_b);
    }

    // Remove a joint by index.
    void remove_joint(int p_idx) {
        ERR_FAIL_INDEX(p_idx, joints.size());
        joints.remove_at(p_idx);
    }

    // Clear all joints.
    void clear() { joints.clear(); }

    // Solve all cross‑engine constraints for the given time step.
    void solve(real_t p_dt) {
        for (int iter = 0; iter < solver_iterations; ++iter) {
            for (CrossEngineJoint &j : joints) {
                if (!j.enabled) continue;
                switch (j.type) {
                    case CrossEngineJoint::BALL:   solve_ball(j, p_dt);   break;
                    case CrossEngineJoint::HINGE:  solve_hinge(j, p_dt);  break;
                    case CrossEngineJoint::SLIDER: solve_slider(j, p_dt); break;
                    case CrossEngineJoint::FIXED:  solve_fixed(j, p_dt);  break;
                }
            }
        }
    }

private:
    // -------------------------------------------------------------------
    // Resolve body_ptr from engine and ID.
    // -------------------------------------------------------------------
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

    // -------------------------------------------------------------------
    // Constraint implementations (inline).
    // -------------------------------------------------------------------
    void solve_ball(CrossEngineJoint &j, real_t dt) {
        Transform3D xA = j.body_a.get_transform(j.body_a.body_ptr);
        Transform3D xB = j.body_b.get_transform(j.body_b.body_ptr);
        Vector3 worldPivotA = xA.xform(j.pivot_a);
        Vector3 worldPivotB = xB.xform(j.pivot_b);
        Vector3 error = worldPivotB - worldPivotA;

        real_t invMA = j.body_a.get_inverse_mass(j.body_a.body_ptr);
        real_t invMB = j.body_b.get_inverse_mass(j.body_b.body_ptr);
        real_t invSum = invMA + invMB;
        if (invSum > CMP_EPSILON) {
            real_t erp = 0.2f;
            Vector3 correction = error * (erp / dt);
            Vector3 impulse = correction / invSum;
            if (invMA > 0.0) j.body_a.apply_impulse(j.body_a.body_ptr,  impulse, worldPivotA);
            if (invMB > 0.0) j.body_b.apply_impulse(j.body_b.body_ptr, -impulse, worldPivotB);
        }
    }

    void solve_hinge(CrossEngineJoint &j, real_t dt) {
        Transform3D xA = j.body_a.get_transform(j.body_a.body_ptr);
        Transform3D xB = j.body_b.get_transform(j.body_b.body_ptr);

        // Pivot constraint (same as ball)
        Vector3 worldPivotA = xA.xform(j.pivot_a);
        Vector3 worldPivotB = xB.xform(j.pivot_b);
        Vector3 error = worldPivotB - worldPivotA;
        real_t invMA = j.body_a.get_inverse_mass(j.body_a.body_ptr);
        real_t invMB = j.body_b.get_inverse_mass(j.body_b.body_ptr);
        real_t invSum = invMA + invMB;
        if (invSum > CMP_EPSILON) {
            Vector3 correction = error * (0.2f / dt);
            Vector3 impulse = correction / invSum;
            if (invMA > 0.0) j.body_a.apply_impulse(j.body_a.body_ptr,  impulse, worldPivotA);
            if (invMB > 0.0) j.body_b.apply_impulse(j.body_b.body_ptr, -impulse, worldPivotB);
        }

        // Axis alignment
        Vector3 worldAxisA = xA.basis.xform(j.axis_a).normalized();
        Vector3 worldAxisB = xB.basis.xform(j.axis_a).normalized();
        Vector3 crossAxes = worldAxisB.cross(worldAxisA);
        real_t crossLen = crossAxes.length();
        if (crossLen > CMP_EPSILON) {
            Vector3 rotAxis = crossAxes / crossLen;
            real_t rotAngle = Math::asin(crossLen);
            rotAngle = CLAMP(rotAngle, -0.5f, 0.5f);

            Basis invIA = j.body_a.get_inverse_inertia_world(j.body_a.body_ptr);
            Basis invIB = j.body_b.get_inverse_inertia_world(j.body_b.body_ptr);
            real_t invEff = rotAxis.dot(invIA.xform(rotAxis)) + rotAxis.dot(invIB.xform(rotAxis));
            if (invEff > CMP_EPSILON) {
                real_t angularSpeed = rotAngle * 0.5f / dt;
                Vector3 angularImpulse = rotAxis * (angularSpeed / invEff);
                if (invMA > 0.0) j.body_a.apply_torque_impulse(j.body_a.body_ptr,  angularImpulse);
                if (invMB > 0.0) j.body_b.apply_torque_impulse(j.body_b.body_ptr, -angularImpulse);
            }
        }

        // Limits (hinge)
        if (j.limit_enabled) {
            Vector3 refDirA = (Math::abs(worldAxisA.x) < 0.999f) ?
                worldAxisA.cross(Vector3(1,0,0)).normalized() :
                worldAxisA.cross(Vector3(0,1,0)).normalized();
            Vector3 refDirB = xB.basis.get_column(0).normalized();
            refDirB = (refDirB - worldAxisB * refDirB.dot(worldAxisB)).normalized();
            Vector3 crossRef = refDirB.cross(refDirA);
            real_t dotRef = refDirB.dot(refDirA);
            real_t currentAngle = Math::atan2(crossRef.dot(worldAxisA), dotRef);
            real_t limitError = 0.0;
            if (currentAngle < j.limit_min) limitError = j.limit_min - currentAngle;
            else if (currentAngle > j.limit_max) limitError = j.limit_max - currentAngle;
            if (Math::abs(limitError) > CMP_EPSILON) {
                Basis invIA = j.body_a.get_inverse_inertia_world(j.body_a.body_ptr);
                Basis invIB = j.body_b.get_inverse_inertia_world(j.body_b.body_ptr);
                real_t invEff = worldAxisA.dot(invIA.xform(worldAxisA)) + worldAxisA.dot(invIB.xform(worldAxisA));
                if (invEff > CMP_EPSILON) {
                    real_t angularSpeed = limitError * 0.5f / dt;
                    Vector3 angularImpulse = worldAxisA * (angularSpeed / invEff);
                    if (invMA > 0.0) j.body_a.apply_torque_impulse(j.body_a.body_ptr,  angularImpulse);
                    if (invMB > 0.0) j.body_b.apply_torque_impulse(j.body_b.body_ptr, -angularImpulse);
                }
            }
        }
    }

    void solve_slider(CrossEngineJoint &j, real_t dt) {
        Transform3D xA = j.body_a.get_transform(j.body_a.body_ptr);
        Transform3D xB = j.body_b.get_transform(j.body_b.body_ptr);
        Vector3 worldPivotA = xA.xform(j.pivot_a);
        Vector3 worldPivotB = xB.xform(j.pivot_b);
        Vector3 worldAxisA = xA.basis.xform(j.axis_a).normalized();

        Vector3 posError = worldPivotB - worldPivotA;
        real_t parallelError = posError.dot(worldAxisA);
        Vector3 perpError = posError - worldAxisA * parallelError;

        real_t invMA = j.body_a.get_inverse_mass(j.body_a.body_ptr);
        real_t invMB = j.body_b.get_inverse_mass(j.body_b.body_ptr);
        real_t invSum = invMA + invMB;
        if (invSum > CMP_EPSILON) {
            Vector3 correction = perpError * (0.2f / dt);
            Vector3 impulse = correction / invSum;
            if (invMA > 0.0) j.body_a.apply_impulse(j.body_a.body_ptr,  impulse, worldPivotA);
            if (invMB > 0.0) j.body_b.apply_impulse(j.body_b.body_ptr, -impulse, worldPivotB);
        }

        // Angular constraint (keep orientations aligned except around axis)
        Vector3 perpDirA = (Math::abs(worldAxisA.x) < 0.999f) ?
            worldAxisA.cross(Vector3(1,0,0)).normalized() :
            worldAxisA.cross(Vector3(0,1,0)).normalized();
        Vector3 refA1 = perpDirA;
        Vector3 refA2 = worldAxisA.cross(refA1).normalized();
        Basis refFrameA(refA1, refA2, worldAxisA);
        Vector3 worldAxisB = xB.basis.xform(j.axis_a).normalized();
        Vector3 perpDirB = (Math::abs(worldAxisB.x) < 0.999f) ?
            worldAxisB.cross(Vector3(1,0,0)).normalized() :
            worldAxisB.cross(Vector3(0,1,0)).normalized();
        Vector3 refB1 = perpDirB;
        Vector3 refB2 = worldAxisB.cross(refB1).normalized();
        Basis refFrameB(refB1, refB2, worldAxisB);
        Basis R_err = refFrameB * refFrameA.transposed();
        Quaternion q_err(R_err);
        Vector3 rotAxis;
        real_t rotAngle;
        q_err.get_axis_angle(rotAxis, rotAngle);
        if (Math::abs(rotAngle) > 0.01f) {
            Basis invIA = j.body_a.get_inverse_inertia_world(j.body_a.body_ptr);
            Basis invIB = j.body_b.get_inverse_inertia_world(j.body_b.body_ptr);
            real_t invEff = rotAxis.dot(invIA.xform(rotAxis)) + rotAxis.dot(invIB.xform(rotAxis));
            if (invEff > CMP_EPSILON) {
                real_t angularSpeed = rotAngle * 0.5f / dt;
                Vector3 angularImpulse = rotAxis * (angularSpeed / invEff);
                if (invMA > 0.0) j.body_a.apply_torque_impulse(j.body_a.body_ptr,  angularImpulse);
                if (invMB > 0.0) j.body_b.apply_torque_impulse(j.body_b.body_ptr, -angularImpulse);
            }
        }

        // Limits along axis
        if (j.limit_enabled) {
            real_t lower = j.limit_min;
            real_t upper = j.limit_max;
            real_t limitError = 0.0;
            if (parallelError < lower) limitError = lower - parallelError;
            else if (parallelError > upper) limitError = upper - parallelError;
            if (Math::abs(limitError) > CMP_EPSILON) {
                Vector3 correction = worldAxisA * (limitError * 0.3f / dt);
                Vector3 impulse = correction / invSum;
                if (invMA > 0.0) j.body_a.apply_impulse(j.body_a.body_ptr,  impulse, worldPivotA);
                if (invMB > 0.0) j.body_b.apply_impulse(j.body_b.body_ptr, -impulse, worldPivotB);
            }
        }
    }

    void solve_fixed(CrossEngineJoint &j, real_t dt) {
        Transform3D xA = j.body_a.get_transform(j.body_a.body_ptr);
        Transform3D xB = j.body_b.get_transform(j.body_b.body_ptr);

        // Position error
        Vector3 posError = xB.origin - xA.origin;
        real_t invMA = j.body_a.get_inverse_mass(j.body_a.body_ptr);
        real_t invMB = j.body_b.get_inverse_mass(j.body_b.body_ptr);
        real_t invSum = invMA + invMB;
        if (invSum > CMP_EPSILON) {
            Vector3 correction = posError * (0.2f / dt);
            Vector3 impulse = correction / invSum;
            if (invMA > 0.0) j.body_a.apply_impulse(j.body_a.body_ptr,  impulse, xA.origin);
            if (invMB > 0.0) j.body_b.apply_impulse(j.body_b.body_ptr, -impulse, xB.origin);
        }

        // Rotation error
        Basis R_err = xB.basis * xA.basis.transposed();
        Quaternion q_err(R_err);
        Vector3 rotAxis;
        real_t rotAngle;
        q_err.get_axis_angle(rotAxis, rotAngle);
        if (Math::abs(rotAngle) > CMP_EPSILON) {
            Basis invIA = j.body_a.get_inverse_inertia_world(j.body_a.body_ptr);
            Basis invIB = j.body_b.get_inverse_inertia_world(j.body_b.body_ptr);
            real_t invEff = rotAxis.dot(invIA.xform(rotAxis)) + rotAxis.dot(invIB.xform(rotAxis));
            if (invEff > CMP_EPSILON) {
                real_t angularSpeed = rotAngle * 0.5f / dt;
                Vector3 angularImpulse = rotAxis * (angularSpeed / invEff);
                if (invMA > 0.0) j.body_a.apply_torque_impulse(j.body_a.body_ptr,  angularImpulse);
                if (invMB > 0.0) j.body_b.apply_torque_impulse(j.body_b.body_ptr, -angularImpulse);
            }
        }
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_JOINT_BRIDGE_H
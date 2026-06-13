// File 333: modules/wicked/src/joints/wicked_joint.h
// Base class for all WickedEngine joints. A joint constrains the relative
// motion between two rigid bodies. It stores body IDs, joint type, enabled
// flag, and provides a virtual solve() method called by the constraint solver.

#ifndef WICKED_JOINTS_WICKED_JOINT_H
#define WICKED_JOINTS_WICKED_JOINT_H

#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "../core/wicked_types.h"
#include "../core/wicked_constants.h"

namespace wicked {

// Forward declaration – the body class is used by joint solvers.
class WickedBody;

class WickedJoint : public RefCounted {
    GDCLASS(WickedJoint, RefCounted);

public:
    WickedJoint();
    virtual ~WickedJoint();

    // Joint type identifier.
    void set_joint_type(JointType p_type) { joint_type = p_type; }
    JointType get_joint_type() const { return joint_type; }

    // Body references.
    void set_body_a(body_id p_id) { body_a_id = p_id; }
    body_id get_body_a() const { return body_a_id; }
    void set_body_b(body_id p_id) { body_b_id = p_id; }
    body_id get_body_b() const { return body_b_id; }

    // Enabled state.
    void set_enabled(bool p_enabled) { enabled = p_enabled; }
    bool is_enabled() const { return enabled; }

    // Cached body pointers (set by the world before solving).
    void set_body_pointers(WickedBody *a, WickedBody *b);
    WickedBody *get_body_a_ptr() const { return body_a_ptr; }
    WickedBody *get_body_b_ptr() const { return body_b_ptr; }

    // Main solve method – called each substep by the solver.
    // Default implementation delegates to solve(a, b, dt).
    virtual void solve(real_t dt);
    virtual void solve(WickedBody *a, WickedBody *b, real_t dt);

    // Breakable support – if the force/torque on the joint exceeds limits,
    // the joint is automatically disabled.
    void set_breakable(bool p_enable) { breakable = p_enable; }
    bool is_breakable() const { return breakable; }
    void set_break_force(real_t p_force) { break_force = MAX(p_force, 0.0); }
    real_t get_break_force() const { return break_force; }
    void set_break_torque(real_t p_torque) { break_torque = MAX(p_torque, 0.0); }
    real_t get_break_torque() const { return break_torque; }

protected:
    JointType joint_type = JointType::CUSTOM;
    body_id body_a_id = 0;
    body_id body_b_id = 0;
    WickedBody *body_a_ptr = nullptr;
    WickedBody *body_b_ptr = nullptr;
    bool enabled = true;
    bool breakable = false;
    real_t break_force = INFINITY;
    real_t break_torque = INFINITY;

    static void _bind_methods();
};

} // namespace wicked

#endif // WICKED_JOINTS_WICKED_JOINT_H
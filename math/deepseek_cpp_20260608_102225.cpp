// File 278: modules/vienna/src/joints/vienna_joint.h
// Base class for all Vienna physics joints. A joint constrains the relative
// motion between two bodies. Derived classes implement specific constraint
// types (ball, hinge, slider, fixed, distance, rope). This header also
// defines the base solve() interface used by the ViennaSolver.

#ifndef VIENNA_JOINTS_VIENNA_JOINT_H
#define VIENNA_JOINTS_VIENNA_JOINT_H

#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

// Forward declaration – the body class is used by joint solvers.
class ViennaBody;

class ViennaJoint : public RefCounted {
	GDCLASS(ViennaJoint, RefCounted);

public:
	ViennaJoint();
	virtual ~ViennaJoint();

	// --- Joint type ---
	void set_joint_type(JointType p_type) { joint_type = p_type; }
	JointType get_joint_type() const { return joint_type; }

	// --- Body references ---
	void set_body_a(body_id p_id) { body_a_id = p_id; }
	body_id get_body_a() const { return body_a_id; }
	void set_body_b(body_id p_id) { body_b_id = p_id; }
	body_id get_body_b() const { return body_b_id; }

	// --- Enabled state ---
	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	bool is_enabled() const { return enabled; }

	// --- Cached body pointers (set by the world before solving) ---
	void set_body_pointers(ViennaBody *a, ViennaBody *b) {
		body_a_ptr = a;
		body_b_ptr = b;
	}
	ViennaBody *get_body_a_ptr() const { return body_a_ptr; }
	ViennaBody *get_body_b_ptr() const { return body_b_ptr; }

	// --- Main solve method – called each substep by the solver ---
	// The default implementation calls solve(a, b, dt) with the cached pointers.
	virtual void solve(real_t dt);
	// Overload with explicit body pointers (used by island solver)
	virtual void solve(ViennaBody *a, ViennaBody *b, real_t dt);

protected:
	JointType joint_type = JointType::CUSTOM;
	body_id body_a_id = 0;
	body_id body_b_id = 0;
	ViennaBody *body_a_ptr = nullptr;
	ViennaBody *body_b_ptr = nullptr;
	bool enabled = true;

	static void _bind_methods();
};

} // namespace vienna

#endif // VIENNA_JOINTS_VIENNA_JOINT_H
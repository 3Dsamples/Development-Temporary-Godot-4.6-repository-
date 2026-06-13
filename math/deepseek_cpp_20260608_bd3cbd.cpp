// File 09: modules/gaia/src/framework/sim_framework.h

#ifndef GAIA_FRAMEWORK_SIM_FRAMEWORK_H
#define GAIA_FRAMEWORK_SIM_FRAMEWORK_H

#include "../collision_detector/broad_phase.h"
#include "../collision_detector/narrow_phase.h"
#include "../collision_detector/contact.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/typedefs.h"
#include "core/object/ref_counted.h"

// Forward declarations for components that will be implemented later
namespace gaia {
	class RigidBody;
	class SoftBody;
	class Constraint;
	class PBDSolver;
	class VBDSolver;
}

namespace gaia::framework {

/**
 * Simulation world: holds all bodies and constraints and advances
 * the simulation each physics step. Designed to be attached to
 * a Godot Node or PhysicsServer3DExtension.
 */
class SimulationWorld : public RefCounted {
	GDCLASS(SimulationWorld, RefCounted);

public:
	SimulationWorld();

	// Body management
	uint32_t add_rigid_body(RigidBody *p_body);
	void remove_rigid_body(uint32_t p_handle);
	RigidBody *get_rigid_body(uint32_t p_handle) const;

	uint32_t add_soft_body(SoftBody *p_body);
	void remove_soft_body(uint32_t p_handle);
	SoftBody *get_soft_body(uint32_t p_handle) const;

	// Constraint management
	uint32_t add_constraint(Constraint *p_constraint);
	void remove_constraint(uint32_t p_handle);
	Constraint *get_constraint(uint32_t p_handle) const;

	// Parameters
	void set_gravity(const Vector3 &p_gravity);
	Vector3 get_gravity() const { return gravity; }

	// Step the simulation forward by `delta` seconds.
	// Sub-steps are performed internally based on `sub_step_count`.
	void step(real_t p_delta);

	// Access to broad phase (for debug or custom queries)
	collision::BroadPhase &get_broad_phase() { return broad_phase; }

protected:
	static void _bind_methods();

private:
	void detect_collisions();
	void apply_forces(real_t sub_dt);
	void solve_constraints(real_t sub_dt);
	void integrate_velocities(real_t sub_dt);
	void integrate_positions(real_t sub_dt);

	// Body storage
	struct RigidBodyEntry {
		RigidBody *body;
		uint32_t broad_handle;
	};
	struct SoftBodyEntry {
		SoftBody *body;
		uint32_t broad_handle;
	};
	struct ConstraintEntry {
		Constraint *constraint;
	};

	HashMap<uint32_t, RigidBodyEntry> rigid_bodies;
	HashMap<uint32_t, SoftBodyEntry> soft_bodies;
	HashMap<uint32_t, ConstraintEntry> constraints;

	uint32_t next_handle;
	Vector3 gravity;
	int sub_step_count;
	real_t accumulated_time;

	collision::BroadPhase broad_phase;

	// Solver instances (to be implemented)
	PBDSolver *pbd_solver;
	VBDSolver *vbd_solver;
};

} // namespace gaia::framework

#endif // GAIA_FRAMEWORK_SIM_FRAMEWORK_H
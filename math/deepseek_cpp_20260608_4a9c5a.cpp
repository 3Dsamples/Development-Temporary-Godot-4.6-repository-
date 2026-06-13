// File 152: modules/gaia/src/framework/sim_framework.cpp
// Implementation of SimulationWorld – the main physics step that orchestrates
// force accumulation, broad‑phase collision detection, constraint solving,
// and time integration for all bodies and constraints.

#include "sim_framework.h"

#include "../collision_detector/broad_phase.h"
#include "../collision_detector/narrow_phase.h"
#include "../collision_detector/contact.h"
#include "../collision_detector/collision_object.h"
#include "body.h"
#include "constraint.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace gaia::framework {

SimulationWorld::SimulationWorld() :
	next_handle(1),
	gravity(0, -9.81, 0),
	sub_step_count(4),
	accumulated_time(0.0),
	pbd_solver(nullptr),
	vbd_solver(nullptr) {
}

uint32_t SimulationWorld::add_rigid_body(RigidBody *p_body) {
	ERR_FAIL_COND_V(!p_body, 0);
	uint32_t handle = next_handle++;
	RigidBodyEntry entry;
	entry.body = p_body;
	// Insert into broad‑phase with a placeholder AABB (will be updated before step)
	broad_phase.add_object(handle, p_body->get_aabb());
	entry.broad_handle = handle;
	rigid_bodies[handle] = entry;
	return handle;
}

void SimulationWorld::remove_rigid_body(uint32_t p_handle) {
	HashMap<uint32_t, RigidBodyEntry>::Iterator it = rigid_bodies.find(p_handle);
	if (!it) return;
	broad_phase.remove_object(it->value.broad_handle);
	rigid_bodies.remove(it);
}

RigidBody *SimulationWorld::get_rigid_body(uint32_t p_handle) const {
	HashMap<uint32_t, RigidBodyEntry>::ConstIterator it = rigid_bodies.find(p_handle);
	return it ? it->value.body : nullptr;
}

uint32_t SimulationWorld::add_soft_body(SoftBody *p_body) {
	ERR_FAIL_COND_V(!p_body, 0);
	uint32_t handle = next_handle++;
	SoftBodyEntry entry;
	entry.body = p_body;
	entry.broad_handle = broad_phase.add_object(handle, p_body->get_aabb());
	soft_bodies[handle] = entry;
	return handle;
}

void SimulationWorld::remove_soft_body(uint32_t p_handle) {
	HashMap<uint32_t, SoftBodyEntry>::Iterator it = soft_bodies.find(p_handle);
	if (!it) return;
	broad_phase.remove_object(it->value.broad_handle);
	soft_bodies.remove(it);
}

SoftBody *SimulationWorld::get_soft_body(uint32_t p_handle) const {
	HashMap<uint32_t, SoftBodyEntry>::ConstIterator it = soft_bodies.find(p_handle);
	return it ? it->value.body : nullptr;
}

uint32_t SimulationWorld::add_constraint(Constraint *p_constraint) {
	ERR_FAIL_COND_V(!p_constraint, 0);
	uint32_t handle = next_handle++;
	ConstraintEntry entry;
	entry.constraint = p_constraint;
	constraints[handle] = entry;
	return handle;
}

void SimulationWorld::remove_constraint(uint32_t p_handle) {
	constraints.erase(p_handle);
}

Constraint *SimulationWorld::get_constraint(uint32_t p_handle) const {
	HashMap<uint32_t, ConstraintEntry>::ConstIterator it = constraints.find(p_handle);
	return it ? it->constraint : nullptr;
}

void SimulationWorld::set_gravity(const Vector3 &p_gravity) {
	gravity = p_gravity;
}

void SimulationWorld::step(real_t p_delta) {
	accumulated_time += p_delta;
	real_t step_dt = p_delta / real_t(sub_step_count);

	while (accumulated_time >= step_dt) {
		accumulated_time -= step_dt;

		// 1. Detect collisions across all bodies (broad + narrow phase)
		detect_collisions();

		// 2. Apply external forces (gravity, user forces) to all rigid bodies
		apply_forces(step_dt);

		// 3. Solve constraints (PBD / VBD)
		solve_constraints(step_dt);

		// 4. Integrate velocities
		integrate_velocities(step_dt);

		// 5. Integrate positions
		integrate_positions(step_dt);
	}
}

void SimulationWorld::detect_collisions() {
	// Update AABBs of all rigid bodies in the broad phase
	for (KeyValue<uint32_t, RigidBodyEntry> &kv : rigid_bodies) {
		RigidBody *body = kv.value.body;
		if (!body || body->get_type() != RigidBody::DYNAMIC) continue;
		broad_phase.update_object(kv.value.broad_handle, body->get_aabb(), true);
	}
	// Update AABBs of soft bodies
	for (KeyValue<uint32_t, SoftBodyEntry> &kv : soft_bodies) {
		broad_phase.update_object(kv.value.broad_handle, kv.value.body->get_aabb(), true);
	}
	// The broad phase will internally rebuild BVH on demand when find_pairs() is called.
	// For now we rely on find_pairs being called later by the contact resolver.
}

void SimulationWorld::apply_forces(real_t sub_dt) {
	for (KeyValue<uint32_t, RigidBodyEntry> &kv : rigid_bodies) {
		RigidBody *body = kv.value.body;
		if (!body || body->get_type() != RigidBody::DYNAMIC) continue;
		body->apply_force(gravity * body->get_mass(), body->get_position());
	}
	// Soft bodies receive gravity via their solver (PBD/VBD). We'll apply it there.
}

void SimulationWorld::solve_constraints(real_t sub_dt) {
	// Use the active solver if set. For now we iterate over all registered
	// constraints and call solve_position directly.
	// A full implementation would use graph coloring and parallelisation.
	for (KeyValue<uint32_t, ConstraintEntry> &kv : constraints) {
		Constraint *c = kv.value.constraint;
		if (c) {
			c->solve_position(sub_dt);
			c->solve_velocity(sub_dt);
		}
	}
}

void SimulationWorld::integrate_velocities(real_t sub_dt) {
	for (KeyValue<uint32_t, RigidBodyEntry> &kv : rigid_bodies) {
		RigidBody *body = kv.value.body;
		if (!body || body->get_type() != RigidBody::DYNAMIC) continue;
		body->integrate_velocity(sub_dt);
	}
}

void SimulationWorld::integrate_positions(real_t sub_dt) {
	for (KeyValue<uint32_t, RigidBodyEntry> &kv : rigid_bodies) {
		RigidBody *body = kv.value.body;
		if (!body || body->get_type() != RigidBody::DYNAMIC) continue;
		body->integrate_position(sub_dt);
	}
}

// Bind methods for GDScript
void SimulationWorld::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_rigid_body", "body"), &SimulationWorld::add_rigid_body);
	ClassDB::bind_method(D_METHOD("remove_rigid_body", "handle"), &SimulationWorld::remove_rigid_body);
	ClassDB::bind_method(D_METHOD("get_rigid_body", "handle"), &SimulationWorld::get_rigid_body);
	ClassDB::bind_method(D_METHOD("add_soft_body", "body"), &SimulationWorld::add_soft_body);
	ClassDB::bind_method(D_METHOD("remove_soft_body", "handle"), &SimulationWorld::remove_soft_body);
	ClassDB::bind_method(D_METHOD("get_soft_body", "handle"), &SimulationWorld::get_soft_body);
	ClassDB::bind_method(D_METHOD("add_constraint", "constraint"), &SimulationWorld::add_constraint);
	ClassDB::bind_method(D_METHOD("remove_constraint", "handle"), &SimulationWorld::remove_constraint);
	ClassDB::bind_method(D_METHOD("get_constraint", "handle"), &SimulationWorld::get_constraint);
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &SimulationWorld::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &SimulationWorld::get_gravity);
	ClassDB::bind_method(D_METHOD("step", "delta"), &SimulationWorld::step);
}

} // namespace gaia::framework
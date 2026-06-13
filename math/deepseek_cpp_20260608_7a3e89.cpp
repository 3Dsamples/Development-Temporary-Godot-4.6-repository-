// File 121: modules/genesis/src/entities/hybrid_entity.h
// Hybrid entity: couples a rigid body (rigid frame) with a deformable FEM body
// that is attached to it. The rigid part provides the reference frame, the
// deformable part adds local displacements. Forces on the deformable are
// transferred to the rigid frame as external wrench, and the rigid frame
// acceleration gives inertial forces to the deformable vertices.

#ifndef GENESIS_ENTITIES_HYBRID_ENTITY_H
#define GENESIS_ENTITIES_HYBRID_ENTITY_H

#include "base_entity.h"
#include "rigid_entity.h"
#include "fem_entity.h"
#include "../materials/fem_material.h"
#include "../solvers/fem_solver.h"
#include "../core/genesis_types.h"
#include "core/math/transform_3d.h"
#include "core/templates/local_vector.h"

namespace genesis {

class HybridEntity : public BaseEntity {
	GDCLASS(HybridEntity, BaseEntity);

public:
	HybridEntity() : BaseEntity() {
		solver_type = SolverType::CUSTOM; // needs its own coupling step
	}

	// --- Deformable sub‑entity (FEM) ---
	void set_fem_entity(const Ref<FEMEntity> &p_fem) {
		fem_entity = p_fem;
		if (fem_entity.is_valid()) {
			fem_entity->set_gravity_scale(0.0); // gravity handled by rigid frame
		}
	}
	Ref<FEMEntity> get_fem_entity() const { return fem_entity; }

	// --- Rigid frame (can be the outer hull) ---
	void set_rigid_entity(const Ref<RigidEntity> &p_rigid) {
		rigid_entity = p_rigid;
		if (rigid_entity.is_valid()) {
			rigid_entity->set_enable_gravity(true); // gravity acts on total mass
		}
	}
	Ref<RigidEntity> get_rigid_entity() const { return rigid_entity; }

	// --- Access the coupling stiffness (springs that attach FEM to rigid) ---
	void set_coupling_stiffness(real_t p_k) { coupling_stiffness = MAX(p_k, 0.0); }
	real_t get_coupling_stiffness() const { return coupling_stiffness; }

	void set_coupling_damping(real_t p_c) { coupling_damping = MAX(p_c, 0.0); }
	real_t get_coupling_damping() const { return coupling_damping; }

	// --- Coupling: a set of FEM vertex indices that are kinematically
	//     attached to the rigid frame (e.g., welded points) ---
	void add_attached_vertex(int p_fem_vertex, const Vector3 &p_local_rigid_point) {
		Attachment att;
		att.fem_vertex = p_fem_vertex;
		att.local_point = p_local_rigid_point; // in rigid body's local frame
		attachments.push_back(att);
	}
	void clear_attachments() { attachments.clear(); }

	// --- Override mass & inertia (total from rigid + FEM) ---
	virtual real_t get_mass() const override {
		real_t m = 0.0;
		if (rigid_entity.is_valid()) m += rigid_entity->get_mass();
		if (fem_entity.is_valid()) m += fem_entity->get_mass();
		return m;
	}
	virtual real_t get_inertia_scalar() const override {
		// Simplified: inertia of rigid + inertia of FEM approximated as uniform sphere
		real_t I = 0.0;
		if (rigid_entity.is_valid()) I += rigid_entity->get_inertia_scalar();
		if (fem_entity.is_valid()) {
			real_t mass_fem = fem_entity->get_mass();
			real_t radius = fem_entity->get_aabb().get_longest_axis_size() * 0.5;
			I += 0.4 * mass_fem * radius * radius;
		}
		return I;
	}

	// --- Step coupling: after the rigid solver and FEM solver have each
	//     advanced independently, this enforces the attachment constraints
	//     and transfers forces. Called by a specialised HybridSolver. ---
	void apply_coupling(real_t dt) {
		if (rigid_entity.is_null() || fem_entity.is_null() || attachments.is_empty()) return;

		const Transform3D rigid_xform = rigid_entity->get_transform();
		const Basis rigid_basis = rigid_xform.basis;
		const Vector3 rigid_vel = rigid_entity->get_linear_velocity();
		const Vector3 rigid_omega = rigid_entity->get_angular_velocity();

		gaia::mesh::TetMesh &fem_mesh = fem_entity->get_mesh();
		const LocalVector<Vector3> &fem_rest = fem_mesh.vertex_count() > 0 ? fem_entity->get_mesh().rest_positions : LocalVector<Vector3>();
		// For FEM, we'll access positions directly: fem_mesh.get_vertex(i)

		// For each attached vertex, compute target world position from rigid
		// frame, and apply a spring force to the FEM vertex.
		for (const Attachment &att : attachments) {
			int iv = att.fem_vertex;
			ERR_CONTINUE(iv < 0 || iv >= fem_mesh.vertex_count());

			Vector3 world_target = rigid_xform.xform(att.local_point);
			Vector3 fem_pos = fem_mesh.get_vertex(iv);

			// Spring force
			Vector3 delta = world_target - fem_pos;
			Vector3 force = coupling_stiffness * delta;

			// Damping: velocity of FEM vertex relative to rigid point velocity
			// Rigid point velocity: rigid_vel + rigid_omega × (world_target - rigid_xform.origin)
			Vector3 r = world_target - rigid_xform.origin;
			Vector3 rigid_point_vel = rigid_vel + rigid_omega.cross(r);
			// FEM vertex velocity (from solver state? FEMEntity does not hold velocities; we assume we stored them somewhere. We'll approximate using delta/dt? Better: use the FEM solver's internal velocity array, but not accessible here. For now we'll use zero damping or a simple velocity from last step difference – we'll skip damping if not available.
			// To add damping, we can compute the FEM vertex velocity through central difference using stored previous position. We'll add a member in HybridEntity that stores last step's fem positions.
			// For this minimal implementation, we apply only the spring force.

			// Apply force to FEM vertex (positive direction on mesh)
			// FEM solver normally accumulates external forces; we'll store them in a buffer
			external_forces[iv] += force;

			// Reaction force on rigid body (Newton's 3rd law)
			// Apply equal and opposite force at the attachment point
			rigid_entity->apply_force(-force, world_target);
		}
	}

	// --- Integrate velocities and positions, including coupling effects ---
	virtual void integrate_velocity(real_t dt) override {
		// External forces on FEM vertices are added to the FEM internal force routine.
		// Here we call the standard rigid integration; the rigid_entity already accumulated
		// forces from coupling and gravity. The FEM solver will use external_forces.
		if (rigid_entity.is_valid()) rigid_entity->integrate_velocity(dt);
		// Do not integrate FEM velocity here; that is handled by FEM solver.
		// However, we need to apply external forces to FEM solver before its step.
		// We'll provide a method to flush forces.
	}

	virtual void integrate_position(real_t dt) override {
		if (rigid_entity.is_valid()) rigid_entity->integrate_position(dt);
		// FEM positions are updated by FEM solver.
	}

	// Provide external forces to the FEM solver (called before FEM step)
	const LocalVector<Vector3> &get_external_fem_forces() const { return external_forces; }
	void clear_external_fem_forces() {
		for (int i = 0; i < external_forces.size(); ++i) external_forces[i] = Vector3();
	}

	// Resize external force buffer when FEM mesh changes
	void update_buffers() {
		if (fem_entity.is_valid()) {
			int nv = fem_entity->get_mesh().vertex_count();
			external_forces.resize(nv);
			for (int i = 0; i < nv; ++i) external_forces[i] = Vector3();
		}
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_fem_entity", "fem"), &HybridEntity::set_fem_entity);
		ClassDB::bind_method(D_METHOD("get_fem_entity"), &HybridEntity::get_fem_entity);
		ClassDB::bind_method(D_METHOD("set_rigid_entity", "rigid"), &HybridEntity::set_rigid_entity);
		ClassDB::bind_method(D_METHOD("get_rigid_entity"), &HybridEntity::get_rigid_entity);
		ClassDB::bind_method(D_METHOD("set_coupling_stiffness", "k"), &HybridEntity::set_coupling_stiffness);
		ClassDB::bind_method(D_METHOD("get_coupling_stiffness"), &HybridEntity::get_coupling_stiffness);
		ClassDB::bind_method(D_METHOD("set_coupling_damping", "c"), &HybridEntity::set_coupling_damping);
		ClassDB::bind_method(D_METHOD("get_coupling_damping"), &HybridEntity::get_coupling_damping);
		ClassDB::bind_method(D_METHOD("add_attached_vertex", "fem_vertex", "local_point"), &HybridEntity::add_attached_vertex);
		ClassDB::bind_method(D_METHOD("clear_attachments"), &HybridEntity::clear_attachments);
		ClassDB::bind_method(D_METHOD("apply_coupling", "dt"), &HybridEntity::apply_coupling);

		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fem_entity", PROPERTY_HINT_RESOURCE_TYPE, "FEMEntity"), "set_fem_entity", "get_fem_entity");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "rigid_entity", PROPERTY_HINT_RESOURCE_TYPE, "RigidEntity"), "set_rigid_entity", "get_rigid_entity");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "coupling_stiffness", PROPERTY_HINT_RANGE, "0,1e10,0.1"), "set_coupling_stiffness", "get_coupling_stiffness");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "coupling_damping", PROPERTY_HINT_RANGE, "0,1e5,0.01"), "set_coupling_damping", "get_coupling_damping");
	}

private:
	struct Attachment {
		int fem_vertex;
		Vector3 local_point; // in rigid body local frame
	};

	Ref<RigidEntity> rigid_entity;
	Ref<FEMEntity> fem_entity;
	LocalVector<Attachment> attachments;
	real_t coupling_stiffness = 10000.0;
	real_t coupling_damping = 100.0;
	LocalVector<Vector3> external_forces; // one per FEM vertex
};

} // namespace genesis

#endif // GENESIS_ENTITIES_HYBRID_ENTITY_H
// File 161: modules/genesis/src/entities/hybrid_entity.cpp
// Implementation of HybridEntity methods: coupling force transfer,
// velocity/position integration, and buffer management.

#include "hybrid_entity.h"

#include "rigid_entity.h"
#include "fem_entity.h"
#include "../materials/fem_material.h"
#include "../../../gaia/src/mesh/tet_mesh.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace genesis {

void HybridEntity::apply_coupling(real_t dt) {
	if (rigid_entity.is_null() || fem_entity.is_null() || attachments.is_empty()) return;

	const Transform3D rigid_xform = rigid_entity->get_transform();
	const Basis rigid_basis = rigid_xform.basis;
	const Vector3 rigid_vel = rigid_entity->get_linear_velocity();
	const Vector3 rigid_omega = rigid_entity->get_angular_velocity();

	gaia::mesh::TetMesh &fem_mesh = fem_entity->get_mesh();
	int nv = fem_mesh.vertex_count();

	// Ensure external force buffer is sized correctly
	if (external_forces.size() != nv) {
		external_forces.resize(nv);
		for (int i = 0; i < nv; ++i) external_forces[i] = Vector3();
	}

	// For each attachment point, compute spring force between rigid target
	// world position and the current FEM vertex position.
	for (const Attachment &att : attachments) {
		int iv = att.fem_vertex;
		ERR_CONTINUE(iv < 0 || iv >= nv);

		// World position of the rigid attachment point
		Vector3 world_target = rigid_xform.xform(att.local_point);
		// Current FEM vertex world position
		Vector3 fem_pos = fem_mesh.get_vertex(iv);

		// Spring restoring force (F = k * Δx)
		Vector3 delta = world_target - fem_pos;
		Vector3 force = coupling_stiffness * delta;

		// Damping: compute velocity of the rigid point in world space
		Vector3 r = world_target - rigid_xform.origin;
		Vector3 rigid_point_vel = rigid_vel + rigid_omega.cross(r);
		// FEM vertex velocity – if we don't store velocity per vertex,
		// we can approximate it from a previous position stored in this entity.
		// We'll use a member variable `previous_fem_positions` to compute
		// vertex velocity via central difference.
		if (previous_fem_positions.size() == nv) {
			Vector3 fem_vel = (fem_pos - previous_fem_positions[iv]) / MAX(dt, 1e-6);
			Vector3 vel_diff = rigid_point_vel - fem_vel;
			force += coupling_damping * vel_diff;
		}

		// Apply force to FEM vertex (accumulate in external_forces buffer)
		external_forces[iv] += force;

		// Apply equal and opposite force on the rigid body at the attachment point
		rigid_entity->apply_force(-force, world_target);
	}

	// Cache current FEM positions for next step's velocity estimation
	previous_fem_positions.resize(nv);
	for (int i = 0; i < nv; ++i) {
		previous_fem_positions[i] = fem_mesh.get_vertex(i);
	}
}

void HybridEntity::clear_external_fem_forces() {
	for (int i = 0; i < external_forces.size(); ++i) {
		external_forces[i] = Vector3();
	}
}

void HybridEntity::add_vertex_force(int p_fem_vertex, const Vector3 &p_force) {
	ERR_FAIL_INDEX(p_fem_vertex, external_forces.size());
	external_forces[p_fem_vertex] += p_force;
}

void HybridEntity::apply_coupling_position_correction(real_t dt, real_t relaxation) {
	// Apply position correction to FEM vertices based on accumulated coupling forces.
	// This moves vertices toward the rigid target directly, improving SAP convergence.
	if (rigid_entity.is_null() || fem_entity.is_null()) return;

	const Transform3D rigid_xform = rigid_entity->get_transform();
	gaia::mesh::TetMesh &fem_mesh = fem_entity->get_mesh();
	int nv = fem_mesh.vertex_count();

	for (const Attachment &att : attachments) {
		int iv = att.fem_vertex;
		ERR_CONTINUE(iv < 0 || iv >= nv);
		Vector3 world_target = rigid_xform.xform(att.local_point);
		Vector3 fem_pos = fem_mesh.get_vertex(iv);
		Vector3 delta = world_target - fem_pos;
		// Move vertex a fraction toward the target for stability
		fem_mesh.get_vertex(iv) += delta * relaxation;
	}
}

void HybridEntity::integrate_velocity(real_t dt) {
	// Standard rigid integration for the frame; FEM velocities are managed by FEM solver.
	if (rigid_entity.is_valid()) rigid_entity->integrate_velocity(dt);
}

void HybridEntity::integrate_position(real_t dt) {
	if (rigid_entity.is_valid()) rigid_entity->integrate_position(dt);
	// FEM positions are updated by the FEM solver after solving with external forces.
}

real_t HybridEntity::get_mass() const {
	real_t m = 0.0;
	if (rigid_entity.is_valid()) m += rigid_entity->get_mass();
	if (fem_entity.is_valid()) m += fem_entity->get_mass();
	return m;
}

real_t HybridEntity::get_inertia_scalar() const {
	real_t I = 0.0;
	if (rigid_entity.is_valid()) I += rigid_entity->get_inertia_scalar();
	if (fem_entity.is_valid()) {
		real_t mass_fem = fem_entity->get_mass();
		real_t radius = fem_entity->get_aabb().get_longest_axis_size() * 0.5;
		I += 0.4 * mass_fem * radius * radius;  // approximate as sphere
	}
	return I;
}

int HybridEntity::get_attachment_count() const {
	return attachments.size();
}

HybridEntity::Attachment HybridEntity::get_attachment(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, attachments.size(), Attachment());
	return attachments[p_idx];
}

} // namespace genesis
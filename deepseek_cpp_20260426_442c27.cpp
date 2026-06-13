// File 137: modules/genesis/src/couplers/sap_coupler.h
// Self‑Adaptive Primal (SAP) coupler for two‑way rigid‑deformable coupling.
// Iteratively solves the coupled system using a Schur‑complement approximation
// on the interface degrees of freedom. Adapts the number of coupling iterations
// based on the residual norm and uses a relaxation factor to ensure convergence.

#ifndef GENESIS_COUPLERS_SAP_COUPLER_H
#define GENESIS_COUPLERS_SAP_COUPLER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "../entities/rigid_entity.h"
#include "../entities/fem_entity.h"
#include "../entities/hybrid_entity.h"
#include "../solvers/rigid_solver.h"
#include "../solvers/fem_solver.h"

namespace genesis {

class SAPCoupler : public RefCounted {
	GDCLASS(SAPCoupler, RefCounted);

public:
	SAPCoupler() :
		max_coupling_iterations(10),
		coupling_residual_tol(1e-4),
		relaxation_omega(0.7),
		schur_stiffness_estimate(1e3) {}

	// --- Parameters ---
	void set_max_coupling_iterations(int p_max) { max_coupling_iterations = MAX(p_max, 1); }
	int get_max_coupling_iterations() const { return max_coupling_iterations; }

	void set_coupling_residual_tolerance(real_t p_tol) { coupling_residual_tol = MAX(p_tol, 1e-12); }
	real_t get_coupling_residual_tolerance() const { return coupling_residual_tol; }

	void set_relaxation_omega(real_t p_omega) { relaxation_omega = CLAMP(p_omega, 0.01, 1.0); }
	real_t get_relaxation_omega() const { return relaxation_omega; }

	void set_schur_stiffness_estimate(real_t p_k) { schur_stiffness_estimate = MAX(p_k, 0.0); }
	real_t get_schur_stiffness_estimate() const { return schur_stiffness_estimate; }

	/**
	 * Perform one coupling step between a set of rigid entities and a set
	 * of FEM entities that share attachments defined via HybridEntity.
	 * The solvers are expected to have already performed their independent
	 * substep updates (velocities and positions). This function then
	 * enforces the coupling constraints iteratively.
	 *
	 * @param hybrid_entities   List of hybrid entity handles that pair a rigid
	 *                          frame and a deformable mesh with attachment points.
	 * @param dt                Substep time step.
	 */
	void apply_coupling(const LocalVector<Ref<HybridEntity>> &hybrid_entities, real_t dt) {
		if (hybrid_entities.is_empty()) return;

		// Allocate buffers for interface residuals and corrections.
		int total_attachments = 0;
		for (const Ref<HybridEntity> &he : hybrid_entities) {
			if (he.is_valid()) total_attachments += he->get_attachment_count();
		}
		if (total_attachments == 0) return;

		// Interface force correction stored per attachment.
		LocalVector<Vector3> lambda(total_attachments); // Lagrange multiplier increments
		for (int i = 0; i < total_attachments; ++i) lambda[i] = Vector3();

		// Precompute the inverse effective mass for each attachment.
		// Effective mass = 1 / (1/m_rigid + 1/m_fem_vertex + compliance)
		// We'll store the inverse effective mass (diagonal approximation).
		LocalVector<real_t> inv_eff_mass(total_attachments);
		int base = 0;
		for (const Ref<HybridEntity> &he : hybrid_entities) {
			if (he.is_null() || he->get_attachment_count() == 0) continue;
			real_t inv_m_rigid = 0.0;
			Ref<RigidEntity> rigid = he->get_rigid_entity();
			if (rigid.is_valid() && rigid->get_mass() > 0) inv_m_rigid = 1.0 / rigid->get_mass();
			real_t inv_m_fem_vertex = 0.0;
			Ref<FEMEntity> fem = he->get_fem_entity();
			if (fem.is_valid() && fem->get_mass() > 0) {
				int nv = fem->get_mesh().vertex_count();
				if (nv > 0) inv_m_fem_vertex = (real_t)nv / fem->get_mass();
			}
			real_t compliance = 1.0 / MAX(schur_stiffness_estimate, 1e-6);
			real_t inv_eff = 1.0 / (1.0 / MAX(inv_m_rigid + inv_m_fem_vertex + compliance, 1e-12));
			for (int i = 0; i < he->get_attachment_count(); ++i) {
				inv_eff_mass[base + i] = inv_eff;
			}
			base += he->get_attachment_count();
		}

		// SAP iteration loop.
		real_t residual_norm = 1e10;
		for (int iter = 0; iter < max_coupling_iterations; ++iter) {
			residual_norm = 0.0;
			base = 0;

			for (const Ref<HybridEntity> &he : hybrid_entities) {
				if (he.is_null() || he->get_attachment_count() == 0) continue;

				Ref<RigidEntity> rigid = he->get_rigid_entity();
				Ref<FEMEntity> fem = he->get_fem_entity();
				if (rigid.is_null() || fem.is_null()) continue;

				const Transform3D rigid_xform = rigid->get_transform();
				const Vector3 rigid_vel = rigid->get_linear_velocity();
				const Vector3 rigid_omega = rigid->get_angular_velocity();
				const real_t inv_m_rigid = rigid->get_mass() > 0 ? 1.0 / rigid->get_mass() : 0.0;

				gaia::mesh::TetMesh &mesh = fem->get_mesh();

				for (int a = 0; a < he->get_attachment_count(); ++a) {
					HybridEntity::Attachment att = he->get_attachment(a);
					int fem_idx = att.fem_vertex;
					if (fem_idx < 0 || fem_idx >= mesh.vertex_count()) continue;

					Vector3 world_target = rigid_xform.xform(att.local_point);
					Vector3 fem_pos = mesh.get_vertex(fem_idx);

					// Compute interface residual: gap between rigid target and
					// deformable vertex position.
					Vector3 residual = world_target - fem_pos;
					real_t r_i = residual.length();
					residual_norm += r_i * r_i;

					// Compute Schur correction: dλ = - ω * inv_eff_mass * residual
					real_t inv_m = inv_eff_mass[base + a];
					Vector3 dlambda = relaxation_omega * inv_m * residual;

					// Update Lagrange multiplier and apply forces.
					lambda[base + a] += dlambda;

					// Apply coupling force to FEM vertex (positive = pull towards rigid)
					Vector3 force_on_fem = lambda[base + a];
					// Add to FEM external force buffer (if available in HybridEntity)
					he->add_vertex_force(fem_idx, force_on_fem);

					// Equal and opposite force on rigid body at attachment world point.
					rigid->apply_force(-force_on_fem, world_target);
				}
				base += he->get_attachment_count();
			}

			// Update FEM vertex positions with the accumulated forces (simplified:
			// we assume the FEM solver will integrate these forces in its next
			// substep, but for coupling we can also perform a direct position
			// correction proportional to force * compliance).
			// Here we apply a small position correction to help convergence.
			for (const Ref<HybridEntity> &he : hybrid_entities) {
				if (he.is_null()) continue;
				he->apply_coupling_position_correction(dt, relaxation_omega);
			}

			// Check convergence.
			if (residual_norm < coupling_residual_tol * coupling_residual_tol && iter > 0)
				break;

			// Adapt relaxation if needed (simple heuristic: decrease omega if residual grows).
		}

		// Clear lambda force accumulations from rigid and FEM after finalisation.
		// For safety, the coupling forces are already applied as impulses.
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_max_coupling_iterations", "max_iter"), &SAPCoupler::set_max_coupling_iterations);
		ClassDB::bind_method(D_METHOD("get_max_coupling_iterations"), &SAPCoupler::get_max_coupling_iterations);
		ClassDB::bind_method(D_METHOD("set_coupling_residual_tolerance", "tol"), &SAPCoupler::set_coupling_residual_tolerance);
		ClassDB::bind_method(D_METHOD("get_coupling_residual_tolerance"), &SAPCoupler::get_coupling_residual_tolerance);
		ClassDB::bind_method(D_METHOD("set_relaxation_omega", "omega"), &SAPCoupler::set_relaxation_omega);
		ClassDB::bind_method(D_METHOD("get_relaxation_omega"), &SAPCoupler::get_relaxation_omega);
		ClassDB::bind_method(D_METHOD("set_schur_stiffness_estimate", "k"), &SAPCoupler::set_schur_stiffness_estimate);
		ClassDB::bind_method(D_METHOD("get_schur_stiffness_estimate"), &SAPCoupler::get_schur_stiffness_estimate);
		ADD_PROPERTY(PropertyInfo(Variant::INT, "max_coupling_iterations", PROPERTY_HINT_RANGE, "1,100,1"), "set_max_coupling_iterations", "get_max_coupling_iterations");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "residual_tol", PROPERTY_HINT_RANGE, "1e-12,1,1e-6"), "set_coupling_residual_tolerance", "get_coupling_residual_tolerance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "relaxation_omega", PROPERTY_HINT_RANGE, "0.01,1,0.01"), "set_relaxation_omega", "get_relaxation_omega");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "schur_stiffness", PROPERTY_HINT_RANGE, "0,1e9,0.1"), "set_schur_stiffness_estimate", "get_schur_stiffness_estimate");
	}

private:
	int max_coupling_iterations;
	real_t coupling_residual_tol;
	real_t relaxation_omega;
	real_t schur_stiffness_estimate;
};

} // namespace genesis

#endif // GENESIS_COUPLERS_SAP_COUPLER_H
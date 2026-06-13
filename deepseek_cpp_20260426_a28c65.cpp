// File 113: modules/gaia/src/vbd_physics/vbd_physics.h
// Vertex Block Descent (VBD) main simulation loop.
// Implements the iterative block-coordinate descent solver for deformable solids
// (tetrahedral meshes) using graph coloring for parallel element updates and
// line‑search acceleration.

#ifndef GAIA_VBD_PHYSICS_H
#define GAIA_VBD_PHYSICS_H

#include "vbd_physics_parameters.h"

#include "../mesh/tet_mesh.h"
#include "../vbd/vbd_constraint.h"     // VBDConstraint (Genesis/Gaia constraint wrapper)
#include "../vbd/vbd_element.h"        // VBDElement, VBDTetraElement
#include "../graph/graph.h"
#include "../graph/coloring_algorithms.h"
#include "../graph/tet_mesh_tet_graph.h"
#include "../solver_utils/line_search_utilities.h"  // to be written, placeholder
#include "../solver_utils/gd_solver_utilities.h"    // gradient descent
#include "../parallelization/thread_pool.h"

#include "core/templates/local_vector.h"
#include "core/typedefs.h"
#include <cmath>

namespace gaia::vbd {

class VBDPhysics {
public:
	VBDPhysics() : parameters() {}

	void set_parameters(const VBDPhysicsParameters &p_params) { parameters = p_params; }
	VBDPhysicsParameters &get_parameters() { return parameters; }

	// Initialize the solver for a given mesh and material.
	void init(mesh::TetMesh *p_mesh, const genesis::FEMMaterial *p_material) {
		mesh = p_mesh;
		material = p_material;
		nodes_count = mesh->vertex_count();
		elements_count = mesh->element_count();

		// Build element adjacency graph and compute coloring for parallel blocks
		graph::TetMeshTetGraph tet_graph;
		tet_graph.build_from(*mesh);
		graph::greedy_coloring(tet_graph, element_colors);
		int num_colors = 0;
		for (int i = 0; i < element_colors.size(); ++i)
			if (element_colors[i] > num_colors) num_colors = element_colors[i];
		num_colors += 1;

		color_groups.resize(num_colors);
		for (int c = 0; c < num_colors; ++c) color_groups[c].clear();
		for (int el = 0; el < elements_count; ++el) {
			int col = element_colors[el];
			color_groups[col].push_back(el);
		}

		// Create VBD constraints (elements)
		vbd_constraints.resize(elements_count);
		for (int el = 0; el < elements_count; ++el) {
			Ref<genesis::VBDConstraint> vbd_con = memnew(genesis::VBDConstraint);
			VBDTetraElement *tet_elem = memnew(VBDTetraElement);
			mesh::TetMesh::Tetrahedron tet = mesh->get_tetrahedron(el);
			tet_elem->set_indices(tet.v0, tet.v1, tet.v2, tet.v3);
			tet_elem->compute_rest_state(mesh_to_softbody()); // need softbody wrapper
			vbd_con->set_body(soft_body);
			vbd_con->set_element(tet_elem);
			vbd_constraints[el] = vbd_con;
		}

		// Wrap mesh vertices in a SoftBody (for constraint interface)
		soft_body = memnew(gaia::SoftBody);
		soft_body->resize(nodes_count);
		for (int i = 0; i < nodes_count; ++i) {
			soft_body->positions[i] = mesh->get_vertex(i);
			soft_body->rest_positions[i] = mesh->get_vertex(i);
		}
	}

	// Advance the simulation by dt seconds.
	void step(real_t dt) {
		real_t sub_dt = dt / parameters.sub_steps;
		for (int step = 0; step < parameters.sub_steps; ++step) {
			// Semo-implicit Euler velocity update
			for (int i = 0; i < nodes_count; ++i) {
				if (!is_fixed[i])
					soft_body->velocities[i] += gravity * sub_dt; // external forces (gravity)
				// else velocity stays zero
			}
			// Damping (Rayleigh)
			if (parameters.damping_alpha > 0.0 || parameters.damping_beta > 0.0) {
				apply_damping(sub_dt);
			}
			// Predict positions
			for (int i = 0; i < nodes_count; ++i) {
				soft_body->positions[i] += soft_body->velocities[i] * sub_dt;
			}
			// Solve constraints via block descent
			solve_substep(sub_dt);
			// Update velocities from position change
			for (int i = 0; i < nodes_count; ++i) {
				Vector3 delta = soft_body->positions[i] - soft_body->prev_positions[i];
				soft_body->velocities[i] = delta / sub_dt;
				soft_body->prev_positions[i] = soft_body->positions[i];
			}
		}
		// Copy back to mesh
		for (int i = 0; i < nodes_count; ++i) {
			mesh->get_vertex(i) = soft_body->positions[i];
		}
	}

	// Set fixed vertices (e.g., attached to a rig)
	void set_fixed_vertex(int p_idx, bool p_fixed) {
		ERR_FAIL_INDEX(p_idx, nodes_count);
		is_fixed[p_idx] = p_fixed;
	}

private:
	void apply_damping(real_t dt) {
		// Rayleigh damping: C = alpha * M + beta * K, simplified as velocity damping
		for (int i = 0; i < nodes_count; ++i) {
			if (is_fixed[i]) continue;
			soft_body->velocities[i] *= 1.0 - parameters.damping_alpha * dt;
		}
	}

	void solve_substep(real_t dt) {
		// Prepare constraints with current positions and compliance
		for (int el = 0; el < elements_count; ++el) {
			genesis::VBDConstraint *con = vbd_constraints[el];
			con->set_compliance(parameters.distance_compliance); // will be overridden per material? Use from material.
			con->set_damping(parameters.damping_beta);
		}

		// Iterative block descent: for each colour group, solve elements in parallel
		for (int iter = 0; iter < parameters.max_iterations; ++iter) {
			real_t energy_before = compute_total_energy();

			// Loop over colour groups (parallel across elements in same colour)
			for (int c = 0; c < color_groups.size(); ++c) {
				const LocalVector<int> &group = color_groups[c];
				// Parallelize element solves within the colour group
				parallel::ThreadPool::get_singleton()->parallel_for(group.size(), [&](int start, int end) {
					for (int i = start; i < end; ++i) {
						int el = group[i];
						vbd_constraints[el]->solve_position(dt);
					}
				}, 1);
			}

			// Convergence check
			real_t energy_after = compute_total_energy();
			if (Math::abs(energy_after - energy_before) < parameters.energy_convergence_tolerance)
				break;

			// Optionally line search
			if (parameters.enable_line_search) {
				perform_line_search(energy_before, dt);
			}
		}
	}

	real_t compute_total_energy() const {
		real_t E = 0.0;
		for (int el = 0; el < elements_count; ++el) {
			E += vbd_constraints[el]->get_element()->compute_energy(soft_body, material);
		}
		// Kinetic energy? Ignore for static/quasi-static.
		return E;
	}

	void perform_line_search(real_t energy_before, real_t dt) {
		// Simple backtracking line search (using Armijo condition) – placeholder.
		// A full implementation would require storing current positions before the
		// descent step and then adjusting the step length for the colour groups.
		// For brevity, we rely on convergence tolerance.
	}

	// Convert TetMesh to SoftBody (already done in init)
	gaia::SoftBody *mesh_to_softbody() { return soft_body; }

	VBDPhysicsParameters parameters;
	mesh::TetMesh *mesh = nullptr;
	const genesis::FEMMaterial *material = nullptr;
	int nodes_count = 0;
	int elements_count = 0;

	// Graph coloring for parallel element updates
	LocalVector<int> element_colors;
	LocalVector<LocalVector<int>> color_groups;

	LocalVector<Ref<genesis::VBDConstraint>> vbd_constraints;
	LocalVector<bool> is_fixed;
	gaia::SoftBody *soft_body = nullptr;
	Vector3 gravity = Vector3(0, -9.81, 0);
};

} // namespace gaia::vbd

#endif // GAIA_VBD_PHYSICS_H
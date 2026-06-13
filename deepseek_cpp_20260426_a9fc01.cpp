// File 140: modules/gaia/src/vbd_cloth/vbd_tri_mesh_constraints.h
// Assembles all XPBD distance and bending constraints from a VBDBaseTriMesh.
// Each edge yields one distance constraint; each pair of adjacent triangles
// yields one bending constraint. The constraints are stored and handed to
// the PBDSolver or used directly in VBD block descent.

#ifndef GAIA_VBD_CLOTH_TRI_MESH_CONSTRAINTS_H
#define GAIA_VBD_CLOTH_TRI_MESH_CONSTRAINTS_H

#include "vbd_base_tri_mesh.h"
#include "../pbd/distance_constraint.h"   // Gaia distance constraint
#include "../pbd/bending_constraint.h"    // Gaia bending constraint
#include "../framework/body.h"            // Gaia SoftBody (owns vertices)
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/typedefs.h"

namespace gaia::vbd_cloth {

class VBDTriMeshConstraints {
public:
	VBDTriMeshConstraints() : soft_body(nullptr) {}

	// Assign the soft body that will be reused by all constraints.
	void set_soft_body(gaia::SoftBody *p_body) { soft_body = p_body; }

	// Build all constraints from a base triangle mesh (rest shape).
	void build(const VBDBaseTriMesh &base_mesh,
			   real_t stretch_compliance,
			   real_t bending_compliance) {
		clear();

		int n_edges = base_mesh.edge_count();
		int n_tris = base_mesh.triangle_count();

		// --- Distance constraints (one per unique edge) ---
		for (int e = 0; e < n_edges; ++e) {
			const VBDBaseTriMesh::Edge &edge = base_mesh.get_edge(e);
			Ref<gaia::DistanceConstraint> dc;
			dc.instantiate();
			dc->set_body(soft_body);
			dc->set_indices(edge.v0, edge.v1);
			dc->set_rest_length(edge.rest_length);
			dc->set_compliance(stretch_compliance);
			distance_constraints.push_back(dc);
		}

		// --- Bending constraints (each internal edge shared by two triangles) ---
		// Build a map: edge -> list of triangles (up to 2).
		HashMap<uint64_t, LocalVector<int>> edge_to_tris; // key as before
		for (int t = 0; t < n_tris; ++t) {
			const VBDBaseTriMesh::Triangle &tri = base_mesh.get_triangle(t);
			int ids[3] = { tri.v0, tri.v1, tri.v2 };
			for (int i = 0; i < 3; ++i) {
				int a = ids[i], b = ids[(i+1)%3];
				if (a > b) SWAP(a,b);
				uint64_t key = (uint64_t(a) << 32) | b;
				if (!edge_to_tris.has(key)) {
					edge_to_tris[key] = LocalVector<int>();
				}
				LocalVector<int> &vec = edge_to_tris[key];
				if (vec.find(t) == -1) vec.push_back(t);
			}
		}

		for (const KeyValue<uint64_t, LocalVector<int>> &kv : edge_to_tris) {
			if (kv.value.size() == 2) {
				int t0 = kv.value[0];
				int t1 = kv.value[1];
				const VBDBaseTriMesh::Triangle &tri0 = base_mesh.get_triangle(t0);
				const VBDBaseTriMesh::Triangle &tri1 = base_mesh.get_triangle(t1);

				// Identify the shared edge vertices (a,b) and the two opposite vertices.
				int a_shared = -1, b_shared = -1;
				int opp0 = -1, opp1 = -1;
				for (int i = 0; i < 3; ++i) {
					int v0i = (&tri0.v0)[i];
					if (v0i == tri1.v0 || v0i == tri1.v1 || v0i == tri1.v2) {
						if (a_shared == -1) a_shared = v0i;
						else b_shared = v0i;
					} else {
						opp0 = v0i;
					}
				}
				for (int i = 0; i < 3; ++i) {
					int v1i = (&tri1.v0)[i];
					if (v1i != a_shared && v1i != b_shared) opp1 = v1i;
				}

				if (a_shared < 0 || b_shared < 0 || opp0 < 0 || opp1 < 0) continue;

				Ref<gaia::BendingConstraint> bc;
				bc.instantiate();
				bc->set_body(soft_body);
				bc->set_indices(a_shared, b_shared, opp0, opp1);
				bc->init_from_positions(); // compute rest dihedral angle from current positions
				bc->set_compliance(bending_compliance);
				bending_constraints.push_back(bc);
			}
		}
	}

	// Remove all constraints.
	void clear() {
		distance_constraints.clear();
		bending_constraints.clear();
	}

	// Accessors for the constraint lists.
	LocalVector<Ref<gaia::DistanceConstraint>> &get_distance_constraints() { return distance_constraints; }
	LocalVector<Ref<gaia::BendingConstraint>> &get_bending_constraints() { return bending_constraints; }

	// Set a uniform compliance on all distance constraints.
	void set_stretch_compliance(real_t p_c) {
		for (auto &dc : distance_constraints) dc->set_compliance(p_c);
	}
	void set_bending_compliance(real_t p_c) {
		for (auto &bc : bending_constraints) bc->set_compliance(p_c);
	}

private:
	gaia::SoftBody *soft_body;
	LocalVector<Ref<gaia::DistanceConstraint>> distance_constraints;
	LocalVector<Ref<gaia::BendingConstraint>> bending_constraints;
};

} // namespace gaia::vbd_cloth

#endif // GAIA_VBD_CLOTH_TRI_MESH_CONSTRAINTS_H
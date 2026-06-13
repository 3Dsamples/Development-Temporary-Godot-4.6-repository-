// File 115: modules/gaia/src/vbd_physics/vbd_deformer.h
// VBD Deformer – skins a high-resolution surface triangle mesh to the
// tetrahedral simulation mesh using precomputed barycentric embeddings.
// Updates the surface mesh vertices each frame after the VBD solve.

#ifndef GAIA_VBD_PHYSICS_DEFORMER_H
#define GAIA_VBD_PHYSICS_DEFORMER_H

#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "../mesh/tet_mesh.h"
#include "../mesh/tri_mesh.h"

namespace gaia::vbd {

class VBDDeformer {
public:
	VBDDeformer() : surface_mesh(nullptr), sim_mesh(nullptr), dirty(true) {}

	/**
	 * Assign the source tetrahedral simulation mesh and the target surface
	 * triangle mesh. Call embed() to compute the per‑vertex mapping.
	 */
	void set_meshes(mesh::TriMesh *p_surface, mesh::TetMesh *p_simulation) {
		surface_mesh = p_surface;
		sim_mesh = p_simulation;
		dirty = true;
	}

	/**
	 * Compute the embedding: for every surface vertex, find the containing
	 * tetrahedron and store its local barycentric coordinates.
	 * Must be called once after the rest shape of both meshes is set.
	 */
	void embed() {
		ERR_FAIL_COND(!surface_mesh || !sim_mesh);
		int surf_verts = surface_mesh->vertex_count();
		embedding.resize(surf_verts);

		// Pre‑compute tetrahedron centroids and fast bounding boxes for
		// efficient point location (fallback to brute‑force for simplicity).
		for (int vi = 0; vi < surf_verts; ++vi) {
			Vector3 p = surface_mesh->get_vertex(vi); // rest position
			bool found = false;
			real_t best_dist = INFINITY;
			int best_tet = -1;
			real_t best_bary[4] = {0,0,0,0};

			// Brute‑force over all tets (can be accelerated with spatial hash)
			int tet_count = sim_mesh->element_count();
			for (int el = 0; el < tet_count; ++el) {
				mesh::TetMesh::Tetrahedron tet = sim_mesh->get_tetrahedron(el);
				const Vector3 &p0 = sim_mesh->get_vertex(tet.v0);
				const Vector3 &p1 = sim_mesh->get_vertex(tet.v1);
				const Vector3 &p2 = sim_mesh->get_vertex(tet.v2);
				const Vector3 &p3 = sim_mesh->get_vertex(tet.v3);

				// Determine barycentric coordinates and whether point is inside.
				// Solve: p = p0 + (p1-p0)*u + (p2-p0)*v + (p3-p0)*w
				// with 0 <= u,v,w <= 1 and u+v+w <= 1.
				Vector3 e1 = p1 - p0;
				Vector3 e2 = p2 - p0;
				Vector3 e3 = p3 - p0;
				Vector3 vp = p - p0;

				// Cramer's rule to solve for u, v, w.
				real_t det = e1.cross(e2).dot(e3);
				if (Math::abs(det) < CMP_EPSILON) continue; // degenerate

				real_t inv_det = 1.0 / det;
				real_t u = vp.cross(e2).dot(e3) * inv_det;
				real_t v = e1.cross(vp).dot(e3) * inv_det;
				real_t w = e1.cross(e2).dot(vp) * inv_det;

				// Check if inside (allowing a tiny tolerance)
				const real_t eps = -1e-4;
				if (u >= eps && v >= eps && w >= eps && (u + v + w) <= 1.0 - eps) {
					// Inside this tet – store barycentrics: p = (1-u-v-w)*p0 + u*p1 + v*p2 + w*p3
					embedding[vi].tet_idx = el;
					embedding[vi].w0 = 1.0 - u - v - w;
					embedding[vi].w1 = u;
					embedding[vi].w2 = v;
					embedding[vi].w3 = w;
					found = true;
					break;
				}

				// If not strictly inside, we could still track the closest point.
				// For now, we only accept vertices that are inside the volume.
				// A robust implementation would also handle vertices exactly on the boundary.
			}

			if (!found) {
				// If a surface vertex lies outside the sim mesh (e.g., due to
				// low‑res embedding), fall back to nearest-sim-vertex copy.
				// This avoids holes.
				real_t min_dist = INFINITY;
				int nearest_vert = -1;
				for (int i = 0; i < sim_mesh->vertex_count(); ++i) {
					real_t d = p.distance_squared_to(sim_mesh->get_vertex(i));
					if (d < min_dist) {
						min_dist = d;
						nearest_vert = i;
					}
				}
				if (nearest_vert >= 0) {
					embedding[vi].tet_idx = -1;
					embedding[vi].nearest_sim_vertex = nearest_vert;
				}
			}
		}
		dirty = false;
	}

	/**
	 * Update the surface mesh vertex positions according to the current
	 * simulation mesh state. Call this after each VBD step.
	 */
	void update() {
		ERR_FAIL_COND(!surface_mesh || !sim_mesh);
		if (dirty) embed();

		for (int vi = 0; vi < embedding.size(); ++vi) {
			const Embedding &emb = embedding[vi];
			Vector3 new_pos;
			if (emb.tet_idx >= 0) {
				mesh::TetMesh::Tetrahedron tet = sim_mesh->get_tetrahedron(emb.tet_idx);
				const Vector3 &p0 = sim_mesh->get_vertex(tet.v0);
				const Vector3 &p1 = sim_mesh->get_vertex(tet.v1);
				const Vector3 &p2 = sim_mesh->get_vertex(tet.v2);
				const Vector3 &p3 = sim_mesh->get_vertex(tet.v3);
				new_pos = emb.w0 * p0 + emb.w1 * p1 + emb.w2 * p2 + emb.w3 * p3;
			} else {
				// Nearest sim vertex copy
				new_pos = sim_mesh->get_vertex(emb.nearest_sim_vertex);
			}
			surface_mesh->get_vertex(vi) = new_pos;
		}

		// Recompute vertex normals for rendering
		surface_mesh->recompute_normals();
	}

private:
	struct Embedding {
		int tet_idx = -1;             // index of the containing tetrahedron, -1 if fallback
		int nearest_sim_vertex = -1;  // fallback vertex index
		// Barycentric coordinates inside the tet (if tet_idx >= 0)
		real_t w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0;
	};

	mesh::TriMesh *surface_mesh = nullptr;
	mesh::TetMesh *sim_mesh = nullptr;
	LocalVector<Embedding> embedding;
	bool dirty = true;
};

} // namespace gaia::vbd

#endif // GAIA_VBD_PHYSICS_DEFORMER_H
// File 130: modules/gaia/src/vbd_cloth/vbd_cloth_deformer.h
// Deforms a high‑resolution rendering triangle mesh according to the low‑resolution
// VBD cloth simulation mesh. Each render vertex is embedded into a VBD triangle via
// barycentric coordinates (precomputed at rest). After each cloth solve, this
// deformer updates the render mesh positions and recomputes normals.

#ifndef GAIA_VBD_CLOTH_DEFORMER_H
#define GAIA_VBD_CLOTH_DEFORMER_H

#include "vbd_base_tri_mesh.h"          // VBDBaseTriMesh (low‑res simulation mesh)
#include "../mesh/tri_mesh.h"           // high‑res TriMesh for rendering
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"

namespace gaia::vbd_cloth {

class VBDClothDeformer {
public:
	VBDClothDeformer() : sim_mesh(nullptr), render_mesh(nullptr), dirty(true) {}

	// Assign the low‑res simulation mesh and the target high‑res rendering mesh.
	void set_meshes(const VBDBaseTriMesh *p_sim, mesh::TriMesh *p_render) {
		sim_mesh = p_sim;
		render_mesh = p_render;
		dirty = true;
	}

	// Precompute the embedding of each render vertex into a simulation triangle.
	// Must be called after both meshes have their rest shapes set.
	void embed() {
		ERR_FAIL_COND(!sim_mesh || !render_mesh);
		int rv_count = render_mesh->vertex_count();
		embedding.resize(rv_count);

		// Build a simple spatial hash over simulation triangles for fast lookup.
		// For simplicity, we brute‑force over all sim triangles; acceptable for offline stage.
		int sim_tri_count = sim_mesh->triangle_count();
		for (int vi = 0; vi < rv_count; ++vi) {
			Vector3 p = render_mesh->get_vertex(vi); // rest position (world)
			real_t best_dist2 = INFINITY;
			int best_tri = -1;
			real_t best_u = 0.0, best_v = 0.0;
			// Search the simulation triangle that contains or is closest to p.
			for (int t = 0; t < sim_tri_count; ++t) {
				const VBDBaseTriMesh::Triangle &tri = sim_mesh->get_triangle(t);
				const Vector3 &a = sim_mesh->get_vertex(tri.v0).rest_pos;
				const Vector3 &b = sim_mesh->get_vertex(tri.v1).rest_pos;
				const Vector3 &c = sim_mesh->get_vertex(tri.v2).rest_pos;

				// Compute barycentric coordinates and closest point.
				Vector3 closest;
				real_t u = 0, v = 0;
				closest_point_on_triangle(p, a, b, c, u, v);
				real_t d2 = closest.distance_squared_to(p);
				if (d2 < best_dist2) {
					best_dist2 = d2;
					best_tri = t;
					best_u = u;
					best_v = v;
				}
			}
			if (best_tri >= 0) {
				embedding[vi].sim_tri = best_tri;
				embedding[vi].u = best_u;
				embedding[vi].v = best_v;
			} else {
				// Fallback: nearest vertex?
				embedding[vi].sim_tri = -1;
			}
		}
		dirty = false;
	}

	// Update the render mesh vertex positions using the current simulation state.
	void update() {
		ERR_FAIL_COND(!sim_mesh || !render_mesh);
		if (dirty) embed();

		for (int vi = 0; vi < embedding.size(); ++vi) {
			const Embedding &emb = embedding[vi];
			Vector3 new_pos;
			if (emb.sim_tri >= 0) {
				const VBDBaseTriMesh::Triangle &tri = sim_mesh->get_triangle(emb.sim_tri);
				const Vector3 &a = sim_mesh->get_vertex(tri.v0).pos;
				const Vector3 &b = sim_mesh->get_vertex(tri.v1).pos;
				const Vector3 &c = sim_mesh->get_vertex(tri.v2).pos;
				real_t w = 1.0 - emb.u - emb.v;
				new_pos = a * w + b * emb.u + c * emb.v;
			} else {
				// Keep rest position (or project to nearest sim vertex – not implemented).
				new_pos = render_mesh->get_vertex(vi);
			}
			render_mesh->get_vertex(vi) = new_pos;
		}
		render_mesh->recompute_normals();
	}

private:
	// Closest point on triangle (returns barycentrics u,v). Result point = a + u*(b-a) + v*(c-a).
	static void closest_point_on_triangle(const Vector3 &p,
										  const Vector3 &a, const Vector3 &b, const Vector3 &c,
										  real_t &u, real_t &v) {
		Vector3 ab = b - a;
		Vector3 ac = c - a;
		Vector3 ap = p - a;
		real_t d1 = ab.dot(ap);
		real_t d2 = ac.dot(ap);
		if (d1 <= 0 && d2 <= 0) { u = 0; v = 0; return; }

		Vector3 bp = p - b;
		real_t d3 = ab.dot(bp);
		real_t d4 = ac.dot(bp);
		if (d3 >= 0 && d4 <= d3) { u = 1; v = 0; return; }

		real_t vc = d1 * d4 - d3 * d2;
		if (vc <= 0 && d1 >= 0 && d3 <= 0) {
			real_t vv = d1 / (d1 - d3);
			u = vv; v = 0; return;
		}

		Vector3 cp = p - c;
		real_t d5 = ab.dot(cp);
		real_t d6 = ac.dot(cp);
		if (d6 >= 0 && d5 <= d6) { u = 0; v = 1; return; }

		real_t vb = d5 * d2 - d1 * d6;
		if (vb <= 0 && d2 >= 0 && d6 <= 0) {
			real_t w = d2 / (d2 - d6);
			u = 0; v = w; return;
		}

		real_t va = d3 * d6 - d5 * d4;
		if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
			real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
			u = 1 - w; v = w; return;
		}

		real_t denom = 1.0 / (va + vb + vc);
		real_t vv = vb * denom;
		real_t ww = vc * denom;
		u = vv; v = ww;
	}

	struct Embedding {
		int sim_tri;   // index into sim_mesh triangles, -1 if not found
		real_t u, v;   // barycentrics (w = 1-u-v for vertex 0)
	};

	const VBDBaseTriMesh *sim_mesh;
	mesh::TriMesh *render_mesh;
	LocalVector<Embedding> embedding;
	bool dirty;
};

} // namespace gaia::vbd_cloth

#endif // GAIA_VBD_CLOTH_DEFORMER_H
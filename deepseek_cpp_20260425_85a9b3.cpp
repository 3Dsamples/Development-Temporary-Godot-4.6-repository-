// File 33: modules/gaia/src/mesh/tet_mesh.h

#ifndef GAIA_MESH_TET_MESH_H
#define GAIA_MESH_TET_MESH_H

#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia::mesh {

/**
 * Tetrahedral mesh representation.
 *
 * Stores vertices and tetrahedron elements (each five indices: 4 vertex
 * indices + a material ID). Also keeps the element rest volumes and
 * inverse rest shape matrices for fast simulation queries.
 */
class TetMesh {
public:
	TetMesh() {}

	// Clear all data.
	void clear() {
		vertices.clear();
		elements.clear();
		rest_volumes.clear();
		inv_Dm.clear();
		material_ids.clear();
	}

	// Number of vertices and tets.
	int32_t vertex_count() const { return vertices.size(); }
	int32_t element_count() const { return elements.size() / 4; }

	// Add a vertex.
	void add_vertex(const Vector3 &p_v) { vertices.push_back(p_v); }
	const Vector3 &get_vertex(int32_t p_idx) const { return vertices[p_idx]; }
	Vector3 &get_vertex(int32_t p_idx) { return vertices[p_idx]; }

	// Add a tetrahedron (4 indices, plus material_id).
	void add_tetrahedron(int32_t p_v0, int32_t p_v1, int32_t p_v2, int32_t p_v3, int32_t p_mat_id = 0) {
		elements.push_back(p_v0);
		elements.push_back(p_v1);
		elements.push_back(p_v2);
		elements.push_back(p_v3);
		material_ids.push_back(p_mat_id);
	}

	// Access a tetrahedron by index.
	struct Tetrahedron {
		int32_t v0, v1, v2, v3;
		int32_t mat_id;
	};
	Tetrahedron get_tetrahedron(int32_t p_el) const {
		Tetrahedron t;
		int idx = p_el * 4;
		ERR_FAIL_INDEX_V(idx + 3, elements.size(), t);
		t.v0 = elements[idx];
		t.v1 = elements[idx + 1];
		t.v2 = elements[idx + 2];
		t.v3 = elements[idx + 3];
		t.mat_id = material_ids[p_el];
		return t;
	}

	void set_tetrahedron(int32_t p_el, const Tetrahedron &p_tet) {
		int idx = p_el * 4;
		ERR_FAIL_INDEX(idx + 3, elements.size());
		elements[idx] = p_tet.v0;
		elements[idx + 1] = p_tet.v1;
		elements[idx + 2] = p_tet.v2;
		elements[idx + 3] = p_tet.v3;
		material_ids[p_el] = p_tet.mat_id;
	}

	// Precompute rest volumes and inverse rest shape matrices for all tets.
	// Must be called after the rest shape vertices are set.
	void precompute_rest_state() {
		int32_t tet_count = element_count();
		rest_volumes.resize(tet_count);
		inv_Dm.resize(tet_count * 3); // store 3 columns per tet

		for (int32_t el = 0; el < tet_count; ++el) {
			Tetrahedron tet = get_tetrahedron(el);
			const Vector3 &p0 = vertices[tet.v0];
			const Vector3 &p1 = vertices[tet.v1];
			const Vector3 &p2 = vertices[tet.v2];
			const Vector3 &p3 = vertices[tet.v3];

			Vector3 e1 = p1 - p0;
			Vector3 e2 = p2 - p0;
			Vector3 e3 = p3 - p0;

			real_t signed_volume6 = e1.cross(e2).dot(e3); // 6 * signed volume
			real_t abs_volume6 = Math::abs(signed_volume6);
			rest_volumes[el] = abs_volume6 / 6.0;

			// Inverse of rest deformation gradient Dm = [e1|e2|e3]
			// Columns of inv(Dm) store transposed cofactor matrix / det
			if (abs_volume6 < CMP_EPSILON) {
				// Degenerate element – use identity
				inv_Dm[el * 3 + 0] = Vector3(1, 0, 0);
				inv_Dm[el * 3 + 1] = Vector3(0, 1, 0);
				inv_Dm[el * 3 + 2] = Vector3(0, 0, 1);
			} else {
				real_t inv_det = 1.0 / signed_volume6;
				inv_Dm[el * 3 + 0] = e2.cross(e3) * inv_det;
				inv_Dm[el * 3 + 1] = e3.cross(e1) * inv_det;
				inv_Dm[el * 3 + 2] = e1.cross(e2) * inv_det;
			}
		}
	}

	// Get rest volume of element.
	real_t get_rest_volume(int32_t p_el) const {
		ERR_FAIL_INDEX_V(p_el, rest_volumes.size(), 0.0);
		return rest_volumes[p_el];
	}

	// Get the three columns of inv(Dm) for element.
	void get_inv_Dm(int32_t p_el, Vector3 &r_col0, Vector3 &r_col1, Vector3 &r_col2) const {
		ERR_FAIL_INDEX(p_el * 3 + 2, inv_Dm.size());
		r_col0 = inv_Dm[p_el * 3 + 0];
		r_col1 = inv_Dm[p_el * 3 + 1];
		r_col2 = inv_Dm[p_el * 3 + 2];
	}

private:
	LocalVector<Vector3> vertices;
	LocalVector<int32_t> elements;       // 4 indices per tet
	LocalVector<real_t> rest_volumes;
	LocalVector<Vector3> inv_Dm;         // 3 columns per tet
	LocalVector<int32_t> material_ids;
};

} // namespace gaia::mesh

#endif // GAIA_MESH_TET_MESH_H
// File 34: modules/gaia/src/mesh/tri_mesh.h

#ifndef GAIA_MESH_TRI_MESH_H
#define GAIA_MESH_TRI_MESH_H

#include "core/math/vector3.h"
#include "core/math/vector2.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia::mesh {

/**
 * Triangle mesh representation (used for cloth and surface collisions).
 *
 * Stores vertices, triangle indices, optionally texture coordinates and
 * precomputed rest edge lengths for bending constraints.
 */
class TriMesh {
public:
	TriMesh() {}

	// Clear all data.
	void clear() {
		vertices.clear();
		triangles.clear();
		rest_edge_lengths.clear();
		uvs.clear();
		normals.clear();
	}

	// Vertex count.
	int32_t vertex_count() const { return vertices.size(); }
	// Triangle count.
	int32_t triangle_count() const { return triangles.size() / 3; }

	// Add a vertex (position only or with UV).
	void add_vertex(const Vector3 &p_pos) {
		vertices.push_back(p_pos);
		if (uvs.size() < vertices.size()) {
			uvs.resize(vertices.size(), Vector2());
		}
	}
	void add_vertex(const Vector3 &p_pos, const Vector2 &p_uv) {
		vertices.push_back(p_pos);
		uvs.push_back(p_uv);
	}

	// Retrieve a vertex position.
	const Vector3 &get_vertex(int32_t p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, vertices.size(), dummy_vec3);
		return vertices[p_idx];
	}
	Vector3 &get_vertex(int32_t p_idx) {
		ERR_FAIL_INDEX_V(p_idx, vertices.size(), dummy_vec3);
		return vertices[p_idx];
	}

	// Add a triangle (3 vertex indices).
	void add_triangle(int32_t p_v0, int32_t p_v1, int32_t p_v2) {
		triangles.push_back(p_v0);
		triangles.push_back(p_v1);
		triangles.push_back(p_v2);
	}

	// Access triangle indices.
	struct Triangle {
		int32_t v0, v1, v2;
	};
	Triangle get_triangle(int32_t p_tri) const {
		Triangle t;
		int idx = p_tri * 3;
		ERR_FAIL_INDEX_V(idx + 2, triangles.size(), t);
		t.v0 = triangles[idx];
		t.v1 = triangles[idx + 1];
		t.v2 = triangles[idx + 2];
		return t;
	}

	void set_triangle(int32_t p_tri, const Triangle &p_t) {
		int idx = p_tri * 3;
		ERR_FAIL_INDEX(idx + 2, triangles.size());
		triangles[idx] = p_t.v0;
		triangles[idx + 1] = p_t.v1;
		triangles[idx + 2] = p_t.v2;
	}

	// Access / set UV coordinates.
	const Vector2 &get_uv(int32_t p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, uvs.size(), dummy_vec2);
		return uvs[p_idx];
	}
	void set_uv(int32_t p_idx, const Vector2 &p_uv) {
		ERR_FAIL_INDEX(p_idx, uvs.size());
		uvs[p_idx] = p_uv;
	}

	// Precompute rest edge lengths (for distance constraints) and rest
	// angles (for bending constraints) from the current vertex positions.
	// Called once after the rest mesh is set.
	void precompute_rest_state() {
		int tri_count = triangle_count();
		// Map edge to its rest length (using sorted indices).
		// For simplicity, we store edge lengths per triangle (three per tri).
		rest_edge_lengths.resize(tri_count * 3);

		for (int t = 0; t < tri_count; ++t) {
			Triangle tri = get_triangle(t);
			const Vector3 &p0 = vertices[tri.v0];
			const Vector3 &p1 = vertices[tri.v1];
			const Vector3 &p2 = vertices[tri.v2];

			rest_edge_lengths[t * 3 + 0] = p0.distance_to(p1);
			rest_edge_lengths[t * 3 + 1] = p1.distance_to(p2);
			rest_edge_lengths[t * 3 + 2] = p2.distance_to(p0);
		}
		// Bending rest angles require shared edges, which can be precomputed
		// later using the adjacent triangle pairs. For now we store edge length
		// data; bending constraints will compute rest angles on the fly or can
		// be extended with a separate precomputation step.
	}

	// Get the rest length of an edge of a specific triangle.
	// Edge index: 0 = v0->v1, 1 = v1->v2, 2 = v2->v0.
	real_t get_rest_edge_length(int32_t p_tri, int32_t p_edge) const {
		ERR_FAIL_INDEX_V(p_tri * 3 + p_edge, rest_edge_lengths.size(), 0.0);
		return rest_edge_lengths[p_tri * 3 + p_edge];
	}

	// Compute face normal (not normalized).
	Vector3 compute_face_normal(int32_t p_tri) const {
		Triangle tri = get_triangle(p_tri);
		const Vector3 &p0 = vertices[tri.v0];
		const Vector3 &p1 = vertices[tri.v1];
		const Vector3 &p2 = vertices[tri.v2];
		return (p1 - p0).cross(p2 - p0);
	}

	// Recompute vertex normals from face normals (averaged).
	void recompute_normals() {
		normals.resize(vertices.size());
		for (int i = 0; i < normals.size(); ++i) {
			normals[i] = Vector3();
		}
		int tri_count = triangle_count();
		for (int t = 0; t < tri_count; ++t) {
			Triangle tri = get_triangle(t);
			Vector3 face_n = compute_face_normal(t);
			normals[tri.v0] += face_n;
			normals[tri.v1] += face_n;
			normals[tri.v2] += face_n;
		}
		for (int i = 0; i < normals.size(); ++i) {
			real_t len = normals[i].length();
			if (len > CMP_EPSILON) {
				normals[i] /= len;
			}
		}
	}

	// Get vertex normal (after recompute).
	Vector3 get_normal(int32_t p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, normals.size(), Vector3(0, 1, 0));
		return normals[p_idx];
	}

private:
	LocalVector<Vector3> vertices;
	LocalVector<int32_t> triangles;        // 3 indices per triangle
	LocalVector<real_t> rest_edge_lengths; // 3 per triangle
	LocalVector<Vector2> uvs;
	LocalVector<Vector3> normals;

	// Dummy return references for error cases.
	static Vector3 dummy_vec3;
	static Vector2 dummy_vec2;
};

// Static dummy definitions (to be placed in a .cpp if not inline).
// Since this is a header-only module, we'll define them inline as extern.
// Actually, for simplicity, we'll just return a default-constructed reference
// from a static local. We'll fix error handling to use a static local.
inline Vector3 TriMesh::dummy_vec3;
inline Vector2 TriMesh::dummy_vec2;

} // namespace gaia::mesh

#endif // GAIA_MESH_TRI_MESH_H
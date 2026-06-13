// File 117: modules/gaia/src/vbd_cloth/vbd_base_tri_mesh.h
// Base triangle mesh for VBD cloth simulation. Stores vertices, edges,
// adjacency, and supports construction of stretch, shear, and bending constraints.

#ifndef GAIA_VBD_CLOTH_BASE_TRI_MESH_H
#define GAIA_VBD_CLOTH_BASE_TRI_MESH_H

#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "../mesh/tri_mesh.h"

namespace gaia::vbd_cloth {

class VBDBaseTriMesh {
public:
	struct Vertex {
		Vector3 rest_pos;
		Vector3 pos;
		Vector3 velocity;
		bool pinned;
		real_t mass;
		Vertex() : rest_pos(), pos(), velocity(), pinned(false), mass(1.0) {}
	};

	struct Edge {
		int v0, v1;
		real_t rest_length;
		Edge(int a, int b) : v0(a), v1(b), rest_length(0) {}
	};

	struct Triangle {
		int v0, v1, v2;
		Triangle(int a, int b, int c) : v0(a), v1(b), v2(c) {}
	};

private:
	LocalVector<Vertex> vertices;
	LocalVector<Edge> edges;
	LocalVector<Triangle> triangles;

	// For quick lookup of edges from vertex pair
	HashMap<uint64_t, int> edge_map; // key = (min_idx << 32 | max_idx)

public:
	VBDBaseTriMesh() {}

	void clear() {
		vertices.clear();
		edges.clear();
		triangles.clear();
		edge_map.clear();
	}

	// Build from a Gaia TriMesh (rest state)
	void build_from(const mesh::TriMesh &tri_mesh) {
		clear();
		int nv = tri_mesh.vertex_count();
		vertices.resize(nv);
		for (int i = 0; i < nv; ++i) {
			vertices[i].rest_pos = tri_mesh.get_vertex(i);
			vertices[i].pos = vertices[i].rest_pos;
			vertices[i].velocity = Vector3();
			vertices[i].mass = 1.0f;
			vertices[i].pinned = false;
		}

		int nt = tri_mesh.triangle_count();
		for (int t = 0; t < nt; ++t) {
			mesh::TriMesh::Triangle tri = tri_mesh.get_triangle(t);
			add_triangle(tri.v0, tri.v1, tri.v2);
		}
	}

	// Direct generation of a rectangular grid (cloth patch)
	void generate_grid(int res_x, int res_y, real_t width, real_t height) {
		clear();
		int nv = res_x * res_y;
		vertices.resize(nv);
		real_t dx = width / (res_x - 1);
		real_t dy = height / (res_y - 1);
		for (int y = 0; y < res_y; ++y) {
			for (int x = 0; x < res_x; ++x) {
				int idx = y * res_x + x;
				Vector3 pos(x * dx - width * 0.5f, 0.0f, y * dy - height * 0.5f);
				vertices[idx].rest_pos = pos;
				vertices[idx].pos = pos;
				vertices[idx].mass = 1.0f;
				vertices[idx].velocity = Vector3();
			}
		}

		for (int y = 0; y < res_y - 1; ++y) {
			for (int x = 0; x < res_x - 1; ++x) {
				int a = y * res_x + x;
				int b = a + 1;
				int c = a + res_x;
				int d = c + 1;
				add_triangle(a, b, c);
				add_triangle(c, b, d);
			}
		}
	}

	int vertex_count() const { return vertices.size(); }
	int edge_count() const { return edges.size(); }
	int triangle_count() const { return triangles.size(); }

	const Vertex &get_vertex(int i) const { return vertices[i]; }
	Vertex &get_vertex(int i) { return vertices[i]; }

	const Edge &get_edge(int i) const { return edges[i]; }
	const Triangle &get_triangle(int i) const { return triangles[i]; }

	// Pin / unpin a vertex (fixed boundary)
	void pin_vertex(int idx, bool pin) {
		ERR_FAIL_INDEX(idx, vertices.size());
		vertices[idx].pinned = pin;
	}
	bool is_vertex_pinned(int idx) const {
		ERR_FAIL_INDEX_V(idx, vertices.size(), false);
		return vertices[idx].pinned;
	}

	// Get edge index from two vertex indices, or -1 if not present
	int find_edge(int v0, int v1) const {
		if (v0 > v1) SWAP(v0, v1);
		uint64_t key = (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
		HashMap<uint64_t, int>::ConstIterator it = edge_map.find(key);
		if (it) return it->value;
		return -1;
	}

private:
	void add_vertex(const Vector3 &rest_pos) {
		vertices.push_back(Vertex());
		vertices.back().rest_pos = rest_pos;
		vertices.back().pos = rest_pos;
	}

	// Add a triangle, updating edges and internal structures
	void add_triangle(int v0, int v1, int v2) {
		triangles.push_back(Triangle(v0, v1, v2));
		add_edge(v0, v1);
		add_edge(v1, v2);
		add_edge(v2, v0);
	}

	void add_edge(int a, int b) {
		if (find_edge(a, b) != -1) return; // already exists
		if (a > b) SWAP(a, b);
		edges.push_back(Edge(a, b));
		Edge &edge = edges.back();
		edge.rest_length = vertices[a].rest_pos.distance_to(vertices[b].rest_pos);

		uint64_t key = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
		edge_map[key] = edges.size() - 1;
	}
};

} // namespace gaia::vbd_cloth

#endif // GAIA_VBD_CLOTH_BASE_TRI_MESH_H
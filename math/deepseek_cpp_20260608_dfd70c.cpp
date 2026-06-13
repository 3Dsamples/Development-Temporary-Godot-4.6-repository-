// File 107: modules/gaia/src/graph/tri_mesh_vertex_graph.h
// Builds a vertex adjacency graph from a triangle mesh.
// Each vertex becomes a node; an edge is added if two vertices share
// at least one triangle.

#ifndef GAIA_GRAPH_TRI_MESH_VERTEX_GRAPH_H
#define GAIA_GRAPH_TRI_MESH_VERTEX_GRAPH_H

#include "graph.h"
#include "../mesh/tri_mesh.h"

namespace gaia::graph {

class TriMeshVertexGraph : public Graph {
public:
	// Build the graph from a TriMesh.
	void build_from(const mesh::TriMesh &p_tri_mesh) {
		int vert_count = p_tri_mesh.vertex_count();
		init(vert_count);
		int tri_count = p_tri_mesh.triangle_count();
		for (int t = 0; t < tri_count; ++t) {
			mesh::TriMesh::Triangle tri = p_tri_mesh.get_triangle(t);
			int ids[3] = { tri.v0, tri.v1, tri.v2 };
			// Connect every pair of distinct vertices in the triangle
			for (int i = 0; i < 3; ++i) {
				for (int j = i + 1; j < 3; ++j) {
					add_edge(ids[i], ids[j]);
				}
			}
		}
	}
};

} // namespace gaia::graph

#endif // GAIA_GRAPH_TRI_MESH_VERTEX_GRAPH_H
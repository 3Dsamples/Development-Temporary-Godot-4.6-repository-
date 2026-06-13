// File 104: modules/gaia/src/graph/tet_mesh_vertex_graph.h
// Builds a vertex adjacency graph from a tetrahedral mesh.
// Each vertex becomes a node; an edge is added if two vertices share
// at least one tetrahedron.

#ifndef GAIA_GRAPH_TET_MESH_VERTEX_GRAPH_H
#define GAIA_GRAPH_TET_MESH_VERTEX_GRAPH_H

#include "graph.h"
#include "../mesh/tet_mesh.h"

namespace gaia::graph {

class TetMeshVertexGraph : public Graph {
public:
	// Build the graph from a TetMesh.
	void build_from(const mesh::TetMesh &p_tet_mesh) {
		int vert_count = p_tet_mesh.vertex_count();
		init(vert_count);
		int tet_count = p_tet_mesh.element_count();
		for (int el = 0; el < tet_count; ++el) {
			mesh::TetMesh::Tetrahedron tet = p_tet_mesh.get_tetrahedron(el);
			int ids[4] = { tet.v0, tet.v1, tet.v2, tet.v3 };
			// Connect every pair of distinct vertices in the tetrahedron
			for (int i = 0; i < 4; ++i) {
				for (int j = i + 1; j < 4; ++j) {
					add_edge(ids[i], ids[j]);
				}
			}
		}
	}
};

} // namespace gaia::graph

#endif // GAIA_GRAPH_TET_MESH_VERTEX_GRAPH_H
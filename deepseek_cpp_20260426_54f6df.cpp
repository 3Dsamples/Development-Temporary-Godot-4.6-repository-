// File 105: modules/gaia/src/graph/tet_mesh_edge_graph.h
// Builds an edge adjacency graph from a tetrahedral mesh.
// Each unique undirected edge in the mesh becomes a node; an edge between
// two edge-nodes is added if they share a common tetrahedron (i.e., both
// belong to the same tet). This graph is used for parallel distance‑constraint coloring.

#ifndef GAIA_GRAPH_TET_MESH_EDGE_GRAPH_H
#define GAIA_GRAPH_TET_MESH_EDGE_GRAPH_H

#include "graph.h"
#include "../mesh/tet_mesh.h"
#include "core/templates/hash_map.h"

namespace gaia::graph {

class TetMeshEdgeGraph : public Graph {
public:
	void build_from(const mesh::TetMesh &p_tet_mesh) {
		// We assign an index to each unique unordered edge (v_a, v_b) with v_a < v_b.
		HashMap<std::pair<int, int>, int> edge_to_index;
		LocalVector<std::pair<int, int>> edge_vertices; // index -> (min_v, max_v)
		int edge_count = 0;

		// First pass: enumerate all edges in all tetrahedra.
		int tet_count = p_tet_mesh.element_count();
		for (int el = 0; el < tet_count; ++el) {
			mesh::TetMesh::Tetrahedron tet = p_tet_mesh.get_tetrahedron(el);
			int v[4] = { tet.v0, tet.v1, tet.v2, tet.v3 };
			for (int i = 0; i < 4; ++i) {
				for (int j = i + 1; j < 4; ++j) {
					int a = MIN(v[i], v[j]);
					int b = MAX(v[i], v[j]);
					auto key = std::make_pair(a, b);
					if (!edge_to_index.has(key)) {
						edge_to_index[key] = edge_count;
						edge_vertices.push_back(key);
						++edge_count;
					}
				}
			}
		}

		init(edge_count);

		// Second pass: for each tet, connect all edge-nodes that belong to this tet.
		for (int el = 0; el < tet_count; ++el) {
			mesh::TetMesh::Tetrahedron tet = p_tet_mesh.get_tetrahedron(el);
			int v[4] = { tet.v0, tet.v1, tet.v2, tet.v3 };
			LocalVector<int> edges_in_tet; // up to 6
			for (int i = 0; i < 4; ++i) {
				for (int j = i + 1; j < 4; ++j) {
					int a = MIN(v[i], v[j]);
					int b = MAX(v[i], v[j]);
					int eid = edge_to_index[std::make_pair(a, b)];
					edges_in_tet.push_back(eid);
				}
			}
			// Connect all distinct edge-node pairs within this tet
			for (int i = 0; i < edges_in_tet.size(); ++i) {
				for (int j = i + 1; j < edges_in_tet.size(); ++j) {
					add_edge(edges_in_tet[i], edges_in_tet[j]);
				}
			}
		}
	}
};

} // namespace gaia::graph

#endif // GAIA_GRAPH_TET_MESH_EDGE_GRAPH_H
// File 106: modules/gaia/src/graph/tet_mesh_tet_graph.h
// Builds an element adjacency graph from a tetrahedral mesh.
// Each tetrahedron (element) becomes a node; an undirected edge is added
// between two elements if they share a common face. This graph is used
// for parallel element‑based operations (e.g., VBD block descent scheduling).

#ifndef GAIA_GRAPH_TET_MESH_TET_GRAPH_H
#define GAIA_GRAPH_TET_MESH_TET_GRAPH_H

#include "graph.h"
#include "../mesh/tet_mesh.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"

namespace gaia::graph {

class TetMeshTetGraph : public Graph {
public:
	void build_from(const mesh::TetMesh &p_tet_mesh) {
		int tet_count = p_tet_mesh.element_count();
		init(tet_count);

		// Map from a canonical face (sorted three vertex indices) to the element that exposes it.
		// When a face is already seen, we connect the previous element with the current one.
		struct FaceKey {
			int v0, v1, v2; // sorted in ascending order
			bool operator==(const FaceKey &other) const {
				return v0 == other.v0 && v1 == other.v1 && v2 == other.v2;
			}
			struct Hash {
				uint32_t operator()(const FaceKey &k) const {
					return (uint32_t(k.v0) * 73856093) ^ (uint32_t(k.v1) * 19349663) ^ (uint32_t(k.v2) * 83492791);
				}
			};
		};

		HashMap<FaceKey, int, FaceKey::Hash> face_to_element;

		for (int el = 0; el < tet_count; ++el) {
			mesh::TetMesh::Tetrahedron tet = p_tet_mesh.get_tetrahedron(el);
			int v[4] = { tet.v0, tet.v1, tet.v2, tet.v3 };

			// Four faces of the tetrahedron, each sorted for canonicality.
			// Convention: omit each vertex in turn.
			const int faces[4][3] = {
				{ v[1], v[2], v[3] },
				{ v[0], v[2], v[3] },
				{ v[0], v[1], v[3] },
				{ v[0], v[1], v[2] }
			};

			for (int f = 0; f < 4; ++f) {
				int a = faces[f][0];
				int b = faces[f][1];
				int c = faces[f][2];
				// Sort the three indices
				if (a > b) SWAP(a, b);
				if (b > c) SWAP(b, c);
				if (a > b) SWAP(a, b);
				FaceKey key{ a, b, c };

				if (face_to_element.has(key)) {
					int neighbor_el = face_to_element[key];
					add_edge(el, neighbor_el);
				} else {
					face_to_element[key] = el;
				}
			}
		}
	}
};

} // namespace gaia::graph

#endif // GAIA_GRAPH_TET_MESH_TET_GRAPH_H
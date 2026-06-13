// File 102: modules/gaia/src/graph/graph.h
// Abstract graph data structure for mesh coloring and parallel constraint scheduling.
// Adapted from Gaia's Graph.h to use Godot core containers.

#ifndef GAIA_GRAPH_GRAPH_H
#define GAIA_GRAPH_GRAPH_H

#include "core/templates/local_vector.h"
#include "core/templates/hash_set.h"
#include "core/typedefs.h"

namespace gaia::graph {

/**
 * A simple undirected graph represented by adjacency lists.
 * Nodes are indexed 0 .. n-1.
 */
class Graph {
private:
	int node_count;
	LocalVector<HashSet<int>> adjacency;
	LocalVector<int> degree;

public:
	Graph() : node_count(0) {}

	void init(int p_count) {
		node_count = p_count;
		adjacency.resize(p_count);
		degree.resize(p_count);
		for (int i = 0; i < p_count; ++i) {
			adjacency[i] = HashSet<int>();
			degree[i] = 0;
		}
	}

	int get_node_count() const { return node_count; }
	int get_degree(int p_node) const {
		ERR_FAIL_INDEX_V(p_node, node_count, 0);
		return degree[p_node];
	}
	const HashSet<int> &get_neighbors(int p_node) const {
		ERR_FAIL_INDEX_V(p_node, node_count, adjacency[p_node]);
		return adjacency[p_node];
	}

	// Add an undirected edge; ignores duplicate.
	void add_edge(int a, int b) {
		ERR_FAIL_INDEX(a, node_count);
		ERR_FAIL_INDEX(b, node_count);
		if (a == b) return;
		if (!adjacency[a].has(b)) {
			adjacency[a].insert(b);
			adjacency[b].insert(a);
			degree[a]++;
			degree[b]++;
		}
	}

	// Remove an undirected edge.
	void remove_edge(int a, int b) {
		ERR_FAIL_INDEX(a, node_count);
		ERR_FAIL_INDEX(b, node_count);
		if (adjacency[a].has(b)) {
			adjacency[a].erase(b);
			adjacency[b].erase(a);
			degree[a]--;
			degree[b]--;
		}
	}

	// Clear the graph.
	void clear() {
		node_count = 0;
		adjacency.clear();
		degree.clear();
	}
};

} // namespace gaia::graph

#endif // GAIA_GRAPH_GRAPH_H
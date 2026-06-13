// File 103: modules/gaia/src/graph/coloring_algorithms.h
// Graph coloring and ordering algorithms for parallel constraint scheduling.
// Rewritten from Gaia's ColoringAlgorithms.h using Godot containers.

#ifndef GAIA_GRAPH_COLORING_ALGORITHMS_H
#define GAIA_GRAPH_COLORING_ALGORITHMS_H

#include "graph.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/typedefs.h"

namespace gaia::graph {

/**
 * Greedy graph coloring: assign a color to each node such that no two
 * adjacent nodes share the same color. Colors are integers starting from 0.
 * The method uses the smallest available color not used by neighbors.
 *
 * @param p_graph   The input graph.
 * @param r_colors  Output array of colors, resized to node count.
 * @return          The number of colors used.
 */
inline int greedy_coloring(const Graph &p_graph, LocalVector<int> &r_colors) {
	int n = p_graph.get_node_count();
	r_colors.resize(n);
	for (int i = 0; i < n; ++i) {
		r_colors[i] = -1;
	}

	int max_color = -1;
	LocalVector<bool> used(n, false); // color availability for current node

	for (int i = 0; i < n; ++i) {
		const HashSet<int> &neighbors = p_graph.get_neighbors(i);
		// Mark colors used by neighbors
		for (const int &nb : neighbors) {
			if (r_colors[nb] != -1) {
				int col = r_colors[nb];
				if (col < used.size()) {
					used[col] = true;
				}
			}
		}
		// Find the smallest unused color
		int chosen_color = 0;
		while (chosen_color < used.size() && used[chosen_color]) {
			++chosen_color;
		}
		r_colors[i] = chosen_color;
		if (chosen_color > max_color) {
			max_color = chosen_color;
		}
		// Reset used flags efficiently: iterate over neighbors again
		for (const int &nb : neighbors) {
			if (r_colors[nb] != -1 && r_colors[nb] < used.size()) {
				used[r_colors[nb]] = false;
			}
		}
	}
	return max_color + 1;
}

/**
 * Maximum Cardinality Search (MCS) ordering.
 * Produces an elimination order starting from a pseudoperipheral node.
 * The resulting permutation can be used for parallel Gauss-Seidel or
 * for coloring. This implementation is sequential.
 *
 * @param p_graph    The input graph.
 * @param r_order    Output: r_order[i] is the i-th node in the MCS order.
 * @param r_weights  Optional: fill with final weights (cardinalities).
 */
inline void mcs_ordering(const Graph &p_graph, LocalVector<int> &r_order, LocalVector<int> *r_weights = nullptr) {
	int n = p_graph.get_node_count();
	r_order.resize(n);
	LocalVector<int> weight(n, 0);
	LocalVector<bool> selected(n, false);

	// Start with node 0 (could be replaced by a pseudoperipheral finder)
	for (int i = 0; i < n; ++i) {
		// Pick unselected node with maximum weight
		int best = -1;
		int best_weight = -1;
		for (int j = 0; j < n; ++j) {
			if (!selected[j] && weight[j] > best_weight) {
				best_weight = weight[j];
				best = j;
			}
		}
		// If all remaining have weight 0 (first or isolated), pick any unselected
		if (best == -1) {
			for (int j = 0; j < n; ++j) {
				if (!selected[j]) { best = j; break; }
			}
		}
		ERR_FAIL_COND(best == -1);

		selected[best] = true;
		r_order[i] = best;

		// Increase weight of all unselected neighbors
		const HashSet<int> &nbrs = p_graph.get_neighbors(best);
		for (const int &nb : nbrs) {
			if (!selected[nb]) {
				weight[nb]++;
			}
		}
	}

	if (r_weights) {
		*r_weights = weight;
	}
}

/**
 * Color the graph using the MCS ordering (often yields fewer colors).
 * This is a common heuristic: apply greedy coloring on the MCS order reversed.
 */
inline int mcs_coloring(const Graph &p_graph, LocalVector<int> &r_colors) {
	int n = p_graph.get_node_count();
	LocalVector<int> order;
	mcs_ordering(p_graph, order);

	// Build reverse mapping
	LocalVector<int> inverse_order(n);
	for (int i = 0; i < n; ++i) {
		inverse_order[order[i]] = i;
	}

	// Color in reverse MCS order
	r_colors.resize(n);
	for (int i = 0; i < n; ++i) {
		r_colors[i] = -1;
	}
	LocalVector<bool> used(n, false);

	int max_color = -1;
	for (int pos = n - 1; pos >= 0; --pos) {
		int node = order[pos];
		const HashSet<int> &nbrs = p_graph.get_neighbors(node);
		for (const int &nb : nbrs) {
			int c = r_colors[nb];
			if (c != -1) used[c] = true;
		}
		int chosen = 0;
		while (chosen < n && used[chosen]) ++chosen;
		r_colors[node] = chosen;
		if (chosen > max_color) max_color = chosen;
		for (const int &nb : nbrs) {
			if (r_colors[nb] != -1) used[r_colors[nb]] = false;
		}
	}
	return max_color + 1;
}

} // namespace gaia::graph

#endif // GAIA_GRAPH_COLORING_ALGORITHMS_H
// File 32: modules/gaia/src/spatial_query/neighbour_query.h

#ifndef GAIA_SPATIAL_NEIGHBOUR_QUERY_H
#define GAIA_SPATIAL_NEIGHBOUR_QUERY_H

#include "spatial_hash.h"

#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia::spatial {

/**
 * Neighbour query utility for finding all pairs of points within a given
 * radius. Uses spatial hashing for efficient O(n) average time.
 */
class NeighbourQuery {
public:
	/**
	 * Find all pairs (i, j) with i < j such that distance(positions[i],
	 * positions[j]) <= p_radius.
	 *
	 * The callback is called with the two indices. The spatial hash cell size
	 * is set to p_radius.
	 */
	template <typename Callback>
	static void find_pairs(const LocalVector<Vector3> &p_positions,
						   real_t p_radius,
						   Callback &&p_callback) {
		int32_t n = p_positions.size();
		if (n < 2) return;

		// Build spatial hash with cell size = radius
		SpatialHash hash(p_radius);
		for (int32_t i = 0; i < n; ++i) {
			hash.insert(i, p_positions[i]);
		}

		// For each point, query its cell and neighbours
		LocalVector<int32_t> neighbors; // reuse buffer
		for (int32_t i = 0; i < n; ++i) {
			neighbors.clear();
			hash.query(p_positions[i], neighbors, true); // include neighbours

			for (int32_t j : neighbors) {
				if (j <= i) continue; // avoid duplicates and self

				real_t dist_sq = p_positions[i].distance_squared_to(p_positions[j]);
				if (dist_sq <= p_radius * p_radius) {
					p_callback(i, j, Math::sqrt(dist_sq));
				}
			}
		}
	}

	/**
	 * Brute-force version for small data or verification.
	 */
	template <typename Callback>
	static void find_pairs_brute_force(const LocalVector<Vector3> &p_positions,
									   real_t p_radius,
									   Callback &&p_callback) {
		int32_t n = p_positions.size();
		real_t r2 = p_radius * p_radius;
		for (int32_t i = 0; i < n; ++i) {
			for (int32_t j = i + 1; j < n; ++j) {
				real_t d2 = p_positions[i].distance_squared_to(p_positions[j]);
				if (d2 <= r2) {
					p_callback(i, j, Math::sqrt(d2));
				}
			}
		}
	}
};

} // namespace gaia::spatial

#endif // GAIA_SPATIAL_NEIGHBOUR_QUERY_H
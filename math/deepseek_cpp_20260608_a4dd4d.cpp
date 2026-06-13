// File 451: modules/integration/unified_sph_neighbour_search.h
// High‑performance SPH neighbour search using the TreeNSearch point‑set BVH.
// Rebuilds a BVH from particle positions each frame, then performs radius
// queries to return neighbour lists for every particle.  Replaces the slower
// Gaia spatial hash in Genesis SPHSolver.  All methods are fully inline
// and use parallel dispatch when multiple cores are available.

#ifndef INTEGRATION_UNIFIED_SPH_NEIGHBOUR_SEARCH_H
#define INTEGRATION_UNIFIED_SPH_NEIGHBOUR_SEARCH_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"
#include "../../treesearch/point_set_search.h"
#include "../../treesearch/parallel_point_search.h"

namespace unified {

class UnifiedSPHNeighbourSearch : public RefCounted {
    GDCLASS(UnifiedSPHNeighbourSearch, RefCounted);

public:
    // Smoothing length (search radius) for nearest neighbour queries.
    real_t smoothing_length = 0.1f;

    // Maximum number of neighbours per particle (used to pre‑allocate buffers).
    int max_neighbours = 64;

    // -------------------------------------------------------------------
    // Rebuild the internal BVH from a set of particle positions.
    // Must be called once per frame before querying.
    // -------------------------------------------------------------------
    void rebuild(const LocalVector<Vector3> &p_positions);

    // -------------------------------------------------------------------
    // For a single particle, return the list of neighbour indices (within
    // smoothing_length).  The output vector is cleared and filled.
    // -------------------------------------------------------------------
    void query_single(const Vector3 &p_position, LocalVector<int> &r_neighbours) const;

    // -------------------------------------------------------------------
    // For all particles, compute neighbour lists and store them in a flat
    // format: for each particle i, neighbours[i] contains the list.
    // This function uses parallel dispatch when enough particles exist.
    // -------------------------------------------------------------------
    void query_all(const LocalVector<Vector3> &p_positions,
                   LocalVector<LocalVector<int>> &r_neighbours) const;

    // -------------------------------------------------------------------
    // Return the total number of points in the BVH.
    // -------------------------------------------------------------------
    int get_point_count() const { return bvh.points.size(); }

protected:
    static void _bind_methods();

private:
    // The underlying TreeNSearch point set BVH.
    treesearch::PointSetSearch bvh;

    // Parallel search helper.
    treesearch::ParallelPointSearch parallel_searcher;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedSPHNeighbourSearch::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_smoothing_length", "h"), &UnifiedSPHNeighbourSearch::set_smoothing_length);
    ClassDB::bind_method(D_METHOD("get_smoothing_length"), &UnifiedSPHNeighbourSearch::get_smoothing_length);
    ClassDB::bind_method(D_METHOD("set_max_neighbours", "max"), &UnifiedSPHNeighbourSearch::set_max_neighbours);
    ClassDB::bind_method(D_METHOD("get_max_neighbours"), &UnifiedSPHNeighbourSearch::get_max_neighbours);
    ClassDB::bind_method(D_METHOD("rebuild", "positions"), &UnifiedSPHNeighbourSearch::rebuild);
    ClassDB::bind_method(D_METHOD("query_single", "position"), &UnifiedSPHNeighbourSearch::query_single);
    ClassDB::bind_method(D_METHOD("query_all", "positions"), &UnifiedSPHNeighbourSearch::query_all);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing_length"), "set_smoothing_length", "get_smoothing_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_neighbours"), "set_max_neighbours", "get_max_neighbours");
}

void UnifiedSPHNeighbourSearch::rebuild(const LocalVector<Vector3> &p_positions) {
    bvh.build(p_positions);
    parallel_searcher.set_point_set(&bvh);
}

void UnifiedSPHNeighbourSearch::query_single(const Vector3 &p_position,
                                             LocalVector<int> &r_neighbours) const {
    r_neighbours.clear();
    bvh.radius(p_position, smoothing_length, r_neighbours);
}

void UnifiedSPHNeighbourSearch::query_all(const LocalVector<Vector3> &p_positions,
                                          LocalVector<LocalVector<int>> &r_neighbours) const {
    int n = p_positions.size();
    r_neighbours.resize(n);
    parallel_searcher.radius_parallel(p_positions, smoothing_length, r_neighbours);
    // Clamp each list to max_neighbours (optional, keeps memory bounded).
    if (max_neighbours > 0) {
        for (int i = 0; i < n; ++i) {
            if (r_neighbours[i].size() > max_neighbours)
                r_neighbours[i].resize(max_neighbours);
        }
    }
}

// Property setters/getters.
void UnifiedSPHNeighbourSearch::set_smoothing_length(real_t v) { smoothing_length = MAX(v, 0.001f); }
real_t UnifiedSPHNeighbourSearch::get_smoothing_length() const { return smoothing_length; }
void UnifiedSPHNeighbourSearch::set_max_neighbours(int v) { max_neighbours = MAX(v, 1); }
int UnifiedSPHNeighbourSearch::get_max_neighbours() const { return max_neighbours; }

} // namespace unified

#endif // INTEGRATION_UNIFIED_SPH_NEIGHBOUR_SEARCH_H
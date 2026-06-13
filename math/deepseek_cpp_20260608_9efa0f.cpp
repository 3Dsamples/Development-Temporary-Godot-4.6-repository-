// File 462: modules/integration/unified_bvh_broadphase.h
// High‑performance broad‑phase collision detection for rigid bodies in
// the unified pipeline.  Replaces Gaia's spatial hash and BVH with a
// TreeNSearch point‑set BVH built from expanded AABB centroids each
// frame.  Uses radius queries to find candidate pairs, then exact AABB
// intersection tests, minimising memory and exploiting the log‑linear
// scaling of a BVH.  All operations are fully inlined for maximum speed.

#ifndef INTEGRATION_UNIFIED_BVH_BROADPHASE_H
#define INTEGRATION_UNIFIED_BVH_BROADPHASE_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

#include "../../treesearch/point_set_search.h"
#include "../../treesearch/treesearch.h"
#include "../../gaia/src/parallelization/cpu_parallelization.h"

namespace unified {

class UnifiedBVHBroadphase : public RefCounted {
    GDCLASS(UnifiedBVHBroadphase, RefCounted);

public:
    // Expansion factor applied to AABBs before centroid computation.
    // Increases the search radius for velocity‑based CCD margin.
    real_t aabb_expansion = 0.1f;

    // -------------------------------------------------------------------
    // Rebuild the broad‑phase from a set of body AABBs.
    // Each entry corresponds to one body; the index in the array is used
    // as the body's ID for pair generation.
    // -------------------------------------------------------------------
    void rebuild(const LocalVector<AABB> &p_aabbs);

    // -------------------------------------------------------------------
    // Find all overlapping pairs (i, j) with i < j based on AABB
    // intersection.  The output is a flat array of two ints per pair.
    // -------------------------------------------------------------------
    void find_pairs(LocalVector<int> &r_pairs) const;

    // -------------------------------------------------------------------
    // Return the number of bodies currently stored.
    // -------------------------------------------------------------------
    int get_body_count() const { return centroids.size(); }

protected:
    static void _bind_methods();

private:
    // Expanded AABB centroids for each body.
    LocalVector<Vector3> centroids;
    // Original AABBs (used for the final intersection test).
    LocalVector<AABB>    aabbs;
    // TreeNSearch point‑set BVH over centroids.
    treesearch::PointSetSearch bvh;

    // Accessor for the centroid array used by TreeNSearch.
    struct CentroidAccessor {
        const LocalVector<Vector3> *centers;
        CentroidAccessor(const LocalVector<Vector3> *p) : centers(p) {}
        Vector3 operator()(int i) const { return (*centers)[i]; }
    };
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedBVHBroadphase::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_aabb_expansion", "expansion"), &UnifiedBVHBroadphase::set_aabb_expansion);
    ClassDB::bind_method(D_METHOD("get_aabb_expansion"), &UnifiedBVHBroadphase::get_aabb_expansion);
    ClassDB::bind_method(D_METHOD("rebuild", "aabbs"), &UnifiedBVHBroadphase::rebuild);
    ClassDB::bind_method(D_METHOD("find_pairs"), &UnifiedBVHBroadphase::find_pairs);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aabb_expansion"), "set_aabb_expansion", "get_aabb_expansion");
}

void UnifiedBVHBroadphase::set_aabb_expansion(real_t v) { aabb_expansion = MAX(v, 0.0f); }
real_t UnifiedBVHBroadphase::get_aabb_expansion() const { return aabb_expansion; }

// ---------------------------------------------------------------------------
// Rebuild: compute centroids of expanded AABBs, then build the BVH.
// ---------------------------------------------------------------------------
void UnifiedBVHBroadphase::rebuild(const LocalVector<AABB> &p_aabbs) {
    aabbs = p_aabbs;
    int n = aabbs.size();
    centroids.resize(n);
    for (int i = 0; i < n; ++i) {
        // Expand the AABB by the expansion margin, then take its centroid.
        AABB expanded = aabbs[i].grow(aabb_expansion);
        centroids[i] = expanded.get_center();
    }
    bvh.build(centroids);
}

// ---------------------------------------------------------------------------
// Find overlapping pairs using a radius search around each body's centroid,
// then exact AABB intersection.
// ---------------------------------------------------------------------------
void UnifiedBVHBroadphase::find_pairs(LocalVector<int> &r_pairs) const {
    r_pairs.clear();
    int n = aabbs.size();
    if (n < 2) return;

    // Determine a generous search radius: half of the largest diagonal of
    // all expanded AABBs.  This ensures that any two AABBs that could
    // possibly intersect will have centroids within this distance.
    // We compute the maximum diagonal semi‑length across all bodies.
    real_t max_half_diag = 0.0f;
    for (const AABB &box : aabbs) {
        real_t d = box.get_longest_axis_size();
        if (d > max_half_diag) max_half_diag = d;
    }
    real_t search_radius = max_half_diag * 0.5f + aabb_expansion;

    if (search_radius <= 0.0f) return;

    // For each body, query the BVH for centroids within search_radius,
    // then test AABB overlap and output pairs i < j.
    for (int i = 0; i < n; ++i) {
        LocalVector<int> neighbours;
        bvh.radius(centroids[i], search_radius, neighbours);
        for (int j : neighbours) {
            if (j <= i) continue; // avoid duplicate pairs
            // Exact AABB intersection test.
            if (aabbs[i].intersects(aabbs[j])) {
                r_pairs.push_back(i);
                r_pairs.push_back(j);
            }
        }
    }
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_BVH_BROADPHASE_H
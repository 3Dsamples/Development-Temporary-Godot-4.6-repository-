// File 450: modules/integration/unified_spatial_query_manager.h
// High‑performance spatial query manager built on the TreeNSearch BVH
// (point‑set search).  Rebuilds a point‑cloud BVH from all active rigid
// body centroids across all registered engines (Newton, Genesis, Vienna,
// Wicked) each frame.  Supports K‑NN and radius queries, returning lists
// of engine‑specific body IDs.  All queries use the parallel TreeNSearch
// templates and Godot's WorkerThreadPool for scalability.

#ifndef INTEGRATION_UNIFIED_SPATIAL_QUERY_MANAGER_H
#define INTEGRATION_UNIFIED_SPATIAL_QUERY_MANAGER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

// TreeNSearch point‑set BVH
#include "../../treesearch/point_set_search.h"

// Forward engine world / body types (used via void* to avoid heavy includes).
namespace newton   { class NewtonWorld; }
namespace genesis  { class GenesisWorld; }
namespace vienna   { class ViennaWorld; }
namespace wicked   { class WickedWorld; }

namespace unified {

class UnifiedSpatialQueryManager : public RefCounted {
    GDCLASS(UnifiedSpatialQueryManager, RefCounted);

public:
    // Engine index constants (same as in other unified classes).
    static constexpr int ENGINE_NEWTON  = 0;
    static constexpr int ENGINE_GENESIS = 1;
    static constexpr int ENGINE_VIENNA  = 2;
    static constexpr int ENGINE_WICKED  = 3;

    // -------------------------------------------------------------------
    // Rebuild the internal BVH from all active bodies in the given worlds.
    // Each body's world position is used as a point.  Body IDs and engine
    // indices are stored for later lookup.
    // Must be called once per physics frame before querying.
    // -------------------------------------------------------------------
    void rebuild(const HashMap<int, void *> &p_worlds);

    // -------------------------------------------------------------------
    // K‑nearest neighbours around a world point.
    // Returns at most p_k body IDs (sorted by distance).  Each body is
    // identified by its engine index and internal ID.
    // -------------------------------------------------------------------
    struct KnnResult {
        int engine;
        uint64_t body_id;
        real_t distance;
    };
    void knn_query(const Vector3 &p_center, int p_k,
                   LocalVector<KnnResult> &r_results) const;

    // -------------------------------------------------------------------
    // Radius query: all bodies whose centroid lies within p_radius.
    // -------------------------------------------------------------------
    void radius_query(const Vector3 &p_center, real_t p_radius,
                      LocalVector<KnnResult> &r_results) const;

    // Return the number of points stored (total active bodies).
    int get_point_count() const { return points.size(); }

protected:
    static void _bind_methods();

private:
    // The BVH over points (centroids).
    treesearch::PointSetSearch bvh;

    // Parallel arrays: for each point, the engine and body ID it belongs to.
    LocalVector<int>      point_engine;
    LocalVector<uint64_t> point_body_id;

    // Raw point positions (centroids).
    LocalVector<Vector3>  points;

    // Cached world pointers for faster rebuild (optional).
    const newton::NewtonWorld   *newton_world = nullptr;
    const genesis::GenesisWorld *genesis_world = nullptr;
    const vienna::ViennaWorld   *vienna_world = nullptr;
    const wicked::WickedWorld   *wicked_world = nullptr;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_SPATIAL_QUERY_MANAGER_H
// File 463: modules/integration/unified_raycast_treesearch.h
// High‑performance ray‑cast engine built on TreeNSearch.  Rebuilds a
// point‑set BVH from the expanded centroids of all active rigid body
// AABBs across all registered engines every frame.  Performs single‑ray
// and batched parallel ray‑AABB intersections, returning sorted hit lists
// with engine and body IDs.  Uses Gaia's fast ray‑AABB test and the
// log‑linear BVH traversal for O(log n) candidate pruning.

#ifndef INTEGRATION_UNIFIED_RAYCAST_TREESEARCH_H
#define INTEGRATION_UNIFIED_RAYCAST_TREESEARCH_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

// TreeNSearch point‑set BVH
#include "../../treesearch/point_set_search.h"

// Gaia ray‑AABB test (very fast)
#include "../../gaia/src/bvh/query.h"

// Gaia parallelisation
#include "../../gaia/src/parallelization/cpu_parallelization.h"

// Forward engine world types (for collecting AABBs)
namespace newton   { class NewtonWorld; class NewtonBody; }
namespace genesis  { class GenesisWorld; class BaseEntity; class RigidEntity; }
namespace vienna   { class ViennaWorld; class ViennaBody; }
namespace wicked   { class WickedWorld; class WickedBody; }

namespace unified {

class UnifiedRaycastTreeSearch : public RefCounted {
    GDCLASS(UnifiedRaycastTreeSearch, RefCounted);

public:
    // Engine index constants.
    static constexpr int ENGINE_NEWTON  = 0;
    static constexpr int ENGINE_GENESIS = 1;
    static constexpr int ENGINE_VIENNA  = 2;
    static constexpr int ENGINE_WICKED  = 3;

    // -------------------------------------------------------------------
    // Rebuild internal structures from the given engine worlds.
    // Must be called once per frame before ray‑casting.
    // -------------------------------------------------------------------
    void rebuild(const HashMap<int, void *> &p_worlds);

    // -------------------------------------------------------------------
    // Cast a single ray and return all hits sorted by distance.
    // `p_origin` and `p_direction` are in world space.  `p_max_dist`
    // can be used to limit the search range.
    // -------------------------------------------------------------------
    struct RayHit {
        int      engine;
        uint64_t body_id;
        Vector3  point;          // world hit point (AABB entry)
        Vector3  normal;         // approximate (body centroid normal)
        real_t   distance;
    };
    void ray_cast(const Vector3 &p_origin, const Vector3 &p_direction,
                  real_t p_max_dist, LocalVector<RayHit> &r_hits) const;

    // -------------------------------------------------------------------
    // Batched ray casts (parallel dispatch).  The results vector is
    // resized to match the number of rays.
    // -------------------------------------------------------------------
    void ray_cast_batch(const LocalVector<Vector3> &p_origins,
                        const LocalVector<Vector3> &p_directions,
                        real_t p_max_dist,
                        LocalVector<LocalVector<RayHit>> &r_all_hits) const;

    // -------------------------------------------------------------------
    // Return the number of bodies currently stored.
    // -------------------------------------------------------------------
    int get_body_count() const { return aabbs.size(); }

protected:
    static void _bind_methods();

private:
    // Per‑body data: world AABB, engine, and body ID.
    struct BodyEntry {
        AABB     aabb;
        int      engine;
        uint64_t body_id;
    };
    LocalVector<BodyEntry> entries;

    // Expanded AABB centroids for the BVH.
    LocalVector<Vector3> centroids;

    // Underlying TreeNSearch point‑set BVH.
    treesearch::PointSetSearch bvh;

    // -------------------------------------------------------------------
    // Collect body data from a specific engine world.
    // -------------------------------------------------------------------
    void collect_from_world(int p_engine, void *p_world);

    // -------------------------------------------------------------------
    // Approximate normal from the closest face of the AABB to the hit point.
    // -------------------------------------------------------------------
    static Vector3 aabb_normal(const AABB &p_box, const Vector3 &p_hit_point);

    // -------------------------------------------------------------------
    // Accessor for the centroid array used by TreeNSearch.
    // -------------------------------------------------------------------
    struct CentroidAccessor { const LocalVector<Vector3> *c; VL_ACCESSOR(c) Vector3 operator()(int i) const { return (*c)[i]; } };

    // Maximum AABB diagonal half‑length for radius search.
    real_t max_half_diag = 0.0f;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedRaycastTreeSearch::_bind_methods() {
    ClassDB::bind_method(D_METHOD("rebuild", "worlds"), &UnifiedRaycastTreeSearch::rebuild);
    ClassDB::bind_method(D_METHOD("ray_cast", "origin", "direction", "max_dist"), &UnifiedRaycastTreeSearch::ray_cast);
    ClassDB::bind_method(D_METHOD("ray_cast_batch", "origins", "directions", "max_dist"), &UnifiedRaycastTreeSearch::ray_cast_batch);
    ClassDB::bind_method(D_METHOD("get_body_count"), &UnifiedRaycastTreeSearch::get_body_count);
}

// ---------------------------------------------------------------------------
// Rebuild: traverse all engine worlds and fill centroids + AABBs.
// ---------------------------------------------------------------------------
void UnifiedRaycastTreeSearch::rebuild(const HashMap<int, void *> &p_worlds) {
    entries.clear();
    centroids.clear();

    for (const KeyValue<int, void *> &kv : p_worlds) {
        collect_from_world(kv.key, kv.value);
    }

    // Determine maximum half‑diagonal for later radius searches.
    max_half_diag = 0.0f;
    for (const BodyEntry &e : entries) {
        real_t d = e.aabb.get_longest_axis_size();
        if (d > max_half_diag) max_half_diag = d;
    }

    // Build the centroid BVH.
    centroids.resize(entries.size());
    for (int i = 0; i < entries.size(); ++i) {
        centroids[i] = entries[i].aabb.get_center();
    }
    bvh.build(centroids);
}

// ---------------------------------------------------------------------------
// Collect bodies from a single engine world.
// ---------------------------------------------------------------------------
void UnifiedRaycastTreeSearch::collect_from_world(int p_engine, void *p_world) {
    if (!p_world) return;

    switch (p_engine) {
        case ENGINE_NEWTON: {
            auto *nw = static_cast<newton::NewtonWorld *>(p_world);
            LocalVector<newton::body_id> ids = nw->get_body_ids();
            for (newton::body_id id : ids) {
                Ref<newton::NewtonBody> body = nw->get_body(id);
                if (body.is_valid() && body->is_active()) {
                    BodyEntry e;
                    e.aabb    = body->get_aabb();
                    e.engine  = ENGINE_NEWTON;
                    e.body_id = id;
                    entries.push_back(e);
                }
            }
        } break;
        case ENGINE_GENESIS: {
            auto *gw = static_cast<genesis::GenesisWorld *>(p_world);
            LocalVector<genesis::entity_id_t> uids = gw->get_all_entity_uids();
            for (genesis::entity_id_t uid : uids) {
                Ref<genesis::BaseEntity> ent = gw->get_entity(uid);
                if (ent.is_valid() && ent->is_active()) {
                    BodyEntry e;
                    e.aabb    = ent->get_aabb();
                    e.engine  = ENGINE_GENESIS;
                    e.body_id = uid;
                    entries.push_back(e);
                }
            }
        } break;
        case ENGINE_VIENNA: {
            auto *vw = static_cast<vienna::ViennaWorld *>(p_world);
            LocalVector<vienna::body_id> ids = vw->get_body_ids();
            for (vienna::body_id id : ids) {
                Ref<vienna::ViennaBody> body = vw->get_body(id);
                if (body.is_valid() && body->is_active()) {
                    BodyEntry e;
                    e.aabb    = body->get_aabb();
                    e.engine  = ENGINE_VIENNA;
                    e.body_id = id;
                    entries.push_back(e);
                }
            }
        } break;
        case ENGINE_WICKED: {
            auto *ww = static_cast<wicked::WickedWorld *>(p_world);
            LocalVector<wicked::body_id> ids = ww->get_body_ids();
            for (wicked::body_id id : ids) {
                Ref<wicked::WickedBody> body = ww->get_body(id);
                if (body.is_valid() && body->get_activation_state() == wicked::ActivationState::ACTIVE_TAG) {
                    BodyEntry e;
                    e.aabb    = body->get_aabb();
                    e.engine  = ENGINE_WICKED;
                    e.body_id = id;
                    entries.push_back(e);
                }
            }
        } break;
    }
}

// ---------------------------------------------------------------------------
// Single ray cast.
// ---------------------------------------------------------------------------
void UnifiedRaycastTreeSearch::ray_cast(const Vector3 &p_origin,
                                        const Vector3 &p_direction,
                                        real_t p_max_dist,
                                        LocalVector<RayHit> &r_hits) const {
    r_hits.clear();
    if (entries.is_empty()) return;

    Vector3 dir = p_direction.normalized();
    // Use a radius search around the ray's midpoint to cull candidate AABBs.
    // The search radius = half diagonal of largest AABB + ray length / 2.
    // We'll query the BVH at the ray midpoint, with a generous radius.
    // A simpler approach: query the BVH at several sample points along the ray
    // (e.g., every meter) and union the results.  But radius search with a
    // large enough radius is fine.
    Vector3 mid = p_origin + dir * (p_max_dist * 0.5f);
    real_t search_radius = max_half_diag + p_max_dist * 0.5f;

    LocalVector<int> candidates;
    bvh.radius(mid, search_radius, candidates);

    // Test each candidate AABB for ray intersection.
    for (int idx : candidates) {
        if (idx < 0 || idx >= entries.size()) continue;
        const AABB &box = entries[idx].aabb;
        real_t t_entry, t_exit;
        if (gaia::bvh::intersect_ray_aabb(p_origin, dir, box, 0.0f, p_max_dist, t_entry, t_exit)) {
            RayHit hit;
            hit.engine   = entries[idx].engine;
            hit.body_id  = entries[idx].body_id;
            hit.point    = p_origin + dir * t_entry;
            hit.normal   = aabb_normal(box, hit.point);
            hit.distance = t_entry;
            r_hits.push_back(hit);
        }
    }

    // Sort hits by distance.
    std::sort(r_hits.begin(), r_hits.end(),
              [](const RayHit &a, const RayHit &b) { return a.distance < b.distance; });
}

// ---------------------------------------------------------------------------
// Batched parallel ray casts.
// ---------------------------------------------------------------------------
void UnifiedRaycastTreeSearch::ray_cast_batch(
        const LocalVector<Vector3> &p_origins,
        const LocalVector<Vector3> &p_directions,
        real_t p_max_dist,
        LocalVector<LocalVector<RayHit>> &r_all_hits) const {

    int n = p_origins.size();
    r_all_hits.resize(n);

    gaia::parallel::CPUParallelization::parallel_for(n,
        [this, &p_origins, &p_directions, p_max_dist, &r_all_hits](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                ray_cast(p_origins[i], p_directions[i], p_max_dist, r_all_hits[i]);
            }
        },
        256);
}

// ---------------------------------------------------------------------------
// Approximate AABB normal: find which face was hit.
// ---------------------------------------------------------------------------
Vector3 UnifiedRaycastTreeSearch::aabb_normal(const AABB &p_box, const Vector3 &p_hit_point) {
    Vector3 min = p_box.position;
    Vector3 max = p_box.position + p_box.size;
    Vector3 center = p_box.get_center();
    Vector3 d = p_hit_point - center;
    real_t best_dist = INFINITY;
    Vector3 best_normal(0, 1, 0);
    for (int i = 0; i < 3; ++i) {
        real_t dist = Math::abs(max[i] - p_hit_point[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_normal = Vector3();
            best_normal[i] = 1.0f;
        }
        dist = Math::abs(min[i] - p_hit_point[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_normal = Vector3();
            best_normal[i] = -1.0f;
        }
    }
    return best_normal;
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_RAYCAST_TREESEARCH_H
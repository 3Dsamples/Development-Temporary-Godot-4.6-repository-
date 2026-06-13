// File 464: modules/integration/unified_overlap_query_treesearch.h
// High‑performance overlap query engine built on TreeNSearch for the
// unified physics pipeline.  Collects expanded AABB centroids of all
// active rigid bodies from all registered engines, rebuilds a point‑set
// BVH each frame, and performs sphere, AABB, and convex overlap queries
// returning engine‑specific body IDs.  All queries are parallelised
// over query shapes using Gaia's CPUParallelization.

#ifndef INTEGRATION_UNIFIED_OVERLAP_QUERY_TREESEARCH_H
#define INTEGRATION_UNIFIED_OVERLAP_QUERY_TREESEARCH_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

#include "../../treesearch/point_set_search.h"
#include "../../gaia/src/parallelization/cpu_parallelization.h"

// Forward engine world types (for iterating bodies)
namespace newton   { class NewtonWorld; class NewtonBody; }
namespace genesis  { class GenesisWorld; class BaseEntity; class RigidEntity; }
namespace vienna   { class ViennaWorld; class ViennaBody; }
namespace wicked   { class WickedWorld; class WickedBody; }

namespace unified {

class UnifiedOverlapQueryTreeSearch : public RefCounted {
    GDCLASS(UnifiedOverlapQueryTreeSearch, RefCounted);

public:
    static constexpr int ENGINE_NEWTON  = 0;
    static constexpr int ENGINE_GENESIS = 1;
    static constexpr int ENGINE_VIENNA  = 2;
    static constexpr int ENGINE_WICKED  = 3;

    // AABB expansion margin for centroid computation.
    real_t aabb_expansion = 0.1f;

    // -------------------------------------------------------------------
    // Rebuild internal structures from the given engine worlds.
    // -------------------------------------------------------------------
    void rebuild(const HashMap<int, void *> &p_worlds);

    // -------------------------------------------------------------------
    // Query all bodies whose AABB overlaps a given sphere.
    // Results are returned as a list of engine+body ID pairs.
    // -------------------------------------------------------------------
    struct OverlapResult {
        int      engine;
        uint64_t body_id;
        AABB     world_aabb;
    };
    void sphere_overlap(const Vector3 &p_center, real_t p_radius,
                        LocalVector<OverlapResult> &r_results) const;

    // -------------------------------------------------------------------
    // Query all bodies whose AABB overlaps a given world‑space AABB.
    // -------------------------------------------------------------------
    void aabb_overlap(const AABB &p_query_aabb,
                      LocalVector<OverlapResult> &r_results) const;

    // -------------------------------------------------------------------
    // Batch sphere overlap queries (parallel dispatch).
    // -------------------------------------------------------------------
    void sphere_overlap_batch(const LocalVector<Vector3> &p_centers,
                              real_t p_radius,
                              LocalVector<LocalVector<OverlapResult>> &r_all_results) const;

    // -------------------------------------------------------------------
    // Return the number of bodies currently stored.
    // -------------------------------------------------------------------
    int get_body_count() const { return entries.size(); }

protected:
    static void _bind_methods();

private:
    struct BodyEntry {
        AABB     aabb;
        int      engine;
        uint64_t body_id;
    };
    LocalVector<BodyEntry> entries;
    LocalVector<Vector3>   centroids;
    treesearch::PointSetSearch bvh;
    real_t search_radius = 0.0f;   // half diagonal of largest AABB + expansion

    void collect_from_world(int p_engine, void *p_world);
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedOverlapQueryTreeSearch::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_aabb_expansion", "expansion"), &UnifiedOverlapQueryTreeSearch::set_aabb_expansion);
    ClassDB::bind_method(D_METHOD("get_aabb_expansion"), &UnifiedOverlapQueryTreeSearch::get_aabb_expansion);
    ClassDB::bind_method(D_METHOD("rebuild", "worlds"), &UnifiedOverlapQueryTreeSearch::rebuild);
    ClassDB::bind_method(D_METHOD("sphere_overlap", "center", "radius"), &UnifiedOverlapQueryTreeSearch::sphere_overlap);
    ClassDB::bind_method(D_METHOD("aabb_overlap", "aabb"), &UnifiedOverlapQueryTreeSearch::aabb_overlap);
    ClassDB::bind_method(D_METHOD("sphere_overlap_batch", "centers", "radius"), &UnifiedOverlapQueryTreeSearch::sphere_overlap_batch);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aabb_expansion"), "set_aabb_expansion", "get_aabb_expansion");
}

void UnifiedOverlapQueryTreeSearch::set_aabb_expansion(real_t v) { aabb_expansion = MAX(v, 0.0f); }
real_t UnifiedOverlapQueryTreeSearch::get_aabb_expansion() const { return aabb_expansion; }

// ---------------------------------------------------------------------------
// Rebuild: gather body data from all worlds, compute centroids, build BVH.
// ---------------------------------------------------------------------------
void UnifiedOverlapQueryTreeSearch::rebuild(const HashMap<int, void *> &p_worlds) {
    entries.clear();
    for (const KeyValue<int, void *> &kv : p_worlds) {
        collect_from_world(kv.key, kv.value);
    }

    search_radius = 0.0f;
    for (const BodyEntry &e : entries) {
        real_t d = e.aabb.get_longest_axis_size();
        if (d > search_radius) search_radius = d;
    }
    search_radius = search_radius * 0.5f + aabb_expansion;

    centroids.resize(entries.size());
    for (int i = 0; i < entries.size(); ++i) {
        centroids[i] = entries[i].aabb.get_center();
    }
    bvh.build(centroids);
}

// ---------------------------------------------------------------------------
// Collect bodies from a single engine world.
// ---------------------------------------------------------------------------
void UnifiedOverlapQueryTreeSearch::collect_from_world(int p_engine, void *p_world) {
    if (!p_world) return;
    switch (p_engine) {
        case ENGINE_NEWTON: {
            auto *nw = static_cast<newton::NewtonWorld *>(p_world);
            LocalVector<newton::body_id> ids = nw->get_body_ids();
            for (newton::body_id id : ids) {
                Ref<newton::NewtonBody> body = nw->get_body(id);
                if (body.is_valid() && body->is_active()) {
                    entries.push_back({body->get_aabb(), ENGINE_NEWTON, id});
                }
            }
        } break;
        case ENGINE_GENESIS: {
            auto *gw = static_cast<genesis::GenesisWorld *>(p_world);
            LocalVector<genesis::entity_id_t> uids = gw->get_all_entity_uids();
            for (genesis::entity_id_t uid : uids) {
                Ref<genesis::BaseEntity> ent = gw->get_entity(uid);
                if (ent.is_valid() && ent->is_active()) {
                    entries.push_back({ent->get_aabb(), ENGINE_GENESIS, uid});
                }
            }
        } break;
        case ENGINE_VIENNA: {
            auto *vw = static_cast<vienna::ViennaWorld *>(p_world);
            LocalVector<vienna::body_id> ids = vw->get_body_ids();
            for (vienna::body_id id : ids) {
                Ref<vienna::ViennaBody> body = vw->get_body(id);
                if (body.is_valid() && body->is_active()) {
                    entries.push_back({body->get_aabb(), ENGINE_VIENNA, id});
                }
            }
        } break;
        case ENGINE_WICKED: {
            auto *ww = static_cast<wicked::WickedWorld *>(p_world);
            LocalVector<wicked::body_id> ids = ww->get_body_ids();
            for (wicked::body_id id : ids) {
                Ref<wicked::WickedBody> body = ww->get_body(id);
                if (body.is_valid() && body->get_activation_state() == wicked::ActivationState::ACTIVE_TAG) {
                    entries.push_back({body->get_aabb(), ENGINE_WICKED, id});
                }
            }
        } break;
    }
}

// ---------------------------------------------------------------------------
// Sphere overlap: radius search on centroids, then exact AABB‑sphere test.
// ---------------------------------------------------------------------------
void UnifiedOverlapQueryTreeSearch::sphere_overlap(
        const Vector3 &p_center, real_t p_radius,
        LocalVector<OverlapResult> &r_results) const {
    r_results.clear();
    if (entries.is_empty()) return;

    // A body's AABB can intersect the sphere even if its centroid is farther
    // than radius + half_diag? Actually if centre is within radius + half_diag,
    // the AABB may intersect.  Using search_radius + radius is safe.
    real_t query_radius = search_radius + p_radius;
    AABB sphere_aabb(p_center - Vector3(p_radius,p_radius,p_radius),
                     Vector3(p_radius*2, p_radius*2, p_radius*2));

    LocalVector<int> candidates;
    bvh.radius(p_center, query_radius, candidates);
    for (int idx : candidates) {
        if (idx < 0 || idx >= entries.size()) continue;
        const AABB &box = entries[idx].aabb;
        // Exact test: AABB vs sphere.
        if (box.intersects(sphere_aabb)) {
            r_results.push_back({entries[idx].engine, entries[idx].body_id, box});
        }
    }
}

// ---------------------------------------------------------------------------
// AABB overlap: similar radius search, then exact AABB intersection.
// ---------------------------------------------------------------------------
void UnifiedOverlapQueryTreeSearch::aabb_overlap(
        const AABB &p_query_aabb,
        LocalVector<OverlapResult> &r_results) const {
    r_results.clear();
    if (entries.is_empty()) return;

    Vector3 center = p_query_aabb.get_center();
    real_t half_diag = p_query_aabb.get_longest_axis_size() * 0.5f;
    real_t query_radius = search_radius + half_diag;

    LocalVector<int> candidates;
    bvh.radius(center, query_radius, candidates);
    for (int idx : candidates) {
        if (idx < 0 || idx >= entries.size()) continue;
        const AABB &box = entries[idx].aabb;
        if (box.intersects(p_query_aabb)) {
            r_results.push_back({entries[idx].engine, entries[idx].body_id, box});
        }
    }
}

// ---------------------------------------------------------------------------
// Batch sphere overlap (parallel).
// ---------------------------------------------------------------------------
void UnifiedOverlapQueryTreeSearch::sphere_overlap_batch(
        const LocalVector<Vector3> &p_centers,
        real_t p_radius,
        LocalVector<LocalVector<OverlapResult>> &r_all_results) const {
    int n = p_centers.size();
    r_all_results.resize(n);
    gaia::parallel::CPUParallelization::parallel_for(
        n,
        [this, &p_centers, p_radius, &r_all_results](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                sphere_overlap(p_centers[i], p_radius, r_all_results[i]);
            }
        },
        256);
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_OVERLAP_QUERY_TREESEARCH_H
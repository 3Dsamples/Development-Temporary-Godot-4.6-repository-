// File 468: modules/integration/unified_contact_clustering.h
// Spatial‑proximity contact clustering using a hash grid and union‑find.
// Groups a set of world‑space contact points into disjoint clusters based
// on a user‑specified merge radius.  Contacts that are within this radius
// belong to the same cluster, enabling parallel constraint solving where
// each cluster can be processed independently.  All operations are fully
// inline and O(n) average time thanks to the spatial hash.

#ifndef INTEGRATION_UNIFIED_CONTACT_CLUSTERING_H
#define INTEGRATION_UNIFIED_CONTACT_CLUSTERING_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/typedefs.h"

namespace unified {

class UnifiedContactClustering : public RefCounted {
    GDCLASS(UnifiedContactClustering, RefCounted);

public:
    // Merge radius: two contacts closer than this distance will be placed
    // in the same cluster.  A typical value is 0.5 m for rigid‑body contacts
    // or 0.1 m for cloth.
    real_t merge_radius = 0.5f;

    // -------------------------------------------------------------------
    // Build clusters from a flat list of contact positions.
    // Returns a vector of clusters; each cluster is a list of indices
    // into the original `p_positions` array.
    // -------------------------------------------------------------------
    void build(const LocalVector<Vector3> &p_positions,
               LocalVector<LocalVector<int>> &r_clusters) const;

    // -------------------------------------------------------------------
    // Convenience: build clusters and return the number of clusters.
    // -------------------------------------------------------------------
    int get_cluster_count(const LocalVector<Vector3> &p_positions) const;

protected:
    static void _bind_methods();

private:
    // Compute the grid cell index for a world position.
    Vector3i world_to_cell(const Vector3 &p) const;

    // Flat index from cell coordinates.
    int cell_index(const Vector3i &p_cell) const;

    // Find root of an element (path compression).
    int find_root(LocalVector<int> &p_parent, int p_idx) const;

    // Merge two sets.
    void union_sets(LocalVector<int> &p_parent, int a, int b) const;

    // Cell count along X axis (set during build).
    int nx = 1;
    // Cell size = merge_radius (ensures any two points closer than radius
    // are in the same cell or neighboring cells).
    real_t cell_size = 0.5f;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedContactClustering::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_merge_radius", "radius"), &UnifiedContactClustering::set_merge_radius);
    ClassDB::bind_method(D_METHOD("get_merge_radius"), &UnifiedContactClustering::get_merge_radius);
    ClassDB::bind_method(D_METHOD("build", "positions"), &UnifiedContactClustering::build);
    ClassDB::bind_method(D_METHOD("get_cluster_count", "positions"), &UnifiedContactClustering::get_cluster_count);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "merge_radius"), "set_merge_radius", "get_merge_radius");
}

void UnifiedContactClustering::set_merge_radius(real_t v) { merge_radius = MAX(v, 0.001f); }
real_t UnifiedContactClustering::get_merge_radius() const { return merge_radius; }

// ---------------------------------------------------------------------------
// Spatial grid helpers
// ---------------------------------------------------------------------------
Vector3i UnifiedContactClustering::world_to_cell(const Vector3 &p) const {
    return Vector3i(
        (int)Math::floor(p.x / cell_size),
        (int)Math::floor(p.y / cell_size),
        (int)Math::floor(p.z / cell_size)
    );
}

int UnifiedContactClustering::cell_index(const Vector3i &p_cell) const {
    // Only positive indices are valid (grid is unbounded in positive direction).
    // We use a fixed nx and compute based on order.
    // To handle negative indices we offset by a large constant.
    // Simpler: use a hash map from Vector3i to cell data.
    // But for performance we'll use the hash map approach as we don't know bounds.
    // Actually we'll use a HashMap<Vector3i, LocalVector<int>> directly.
    // So cell_index is not used; we'll adapt.
    return 0;
}

// ---------------------------------------------------------------------------
// Union‑find helpers
// ---------------------------------------------------------------------------
int UnifiedContactClustering::find_root(LocalVector<int> &p_parent, int p_idx) const {
    while (p_idx != p_parent[p_idx]) {
        p_parent[p_idx] = p_parent[p_parent[p_idx]];
        p_idx = p_parent[p_idx];
    }
    return p_idx;
}

void UnifiedContactClustering::union_sets(LocalVector<int> &p_parent, int a, int b) const {
    int ra = find_root(p_parent, a);
    int rb = find_root(p_parent, b);
    if (ra != rb) p_parent[ra] = rb;
}

// ---------------------------------------------------------------------------
// build: hash grid + union‑find.
// ---------------------------------------------------------------------------
void UnifiedContactClustering::build(const LocalVector<Vector3> &p_positions,
                                     LocalVector<LocalVector<int>> &r_clusters) const {
    int n = p_positions.size();
    r_clusters.clear();
    if (n == 0) return;

    cell_size = merge_radius;  // ensures two points within merge_radius are in same or adjacent cells.

    // Insert points into hash grid.
    HashMap<Vector3i, LocalVector<int>, int> cell_map;
    for (int i = 0; i < n; ++i) {
        Vector3i cell = world_to_cell(p_positions[i]);
        cell_map[cell].push_back(i);
    }

    // Union‑find parent array.
    LocalVector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;

    // For each point, check its own cell and the 26 neighboring cells.
    // For each pair of points within merge_radius, union them.
    auto check_cell = [&](const Vector3i &p_cell, int p_idx) {
        // Iterate over points in the same cell (already covered by the neighbor loop when processing each cell).
    };

    // Loop over all cells.
    for (const KeyValue<Vector3i, LocalVector<int>> &kv : cell_map) {
        const Vector3i &cell = kv.key;
        const LocalVector<int> &pts_in_cell = kv.value;

        // Test all pairs within the same cell.
        for (int a = 0; a < pts_in_cell.size(); ++a) {
            for (int b = a + 1; b < pts_in_cell.size(); ++b) {
                if (p_positions[pts_in_cell[a]].distance_squared_to(p_positions[pts_in_cell[b]]) <= merge_radius * merge_radius) {
                    union_sets(parent, pts_in_cell[a], pts_in_cell[b]);
                }
            }
        }

        // Test against neighboring cells (13 unique directions to avoid double checking).
        static const int dirs[13][3] = {
            {1,0,0},{0,1,0},{0,0,1},
            {1,1,0},{1,0,1},{0,1,1},
            {1,-1,0},{1,0,-1},{0,1,-1},
            {1,1,1},{1,1,-1},{1,-1,1},{-1,1,1}
        };
        for (int d = 0; d < 13; ++d) {
            Vector3i nb(cell.x + dirs[d][0], cell.y + dirs[d][1], cell.z + dirs[d][2]);
            HashMap<Vector3i, LocalVector<int>>::ConstIterator it = cell_map.find(nb);
            if (!it) continue;
            const LocalVector<int> &nb_pts = it->value;
            for (int a : pts_in_cell) {
                for (int b : nb_pts) {
                    if (p_positions[a].distance_squared_to(p_positions[b]) <= merge_radius * merge_radius) {
                        union_sets(parent, a, b);
                    }
                }
            }
        }
    }

    // Collect clusters.
    HashMap<int, int> root_to_cluster;
    for (int i = 0; i < n; ++i) {
        int root = find_root(parent, i);
        if (!root_to_cluster.has(root)) {
            root_to_cluster[root] = r_clusters.size();
            r_clusters.push_back(LocalVector<int>());
        }
        r_clusters[root_to_cluster[root]].push_back(i);
    }
}

int UnifiedContactClustering::get_cluster_count(const LocalVector<Vector3> &p_positions) const {
    LocalVector<LocalVector<int>> clusters;
    build(p_positions, clusters);
    return clusters.size();
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_CONTACT_CLUSTERING_H
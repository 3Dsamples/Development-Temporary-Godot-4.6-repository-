// File 458: modules/integration/unified_soft_self_collision_detector.h
// Detects vertex‑face self‑collisions within a single tetrahedral mesh.
// Builds a TreeNSearch BVH over the centroids of boundary faces (surface
// triangles) and performs radius queries per vertex to quickly find
// candidate faces.  Exact closest‑point distances are then computed.
// All detection is fully inlined and parallelised over vertices using
// Gaia's CPUParallelization.  Output contacts are ready for the unified
// penalty or IPC solvers.

#ifndef INTEGRATION_UNIFIED_SOFT_SELF_COLLISION_DETECTOR_H
#define INTEGRATION_UNIFIED_SOFT_SELF_COLLISION_DETECTOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

#include "../../treesearch/point_set_search.h"
#include "../../gaia/src/mesh/tet_mesh.h"
#include "../../gaia/src/parallelization/cpu_parallelization.h"

namespace unified {

class UnifiedSoftSelfCollisionDetector : public RefCounted {
    GDCLASS(UnifiedSoftSelfCollisionDetector, RefCounted);

public:
    // Collision margin: distance below which a contact is generated.
    real_t collision_margin = 0.01f;

    // Maximum number of candidate faces per vertex (KNN search).
    int max_search_faces = 16;

    // Contact descriptor for self‑collision.
    struct SelfContact {
        int     vertex_idx;           // index of the soft vertex
        int     face_v0;              // three vertex indices of the face
        int     face_v1;
        int     face_v2;
        Vector3 point_on_face;        // closest point on the face
        Vector3 normal;               // from face to vertex
        real_t  distance;             // separation (negative if vertex is inside? positive = gap)
    };

    // -------------------------------------------------------------------
    // Rebuild internal data structures from the current mesh.
    // Must be called once per frame after vertex positions are updated.
    // -------------------------------------------------------------------
    void rebuild(const gaia::mesh::TetMesh &p_mesh);

    // -------------------------------------------------------------------
    // Run self‑collision detection over all vertices.  The mesh must have
    // been previously rebuilt via rebuild().
    //
    // @param r_contacts  Output: per vertex, a list of self‑contacts.
    // -------------------------------------------------------------------
    void detect_collisions(
            LocalVector<LocalVector<SelfContact>> &r_contacts) const;

    // -------------------------------------------------------------------
    // Return the number of boundary faces stored.
    // -------------------------------------------------------------------
    int get_boundary_face_count() const { return boundary_faces.size(); }

protected:
    static void _bind_methods();

private:
    // Position accessor for the TreeNSearch BVH (face centroids -> index).
    struct FaceAccessor {
        const LocalVector<Vector3> *centroids;
        FaceAccessor(const LocalVector<Vector3> *p) : centroids(p) {}
        Vector3 operator()(int idx) const { return (*centroids)[idx]; }
    };

    // Triangular face (boundary) with its pre‑computed AABB (optional, not stored).
    struct BoundaryFace {
        int v0, v1, v2;
        Vector3 centroid;          // pre‑computed for BVH
    };

    // Current mesh pointer (non‑owning).
    const gaia::mesh::TetMesh *mesh = nullptr;

    // Boundary faces extracted from the mesh.
    LocalVector<BoundaryFace> boundary_faces;
    // Their centroids (parallel array for the TreeNSearch BVH).
    LocalVector<Vector3> face_centroids;

    // TreeNSearch point‑set BVH built over face centroids.
    treesearch::PointSetSearch face_bvh;

    // Pre‑computed vertex positions (copy for fast access during parallel loops).
    LocalVector<Vector3> vertex_positions;

    // -------------------------------------------------------------------
    // Single vertex‑face distance test.
    // -------------------------------------------------------------------
    SelfContact test_vertex_against_face(int p_vertex_idx,
                                         const BoundaryFace &p_face) const;

    // -------------------------------------------------------------------
    // Closest point on triangle.
    // -------------------------------------------------------------------
    static Vector3 closest_point_on_triangle(const Vector3 &p,
                                             const Vector3 &a, const Vector3 &b, const Vector3 &c,
                                             real_t *r_u = nullptr, real_t *r_v = nullptr);
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedSoftSelfCollisionDetector::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_collision_margin", "margin"), &UnifiedSoftSelfCollisionDetector::set_collision_margin);
    ClassDB::bind_method(D_METHOD("get_collision_margin"), &UnifiedSoftSelfCollisionDetector::get_collision_margin);
    ClassDB::bind_method(D_METHOD("rebuild", "mesh"), &UnifiedSoftSelfCollisionDetector::rebuild);
    ClassDB::bind_method(D_METHOD("detect_collisions"), &UnifiedSoftSelfCollisionDetector::detect_collisions);
    ClassDB::bind_method(D_METHOD("get_boundary_face_count"), &UnifiedSoftSelfCollisionDetector::get_boundary_face_count);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_margin"), "set_collision_margin", "get_collision_margin");
}

void UnifiedSoftSelfCollisionDetector::set_collision_margin(real_t v) { collision_margin = MAX(v, 0.0f); }
real_t UnifiedSoftSelfCollisionDetector::get_collision_margin() const { return collision_margin; }

// ---------------------------------------------------------------------------
// Rebuild: extract boundary faces and build face‑centroid BVH.
// ---------------------------------------------------------------------------
void UnifiedSoftSelfCollisionDetector::rebuild(const gaia::mesh::TetMesh &p_mesh) {
    mesh = &p_mesh;

    // Copy vertex positions for fast access during detection.
    int nv = p_mesh.vertex_count();
    vertex_positions.resize(nv);
    for (int i = 0; i < nv; ++i) {
        vertex_positions[i] = p_mesh.get_vertex(i);
    }

    // Identify boundary faces: faces that appear exactly once.
    // A tetrahedron face is defined by a sorted triplet of vertex indices.
    struct FaceKey {
        int v0, v1, v2;
        FaceKey(int a, int b, int c) {
            // Sort the three indices.
            if (a > b) SWAP(a, b);
            if (b > c) SWAP(b, c);
            if (a > b) SWAP(a, b);
            v0 = a; v1 = b; v2 = c;
        }
        bool operator==(const FaceKey &o) const { return v0==o.v0 && v1==o.v1 && v2==o.v2; }
        struct Hash {
            uint32_t operator()(const FaceKey &k) const {
                return (uint32_t(k.v0)*73856093)^(uint32_t(k.v1)*19349663)^(uint32_t(k.v2)*83492791);
            }
        };
    };

    HashMap<FaceKey, int, FaceKey::Hash> face_count;
    int nt = p_mesh.element_count();
    for (int t = 0; t < nt; ++t) {
        auto tet = p_mesh.get_tetrahedron(t);
        int v[4] = {tet.v0, tet.v1, tet.v2, tet.v3};
        // Four faces.
        for (int i = 0; i < 4; ++i) {
            FaceKey key(v[(i+1)%4], v[(i+2)%4], v[(i+3)%4]);
            face_count[key]++;
        }
    }

    // Collect boundary faces (count == 1).
    boundary_faces.clear();
    face_centroids.clear();
    for (const KeyValue<FaceKey, int> &kv : face_count) {
        if (kv.value == 1) {
            BoundaryFace bf;
            bf.v0 = kv.key.v0;
            bf.v1 = kv.key.v1;
            bf.v2 = kv.key.v2;
            const Vector3 &p0 = vertex_positions[bf.v0];
            const Vector3 &p1 = vertex_positions[bf.v1];
            const Vector3 &p2 = vertex_positions[bf.v2];
            bf.centroid = (p0 + p1 + p2) / 3.0f;
            boundary_faces.push_back(bf);
            face_centroids.push_back(bf.centroid);
        }
    }

    // Build the face‑centroid BVH.
    face_bvh.build(face_centroids);
}

// ---------------------------------------------------------------------------
// Detect self‑collisions: for each vertex, find nearby faces and test.
// ---------------------------------------------------------------------------
void UnifiedSoftSelfCollisionDetector::detect_collisions(
        LocalVector<LocalVector<SelfContact>> &r_contacts) const {

    int nv = vertex_positions.size();
    r_contacts.resize(nv);
    for (int i = 0; i < nv; ++i) r_contacts[i].clear();

    if (boundary_faces.is_empty()) return;

    // Parallel loop over vertices.
    gaia::parallel::CPUParallelization::parallel_for(nv,
        [this, &r_contacts](int64_t start, int64_t end) {
            for (int64_t vi = start; vi < end; ++vi) {
                const Vector3 &vertex = vertex_positions[vi];
                // Find the nearest face centroids via KNN search.
                LocalVector<treesearch::KnnSearch<FaceAccessor>::Result> knn;
                FaceAccessor accessor(&face_centroids);
                treesearch::KnnSearch<FaceAccessor>::search(
                    face_bvh.nodes, accessor, vertex,
                    max_search_faces, knn);

                // For each candidate face, test distance.
                for (const auto &res : knn) {
                    int face_idx = res.index;
                    if (face_idx < 0 || face_idx >= boundary_faces.size()) continue;
                    const BoundaryFace &face = boundary_faces[face_idx];
                    // Skip faces that include the vertex itself.
                    if (face.v0 == vi || face.v1 == vi || face.v2 == vi) continue;

                    SelfContact sc = test_vertex_against_face(vi, face);
                    if (sc.distance <= collision_margin) {
                        r_contacts[vi].push_back(sc);
                    }
                }
            }
        },
        256); // min batch size 256 vertices per thread
}

// ---------------------------------------------------------------------------
// Single vertex‑face test.
// ---------------------------------------------------------------------------
UnifiedSoftSelfCollisionDetector::SelfContact
UnifiedSoftSelfCollisionDetector::test_vertex_against_face(
        int p_vertex_idx,
        const BoundaryFace &p_face) const {

    SelfContact result;
    result.vertex_idx = p_vertex_idx;
    result.face_v0 = p_face.v0;
    result.face_v1 = p_face.v1;
    result.face_v2 = p_face.v2;

    const Vector3 &vert = vertex_positions[p_vertex_idx];
    const Vector3 &v0 = vertex_positions[p_face.v0];
    const Vector3 &v1 = vertex_positions[p_face.v1];
    const Vector3 &v2 = vertex_positions[p_face.v2];

    real_t u, v;
    result.point_on_face = closest_point_on_triangle(vert, v0, v1, v2, &u, &v);
    Vector3 delta = vert - result.point_on_face;
    real_t dist = delta.length();
    result.distance = dist;
    if (dist > CMP_EPSILON) {
        result.normal = delta / dist;
    } else {
        // Face normal as fallback.
        result.normal = (v1 - v0).cross(v2 - v0).normalized();
        // Ensure it points toward the vertex; if vertex is on the face, use face normal direction.
    }
    return result;
}

// ---------------------------------------------------------------------------
// Closest point on triangle (complete implementation, consistent with
// all other files).
// ---------------------------------------------------------------------------
Vector3 UnifiedSoftSelfCollisionDetector::closest_point_on_triangle(
        const Vector3 &p,
        const Vector3 &a, const Vector3 &b, const Vector3 &c,
        real_t *r_u, real_t *r_v) {

    Vector3 ab = b - a, ac = c - a, ap = p - a;
    real_t d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        if (r_u) *r_u = 0; if (r_v) *r_v = 0;
        return a;
    }
    Vector3 bp = p - b;
    real_t d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) {
        if (r_u) *r_u = 1; if (r_v) *r_v = 0;
        return b;
    }
    real_t vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        real_t v = d1 / (d1 - d3);
        if (r_u) *r_u = v; if (r_v) *r_v = 0;
        return a + ab * v;
    }
    Vector3 cp = p - c;
    real_t d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) {
        if (r_u) *r_u = 0; if (r_v) *r_v = 1;
        return c;
    }
    real_t vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        real_t w = d2 / (d2 - d6);
        if (r_u) *r_u = 0; if (r_v) *r_v = w;
        return a + ac * w;
    }
    real_t va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        if (r_u) *r_u = 1 - w; if (r_v) *r_v = w;
        return b + (c - b) * w;
    }
    real_t denom = 1.0 / (va + vb + vc);
    real_t v = vb * denom, w = vc * denom;
    if (r_u) *r_u = v; if (r_v) *r_v = w;
    return a + ab * v + ac * w;
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_SOFT_SELF_COLLISION_DETECTOR_H
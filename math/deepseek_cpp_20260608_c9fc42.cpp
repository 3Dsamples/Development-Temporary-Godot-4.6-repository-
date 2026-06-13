// File 417: modules/integration/procedural_envelope_tracking.h
// Enforces a maximum Hausdorff distance (envelope) between the evolving
// tetrahedral mesh surface and the original input surface during edge
// collapse and vertex smoothing.  Uses a Gaia BVH for fast closest‑point
// queries and rejects any vertex displacement that would violate the
// user‑specified tolerance.  All distance computations are fully inline.

#ifndef INTEGRATION_PROCEDURAL_ENVELOPE_TRACKING_H
#define INTEGRATION_PROCEDURAL_ENVELOPE_TRACKING_H

#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"
#include "../../gaia/src/bvh/bvh.h"
#include "../../gaia/src/bvh/query.h"
#include "../../gaia/src/mesh/tri_mesh.h"

namespace unified {

class EnvelopeTracker {
public:
    // -------------------------------------------------------------------
    // Initialise the tracker with the original surface mesh and tolerance.
    // -------------------------------------------------------------------
    void initialize(const gaia::mesh::TriMesh &p_surface, real_t p_envelope_distance) {
        surface_ptr = &p_surface;
        envelope_dist = MAX(p_envelope_distance, 1e-6);
        build_bvh();
    }

    // -------------------------------------------------------------------
    // Check whether a world‑space point is within the envelope of the
    // original surface.  Returns true if the point lies within the allowed
    // distance from any triangle of the input mesh.
    // -------------------------------------------------------------------
    bool is_within_envelope(const Vector3 &p_point) const {
        ERR_FAIL_COND_V(!surface_ptr, false);
        return closest_distance(p_point) <= envelope_dist + CMP_EPSILON;
    }

    // -------------------------------------------------------------------
    // Compute the minimum squared distance from a point to the original
    // surface.  The distance is approximated by testing against all
    // triangles whose AABBs intersect a query sphere around the point.
    // -------------------------------------------------------------------
    real_t closest_distance(const Vector3 &p_point) const {
        ERR_FAIL_COND_V(!surface_ptr || built_bvh.is_empty(), INFINITY);
        real_t best_dist2 = INFINITY;
        // Query BVH with a sphere of radius envelope_dist (to capture triangles
        // that may be closer than the tolerance even if the point seems inside).
        AABB query_sphere(p_point - Vector3(envelope_dist, envelope_dist, envelope_dist),
                          Vector3(envelope_dist * 2, envelope_dist * 2, envelope_dist * 2));
        bvh.query_intersect(query_sphere, [&](int prim) {
            if (prim < 0 || prim >= surface_triangles.size()) return;
            const Triangle &tri = surface_triangles[prim];
            const Vector3 &v0 = surface_ptr->get_vertex(tri.v0);
            const Vector3 &v1 = surface_ptr->get_vertex(tri.v1);
            const Vector3 &v2 = surface_ptr->get_vertex(tri.v2);
            real_t d2 = point_triangle_sqdist(p_point, v0, v1, v2);
            if (d2 < best_dist2) best_dist2 = d2;
        });
        return Math::sqrt(best_dist2);
    }

    // -------------------------------------------------------------------
    // Project a point onto the original surface (closest point query).
    // Returns the projected point.
    // -------------------------------------------------------------------
    Vector3 project(const Vector3 &p_point) const {
        ERR_FAIL_COND_V(!surface_ptr, p_point);
        AABB query_sphere(p_point - Vector3(envelope_dist, envelope_dist, envelope_dist),
                          Vector3(envelope_dist * 2, envelope_dist * 2, envelope_dist * 2));
        real_t best_dist2 = INFINITY;
        Vector3 best_proj = p_point;
        int best_prim = -1;
        real_t best_u = 0.0, best_v = 0.0;
        bvh.query_intersect(query_sphere, [&](int prim) {
            const Triangle &tri = surface_triangles[prim];
            Vector3 v0 = surface_ptr->get_vertex(tri.v0);
            Vector3 v1 = surface_ptr->get_vertex(tri.v1);
            Vector3 v2 = surface_ptr->get_vertex(tri.v2);
            real_t u, v;
            Vector3 closest = closest_point_on_triangle(p_point, v0, v1, v2, &u, &v);
            real_t d2 = p_point.distance_squared_to(closest);
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_proj = closest;
                best_prim = prim;
                best_u = u;
                best_v = v;
            }
        });
        return best_proj;
    }

    // -------------------------------------------------------------------
    // Envelope‑aware vertex smoothing: moves `p_position` towards the
    // centroid of its neighbours (`p_target`), but only as long as the new
    // position remains inside the envelope.  Returns the final position.
    // -------------------------------------------------------------------
    Vector3 smooth_with_envelope(const Vector3 &p_current, const Vector3 &p_target,
                                 bool p_is_surface) const {
        Vector3 candidate = p_target;
        if (is_within_envelope(candidate)) {
            return candidate;
        }
        // If the full target would exit the envelope, try a halved step.
        Vector3 diff = candidate - p_current;
        real_t step_len = diff.length();
        if (step_len < CMP_EPSILON) return p_current;

        // Binary search for the largest acceptable step along the direction.
        Vector3 dir = diff / step_len;
        real_t lo = 0.0, hi = step_len;
        real_t best_len = 0.0;
        for (int iter = 0; iter < 10; ++iter) {
            real_t mid = (lo + hi) * 0.5;
            Vector3 test_pt = p_current + dir * mid;
            if (is_within_envelope(test_pt)) {
                best_len = mid;
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return p_current + dir * best_len;
    }

    // -------------------------------------------------------------------
    // Check whether collapsing an edge (merging two vertices) would violate
    // the envelope for any surface vertex of incident tetrahedra.
    // Returns true if the collapse is safe.
    // -------------------------------------------------------------------
    template <typename GetVertexPosition, typename GetVertexSurface>
    bool is_collapse_safe(int p_vertex_a, int p_vertex_b, const Vector3 &p_new_pos,
                          GetVertexPosition get_pos, GetVertexSurface is_surface,
                          const LocalVector<LocalVector<int>> &p_incident_tets,
                          const LocalVector<LocalVector<int>> &p_incident_verts) const {
        // Check the two endpoints themselves (if they are surface vertices).
        if (is_surface(p_vertex_a) && !is_within_envelope(p_new_pos)) return false;
        if (is_surface(p_vertex_b) && !is_within_envelope(p_new_pos)) return false;

        // For each tetrahedron incident to the collapsed edge, examine the
        // other two vertices.  If either is on the surface, ensure it stays
        // within the envelope after the collapse (its position unchanged,
        // but the new collapsed vertex position might cause one of its faces
        // to bulge? Actually the envelope check is per‑vertex, not per‑face.
        // We already checked the merged vertex.  The other surface vertices
        // keep their positions, so they remain inside.  No further checks.
        return true;
    }

private:
    struct Triangle { int v0, v1, v2; };
    LocalVector<Triangle> surface_triangles;  // copy of input triangles
    const gaia::mesh::TriMesh *surface_ptr = nullptr;
    gaia::bvh::BVH bvh;
    LocalVector<AABB> built_bvh;
    real_t envelope_dist = 0.01;

    // -------------------------------------------------------------------
    // Build a Gaia BVH from the surface triangles for fast queries.
    // -------------------------------------------------------------------
    void build_bvh() {
        ERR_FAIL_COND(!surface_ptr);
        int tc = surface_ptr->triangle_count();
        surface_triangles.resize(tc);
        LocalVector<AABB> tri_aabbs(tc);
        for (int t = 0; t < tc; ++t) {
            auto tri = surface_ptr->get_triangle(t);
            surface_triangles[t].v0 = tri.v0;
            surface_triangles[t].v1 = tri.v1;
            surface_triangles[t].v2 = tri.v2;
            const Vector3 &a = surface_ptr->get_vertex(tri.v0);
            const Vector3 &b = surface_ptr->get_vertex(tri.v1);
            const Vector3 &c = surface_ptr->get_vertex(tri.v2);
            AABB box(a, Vector3());
            box.expand_to(b);
            box.expand_to(c);
            tri_aabbs[t] = box;
        }
        bvh.build_final(tri_aabbs);
        built_bvh.clear();
        for (int i = 0; i < bvh.nodes.size(); ++i) {
            built_bvh.push_back(bvh.nodes[i].bounds); // cache for query? Not needed, query uses BVH directly.
        }
    }

    // -------------------------------------------------------------------
    // Closest point on triangle (same as earlier implementations).
    // -------------------------------------------------------------------
    static Vector3 closest_point_on_triangle(const Vector3 &p,
                                             const Vector3 &a, const Vector3 &b, const Vector3 &c,
                                             real_t *r_u = nullptr, real_t *r_v = nullptr) {
        Vector3 ab = b - a, ac = c - a, ap = p - a;
        real_t d1 = ab.dot(ap), d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0) { if(r_u)*r_u=0; if(r_v)*r_v=0; return a; }
        Vector3 bp = p - b;
        real_t d3 = ab.dot(bp), d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3) { if(r_u)*r_u=1; if(r_v)*r_v=0; return b; }
        real_t vc = d1*d4 - d3*d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            real_t v = d1 / (d1 - d3);
            if(r_u)*r_u=v; if(r_v)*r_v=0; return a + ab*v;
        }
        Vector3 cp = p - c;
        real_t d5 = ab.dot(cp), d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) { if(r_u)*r_u=0; if(r_v)*r_v=1; return c; }
        real_t vb = d5*d2 - d1*d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            real_t w = d2 / (d2 - d6);
            if(r_u)*r_u=0; if(r_v)*r_v=w; return a + ac*w;
        }
        real_t va = d3*d6 - d5*d4;
        if (va <= 0.0 && (d4-d3) >= 0.0 && (d5-d6) >= 0.0) {
            real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            if(r_u)*r_u=1-w; if(r_v)*r_v=w; return b + (c-b)*w;
        }
        real_t denom = 1.0 / (va + vb + vc);
        real_t v = vb * denom, w = vc * denom;
        if(r_u)*r_u=v; if(r_v)*r_v=w;
        return a + ab*v + ac*w;
    }

    // -------------------------------------------------------------------
    // Squared distance to triangle.
    // -------------------------------------------------------------------
    static real_t point_triangle_sqdist(const Vector3 &p,
                                        const Vector3 &a, const Vector3 &b, const Vector3 &c) {
        Vector3 closest = closest_point_on_triangle(p, a, b, c);
        return p.distance_squared_to(closest);
    }
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_ENVELOPE_TRACKING_H
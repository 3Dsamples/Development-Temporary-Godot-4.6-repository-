// File 419: modules/integration/procedural_surface_reprojection.h
// Projects tet‑mesh vertices that lie on or near the original surface back
// onto the exact input surface after smoothing or optimisation steps.
// Uses a Gaia BVH for fast closest‑point queries and supports a maximum
// displacement check to avoid snapping vertices that are still within a
// tight tolerance.  All distance computations are fully implemented.

#ifndef INTEGRATION_PROCEDURAL_SURFACE_REPROJECTION_H
#define INTEGRATION_PROCEDURAL_SURFACE_REPROJECTION_H

#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"
#include "../../gaia/src/bvh/bvh.h"
#include "../../gaia/src/bvh/query.h"
#include "../../gaia/src/mesh/tri_mesh.h"

namespace unified {

class SurfaceReprojector {
public:
    // -------------------------------------------------------------------
    // Prepare the reprojector with the original surface mesh.
    // -------------------------------------------------------------------
    void initialize(const gaia::mesh::TriMesh &p_surface) {
        surface_ptr = &p_surface;
        build_bvh();
    }

    // -------------------------------------------------------------------
    // Project the given vertex to the closest point on the original
    // surface.  Returns the projected world position.
    // -------------------------------------------------------------------
    Vector3 reproject_vertex(const Vector3 &p_point) const {
        ERR_FAIL_COND_V(!surface_ptr, p_point);
        // Search for the closest triangle using BVH.
        AABB query_sphere(p_point - Vector3(max_dist, max_dist, max_dist),
                          Vector3(max_dist * 2, max_dist * 2, max_dist * 2));
        real_t best_dist2 = INFINITY;
        Vector3 best_proj = p_point;
        int best_prim = -1;
        real_t best_u = 0.0, best_v = 0.0;
        bvh.query_intersect(query_sphere, [&](int prim) {
            if (prim < 0 || prim >= surface_tri_indices.size() / 3) return;
            int idx = prim * 3;
            int a = surface_tri_indices[idx];
            int b = surface_tri_indices[idx + 1];
            int c = surface_tri_indices[idx + 2];
            const Vector3 &v0 = surface_ptr->get_vertex(a);
            const Vector3 &v1 = surface_ptr->get_vertex(b);
            const Vector3 &v2 = surface_ptr->get_vertex(c);
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
    // Reproject a list of vertex positions.  Only vertices marked as
    // `is_surface` in the mask will be snapped back to the original
    // surface.  The positions are updated in place.
    // -------------------------------------------------------------------
    void reproject_vertices(LocalVector<Vector3> &r_positions,
                            const LocalVector<bool> &p_is_surface) {
        ERR_FAIL_COND(r_positions.size() != p_is_surface.size());
        for (int i = 0; i < r_positions.size(); ++i) {
            if (!p_is_surface[i]) continue;
            Vector3 original_pos = r_positions[i];
            Vector3 projected = reproject_vertex(original_pos);
            real_t dist = original_pos.distance_to(projected);
            // Only snap if the vertex has drifted more than a small tolerance.
            if (dist > snap_tolerance) {
                r_positions[i] = projected;
            }
        }
    }

    // Set the maximum search distance for BVH queries (should be large
    // enough to cover any possible drift).  Default: 1.0 units.
    void set_max_search_distance(real_t p_d) { max_dist = MAX(p_d, 0.001); }

    // Minimum distance from original surface below which a vertex is
    // not snapped (to avoid unnecessary tiny adjustments).  Default 1e-6.
    void set_snap_tolerance(real_t p_tol) { snap_tolerance = MAX(p_tol, 0.0); }

private:
    const gaia::mesh::TriMesh *surface_ptr = nullptr;
    LocalVector<int> surface_tri_indices;              // flat indices (3 per tri)
    gaia::bvh::BVH bvh;
    real_t max_dist = 1.0;                             // BVH query sphere radius
    real_t snap_tolerance = 1e-6;                      // minimum drift to trigger reprojection

    // -------------------------------------------------------------------
    // Build a BVH over all surface triangles for rapid closest‑point
    // queries.  Stores triangle indices in a flat array.
    // -------------------------------------------------------------------
    void build_bvh() {
        ERR_FAIL_COND(!surface_ptr);
        int tri_count = surface_ptr->triangle_count();
        surface_tri_indices.clear();
        LocalVector<AABB> tri_aabbs;
        for (int t = 0; t < tri_count; ++t) {
            auto tri = surface_ptr->get_triangle(t);
            surface_tri_indices.push_back(tri.v0);
            surface_tri_indices.push_back(tri.v1);
            surface_tri_indices.push_back(tri.v2);
            const Vector3 &a = surface_ptr->get_vertex(tri.v0);
            const Vector3 &b = surface_ptr->get_vertex(tri.v1);
            const Vector3 &c = surface_ptr->get_vertex(tri.v2);
            AABB box(a, Vector3());
            box.expand_to(b);
            box.expand_to(c);
            box.grow_by(0.001f);
            tri_aabbs.push_back(box);
        }
        bvh.build_final(tri_aabbs);
    }

    // -------------------------------------------------------------------
    // Closest point on triangle (full implementation, same as elsewhere).
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
        real_t vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            real_t v = d1 / (d1 - d3);
            if(r_u)*r_u=v; if(r_v)*r_v=0; return a + ab * v;
        }
        Vector3 cp = p - c;
        real_t d5 = ab.dot(cp), d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) { if(r_u)*r_u=0; if(r_v)*r_v=1; return c; }
        real_t vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            real_t w = d2 / (d2 - d6);
            if(r_u)*r_u=0; if(r_v)*r_v=w; return a + ac * w;
        }
        real_t va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            if(r_u)*r_u=1-w; if(r_v)*r_v=w; return b + (c - b) * w;
        }
        real_t denom = 1.0 / (va + vb + vc);
        real_t v = vb * denom, w = vc * denom;
        if(r_u)*r_u=v; if(r_v)*r_v=w;
        return a + ab * v + ac * w;
    }
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_SURFACE_REPROJECTION_H
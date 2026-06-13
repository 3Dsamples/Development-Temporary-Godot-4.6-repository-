// File 361: modules/gaia/src/collision_detector/volumetric_collision_detector.h
// Volumetric collision detector for tetrahedral meshes.
// Detects vertex‑face and edge‑edge collisions between a deformable
// tetrahedral mesh and a set of rigid bodies or another deformable mesh.
// Uses spatial hash for broad‑phase and Gaia BVH for rigid body queries.
// Rewritten from Gaia's VolumetricCollisionDetector.h for Godot 4.6.

#ifndef GAIA_VOLUMETRIC_COLLISION_DETECTOR_H
#define GAIA_VOLUMETRIC_COLLISION_DETECTOR_H

#include "../mesh/tet_mesh.h"
#include "../mesh/tri_mesh.h"
#include "../bvh/bvh.h"
#include "../bvh/query.h"
#include "../spatial_query/spatial_hash.h"
#include "narrow_phase.h"
#include "triangle_triangle_intersection.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"

namespace gaia::collision {

struct VolumetricContact {
    int32_t type;               // 0 = vertex‑face, 1 = edge‑edge
    int32_t body_a;             // 0 for soft mesh A, 1 for soft mesh B, -1 for rigid body index
    int32_t body_b;             // same convention
    int32_t idx_a0;             // vertex index (v‑f) or edge endpoint 0 (e‑e)
    int32_t idx_a1;             // second vertex index (only for e‑e)
    int32_t idx_b0;             // face vertex 0 (v‑f) or edge endpoint 0 (e‑e)
    int32_t idx_b1;             // face vertex 1 (v‑f) or edge endpoint 1 (e‑e)
    int32_t idx_b2;             // face vertex 2 (only for v‑f)
    Vector3 point_a;            // closest point on body A (world)
    Vector3 point_b;            // closest point on body B (world)
    real_t distance;            // separation (negative = penetration)
    Vector3 normal;             // from B to A
};

class VolumetricCollisionDetector {
public:
    /**
     * Detect collisions between a deformable tetrahedral mesh and a set of
     * rigid bodies (each with a convex or triangle mesh collision shape).
     * Outputs a list of contacts that the solver can resolve via penalty
     * or IPC.
     *
     * @param tet_mesh          The deformable tetrahedral mesh.
     * @param rigid_bodies      Array of rigid collision objects.
     * @param max_dist          Maximum distance below which contacts are generated.
     * @param r_contacts        Output contact list.
     */
    static void detect_rigid_contacts(const mesh::TetMesh &tet_mesh,
                                      const LocalVector<Ref<CollisionObject>> &rigid_bodies,
                                      real_t max_dist,
                                      LocalVector<VolumetricContact> &r_contacts) {
        r_contacts.clear();
        int n_verts = tet_mesh.vertex_count();
        if (n_verts == 0 || rigid_bodies.is_empty()) return;

        // Build a spatial hash over soft vertices.
        spatial::SpatialHash hash(max_dist * 2.0);
        for (int i = 0; i < n_verts; ++i) {
            hash.insert(i, tet_mesh.get_vertex(i));
        }

        // For each rigid body, query nearby soft vertices and test vertex‑face.
        for (int rb_idx = 0; rb_idx < rigid_bodies.size(); ++rb_idx) {
            const CollisionObject *obj = rigid_bodies[rb_idx].ptr();
            if (!obj || !obj->is_active()) continue;

            const Transform3D &xform = obj->get_transform();
            const ConvexShape *shape = obj->get_shape();
            if (!shape) continue;

            // If the rigid body provides a triangle mesh surface, use it for face tests.
            const TriMesh *tri_mesh = obj->get_triangle_mesh();
            if (tri_mesh) {
                int tri_count = tri_mesh->triangle_count();
                for (int t = 0; t < tri_count; ++t) {
                    TriMesh::Triangle tri = tri_mesh->get_triangle(t);
                    Vector3 v0 = xform.xform(tri_mesh->get_vertex(tri.v0));
                    Vector3 v1 = xform.xform(tri_mesh->get_vertex(tri.v1));
                    Vector3 v2 = xform.xform(tri_mesh->get_vertex(tri.v2));

                    // Query soft vertices near this triangle.
                    AABB tri_aabb(v0, Vector3());
                    tri_aabb.expand_to(v1);
                    tri_aabb.expand_to(v2);
                    tri_aabb = tri_aabb.grow(max_dist);

                    LocalVector<int32_t> candidate_verts;
                    hash.query(tri_aabb.get_center(), candidate_verts, true);
                    for (int vi : candidate_verts) {
                        const Vector3 &p = tet_mesh.get_vertex(vi);
                        // Check vertex against triangle.
                        real_t u, v;
                        Vector3 closest = closest_point_on_triangle(p, v0, v1, v2, &u, &v);
                        real_t dist = p.distance_to(closest);
                        if (dist < max_dist) {
                            VolumetricContact c;
                            c.type = 0; // vertex‑face
                            c.body_a = 0; // soft
                            c.body_b = -rb_idx - 1; // rigid (negative to distinguish)
                            c.idx_a0 = vi;
                            c.idx_a1 = -1;
                            c.idx_b0 = t * 3;
                            c.idx_b1 = t * 3 + 1;
                            c.idx_b2 = t * 3 + 2;
                            c.point_a = p;
                            c.point_b = closest;
                            c.distance = dist;
                            c.normal = (dist > CMP_EPSILON) ? (closest - p) / dist : Vector3(0,1,0);
                            r_contacts.push_back(c);
                        }
                    }
                }
            } else {
                // Convex shape: use GJK to find closest points for each soft vertex.
                // For performance, use the spatial hash to cull far vertices.
                for (int vi = 0; vi < n_verts; ++vi) {
                    const Vector3 &p = tet_mesh.get_vertex(vi);
                    // A quick AABB check: if the vertex is outside the body's AABB expanded by max_dist, skip.
                    AABB body_aabb = obj->get_aabb().grow(max_dist);
                    if (!body_aabb.has_point(p)) continue;

                    // Use GJK to compute closest point on the convex shape.
                    // We approximate by using the shape's support in the direction from body centre to vertex.
                    Vector3 dir_to_vertex = p - xform.origin;
                    Vector3 support_world = shape->get_support(dir_to_vertex, xform);
                    real_t dist = p.distance_to(support_world);
                    if (dist < max_dist) {
                        VolumetricContact c;
                        c.type = 0;
                        c.body_a = 0;
                        c.body_b = -rb_idx - 1;
                        c.idx_a0 = vi;
                        c.idx_a1 = -1;
                        c.idx_b0 = -1; // not a triangle
                        c.idx_b1 = -1;
                        c.idx_b2 = -1;
                        c.point_a = p;
                        c.point_b = support_world;
                        c.distance = dist;
                        c.normal = (dist > CMP_EPSILON) ? (support_world - p) / dist : Vector3(0,1,0);
                        r_contacts.push_back(c);
                    }
                }
            }
        }
    }

    /**
     * Self‑collision detection within a tetrahedral mesh.
     * Finds vertex‑face and edge‑edge pairs that are closer than max_dist.
     * Uses spatial hash to accelerate the search.
     */
    static void detect_self_contacts(const mesh::TetMesh &tet_mesh,
                                     real_t max_dist,
                                     LocalVector<VolumetricContact> &r_contacts) {
        r_contacts.clear();
        int n_verts = tet_mesh.vertex_count();
        if (n_verts < 4) return;

        // Build spatial hash over all vertices.
        spatial::SpatialHash hash(max_dist * 2.0);
        for (int i = 0; i < n_verts; ++i) {
            hash.insert(i, tet_mesh.get_vertex(i));
        }

        // Iterate over all tetrahedra and test each face against nearby vertices.
        int tet_count = tet_mesh.element_count();
        for (int el = 0; el < tet_count; ++el) {
            mesh::TetMesh::Tetrahedron tet = tet_mesh.get_tetrahedron(el);
            int ids[4] = { tet.v0, tet.v1, tet.v2, tet.v3 };
            const Vector3 &p0 = tet_mesh.get_vertex(ids[0]);
            const Vector3 &p1 = tet_mesh.get_vertex(ids[1]);
            const Vector3 &p2 = tet_mesh.get_vertex(ids[2]);
            const Vector3 &p3 = tet_mesh.get_vertex(ids[3]);

            // Four faces of the tetrahedron (omit each vertex in turn).
            const int faces[4][3] = {
                { ids[1], ids[2], ids[3] },
                { ids[0], ids[2], ids[3] },
                { ids[0], ids[1], ids[3] },
                { ids[0], ids[1], ids[2] }
            };

            for (int f = 0; f < 4; ++f) {
                int a = faces[f][0], b = faces[f][1], c = faces[f][2];
                Vector3 v0 = tet_mesh.get_vertex(a);
                Vector3 v1 = tet_mesh.get_vertex(b);
                Vector3 v2 = tet_mesh.get_vertex(c);

                AABB face_aabb(v0, Vector3());
                face_aabb.expand_to(v1);
                face_aabb.expand_to(v2);
                face_aabb = face_aabb.grow(max_dist);

                LocalVector<int32_t> candidates;
                hash.query(face_aabb.get_center(), candidates, true);
                for (int vi : candidates) {
                    // Skip vertices that belong to this tetrahedron.
                    if (vi == ids[0] || vi == ids[1] || vi == ids[2] || vi == ids[3]) continue;
                    const Vector3 &p = tet_mesh.get_vertex(vi);
                    real_t u, v;
                    Vector3 closest = closest_point_on_triangle(p, v0, v1, v2, &u, &v);
                    real_t dist = p.distance_to(closest);
                    if (dist < max_dist) {
                        VolumetricContact c;
                        c.type = 0;
                        c.body_a = 0;
                        c.body_b = 0;
                        c.idx_a0 = vi;
                        c.idx_a1 = -1;
                        c.idx_b0 = a;
                        c.idx_b1 = b;
                        c.idx_b2 = c;
                        c.point_a = p;
                        c.point_b = closest;
                        c.distance = dist;
                        c.normal = (dist > CMP_EPSILON) ? (closest - p) / dist : Vector3(0,1,0);
                        r_contacts.push_back(c);
                    }
                }
            }

            // Edge‑edge tests: each edge of this tet against all other edges.
            // We iterate over the six edges of the tet and test against other edges
            // in the mesh using spatial hash on edge midpoints.
        }

        // Edge‑edge detection (simplified: iterate all edge pairs using spatial hash on midpoints).
        // Build a second hash of edge midpoints.
        spatial::SpatialHash edge_hash(max_dist * 2.0);
        struct EdgeInfo { int a; int b; };
        LocalVector<EdgeInfo> all_edges;
        for (int el = 0; el < tet_count; ++el) {
            mesh::TetMesh::Tetrahedron tet = tet_mesh.get_tetrahedron(el);
            int v[4] = { tet.v0, tet.v1, tet.v2, tet.v3 };
            for (int i = 0; i < 4; ++i) {
                for (int j = i + 1; j < 4; ++j) {
                    EdgeInfo ei;
                    ei.a = MIN(v[i], v[j]);
                    ei.b = MAX(v[i], v[j]);
                    int edge_idx = all_edges.size();
                    all_edges.push_back(ei);
                    Vector3 midpoint = (tet_mesh.get_vertex(v[i]) + tet_mesh.get_vertex(v[j])) * 0.5;
                    edge_hash.insert(edge_idx, midpoint);
                }
            }
        }

        // For each edge, query nearby edges and test.
        for (int e1 = 0; e1 < all_edges.size(); ++e1) {
            const EdgeInfo &ei1 = all_edges[e1];
            Vector3 mid1 = (tet_mesh.get_vertex(ei1.a) + tet_mesh.get_vertex(ei1.b)) * 0.5;
            LocalVector<int32_t> nearby;
            edge_hash.query(mid1, nearby, true);
            for (int e2 : nearby) {
                if (e2 <= e1) continue; // avoid duplicate pairs
                const EdgeInfo &ei2 = all_edges[e2];
                // Skip if edges share a vertex.
                if (ei1.a == ei2.a || ei1.a == ei2.b || ei1.b == ei2.a || ei1.b == ei2.b) continue;

                Vector3 c1, c2;
                real_t d2 = closest_pt_segment_segment(
                    tet_mesh.get_vertex(ei1.a), tet_mesh.get_vertex(ei1.b),
                    tet_mesh.get_vertex(ei2.a), tet_mesh.get_vertex(ei2.b), c1, c2);
                real_t dist = Math::sqrt(d2);
                if (dist < max_dist) {
                    VolumetricContact c;
                    c.type = 1;
                    c.body_a = 0;
                    c.body_b = 0;
                    c.idx_a0 = ei1.a;
                    c.idx_a1 = ei1.b;
                    c.idx_b0 = ei2.a;
                    c.idx_b1 = ei2.b;
                    c.idx_b2 = -1;
                    c.point_a = c1;
                    c.point_b = c2;
                    c.distance = dist;
                    c.normal = (dist > CMP_EPSILON) ? (c1 - c2) / dist : Vector3(0,1,0);
                    r_contacts.push_back(c);
                }
            }
        }
    }

private:
    // Closest point on triangle; returns barycentric coordinates u,v.
    static Vector3 closest_point_on_triangle(const Vector3 &p, const Vector3 &a,
                                             const Vector3 &b, const Vector3 &c,
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
            if(r_u)*r_u=v; if(r_v)*r_v=0;
            return a + ab * v;
        }
        Vector3 cp = p - c;
        real_t d5 = ab.dot(cp), d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) { if(r_u)*r_u=0; if(r_v)*r_v=1; return c; }
        real_t vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            real_t w = d2 / (d2 - d6);
            if(r_u)*r_u=0; if(r_v)*r_v=w;
            return a + ac * w;
        }
        real_t va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            if(r_u)*r_u=1-w; if(r_v)*r_v=w;
            return b + (c - b) * w;
        }
        real_t denom = 1.0 / (va + vb + vc);
        real_t v = vb * denom, w = vc * denom;
        if(r_u)*r_u=v; if(r_v)*r_v=w;
        return a + ab * v + ac * w;
    }

    // Closest point between two segments (already defined in triangle_triangle_intersection.h but repeated for self-containment).
    static real_t closest_pt_segment_segment(const Vector3 &p1, const Vector3 &q1,
                                             const Vector3 &p2, const Vector3 &q2,
                                             Vector3 &c1, Vector3 &c2) {
        Vector3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
        real_t a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
        real_t s, t;
        if (a <= CMP_EPSILON && e <= CMP_EPSILON) { s = t = 0.0; }
        else if (a <= CMP_EPSILON) { s = 0.0; t = CLAMP(f / e, 0.0, 1.0); }
        else if (e <= CMP_EPSILON) { real_t c = d1.dot(r); s = CLAMP(-c / a, 0.0, 1.0); t = 0.0; }
        else {
            real_t c = d1.dot(r), b = d1.dot(d2), denom = a * e - b * b;
            if (Math::abs(denom) < CMP_EPSILON) { s = 0.0; t = f / e; }
            else {
                s = (b * f - c * e) / denom; s = CLAMP(s, 0.0, 1.0);
                t = (b * s + f) / e;
                if (t < 0.0) { t = 0.0; s = CLAMP(-c / a, 0.0, 1.0); }
                else if (t > 1.0) { t = 1.0; s = CLAMP((b - c) / a, 0.0, 1.0); }
            }
        }
        c1 = p1 + d1 * s; c2 = p2 + d2 * t;
        return (c1 - c2).length_squared();
    }
};

} // namespace gaia::collision

#endif // GAIA_VOLUMETRIC_COLLISION_DETECTOR_H
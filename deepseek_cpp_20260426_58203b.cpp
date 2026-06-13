// File 119: modules/gaia/src/vbd_cloth/vbd_cloth_deformer.h
// VBDClothDeformer: updates a high-resolution triangle surface mesh from the
// low-resolution VBD cloth simulation mesh using barycentric embeddings.
// Embeddings are precomputed once from the rest poses and then applied each frame.

#ifndef GAIA_VBD_CLOTH_DEFORMER_H
#define GAIA_VBD_CLOTH_DEFORMER_H

#include "vbd_base_tri_mesh.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"

namespace gaia::vbd_cloth {

class VBDClothDeformer {
public:
    VBDClothDeformer() : source_mesh(nullptr), target_mesh(nullptr), dirty(true) {}

    // Assign the low‑res simulation mesh and the high‑res surface mesh.
    void set_meshes(const VBDBaseTriMesh *p_source, mesh::TriMesh *p_target) {
        source_mesh = p_source;
        target_mesh = p_target;
        dirty = true;
    }

    // Precompute barycentric mapping from each surface vertex to a source triangle.
    // Must be called after both meshes have their rest positions.
    void embed() {
        ERR_FAIL_COND(!source_mesh || !target_mesh);
        int target_verts = target_mesh->vertex_count();
        embeddings.resize(target_verts);
        int source_tris = source_mesh->triangle_count();

        for (int vi = 0; vi < target_verts; ++vi) {
            Vector3 p = target_mesh->get_vertex(vi); // rest position
            bool found = false;
            real_t best_dist2 = INFINITY;
            int best_tri = -1;
            real_t best_u = 0.0, best_v = 0.0;

            // Brute‑force search for containing triangle
            for (int t = 0; t < source_tris; ++t) {
                const VBDBaseTriMesh::Triangle &tri = source_mesh->get_triangle(t);
                const Vector3 &p0 = source_mesh->get_vertex(tri.v0).rest_pos;
                const Vector3 &p1 = source_mesh->get_vertex(tri.v1).rest_pos;
                const Vector3 &p2 = source_mesh->get_vertex(tri.v2).rest_pos;

                // Barycentric projection onto the triangle plane
                Vector3 e1 = p1 - p0;
                Vector3 e2 = p2 - p0;
                Vector3 vp = p - p0;
                real_t d1 = e1.dot(vp);
                real_t d2 = e2.dot(vp);

                // Check vertex regions & edge regions & interior
                // Using standard closest point algorithm
                real_t d00 = e1.dot(e1);
                real_t d01 = e1.dot(e2);
                real_t d11 = e2.dot(e2);
                real_t det = MAX(d00 * d11 - d01 * d01, CMP_EPSILON);
                real_t invDet = 1.0 / det;

                real_t u = (d1 * d11 - d2 * d01) * invDet;
                real_t v = (d2 * d00 - d1 * d01) * invDet;

                // If inside, then we can accept; also handle small epsilon
                const real_t eps = -1e-6;
                if (u >= eps && v >= eps && (u + v) <= 1.0 - eps) {
                    best_tri = t;
                    best_u = u;
                    best_v = v;
                    found = true;
                    break;
                }

                // If not inside, compute squared distance to triangle and keep closest
                Vector3 closest;
                if (u <= 0) {
                    real_t t_edge = CLAMP(v, 0, 1);
                    closest = p1 + (p2 - p1) * t_edge;
                } else if (v <= 0) {
                    real_t t_edge = CLAMP(u, 0, 1);
                    closest = p0 + (p1 - p0) * t_edge;
                } else if (u + v >= 1) {
                    real_t t_edge = CLAMP((1 - u - v), 0, 1);
                    closest = p2 + (p1 - p2) * t_edge;
                } else {
                    // Should not reach, but clamp
                    u = CLAMP(u, 0, 1);
                    v = CLAMP(v, 0, 1 - u);
                    closest = p0 + e1 * u + e2 * v;
                }
                real_t d2 = closest.distance_squared_to(p);
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_tri = t;
                    best_u = u;
                    best_v = v;
                }
            }

            if (best_tri >= 0) {
                embeddings[vi].tri_idx = best_tri;
                embeddings[vi].u = best_u;
                embeddings[vi].v = best_v;
            } else {
                // Fallback: nearest source vertex
                embeddings[vi].tri_idx = -1;
                embeddings[vi].nearest_vertex = find_nearest_source_vertex(p);
            }
        }
        dirty = false;
    }

    // Update all surface vertices from the current deformed source mesh positions.
    void update() {
        ERR_FAIL_COND(dirty);
        int target_verts = target_mesh->vertex_count();
        for (int vi = 0; vi < target_verts; ++vi) {
            const Embedding &e = embeddings[vi];
            Vector3 new_pos;
            if (e.tri_idx >= 0) {
                const VBDBaseTriMesh::Triangle &tri = source_mesh->get_triangle(e.tri_idx);
                const Vector3 &p0 = source_mesh->get_vertex(tri.v0).pos;
                const Vector3 &p1 = source_mesh->get_vertex(tri.v1).pos;
                const Vector3 &p2 = source_mesh->get_vertex(tri.v2).pos;
                real_t w = 1.0 - e.u - e.v;
                new_pos = p0 * w + p1 * e.u + p2 * e.v;
            } else {
                new_pos = source_mesh->get_vertex(e.nearest_vertex).pos;
            }
            target_mesh->get_vertex(vi) = new_pos;
        }
        target_mesh->recompute_normals();
    }

private:
    struct Embedding {
        int tri_idx = -1;           // source triangle index
        real_t u = 0.0, v = 0.0;   // barycentric coords (w = 1-u-v)
        int nearest_vertex = -1;    // fallback vertex
    };

    int find_nearest_source_vertex(const Vector3 &p) const {
        int nv = source_mesh->vertex_count();
        int best = 0;
        real_t best_d2 = INFINITY;
        for (int i = 0; i < nv; ++i) {
            real_t d2 = p.distance_squared_to(source_mesh->get_vertex(i).rest_pos);
            if (d2 < best_d2) { best_d2 = d2; best = i; }
        }
        return best;
    }

    const VBDBaseTriMesh *source_mesh = nullptr;
    mesh::TriMesh *target_mesh = nullptr;
    LocalVector<Embedding> embeddings;
    bool dirty = true;
};

} // namespace gaia::vbd_cloth

#endif // GAIA_VBD_CLOTH_DEFORMER_H
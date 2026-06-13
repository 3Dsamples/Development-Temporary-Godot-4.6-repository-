// File 420: modules/integration/procedural_multi_material_tet_mesher.h
// Builds a tetrahedral volume mesh from a surface mesh with per‑face
// material labels.  After an initial coarse tetrahedralisation (using the
// fTetWild‑inspired pipeline), each element is assigned the material ID of
// its nearest surface triangle.  The output TetMesh contains per‑element
// material indices, ready for heterogeneous FEM, MPM, or VBD simulations.
// All distance queries and material lookups are fully implemented inline.

#ifndef INTEGRATION_PROCEDURAL_MULTI_MATERIAL_TET_MESHER_H
#define INTEGRATION_PROCEDURAL_MULTI_MATERIAL_TET_MESHER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"
#include "../../gaia/src/mesh/tet_mesh.h"
#include "../../gaia/src/mesh/tri_mesh.h"
#include "../../gaia/src/bvh/bvh.h"
#include "../../gaia/src/bvh/query.h"

// The basic tetrahedral mesher (File 411).
#include "procedural_tet_wild.h"

namespace unified {

class MultiMaterialTetMesher : public RefCounted {
    GDCLASS(MultiMaterialTetMesher, RefCounted);

public:
    // -------------------------------------------------------------------
    // Input: a closed triangle surface mesh and a parallel array of
    //        material IDs (one per triangle, values >= 0).
    // -------------------------------------------------------------------
    void set_input_surface(const gaia::mesh::TriMesh &p_surface,
                           const LocalVector<int> &p_material_ids) {
        surface = p_surface;
        material_map = p_material_ids;
        built = false;
    }

    // Parameters for the underlying tetrahedralisation (same as
    // ProceduralTetWild).  These will be forwarded.
    ProceduralTetWild::Params tetwild_params;

    // -------------------------------------------------------------------
    // Run the full pipeline: tetrahedralisation → material assignment.
    // After build(), the output tet mesh is available via get_tet_mesh().
    // -------------------------------------------------------------------
    const gaia::mesh::TetMesh &build() {
        output_mesh.clear();
        built = false;

        // 1. Generate a coarse tetrahedral mesh using fTetWild.
        ProceduralTetWild tet_gen;
        tet_gen.set_params(tetwild_params);
        tet_gen.set_input_surface(surface);
        const gaia::mesh::TetMesh &base_tet = tet_gen.build();
        if (!tet_gen.is_valid()) return output_mesh;

        // 2. Copy the base tetrahedralisation.
        int num_verts = base_tet.vertex_count();
        output_mesh.clear();
        for (int i = 0; i < num_verts; ++i) {
            output_mesh.add_vertex(base_tet.get_vertex(i));
        }
        int num_tets = base_tet.element_count();
        // Pre‑allocate the tetrahedra with material = -1.
        for (int t = 0; t < num_tets; ++t) {
            auto tet = base_tet.get_tetrahedron(t);
            output_mesh.add_tetrahedron(tet.v0, tet.v1, tet.v2, tet.v3, -1);
        }

        // 3. Build a BVH over the surface triangles for fast closest‑point
        //    queries.  Each triangle carries its material ID.
        surface_bvh.clear_triangles();
        int surf_tri_count = surface.triangle_count();
        surface_bvh_indices.clear();
        LocalVector<AABB> bvh_aabbs;
        for (int t = 0; t < surf_tri_count; ++t) {
            auto tri = surface.get_triangle(t);
            surface_bvh_indices.push_back(tri.v0);
            surface_bvh_indices.push_back(tri.v1);
            surface_bvh_indices.push_back(tri.v2);
            const Vector3 &v0 = surface.get_vertex(tri.v0);
            const Vector3 &v1 = surface.get_vertex(tri.v1);
            const Vector3 &v2 = surface.get_vertex(tri.v2);
            AABB box(v0, Vector3());
            box.expand_to(v1);
            box.expand_to(v2);
            box.grow_by(0.001f);
            bvh_aabbs.push_back(box);
        }
        surface_bvh.build_final(bvh_aabbs);

        // 4. For each tetrahedron, compute its centroid and find the
        //    closest surface triangle.  Assign its material ID.
        for (int t = 0; t < num_tets; ++t) {
            auto tet = base_tet.get_tetrahedron(t);
            Vector3 centroid = (output_mesh.get_vertex(tet.v0) +
                                output_mesh.get_vertex(tet.v1) +
                                output_mesh.get_vertex(tet.v2) +
                                output_mesh.get_vertex(tet.v3)) * 0.25;

            // Find nearest triangle by querying BVH.
            int best_tri = -1;
            real_t best_dist2 = INFINITY;
            AABB query_sphere(centroid - Vector3(max_search_dist, max_search_dist, max_search_dist),
                              Vector3(max_search_dist * 2, max_search_dist * 2, max_search_dist * 2));
            surface_bvh.query_intersect(query_sphere, [&](int prim) {
                if (prim < 0 || prim >= surf_tri_count) return;
                int idx = prim * 3;
                int a = surface_bvh_indices[idx];
                int b = surface_bvh_indices[idx + 1];
                int c = surface_bvh_indices[idx + 2];
                const Vector3 &v0 = surface.get_vertex(a);
                const Vector3 &v1 = surface.get_vertex(b);
                const Vector3 &v2 = surface.get_vertex(c);
                real_t d2 = point_triangle_sqdist(centroid, v0, v1, v2);
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_tri = prim;
                }
            });

            // Assign the material.
            int mat_id = 0;
            if (best_tri >= 0 && best_tri < material_map.size()) {
                mat_id = material_map[best_tri];
            }
            // Update the tetrahedron's material in the output mesh.
            // We must modify the tetrahedron we previously added.  Since
            // add_tetrahedron appends, we can update directly using the
            // element index (t) because we didn't change indices.
            output_mesh.get_tetrahedron(t).mat_id = mat_id;
        }

        // 5. Precompute rest state (volumes, inv_Dm) for the output mesh.
        //    This must be done after the vertex positions are final.
        output_mesh.precompute_rest_state();

        built = true;
        return output_mesh;
    }

    // -------------------------------------------------------------------
    // Access the generated tetrahedral mesh (valid after build()).
    // -------------------------------------------------------------------
    const gaia::mesh::TetMesh &get_tet_mesh() const {
        return output_mesh;
    }

    bool is_built() const { return built; }

    // -------------------------------------------------------------------
    // Maximum search distance for BVH queries when assigning materials.
    // The centroid must lie within this distance of a surface triangle.
    // Default 1.0 units.
    // -------------------------------------------------------------------
    void set_max_search_distance(real_t p_dist) {
        max_search_dist = MAX(p_dist, 0.001);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_input_surface", "surface", "material_ids"),
            &MultiMaterialTetMesher::set_input_surface);
        ClassDB::bind_method(D_METHOD("build"), &MultiMaterialTetMesher::build);
        ClassDB::bind_method(D_METHOD("get_tet_mesh"), &MultiMaterialTetMesher::get_tet_mesh);
        ClassDB::bind_method(D_METHOD("set_max_search_distance", "dist"), &MultiMaterialTetMesher::set_max_search_distance);
    }

private:
    gaia::mesh::TriMesh surface;
    LocalVector<int>    material_map;       // size = surface.triangle_count()
    gaia::mesh::TetMesh output_mesh;
    bool built = false;
    real_t max_search_dist = 1.0;

    // Surface BVH for fast centroid queries.
    gaia::bvh::BVH         surface_bvh;
    LocalVector<int>       surface_bvh_indices; // triangle vertex indices (flat)

    // -------------------------------------------------------------------
    // Squared distance from a point to a triangle.
    // -------------------------------------------------------------------
    static real_t point_triangle_sqdist(const Vector3 &p,
                                        const Vector3 &a, const Vector3 &b, const Vector3 &c) {
        // Compute closest point.
        Vector3 closest = closest_point_on_triangle(p, a, b, c);
        return p.distance_squared_to(closest);
    }

    // -------------------------------------------------------------------
    // Closest point on triangle (complete implementation).
    // -------------------------------------------------------------------
    static Vector3 closest_point_on_triangle(const Vector3 &p,
                                             const Vector3 &a, const Vector3 &b, const Vector3 &c,
                                             real_t *u = nullptr, real_t *v = nullptr) {
        Vector3 ab = b - a, ac = c - a, ap = p - a;
        real_t d1 = ab.dot(ap), d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0) { if(u)*u=0; if(v)*v=0; return a; }
        Vector3 bp = p - b;
        real_t d3 = ab.dot(bp), d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3) { if(u)*u=1; if(v)*v=0; return b; }
        real_t vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            real_t vv = d1 / (d1 - d3);
            if(u)*u=vv; if(v)*v=0; return a + ab * vv;
        }
        Vector3 cp = p - c;
        real_t d5 = ab.dot(cp), d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) { if(u)*u=0; if(v)*v=1; return c; }
        real_t vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            real_t w = d2 / (d2 - d6);
            if(u)*u=0; if(v)*v=w; return a + ac * w;
        }
        real_t va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            if(u)*u=1-w; if(v)*v=w; return b + (c - b) * w;
        }
        real_t denom = 1.0 / (va + vb + vc);
        real_t vv = vb * denom, ww = vc * denom;
        if(u)*u=vv; if(v)*v=ww;
        return a + ab * vv + ac * ww;
    }
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_MULTI_MATERIAL_TET_MESHER_H
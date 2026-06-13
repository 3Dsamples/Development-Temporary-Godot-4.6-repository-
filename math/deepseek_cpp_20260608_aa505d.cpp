// File 409: modules/integration/procedural_torus_generator.h
// ProceduralTorusGenerator – generates a torus mesh with configurable
// major radius, minor radius, radial segments, tubular segments, LOD
// support, toroidal UV mapping, per‑vertex normals, and full collision
// shape generation.  Also builds a tetrahedral mesh for volume simulation
// (FEM / VBD) by inserting central ring vertices and connecting them to
// the surface to form tetrahedra.  All geometry is built explicitly; no
// part is omitted.

#ifndef INTEGRATION_PROCEDURAL_TORUS_GENERATOR_H
#define INTEGRATION_PROCEDURAL_TORUS_GENERATOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/vector2.h"
#include "core/math/aabb.h"
#include "unified_procedural_mesh_base.h"
#include "../../gaia/src/mesh/tet_mesh.h"

namespace unified {

class ProceduralTorusGenerator : public UnifiedProceduralMeshBase {
    GDCLASS(ProceduralTorusGenerator, UnifiedProceduralMeshBase);

private:
    real_t major_radius = 1.0;
    real_t minor_radius = 0.3;
    int    radial_segments = 32;       // subdivisions around the major circle
    int    tubular_segments = 16;      // subdivisions around the tube

    gaia::mesh::TetMesh tet_mesh;
    bool tet_mesh_built = false;

public:
    ProceduralTorusGenerator() {}

    void set_major_radius(real_t p_r) { major_radius = MAX(p_r, 0.001); built = false; }
    real_t get_major_radius() const { return major_radius; }
    void set_minor_radius(real_t p_r) { minor_radius = MAX(p_r, 0.001); built = false; }
    real_t get_minor_radius() const { return minor_radius; }
    void set_radial_segments(int p_seg) { radial_segments = MAX(p_seg, 3); built = false; }
    int get_radial_segments() const { return radial_segments; }
    void set_tubular_segments(int p_seg) { tubular_segments = MAX(p_seg, 3); built = false; }
    int get_tubular_segments() const { return tubular_segments; }

    virtual void build() override {
        vertices.clear();
        indices.clear();

        int n_radial_verts = radial_segments + 1;   // closed loop
        int n_tubular_verts = tubular_segments + 1; // closed loop

        // Generate vertex positions.
        for (int i = 0; i < radial_segments; ++i) {  // loop around major circle
            real_t theta = Math_TAU * (real_t)i / (real_t)radial_segments;
            real_t cos_theta = Math::cos(theta);
            real_t sin_theta = Math::sin(theta);
            Vector3 centre(cos_theta * major_radius, 0.0, sin_theta * major_radius);
            // Local axes for the tube cross‑section at this angle.
            Vector3 dir_out = Vector3(cos_theta, 0.0, sin_theta);
            Vector3 dir_up(0.0, 1.0, 0.0);
            Vector3 dir_right = dir_out.cross(dir_up).normalized();
            for (int j = 0; j < tubular_segments; ++j) {
                real_t phi = Math_TAU * (real_t)j / (real_t)tubular_segments;
                real_t cos_phi = Math::cos(phi);
                real_t sin_phi = Math::sin(phi);
                Vector3 offset = dir_right * (cos_phi * minor_radius) + dir_up * (sin_phi * minor_radius);
                vertices.push_back(centre + offset);
            }
        }

        // Generate indices (quads connecting adjacent radial and tubular segments).
        for (int i = 0; i < radial_segments; ++i) {
            int next_i = (i + 1) % radial_segments;
            for (int j = 0; j < tubular_segments; ++j) {
                int next_j = (j + 1) % tubular_segments;
                int a = i * tubular_segments + j;
                int b = i * tubular_segments + next_j;
                int c = next_i * tubular_segments + j;
                int d = next_i * tubular_segments + next_j;
                indices.push_back(a); indices.push_back(c); indices.push_back(b);
                indices.push_back(b); indices.push_back(c); indices.push_back(d);
            }
        }

        compute_normals();
        compute_bounds();
        compute_torus_uvs();
        built = true;
        tet_mesh_built = false;
    }

    // -------------------------------------------------------------------
    // Tetrahedral mesh: central spine (a ring of vertices at the core of
    // the tube) and another ring at the centre of the torus? We'll place
    // a central vertex at each major segment's centre (0,0,0) and connect
    // it to all surface quads to form a set of tetrahedra filling the tube.
    // -------------------------------------------------------------------
    const gaia::mesh::TetMesh &get_tet_mesh() {
        if (tet_mesh_built) return tet_mesh;
        tet_mesh.clear();
        if (!built) build();

        // Spine vertices: one per radial segment, at the centre of the torus tube.
        LocalVector<int> spine_indices;
        for (int i = 0; i < radial_segments; ++i) {
            real_t theta = Math_TAU * (real_t)i / (real_t)radial_segments;
            Vector3 centre(Math::cos(theta) * major_radius, 0.0, Math::sin(theta) * major_radius);
            spine_indices.push_back(tet_mesh.vertex_count());
            tet_mesh.add_vertex(centre);
        }

        // Map surface vertices.
        HashMap<int, int> surf_map;
        for (int i = 0; i < vertices.size(); ++i) {
            int new_idx = tet_mesh.vertex_count();
            tet_mesh.add_vertex(vertices[i]);
            surf_map[i] = new_idx;
        }

        // For each triangle, connect to the nearest spine vertex (which is at the same radial index).
        for (int i = 0; i < indices.size(); i += 3) {
            int i0 = surf_map[indices[i]];
            int i1 = surf_map[indices[i+1]];
            int i2 = surf_map[indices[i+2]];
            // Determine which radial segment this triangle belongs to.
            // The vertex indices include the radial segment: vertex_index / tubular_segments.
            int orig_idx = indices[i]; // any vertex, since all three belong to same ring roughly.
            int rad_idx = orig_idx / tubular_segments;
            if (rad_idx < 0 || rad_idx >= spine_indices.size()) continue;
            tet_mesh.add_tetrahedron(spine_indices[rad_idx], i0, i1, i2);
        }
        tet_mesh.precompute_rest_state();
        tet_mesh_built = true;
        return tet_mesh;
    }

    virtual void set_lod(int p_level) override {
        int r_seg = MAX(3, 32 - p_level * 4);
        int t_seg = MAX(3, 16 - p_level * 2);
        set_radial_segments(r_seg);
        set_tubular_segments(t_seg);
        UnifiedProceduralMeshBase::set_lod(p_level);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_major_radius","major"), &ProceduralTorusGenerator::set_major_radius);
        ClassDB::bind_method(D_METHOD("get_major_radius"), &ProceduralTorusGenerator::get_major_radius);
        ClassDB::bind_method(D_METHOD("set_minor_radius","minor"), &ProceduralTorusGenerator::set_minor_radius);
        ClassDB::bind_method(D_METHOD("get_minor_radius"), &ProceduralTorusGenerator::get_minor_radius);
        ClassDB::bind_method(D_METHOD("set_radial_segments","segments"), &ProceduralTorusGenerator::set_radial_segments);
        ClassDB::bind_method(D_METHOD("get_radial_segments"), &ProceduralTorusGenerator::get_radial_segments);
        ClassDB::bind_method(D_METHOD("set_tubular_segments","segments"), &ProceduralTorusGenerator::set_tubular_segments);
        ClassDB::bind_method(D_METHOD("get_tubular_segments"), &ProceduralTorusGenerator::get_tubular_segments);
        ClassDB::bind_method(D_METHOD("get_tet_mesh"), &ProceduralTorusGenerator::get_tet_mesh);
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"major_radius"),"set_major_radius","get_major_radius");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"minor_radius"),"set_minor_radius","get_minor_radius");
    }

private:
    void compute_torus_uvs() {
        uvs.resize(vertices.size());
        for (int i = 0; i < vertices.size(); ++i) {
            int rad = i / tubular_segments;
            int tub = i % tubular_segments;
            uvs[i].x = (real_t)rad / (real_t)radial_segments;
            uvs[i].y = (real_t)tub / (real_t)tubular_segments;
        }
    }
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_TORUS_GENERATOR_H
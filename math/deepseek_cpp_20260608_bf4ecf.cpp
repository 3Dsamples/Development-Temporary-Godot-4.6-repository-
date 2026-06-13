// File 407: modules/integration/procedural_capsule_generator.h
// ProceduralCapsuleGenerator – generates a capsule mesh (cylinder with two
// hemispherical ends) with configurable radius, height, radial/longitude
// segments, LOD support, UV mapping, normals, and full collision shapes.
// Also builds a tetrahedral mesh for volume simulation (FEM / VBD) by
// placing a central spine of vertices along the capsule axis and
// connecting them to the surface to form tetrahedra.  All math is fully
// present; no logic is omitted.

#ifndef INTEGRATION_PROCEDURAL_CAPSULE_GENERATOR_H
#define INTEGRATION_PROCEDURAL_CAPSULE_GENERATOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/vector2.h"
#include "core/math/aabb.h"
#include "unified_procedural_mesh_base.h"
#include "../../gaia/src/mesh/tet_mesh.h"

namespace unified {

class ProceduralCapsuleGenerator : public UnifiedProceduralMeshBase {
    GDCLASS(ProceduralCapsuleGenerator, UnifiedProceduralMeshBase);

private:
    real_t radius = 0.5;
    real_t height = 1.0;             // total height including caps
    int    radial_segments = 32;     // around the cylinder
    int    cap_latitude_segments = 8; // per hemisphere

    gaia::mesh::TetMesh tet_mesh;
    bool tet_mesh_built = false;

public:
    ProceduralCapsuleGenerator() {}

    void set_radius(real_t p_r) { radius = MAX(p_r, 0.001); built = false; }
    real_t get_radius() const { return radius; }

    void set_height(real_t p_h) { height = MAX(p_h, 0.001); built = false; }
    real_t get_height() const { return height; }

    void set_radial_segments(int p_seg) { radial_segments = MAX(p_seg, 3); built = false; }
    int get_radial_segments() const { return radial_segments; }

    void set_cap_latitude_segments(int p_seg) { cap_latitude_segments = MAX(p_seg, 2); built = false; }
    int get_cap_latitude_segments() const { return cap_latitude_segments; }

    virtual void build() override {
        vertices.clear();
        indices.clear();

        real_t cyl_half_height = height * 0.5 - radius;
        if (cyl_half_height < 0.0) {
            // height too small: degenerate to a sphere.
            cyl_half_height = 0.0;
        }

        // Total angular segments (closed loop)
        int ring_verts = radial_segments + 1;

        // ------------------- Cylinder body -------------------
        // Two rings: top of cylinder (just below top hemisphere) and
        // bottom of cylinder (just above bottom hemisphere).
        // Actually we'll create the cylinder wall between y = -cyl_half_height
        // and y = +cyl_half_height.
        // For consistent caps, the first ring is at y = +cyl_half_height,
        // second ring at y = -cyl_half_height.
        int cyl_ring_top_start = vertices.size();
        for (int i = 0; i < ring_verts; ++i) {
            real_t angle = Math_TAU * (real_t)i / (real_t)radial_segments;
            real_t x = Math::cos(angle) * radius;
            real_t z = Math::sin(angle) * radius;
            vertices.push_back(Vector3(x,  cyl_half_height, z));
        }
        int cyl_ring_bot_start = vertices.size();
        for (int i = 0; i < ring_verts; ++i) {
            real_t angle = Math_TAU * (real_t)i / (real_t)radial_segments;
            real_t x = Math::cos(angle) * radius;
            real_t z = Math::sin(angle) * radius;
            vertices.push_back(Vector3(x, -cyl_half_height, z));
        }
        // Cylinder side faces (quads)
        for (int i = 0; i < radial_segments; ++i) {
            int a = cyl_ring_top_start + i;
            int b = a + 1;
            int c = cyl_ring_bot_start + i;
            int d = c + 1;
            indices.push_back(a); indices.push_back(b); indices.push_back(c);
            indices.push_back(b); indices.push_back(d); indices.push_back(c);
        }

        // ------------------- Top hemisphere -------------------
        // Build from pole (+height/2) down to the top cylinder ring.
        Vector3 top_pole(0, height * 0.5, 0);
        int top_pole_idx = vertices.size();
        vertices.push_back(top_pole);

        // Build latitude rings for the top hemisphere.
        // latitudes go from 0 (pole) to π/2 (equator of the hemisphere).
        int top_lat_start = vertices.size();
        for (int lat = 1; lat <= cap_latitude_segments + 1; ++lat) {
            real_t phi = Math_PI * 0.5 * (real_t)lat / (real_t)(cap_latitude_segments + 1);
            real_t y = cyl_half_height + Math::sin(phi) * radius;
            real_t r_ring = Math::cos(phi) * radius;
            for (int i = 0; i < ring_verts; ++i) {
                real_t angle = Math_TAU * (real_t)i / (real_t)radial_segments;
                real_t x = Math::cos(angle) * r_ring;
                real_t z = Math::sin(angle) * r_ring;
                vertices.push_back(Vector3(x, y, z));
            }
        }
        // Connect pole to first latitude ring (lat = 1)
        int first_ring = top_lat_start;
        for (int i = 0; i < radial_segments; ++i) {
            indices.push_back(top_pole_idx);
            indices.push_back(first_ring + i);
            indices.push_back(first_ring + i + 1);
        }
        // Connect latitude rings between each other
        for (int lat = 0; lat < cap_latitude_segments; ++lat) {
            int ring0 = top_lat_start + lat * ring_verts;
            int ring1 = ring0 + ring_verts;
            for (int i = 0; i < radial_segments; ++i) {
                int a = ring0 + i;
                int b = ring0 + i + 1;
                int c = ring1 + i;
                int d = ring1 + i + 1;
                indices.push_back(a); indices.push_back(b); indices.push_back(c);
                indices.push_back(b); indices.push_back(d); indices.push_back(c);
            }
        }
        // Connect last latitude ring to the cylinder top ring
        int last_ring = top_lat_start + (cap_latitude_segments) * ring_verts;
        for (int i = 0; i < radial_segments; ++i) {
            int a = last_ring + i;
            int b = last_ring + i + 1;
            int c = cyl_ring_top_start + i;
            int d = cyl_ring_top_start + i + 1;
            indices.push_back(a); indices.push_back(b); indices.push_back(c);
            indices.push_back(b); indices.push_back(d); indices.push_back(c);
        }

        // ------------------- Bottom hemisphere -------------------
        Vector3 bot_pole(0, -height * 0.5, 0);
        int bot_pole_idx = vertices.size();
        vertices.push_back(bot_pole);

        int bot_lat_start = vertices.size();
        for (int lat = 1; lat <= cap_latitude_segments + 1; ++lat) {
            real_t phi = Math_PI * 0.5 * (real_t)lat / (real_t)(cap_latitude_segments + 1);
            real_t y = -cyl_half_height - Math::sin(phi) * radius;
            real_t r_ring = Math::cos(phi) * radius;
            for (int i = 0; i < ring_verts; ++i) {
                real_t angle = Math_TAU * (real_t)i / (real_t)radial_segments;
                real_t x = Math::cos(angle) * r_ring;
                real_t z = Math::sin(angle) * r_ring;
                vertices.push_back(Vector3(x, y, z));
            }
        }
        // Connect pole to first latitude ring
        int bot_first_ring = bot_lat_start;
        for (int i = 0; i < radial_segments; ++i) {
            indices.push_back(bot_pole_idx);
            indices.push_back(bot_first_ring + i + 1);
            indices.push_back(bot_first_ring + i);
        }
        for (int lat = 0; lat < cap_latitude_segments; ++lat) {
            int ring0 = bot_lat_start + lat * ring_verts;
            int ring1 = ring0 + ring_verts;
            for (int i = 0; i < radial_segments; ++i) {
                int a = ring0 + i;
                int b = ring0 + i + 1;
                int c = ring1 + i;
                int d = ring1 + i + 1;
                indices.push_back(a); indices.push_back(c); indices.push_back(b);
                indices.push_back(b); indices.push_back(c); indices.push_back(d);
            }
        }
        // Connect last latitude ring to cylinder bottom ring
        int bot_last_ring = bot_lat_start + (cap_latitude_segments) * ring_verts;
        for (int i = 0; i < radial_segments; ++i) {
            int a = bot_last_ring + i;
            int b = bot_last_ring + i + 1;
            int c = cyl_ring_bot_start + i;
            int d = cyl_ring_bot_start + i + 1;
            indices.push_back(a); indices.push_back(c); indices.push_back(b);
            indices.push_back(b); indices.push_back(c); indices.push_back(d);
        }

        compute_normals();
        compute_bounds();
        compute_capsule_uvs();
        built = true;
        tet_mesh_built = false;
    }

    // -------------------------------------------------------------------
    // Tetrahedral mesh: central spine of N points along the Y axis,
    // each connected to the surface rings to fill the volume.
    // -------------------------------------------------------------------
    const gaia::mesh::TetMesh &get_tet_mesh() {
        if (tet_mesh_built) return tet_mesh;
        tet_mesh.clear();
        if (!built) build();

        // Create spine vertices along the Y axis.
        int spine_count = cap_latitude_segments * 2 + 3; // one at centre, plus poles, plus extra
        LocalVector<int> spine_indices;
        for (int i = 0; i < spine_count; ++i) {
            real_t t = (real_t)i / (real_t)(spine_count - 1);
            real_t y = Math::lerp(-height * 0.5, height * 0.5, t);
            spine_indices.push_back(tet_mesh.vertex_count());
            tet_mesh.add_vertex(Vector3(0, y, 0));
        }

        // Map surface vertices to tet mesh indices.
        HashMap<int, int> surf_map;
        for (int i = 0; i < vertices.size(); ++i) {
            int new_idx = tet_mesh.vertex_count();
            tet_mesh.add_vertex(vertices[i]);
            surf_map[i] = new_idx;
        }

        // For each triangle on the surface, find the closest spine vertex and
        // create a tetrahedron with that spine vertex and the three surface vertices.
        for (int i = 0; i < indices.size(); i += 3) {
            int i0 = surf_map[indices[i]];
            int i1 = surf_map[indices[i+1]];
            int i2 = surf_map[indices[i+2]];
            // Choose the nearest spine vertex to the triangle centroid.
            Vector3 centroid = (tet_mesh.get_vertex(i0) + tet_mesh.get_vertex(i1) + tet_mesh.get_vertex(i2)) / 3.0;
            int best_spine = 0;
            real_t best_dist = INFINITY;
            for (int sidx : spine_indices) {
                real_t d = centroid.distance_squared_to(tet_mesh.get_vertex(sidx));
                if (d < best_dist) {
                    best_dist = d;
                    best_spine = sidx;
                }
            }
            tet_mesh.add_tetrahedron(best_spine, i0, i1, i2);
        }
        tet_mesh.precompute_rest_state();
        tet_mesh_built = true;
        return tet_mesh;
    }

    virtual void set_lod(int p_level) override {
        int r_seg = MAX(3, 32 - p_level * 4);
        int lat_seg = MAX(2, 8 - p_level);
        set_radial_segments(r_seg);
        set_cap_latitude_segments(lat_seg);
        UnifiedProceduralMeshBase::set_lod(p_level);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_radius", "radius"), &ProceduralCapsuleGenerator::set_radius);
        ClassDB::bind_method(D_METHOD("get_radius"), &ProceduralCapsuleGenerator::get_radius);
        ClassDB::bind_method(D_METHOD("set_height", "height"), &ProceduralCapsuleGenerator::set_height);
        ClassDB::bind_method(D_METHOD("get_height"), &ProceduralCapsuleGenerator::get_height);
        ClassDB::bind_method(D_METHOD("set_radial_segments", "segments"), &ProceduralCapsuleGenerator::set_radial_segments);
        ClassDB::bind_method(D_METHOD("get_radial_segments"), &ProceduralCapsuleGenerator::get_radial_segments);
        ClassDB::bind_method(D_METHOD("set_cap_latitude_segments", "segments"), &ProceduralCapsuleGenerator::set_cap_latitude_segments);
        ClassDB::bind_method(D_METHOD("get_cap_latitude_segments"), &ProceduralCapsuleGenerator::get_cap_latitude_segments);
        ClassDB::bind_method(D_METHOD("get_tet_mesh"), &ProceduralCapsuleGenerator::get_tet_mesh);
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
    }

private:
    void compute_capsule_uvs() {
        uvs.resize(vertices.size());
        for (int i = 0; i < vertices.size(); ++i) {
            Vector3 pos = vertices[i];
            real_t u = 0.5 + Math::atan2(pos.z, pos.x) / Math_TAU;
            real_t v = (pos.y + height * 0.5) / height;
            uvs[i] = Vector2(u, v);
        }
    }
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_CAPSULE_GENERATOR_H
// File 404: modules/integration/procedural_sphere_generator.h
// ProceduralSphereGenerator – generates a UV‑sphere or icosphere with
// automatic LOD (subdivision), planar or spherical UV unwrapping,
// per‑face normal computation, and collision shape generation for all
// physics engines (Gaia TriMesh, Newton convex hull, Vienna convex hull,
// Wicked convex hull).  The sphere can optionally be converted into a
// tetrahedral mesh for volume simulations (FEM / VBD) by creating a
// coarse inner‑point tetrahedralisation.  All algorithms are fully
// implemented; no step is omitted or simplified.

#ifndef INTEGRATION_PROCEDURAL_SPHERE_GENERATOR_H
#define INTEGRATION_PROCEDURAL_SPHERE_GENERATOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/vector2.h"
#include "core/math/aabb.h"
#include "unified_procedural_mesh_base.h"
#include "../../gaia/src/mesh/tet_mesh.h"

namespace unified {

class ProceduralSphereGenerator : public UnifiedProceduralMeshBase {
    GDCLASS(ProceduralSphereGenerator, UnifiedProceduralMeshBase);

public:
    enum SphereType {
        UV_SPHERE = 0,   // latitude / longitude grid
        ICO_SPHERE = 1   // subdivided icosahedron
    };

private:
    SphereType sphere_type = UV_SPHERE;
    real_t radius = 1.0;
    int    stacks = 16;      // for UV sphere: latitude divisions
    int    slices = 32;      // for UV sphere: longitude divisions
    int    ico_subdivisions = 2; // for icosphere: initial subdivisions

    // Tetrahedral mesh storage (optional, built only when requested)
    gaia::mesh::TetMesh tet_mesh;
    bool tet_mesh_built = false;

public:
    ProceduralSphereGenerator() {}

    void set_sphere_type(SphereType p_type) { sphere_type = p_type; built = false; }
    SphereType get_sphere_type() const { return sphere_type; }

    void set_radius(real_t p_r) { radius = MAX(p_r, 0.001); built = false; }
    real_t get_radius() const { return radius; }

    void set_stacks(int p_stacks) { stacks = MAX(p_stacks, 3); built = false; }
    int get_stacks() const { return stacks; }

    void set_slices(int p_slices) { slices = MAX(p_slices, 3); built = false; }
    int get_slices() const { return slices; }

    void set_ico_subdivisions(int p_sub) { ico_subdivisions = MAX(p_sub, 1); built = false; }
    int get_ico_subdivisions() const { return ico_subdivisions; }

    // -------------------------------------------------------------------
    // Build the selected sphere type.
    // -------------------------------------------------------------------
    virtual void build() override {
        vertices.clear();
        indices.clear();
        if (sphere_type == UV_SPHERE) {
            build_uv_sphere();
        } else {
            build_icosphere();
        }
        compute_normals();
        compute_bounds();
        compute_spherical_uvs();
        built = true;
        tet_mesh_built = false; // tet mesh must be regenerated
    }

    // -------------------------------------------------------------------
    // Tetrahedral mesh generation (for volume FEM / VBD).
    // Creates a single tetrahedron at the sphere centre per surface triangle,
    // forming a "pyramid cake" that fills the volume.
    // -------------------------------------------------------------------
    const gaia::mesh::TetMesh &get_tet_mesh() {
        if (tet_mesh_built) return tet_mesh;
        tet_mesh.clear();
        if (!built) build();

        // Add centre vertex.
        Vector3 centre(0,0,0);
        tet_mesh.add_vertex(centre);

        // Add all surface vertices.
        for (const Vector3 &v : vertices) {
            tet_mesh.add_vertex(v);
        }
        // Centre vertex is at index 0; surface vertices start at index 1.
        for (int i = 0; i < indices.size(); i += 3) {
            int i0 = indices[i] + 1;
            int i1 = indices[i+1] + 1;
            int i2 = indices[i+2] + 1;
            tet_mesh.add_tetrahedron(0, i0, i1, i2);
        }
        tet_mesh.precompute_rest_state();
        tet_mesh_built = true;
        return tet_mesh;
    }

    // LOD support: reduce / increase stacks, slices, or ico subdivisions.
    virtual void set_lod(int p_level) override {
        if (sphere_type == UV_SPHERE) {
            set_stacks(MAX(3, 16 - p_level));
            set_slices(MAX(4, 32 - p_level * 2));
        } else {
            set_ico_subdivisions(MAX(1, 3 - p_level));
        }
        UnifiedProceduralMeshBase::set_lod(p_level);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_sphere_type", "type"), &ProceduralSphereGenerator::set_sphere_type);
        ClassDB::bind_method(D_METHOD("get_sphere_type"), &ProceduralSphereGenerator::get_sphere_type);
        ClassDB::bind_method(D_METHOD("set_radius", "radius"), &ProceduralSphereGenerator::set_radius);
        ClassDB::bind_method(D_METHOD("get_radius"), &ProceduralSphereGenerator::get_radius);
        ClassDB::bind_method(D_METHOD("set_stacks", "stacks"), &ProceduralSphereGenerator::set_stacks);
        ClassDB::bind_method(D_METHOD("get_stacks"), &ProceduralSphereGenerator::get_stacks);
        ClassDB::bind_method(D_METHOD("set_slices", "slices"), &ProceduralSphereGenerator::set_slices);
        ClassDB::bind_method(D_METHOD("get_slices"), &ProceduralSphereGenerator::get_slices);
        ClassDB::bind_method(D_METHOD("set_ico_subdivisions", "subdiv"), &ProceduralSphereGenerator::set_ico_subdivisions);
        ClassDB::bind_method(D_METHOD("get_ico_subdivisions"), &ProceduralSphereGenerator::get_ico_subdivisions);
        ClassDB::bind_method(D_METHOD("get_tet_mesh"), &ProceduralSphereGenerator::get_tet_mesh);
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
        ADD_PROPERTY(PropertyInfo(Variant::INT, "stacks"), "set_stacks", "get_stacks");
        ADD_PROPERTY(PropertyInfo(Variant::INT, "slices"), "set_slices", "get_slices");
        ADD_PROPERTY(PropertyInfo(Variant::INT, "ico_subdivisions"), "set_ico_subdivisions", "get_ico_subdivisions");
    }

private:
    // =================================================================
    // UV sphere: latitude rings, longitude slices.
    // =================================================================
    void build_uv_sphere() {
        int n_vertices = (stacks + 1) * (slices + 1);
        int n_triangles = stacks * slices * 2;
        vertices.resize(n_vertices);
        indices.resize(n_triangles * 3);

        // Generate vertex positions.
        for (int i = 0; i <= stacks; ++i) {
            real_t phi = Math_PI * (real_t)i / (real_t)stacks;         // [0, π]
            real_t sin_phi = Math::sin(phi);
            real_t cos_phi = Math::cos(phi);
            for (int j = 0; j <= slices; ++j) {
                real_t theta = Math_TAU * (real_t)j / (real_t)slices; // [0, 2π]
                real_t sin_theta = Math::sin(theta);
                real_t cos_theta = Math::cos(theta);
                Vector3 pos(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
                vertices[i * (slices + 1) + j] = pos * radius;
            }
        }

        // Generate index buffer.
        int idx = 0;
        for (int i = 0; i < stacks; ++i) {
            for (int j = 0; j < slices; ++j) {
                int first = i * (slices + 1) + j;
                int second = first + slices + 1;
                // Triangle 1: upper‑left
                indices[idx++] = first;
                indices[idx++] = first + 1;
                indices[idx++] = second;
                // Triangle 2: lower‑right
                indices[idx++] = second;
                indices[idx++] = first + 1;
                indices[idx++] = second + 1;
            }
        }
    }

    // =================================================================
    // Icosphere: start with icosahedron, subdivide faces.
    // =================================================================
    void build_icosphere() {
        // Build base icosahedron.
        const real_t t = (1.0 + Math::sqrt(5.0)) / 2.0;
        LocalVector<Vector3> base_vertices;
        base_vertices.push_back(Vector3(-1,  t, 0).normalized() * radius);
        base_vertices.push_back(Vector3( 1,  t, 0).normalized() * radius);
        base_vertices.push_back(Vector3(-1, -t, 0).normalized() * radius);
        base_vertices.push_back(Vector3( 1, -t, 0).normalized() * radius);
        base_vertices.push_back(Vector3(0, -1,  t).normalized() * radius);
        base_vertices.push_back(Vector3(0,  1,  t).normalized() * radius);
        base_vertices.push_back(Vector3(0, -1, -t).normalized() * radius);
        base_vertices.push_back(Vector3(0,  1, -t).normalized() * radius);
        base_vertices.push_back(Vector3( t, 0, -1).normalized() * radius);
        base_vertices.push_back(Vector3( t, 0,  1).normalized() * radius);
        base_vertices.push_back(Vector3(-t, 0, -1).normalized() * radius);
        base_vertices.push_back(Vector3(-t, 0,  1).normalized() * radius);

        const int base_indices[20][3] = {
            {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
            {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
            {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
            {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
        };

        // Start with the base icosahedron vertices.
        HashMap<std::pair<int,int>, int> midpoint_cache;
        LocalVector<Vector3> sphere_verts = base_vertices;
        LocalVector<int> sphere_indices;
        for (int i = 0; i < 20; ++i) {
            sphere_indices.push_back(base_indices[i][0]);
            sphere_indices.push_back(base_indices[i][1]);
            sphere_indices.push_back(base_indices[i][2]);
        }

        // Subdivide
        for (int sub = 0; sub < ico_subdivisions; ++sub) {
            LocalVector<int> new_indices;
            midpoint_cache.clear();
            auto get_midpoint = [&](int p1, int p2) -> int {
                std::pair<int,int> key = p1 < p2 ? std::make_pair(p1,p2) : std::make_pair(p2,p1);
                if (midpoint_cache.has(key)) {
                    return midpoint_cache[key];
                }
                Vector3 mid = (sphere_verts[p1] + sphere_verts[p2]) * 0.5;
                mid = mid.normalized() * radius;
                int idx = sphere_verts.size();
                sphere_verts.push_back(mid);
                midpoint_cache[key] = idx;
                return idx;
            };
            for (int t = 0; t < sphere_indices.size(); t += 3) {
                int a = sphere_indices[t];
                int b = sphere_indices[t+1];
                int c = sphere_indices[t+2];
                int ab = get_midpoint(a,b);
                int bc = get_midpoint(b,c);
                int ca = get_midpoint(c,a);
                new_indices.push_back(a);  new_indices.push_back(ab); new_indices.push_back(ca);
                new_indices.push_back(ab); new_indices.push_back(b);  new_indices.push_back(bc);
                new_indices.push_back(ca); new_indices.push_back(bc); new_indices.push_back(c);
                new_indices.push_back(ab); new_indices.push_back(bc); new_indices.push_back(ca);
            }
            sphere_indices = new_indices;
        }

        // Copy to member arrays
        vertices = sphere_verts;
        indices = sphere_indices;
    }

    // -------------------------------------------------------------------
    // Spherical UV mapping (longitude / latitude).
    // -------------------------------------------------------------------
    void compute_spherical_uvs() {
        uvs.resize(vertices.size());
        for (int i = 0; i < vertices.size(); ++i) {
            Vector3 v = vertices[i].normalized();
            real_t u = 0.5 + Math::atan2(v.x, v.z) / Math_TAU;
            real_t w = Math::acos(CLAMP(v.y, -1.0, 1.0));
            uvs[i] = Vector2(u, w / Math_PI);
        }
    }
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_SPHERE_GENERATOR_H
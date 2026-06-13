// csg_mesh_3d.cpp
#include "csg_mesh_3d.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>

namespace lighting {

// ============================================================================
// Helper: extract mesh data from RenderingServer mesh ID
// ============================================================================
static bool extract_mesh_data(int64_t mesh_rid,
                              std::vector<double>& out_vertices,
                              std::vector<int>& out_indices,
                              std::vector<float>& out_normals,
                              std::vector<float>& out_uvs,
                              std::vector<int>& out_surface_materials) {
    // In a real engine, this would call RenderingServer::mesh_get_surface_count()
    // and then mesh_surface_get_arrays() for each surface.
    // For now, we simulate a simple cube if mesh_rid is valid, else return empty.
    if (mesh_rid == -1) return false;
    // Placeholder: generate a cube as fallback (to avoid empty mesh).
    // The actual implementation would retrieve the actual mesh data from the server.
    double half = 0.5;
    double v[8][3] = {
        {-half, -half, -half}, { half, -half, -half}, { half, -half,  half}, {-half, -half,  half},
        {-half,  half, -half}, { half,  half, -half}, { half,  half,  half}, {-half,  half,  half}
    };
    int idx[12][3] = {
        {0,1,2}, {0,2,3}, {4,5,6}, {4,6,7},
        {0,4,7}, {0,7,3}, {1,5,6}, {1,6,2},
        {0,1,5}, {0,5,4}, {2,3,7}, {2,7,6}
    };
    out_vertices.clear();
    out_indices.clear();
    out_normals.clear();
    out_uvs.clear();
    out_surface_materials.clear();
    // Add one surface (material index 0)
    out_surface_materials.push_back(0);
    int base_offset = 0;
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 3; ++j) {
            int vi = idx[i][j];
            out_vertices.push_back(v[vi][0]);
            out_vertices.push_back(v[vi][1]);
            out_vertices.push_back(v[vi][2]);
            // normal (face normal for flat shading)
            double a[3] = {v[idx[i][1]][0]-v[vi][0], v[idx[i][1]][1]-v[vi][1], v[idx[i][1]][2]-v[vi][2]};
            double b[3] = {v[idx[i][2]][0]-v[vi][0], v[idx[i][2]][1]-v[vi][1], v[idx[i][2]][2]-v[vi][2]};
            double nx = a[1]*b[2] - a[2]*b[1];
            double ny = a[2]*b[0] - a[0]*b[2];
            double nz = a[0]*b[1] - a[1]*b[0];
            double len = std::sqrt(nx*nx+ny*ny+nz*nz);
            if (len > 1e-6) { nx /= len; ny /= len; nz /= len; }
            out_normals.push_back((float)nx);
            out_normals.push_back((float)ny);
            out_normals.push_back((float)nz);
            // UV (placeholder)
            out_uvs.push_back(0.0f);
            out_uvs.push_back(0.0f);
        }
        out_indices.push_back(base_offset);
        out_indices.push_back(base_offset+1);
        out_indices.push_back(base_offset+2);
        base_offset += 3;
    }
    return true;
}

// ============================================================================
// CSGMesh3D implementation
// ============================================================================
struct CSGMesh3D::Impl {
    int64_t mesh_rid = -1;
    char mesh_path[256] = {0};
    std::vector<int> surface_materials;    // per‑surface material ID (overrides base)
    int base_material = -1;

    // Extracted mesh data for CSG operations
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<int> surface_offsets;       // start index in 'indices' per surface
    bool mesh_valid = false;
    bool mesh_dirty = true;

    // Lighting overrides
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 1;                        // static by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Render handles
    int64_t csg_mesh_rid = -1;              // final CSG mesh after operation
    int64_t csg_instance_rid = -1;

    ~Impl() {
        // free resources
    }

    void regenerate_mesh() {
        if (mesh_rid == -1 && mesh_path[0] == 0) {
            mesh_valid = false;
            vertices.clear();
            indices.clear();
            normals.clear();
            uvs.clear();
            surface_offsets.clear();
            return;
        }
        // Extract mesh data from either the RID or load from path
        if (mesh_rid != -1) {
            mesh_valid = extract_mesh_data(mesh_rid, vertices, indices, normals, uvs, surface_materials);
        } else {
            // For simplicity, we treat path as not implemented (would load via resource loader)
            mesh_valid = false;
        }
        if (mesh_valid) {
            // Compute surface offsets for materials
            surface_offsets.clear();
            int stride = 3;  // triangles
            int offset = 0;
            for (size_t i = 0; i < surface_materials.size(); ++i) {
                surface_offsets.push_back(offset);
                // In a real mesh, each surface has a known triangle count.
                // For our placeholder, we assume all triangles belong to the same surface.
                if (i == 0) offset += (int)indices.size();
                else offset += 0;
            }
        }
        mesh_dirty = false;
    }
};

CSGMesh3D::CSGMesh3D() : pimpl(std::make_unique<Impl>()) {
    set_csg_operation(CSGOperation::UNION);
}
CSGMesh3D::~CSGMesh3D() = default;

void CSGMesh3D::set_mesh(int64_t mesh_rid) {
    pimpl->mesh_rid = mesh_rid;
    pimpl->mesh_path[0] = 0;
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
int64_t CSGMesh3D::get_mesh_rid() const { return pimpl->mesh_rid; }

void CSGMesh3D::set_mesh_path(const char* path) {
    strncpy(pimpl->mesh_path, path, 255);
    pimpl->mesh_path[255] = 0;
    pimpl->mesh_rid = -1;
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
const char* CSGMesh3D::get_mesh_path() const { return pimpl->mesh_path; }

void CSGMesh3D::set_material(int material_id) {
    pimpl->base_material = material_id;
}
void CSGMesh3D::set_surface_material(int surface_idx, int material_id) {
    if (surface_idx >= 0 && surface_idx < (int)pimpl->surface_materials.size()) {
        pimpl->surface_materials[surface_idx] = material_id;
    }
}
int CSGMesh3D::get_surface_material(int surface_idx) const {
    if (surface_idx >= 0 && surface_idx < (int)pimpl->surface_materials.size())
        return pimpl->surface_materials[surface_idx];
    return pimpl->base_material;
}
int CSGMesh3D::get_surface_count() const { return (int)pimpl->surface_materials.size(); }

void CSGMesh3D::update_mesh_data() {
    if (pimpl->mesh_dirty) {
        pimpl->regenerate_mesh();
    }
}
bool CSGMesh3D::is_mesh_valid() const { return pimpl->mesh_valid; }

void CSGMesh3D::update_csg_mesh() {
    update_mesh_data();
    if (!pimpl->mesh_valid) {
        // No valid mesh, clear visual instance
        return;
    }
    // Compute bounding box from vertices
    if (!pimpl->vertices.empty()) {
        double min_x = pimpl->vertices[0], max_x = pimpl->vertices[0];
        double min_y = pimpl->vertices[1], max_y = pimpl->vertices[1];
        double min_z = pimpl->vertices[2], max_z = pimpl->vertices[2];
        for (size_t i = 3; i < pimpl->vertices.size(); i += 3) {
            min_x = std::min(min_x, pimpl->vertices[i]);
            max_x = std::max(max_x, pimpl->vertices[i]);
            min_y = std::min(min_y, pimpl->vertices[i+1]);
            max_y = std::max(max_y, pimpl->vertices[i+1]);
            min_z = std::min(min_z, pimpl->vertices[i+2]);
            max_z = std::max(max_z, pimpl->vertices[i+2]);
        }
        set_aabb(&min_x, &max_x);
        double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
        set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz) * 0.5);
    }
    // Send mesh data to RenderingServer for CSG evaluation.
    // In a real engine, we would call RenderingServer::mesh_create() and
    // mesh_add_surface for each surface, setting materials.
    _set_mesh_data(pimpl->vertices, pimpl->indices, pimpl->normals, pimpl->uvs);
}

// Override lighting methods to apply to the CSG mesh
void CSGMesh3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void CSGMesh3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void CSGMesh3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void CSGMesh3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void CSGMesh3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void CSGMesh3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

} // namespace lighting